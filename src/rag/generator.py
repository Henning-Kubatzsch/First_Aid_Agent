# Wrap llama_cpp.Llama; builds prompt + generates test (temperature, max_tokens)

from __future__ import annotations

import os
from typing import Dict, List, Iterable, Optional, Any
from dataclasses import dataclass

import yaml
from llama_cpp import Llama

# ---- 1) Small dataclass to hold LLM config ----
@dataclass
class LLMConfig:
    model_path: str
    family: str = "qwen2"                # qwen2 | llama3 | phi3 | mistral
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_threads: Optional[int] = None
    seed: int = 42
    temperature: float = 0.2
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    max_tokens: int = 512
    stop: Optional[List[str]] = None

# ---- 2) Load YAML and extract LLM config ----
def load_llm_config(path: str) -> LLMConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)                     # cfg: python dictionary
    llm_cfg = cfg.get("llm", {})                    # get key "llm" from the dictionary
    return LLMConfig(**llm_cfg)

# ---- 3) Map "family" to llama.cpp chat_format + default stop tokens ----
def family_to_chat_format_and_stops(family: str) -> Dict[str, Any]:
    family = family.lower()                         # converts all chars to lowercase
    if family in ("qwen", "qwen2", "qwen2.5"):
        return {"chat_format": "qwen2", "extra_stops": ["<|im_end|>", "<|endoftext|>"]}
    if family in ("llama3", "llama-3"):
        return {"chat_format": "llama-3", "extra_stops": ["<|eot_id|>", "<|end_of_text|>"]}
    if family in ("phi3", "phi-3", "phi3-mini"):
        return {"chat_format": "phi3", "extra_stops": ["<|end|>", "<|endoftext|>"]}
    if family in ("mistral", "mistral-instruct"):
        return {"chat_format": "mistral-instruct", "extra_stops": ["</s>"]}
    # fallback: raw completion mode (not recommended)
    return {"chat_format": None, "extra_stops": []}

# ---- 4) The LocalLLM class: constructs the llama and runs chat ----
class LocalLLM:
    def __init__(self, cfg: LLMConfig):
        assert os.path.exists(cfg.model_path), f"Model not found: {cfg.model_path}"
        mapping = family_to_chat_format_and_stops(cfg.family)

        # Merge user-provided stops with family defaults
        stops = list(mapping["extra_stops"])
        if cfg.stop:
            stops.extend(s for s in cfg.stop if s not in stops)

        # llama.cpp model handle (loads weights into RAM; Metal offload if enabled)
        self.llama = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads or os.cpu_count(),
            n_gpu_layers=cfg.n_gpu_layers,          # -1 = try to offload all layers to GPU (Metal)
            seed=cfg.seed,
            logits_all=False,                       # saves memory; we only need final tokens
            verbose=False,                          # log all llama.cpp infos (Layer, Offload, Tokenization ...)
            chat_format=mapping["chat_format"],     # selects the built-in chat template
        )

        self.cfg = cfg
        self.stop = stops
        self.chat_format = mapping["chat_format"]

    # Build messages in OpenAI-style schema so llama.cpp can format them
    def make_messages(
        self,
        user: str,
        system: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        return messages

    # Streaming generator: yields tokens as they arrive
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        params = dict(
            temperature=temperature if temperature is not None else self.cfg.temperature,
            top_p=top_p if top_p is not None else self.cfg.top_p,
            max_tokens=max_tokens if max_tokens is not None else self.cfg.max_tokens,
            repeat_penalty=self.cfg.repeat_penalty,
            stop=self.stop,
            stream=True,  # <-- important
        )

        # llama.cpp auto-applies the correct prompt template if chat_format is set
        stream = self.llama.create_chat_completion(messages=messages, **params)
        for part in stream:
            # The event stream yields incremental deltas
            delta = part["choices"][0]["delta"].get("content", "")
            if delta:
                yield delta

    # Non-streaming call: returns the full string at once
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        params = dict(
            temperature=temperature if temperature is not None else self.cfg.temperature,
            top_p=top_p if top_p is not None else self.cfg.top_p,
            max_tokens=max_tokens if max_tokens is not None else self.cfg.max_tokens,
            repeat_penalty=self.cfg.repeat_penalty,
            stop=self.stop,
            stream=False,
        )
        out = self.llama.create_chat_completion(messages=messages, **params)
        return out["choices"][0]["message"]["content"]

# ---- 5) Convenience helpers for your pipeline ----
def simple_answer(question: str, system: Optional[str], cfg_path: str) -> str:
    cfg = load_llm_config(cfg_path)
    llm = LocalLLM(cfg)
    messages = llm.make_messages(user=question, system=system)
    return llm.chat(messages)
