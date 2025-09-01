# src/rag/interfaces.py
from __future__ import annotations
from typing import Dict, List, Iterable, Optional, Protocol


# Protocol:
# acts as typechecker for chat models
# if you want to implement a new chat model, just make it conform to this protocol

class ChatModel(Protocol):
    """Minimal interface every chat-capable LLM adapter must implement."""

    def make_messages(
        self,
        user: str,
        system: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        ...

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        ...

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        ...
