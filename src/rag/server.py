# src/rag/server.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Iterable
import orjson
import yaml
from contextlib import asynccontextmanager

from rag.generator import LocalLLM, LLMConfig
from rag.embed import SBertEmbeddings
from rag.indexer import HnswIndex
from rag.retriever import Retriever
# from rag.prompt import build_prompt  # deine baseline-Funktion


class State:
    llm: LocalLLM | None = None
    embedder: SBertEmbeddings | None = None
    index: HnswIndex | None = None
    retriever: Retriever | None = None

def load_llm_config(path: str) -> LLMConfig:
    """Read YAML and construct LLMConfig."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm_cfg = cfg.get("llm", {})
    return LLMConfig(**llm_cfg)

S = State()

yaml_path = "configs/rag.yaml"

@asynccontextmanager
async def lifespan(app: FastAPI):
    #_startup()
    # 1) LLM warm
 
    cfg = load_llm_config(yaml_path)

    S.llm = LocalLLM(cfg)

    # 2) Embedder warm (CPU zuerst; MPS optional testen)
    S.embedder = SBertEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",  # oder "mps" – s.u.
    )
    # 3) Index warm
    S.index = HnswIndex()
    S.index.load("data/index")

    # 4) Retriever
    S.retriever = Retriever(S.embedder, S.index, k=5)
    yield
    #_shutdown()


#app.add_event_handler("startup", _startup)
#app.add_event_handler("shutdown", _shutdown)

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/rag")
def rag(query: dict):
    q = query["q"]
    hits = S.retriever.search(q)
    # prompt = build_prompt(q, hits)
    prormpt = "Put shit on a stick. Now!"
    # Stream für perceived latency
    def gen() -> Iterable[bytes]:
        for tok in S.llm.chat_stream(prompt, max_tokens=256, temperature=0.2):
            yield tok.encode("utf-8")
    return StreamingResponse(gen(), media_type="text/plain")






def _startup():
    # 1) LLM warm
    cfg = LLMConfig.from_yaml("configs/rag.yaml")
    S.llm = LocalLLM(cfg)

    # 2) Embedder warm (CPU zuerst; MPS optional testen)
    S.embedder = SBertEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",  # oder "mps" – s.u.
        batch_size=1
    )
    # 3) Index warm
    S.index = HnswIndex()
    S.index.load("data/index")

    # 4) Retriever
    S.retriever = Retriever(S.embedder, S.index, k=5)
