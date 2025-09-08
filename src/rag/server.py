# src/rag/server.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Iterable
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

# method to check if the server is running
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/rag")
def rag(query: dict):
    q = query["q"]

    # 1) Retrieve
    hits = S.retriever.search(q)

    # 2) Kontext bauen (kurz halten!)
    context = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
    system = (
        "You are a concise, ERC-aligned training assistant. "
        "Answer with short, safe, step-by-step instructions and cite [1], [2] as needed."
    )
    user = f"Context:\n{context}\n\nQuestion:\n{q}"

    # 3) Messages korrekt aufbauen
    msgs = S.llm.make_messages(user=user, system=system)

    # 4) Streaming-Generator mit Fehlerfang
    def gen() -> Iterable[bytes]:
        try:
            for tok in S.llm.chat_stream(
                msgs, max_tokens=256, temperature=0.2
            ):
                # einzelne Tokens rausreichen
                yield tok.encode("utf-8")
        except Exception as e:
            # Fehler sauber signalisieren statt abrupt zu schließen
            err = f"\n\n[stream-error] {type(e).__name__}: {e}\n"
            yield err.encode("utf-8")

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

#@app.post("/rag")
#def rag(query: dict):
#    q = query["q"]
#    hits = S.retriever.search(q)
#    print(hits)
#    # prompt = build_prompt(q, hits)
#    prompt = q
#    # Stream für perceived latency
#    def gen() -> Iterable[bytes]:
#        for tok in S.llm.chat_stream(prompt, max_tokens=256, temperature=0.2):
#            yield tok.encode("utf-8")
#    return StreamingResponse(gen(), media_type="text/plain")



# server.py
@app.post("/rag_once")
def rag_once(query: dict):
    q = query["q"]
    hits = S.retriever.search(q)
    msgs = S.llm.make_messages(
        user=f"Context:\n" + "\n\n".join(f"[{i+1}] {d['text']}" for i,d in enumerate(hits))
             + f"\n\nQuestion:\n{q}",
        system="You are a concise, ERC-aligned training assistant.",
    )
    out = S.llm.chat(msgs, max_tokens=256, temperature=0.2)
    return {"answer": out}


