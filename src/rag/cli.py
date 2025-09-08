# src/rag/cli.py
from __future__ import annotations
import typer, json, os
from rag.generator import simple_answer, load_llm_config, LocalLLM
from rag.embed import SBertEmbeddings
from rag.indexer import HnswIndex
from rag.retriever import Retriever
import time

app = typer.Typer(help="RAG CLI")

# ===============================
# Commands for document retrieval
# ===============================

@app.command("ingest")
def ingest(
    docs_dir: str = typer.Option("data/docs/en", "--docs", help="Folder with ERC .txt/.md"),
    out_dir: str  = typer.Option("data/index", "--out", help="Where to store HNSW index"),
    embed_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--embed-model"),
):
    from scripts.build_index import build_erc_index
    build_erc_index(docs_dir=docs_dir, out_dir=out_dir, model_name=embed_model)
    typer.echo(f"Index written to {out_dir}")

@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="User question"),
    config: str = typer.Option("configs/rag.yaml", "--config", "-c"),
    index_dir: str = typer.Option("data/index", "--index"),
    embed_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--embed-model"),
    k: int = typer.Option(2, "--k"),
):
    t0 = time.perf_counter()    
    # 1) Load LLM
    llm_cfg = load_llm_config(config)
    llm = LocalLLM(llm_cfg)
    print(f"---- Loaded LLM in {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()    


    # 2) Load embedder + index
    embedder = SBertEmbeddings(model_name=embed_model)
    # dim = len(embedder.embed_one("probe"))
    index = HnswIndex()
    index.load(index_dir)  # uses header stored during build

    print(f"---- Loaded Index in {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()    


    # 3) Retrieve
    retriever = Retriever(embedder, index, k=k)
    docs = retriever.search(question)

    print(f"---- Retrieving done in {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()    


    print(docs)

    # 4) Build prompt with context and ask LLM
    context = "\n\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
    system = "You are a concise, ERC-aligned training assistant. Answer with short, safe, step-by-step instructions and cite [1], [2] as needed."
    user = f"Context:\n{context}\n\nQuestion:\n{question}"

    print(f"---- Prompt built in {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()    
    
    # \n\nIf unsure, say you are unsure."

    msgs = llm.make_messages(user=user, system=system)
    answer = llm.chat(msgs, max_tokens=llm_cfg.max_tokens)

    print(f"---- LLM response in {time.perf_counter() - t0:.2f} seconds")

    print(answer)


@app.command("llm-stream")
def llm_stream(
    question: str = typer.Argument(..., help="User question"),
    config: str = typer.Option("configs/rag.yaml", "--config", "-c"),
    index_dir: str = typer.Option("data/index", "--index"),
    embed_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--embed-model"),
    k: int = typer.Option(2, "--k"),
):
    
    print("\n")

    t0 = time.perf_counter()    
    # 1) Load LLM
    llm_cfg = load_llm_config(config)
    llm = LocalLLM(llm_cfg)
    print(f"---- Loaded LLM in {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()    


    # 2) Load embedder + index
    embedder = SBertEmbeddings(model_name=embed_model)
    dim = len(embedder.embed_one("probe"))

    index = HnswIndex()
    index.load(index_dir)  # uses header stored during build

    print(f"---- Loaded Index in {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()    


    # 3) Retrieve
    retriever = Retriever(embedder, index, k=k)
    docs = retriever.search(question)

    print(f"---- Retrieving done in {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()    


    # print(docs)

    # 4) Build prompt with context and ask LLM
    context = "\n\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
    # system = "You are a concise, ERC-aligned training assistant. Answer with short, safe, step-by-step instructions."
    # system = "You are a concise, ERC-aligned training assistant. Answer the users question given the context provided."
    # system = "Du bist ein knapper, ERC-konformer Erste-Hilfe-Microcoach. Antworte in 1–2 kurzen Sätzen in Alltagssprache. Gib genau EINE konkrete Handlung, keine Aufzählungen, keine Überschriften, keine Zitate/Quellen/Links. Bei Lebensgefahr beginne mit „Notfall:“ und nenne zuerst den wichtigsten Schritt. Keine Diagnosen, keine Medikamentenhinweise. Beende JEDE Antwort exakt mit: „Wenn du das gemacht hast, sag Bescheid — ich gebe dir den nächsten Schritt."
    system = "give short answers"

    user = f"Context:\n{context}\n\nQuestion:\n{question}"

    print(f"---- Prompt built in {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()    
    print("\n")

    # \n\nIf unsure, say you are unsure."

    msgs = llm.make_messages(user=user, system=system)

    for tok in llm.chat_stream(msgs):
        print(tok, end="", flush=True)
    
    
# ======================================
# Direct commands to LLM -> no retrieval
# ======================================

# using simple_answer creates model each time
@app.command("ask-no-retrieval")
def ask(
    question: str = typer.Argument(..., help="Prompt to send to the local LLM"),
    config: str = typer.Option("configs/rag.yaml", "--config", "-c"),
):
    system = ("You are a concise, safe training assistant. "
              "Give step-by-step ERC-aligned instructions.")
    out = simple_answer(question, system, config)
    typer.echo(out)

# using simple_answer creates model each time
@app.command("llm-sanity-no-retrieval")
def llm_sanity(
    question: str = typer.Argument(...),
    config: str = typer.Option("configs/rag.yaml", "--config", "-c"),
):
    system = ("You are a concise, safe training assistant. "
              "Give step-by-step ERC-aligned instructions.")
    out = simple_answer(question, system, config)
    typer.echo(out)


# create model then stream tokens to stdout
@app.command("llm-stream-no-retrieval")
def llm_stream_simple(
    question: str = typer.Argument(...),
    config: str = typer.Option("configs/rag.yaml", "--config", "-c"),
):
    cfg = load_llm_config(config)
    llm = LocalLLM(cfg)
    msgs = llm.make_messages(
        user=question,
        system="You are a concise ERC training assistant.",
    )
    for tok in llm.chat_stream(msgs):
        print(tok, end="", flush=True)
    print()

if __name__ == "__main__":
    app()
