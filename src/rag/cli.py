# src/rag/cli.py
from __future__ import annotations
import typer
from rag.generator import simple_answer, load_llm_config, LocalLLM

app = typer.Typer(help="RAG CLI")



@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Prompt to send to the local LLM"),
    config: str = typer.Option("configs/rag.yaml", "--config", "-c"),
):
    system = ("You are a concise, safe training assistant. "
              "Give step-by-step ERC-aligned instructions.")
    out = simple_answer(question, system, config)
    typer.echo(out)

@app.command("llm-sanity")
def llm_sanity(
    question: str = typer.Argument(...),
    config: str = typer.Option("configs/rag.yaml", "--config", "-c"),
):
    system = ("You are a concise, safe training assistant. "
              "Give step-by-step ERC-aligned instructions.")
    out = simple_answer(question, system, config)
    typer.echo(out)

@app.command("llm-stream")
def llm_stream(
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
