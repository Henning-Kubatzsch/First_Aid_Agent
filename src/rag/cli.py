# cli: command line interface
# ingest --path --out
# ask "question" (uses pipeline.ask_once)

# src/rag/cli.py (append this command)
import typer
from rag.generator import simple_answer

app = typer.Typer(help="RAG CLI")

@app.command()      # decorator - defines a command for the CLI
def llm_sanity(
    question: str = typer.Argument(..., help="Prompt to send to the local LLM"),
    config: str = typer.Option("configs/rag.yaml", "--config", "-c"),
):
    """
    Quick check: loads the local GGUF via llama.cpp and prints the answer.
    """
    system = (
        "You are a concise, safe training assistant. "
        "Give step-by-step ERC-aligned instructions. If unsure, say so."
    )
    out = simple_answer(question, system, config)
    typer.echo(out)

# keep your other commands (ingest, ask, etc.)
if __name__ == "__main__":
    app()
