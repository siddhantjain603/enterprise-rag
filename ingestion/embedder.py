"""
Stage 1d — Embedder
--------------------
Generates embeddings for each child chunk using Azure OpenAI.
Prints vector dimensions, sample values, and cost estimate.

NOTE: Only child chunks are embedded — parent chunks are stored as plain text.
      Parents are fetched by ID at query time, never searched by vector.

Usage:
    python ingestion/embedder.py
"""

from pathlib import Path
from openai import AzureOpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))
from ingestion.document_loader import load_all_documents
from ingestion.chunker import build_child_chunks, build_parent_chunks, link_children_to_parents
import config

console = Console()

# ── Azure OpenAI client ──────────────────────────────────────────────
client = AzureOpenAI(
    azure_endpoint = config.AZURE_OPENAI_ENDPOINT,
    api_key        = config.AZURE_OPENAI_API_KEY,
    api_version    = config.AZURE_OPENAI_API_VERSION
)


def embed_text(text: str) -> list[float]:
    """Generate embedding for a single text string."""
    response = client.embeddings.create(
        input=text,
        model=config.EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding


def embed_child_chunks(child_chunks: list[dict]) -> list[dict]:
    """Embed all child chunks and attach vector to each."""
    console.print(f"\n[bold yellow]🔢 Generating Embeddings for {len(child_chunks)} Child Chunks[/bold yellow]\n")
    console.print(f"   Model      : [cyan]{config.EMBEDDING_DEPLOYMENT}[/cyan]")
    console.print(f"   Endpoint   : [cyan]{config.AZURE_OPENAI_ENDPOINT}[/cyan]\n")

    for i, chunk in enumerate(child_chunks):
        with console.status(f"Embedding chunk [cyan]{chunk['chunk_id']}[/cyan] ({i+1}/{len(child_chunks)})..."):
            start = time.time()
            embedding = embed_text(chunk["text"])
            latency = round(time.time() - start, 3)

        chunk["embedding"]          = embedding
        chunk["embedding_dim"]      = len(embedding)
        chunk["embedding_latency"]  = latency

        console.print(
            f"  ✅ [cyan]{chunk['chunk_id']}[/cyan] | "
            f"dim: [green]{len(embedding)}[/green] | "
            f"latency: [yellow]{latency}s[/yellow]"
        )

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    return child_chunks


def print_vector_preview(child_chunks: list[dict]):
    """Print first 5 vector values for each chunk so you can see the actual numbers."""
    console.print("\n[bold yellow]🔍 Vector Preview (first 5 dimensions of each chunk)[/bold yellow]\n")

    table = Table(show_lines=True, title="Embedding Vectors — Sample Values")
    table.add_column("Chunk ID",    style="cyan",   no_wrap=True)
    table.add_column("Document",    style="white",  no_wrap=True)
    table.add_column("Dimensions",  style="green",  justify="center")
    table.add_column("First 5 Values",              style="dim")
    table.add_column("Text Preview",style="dim",    max_width=40)

    for chunk in child_chunks:
        first_5 = [round(v, 6) for v in chunk["embedding"][:5]]
        table.add_row(
            chunk["chunk_id"],
            chunk["doc_name"],
            str(chunk["embedding_dim"]),
            str(first_5),
            chunk["text"][:60] + "..."
        )

    console.print(table)


def print_embedding_stats(child_chunks: list[dict]):
    """Print embedding statistics."""
    console.print("\n[bold yellow]📊 Embedding Stats[/bold yellow]\n")

    total_dims      = child_chunks[0]["embedding_dim"]
    avg_latency     = sum(c["embedding_latency"] for c in child_chunks) / len(child_chunks)
    total_latency   = sum(c["embedding_latency"] for c in child_chunks)
    total_tokens    = sum(c["token_count"] for c in child_chunks)

    # Cost estimate: text-embedding-3-small = $0.00002 per 1K tokens
    cost_per_1k     = 0.00002
    estimated_cost  = (total_tokens / 1000) * cost_per_1k

    table = Table(show_lines=True, title="Embedding Summary")
    table.add_column("Metric",  style="white")
    table.add_column("Value",   style="cyan", justify="right")

    table.add_row("Embedding model",            config.EMBEDDING_DEPLOYMENT)
    table.add_row("Vector dimensions",          str(total_dims))
    table.add_row("Total chunks embedded",      str(len(child_chunks)))
    table.add_row("Total tokens processed",     str(total_tokens))
    table.add_row("Avg embedding latency",      f"{avg_latency:.3f}s")
    table.add_row("Total embedding time",       f"{total_latency:.3f}s")
    table.add_row("Estimated cost (USD)",       f"${estimated_cost:.6f}")

    console.print(table)

    # Important note about parents
    console.print(Panel(
        "[yellow]Parent chunks are NOT embedded.[/yellow]\n"
        "They are stored as plain text and fetched by parent_id at query time.\n"
        "Only child chunks live in the vector index.",
        title="ℹ️  Why Only Child Chunks?",
        border_style="blue"
    ))


def main():
    console.rule("[bold cyan]Stage 1d — Embedder[/bold cyan]")

    # ── Load and chunk documents ─────────────────────────────────────
    data_dir = Path(__file__).parent.parent / "data"
    console.print(f"\n📁 Loading documents from: [green]{data_dir}[/green]")
    documents = load_all_documents(data_dir)
    if not documents:
        return

    child_chunks  = build_child_chunks(documents)
    parent_chunks = build_parent_chunks(documents)
    child_chunks  = link_children_to_parents(child_chunks, parent_chunks)

    console.print(f"\n   Child chunks to embed : [cyan]{len(child_chunks)}[/cyan]")
    console.print(f"   Parent chunks (no embed): [cyan]{len(parent_chunks)}[/cyan]\n")

    # ── Generate embeddings ──────────────────────────────────────────
    child_chunks = embed_child_chunks(child_chunks)

    # ── Print results ────────────────────────────────────────────────
    print_vector_preview(child_chunks)
    print_embedding_stats(child_chunks)

    console.print(f"\n[bold green]✅ Embeddings generated for all {len(child_chunks)} child chunks[/bold green]")
    console.print(f"[dim]Next step → run: python ingestion/indexer.py[/dim]\n")

    return child_chunks, parent_chunks


if __name__ == "__main__":
    main()