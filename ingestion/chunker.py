"""
Stage 1c — Chunker
-------------------
Splits extracted text into Child chunks (for searching)
and Parent chunks (for LLM context).

Prints every chunk with token count, word count, and overlap verification.

Usage:
    python ingestion/chunker.py
"""

from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import tiktoken
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))
from ingestion.document_loader import load_all_documents

console = Console()

# ── Chunking config ──────────────────────────────────────────────────
CHILD_CHUNK_SIZE    = 200   # tokens  — small, for precise searching
CHILD_CHUNK_OVERLAP = 50    # tokens  — overlap between child chunks
PARENT_CHUNK_SIZE   = 600   # tokens  — large, for LLM context

ENCODING = tiktoken.get_encoding("cl100k_base")  # same as text-embedding-3-small


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by token count."""
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = ENCODING.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return [c for c in chunks if c]


def build_child_chunks(documents: list[dict]) -> list[dict]:
    """Create child chunks from all documents."""
    child_chunks = []
    chunk_id = 0

    for doc in documents:
        doc_name = doc["filename"].replace(".pdf", "")
        chunks = chunk_text(doc["full_text"], CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            child_chunks.append({
                "chunk_id"      : f"child_{chunk_id:04d}",
                "doc_name"      : doc_name,
                "chunk_index"   : i,
                "text"          : chunk,
                "token_count"   : count_tokens(chunk),
                "word_count"    : len(chunk.split()),
                "char_count"    : len(chunk),
            })
            chunk_id += 1

    return child_chunks


def build_parent_chunks(documents: list[dict]) -> list[dict]:
    """Create parent chunks from all documents."""
    parent_chunks = []
    parent_id = 0

    for doc in documents:
        doc_name = doc["filename"].replace(".pdf", "")
        chunks = chunk_text(doc["full_text"], PARENT_CHUNK_SIZE, overlap=0)

        for i, chunk in enumerate(chunks):
            parent_chunks.append({
                "parent_id"     : f"parent_{parent_id:04d}",
                "doc_name"      : doc_name,
                "chunk_index"   : i,
                "text"          : chunk,
                "token_count"   : count_tokens(chunk),
                "word_count"    : len(chunk.split()),
                "char_count"    : len(chunk),
            })
            parent_id += 1

    return parent_chunks


def link_children_to_parents(
    child_chunks: list[dict],
    parent_chunks: list[dict]
) -> list[dict]:
    """
    Assign each child chunk a parent_id.
    Logic: child belongs to the parent from the same doc
    whose token range best contains the child's position.
    """
    # Build a lookup: doc_name -> list of parents in order
    parent_lookup: dict[str, list[dict]] = {}
    for p in parent_chunks:
        parent_lookup.setdefault(p["doc_name"], []).append(p)

    # Track cumulative token offset per doc to find which parent a child falls in
    doc_token_offset: dict[str, int] = {}

    for child in child_chunks:
        doc = child["doc_name"]
        parents_for_doc = parent_lookup.get(doc, [])

        # Find cumulative offset for this doc
        if doc not in doc_token_offset:
            doc_token_offset[doc] = 0

        offset = doc_token_offset[doc]

        # Find which parent contains this offset
        cumulative = 0
        assigned_parent = parents_for_doc[0]["parent_id"] if parents_for_doc else "unknown"

        for parent in parents_for_doc:
            cumulative += parent["token_count"]
            if offset < cumulative:
                assigned_parent = parent["parent_id"]
                break

        child["parent_id"] = assigned_parent
        doc_token_offset[doc] = offset + (child["token_count"] - CHILD_CHUNK_OVERLAP)

    return child_chunks


def print_chunk_config():
    """Print chunking configuration."""
    console.print("\n[bold yellow]⚙️  Chunking Configuration[/bold yellow]")
    console.print(f"   Child chunk size    : [cyan]{CHILD_CHUNK_SIZE} tokens[/cyan]")
    console.print(f"   Child chunk overlap : [cyan]{CHILD_CHUNK_OVERLAP} tokens[/cyan]")
    console.print(f"   Parent chunk size   : [cyan]{PARENT_CHUNK_SIZE} tokens[/cyan]")
    console.print(f"   Tokenizer           : [cyan]cl100k_base (same as text-embedding-3-small)[/cyan]\n")


def print_child_chunks(child_chunks: list[dict]):
    """Print all child chunks with details."""
    console.print("\n[bold yellow]🔹 Child Chunks (used for SEARCHING)[/bold yellow]\n")

    table = Table(show_lines=True, title="Child Chunks")
    table.add_column("ID",          style="cyan",   no_wrap=True)
    table.add_column("Document",    style="white",  no_wrap=True)
    table.add_column("Tokens",      style="green",  justify="right")
    table.add_column("Words",       style="green",  justify="right")
    table.add_column("Parent ID",   style="magenta",no_wrap=True)
    table.add_column("Preview",     style="dim",    max_width=50)

    for chunk in child_chunks:
        table.add_row(
            chunk["chunk_id"],
            chunk["doc_name"],
            str(chunk["token_count"]),
            str(chunk["word_count"]),
            chunk["parent_id"],
            chunk["text"][:80] + "..."
        )

    console.print(table)


def print_parent_chunks(parent_chunks: list[dict]):
    """Print all parent chunks with details."""
    console.print("\n[bold yellow]🔶 Parent Chunks (sent to LLM)[/bold yellow]\n")

    table = Table(show_lines=True, title="Parent Chunks")
    table.add_column("ID",          style="magenta",no_wrap=True)
    table.add_column("Document",    style="white",  no_wrap=True)
    table.add_column("Tokens",      style="green",  justify="right")
    table.add_column("Words",       style="green",  justify="right")
    table.add_column("Preview",     style="dim",    max_width=60)

    for chunk in parent_chunks:
        table.add_row(
            chunk["parent_id"],
            chunk["doc_name"],
            str(chunk["token_count"]),
            str(chunk["word_count"]),
            chunk["text"][:100] + "..."
        )

    console.print(table)


def verify_overlap(child_chunks: list[dict]):
    """
    Verify overlap between consecutive child chunks from the same doc.
    Prints last N words of chunk[i] and first N words of chunk[i+1].
    """
    console.print("\n[bold yellow]🔍 Overlap Verification (consecutive chunks, same document)[/bold yellow]\n")

    verified = 0
    for i in range(len(child_chunks) - 1):
        curr = child_chunks[i]
        nxt  = child_chunks[i + 1]

        if curr["doc_name"] != nxt["doc_name"]:
            continue

        curr_words = curr["text"].split()
        nxt_words  = nxt["text"].split()

        last_10  = " ".join(curr_words[-10:])
        first_10 = " ".join(nxt_words[:10])

        console.print(f"[cyan]{curr['chunk_id']}[/cyan] → [cyan]{nxt['chunk_id']}[/cyan]  ([dim]{curr['doc_name']}[/dim])")
        console.print(f"  End of chunk   : [green]...{last_10}[/green]")
        console.print(f"  Start of next  : [yellow]{first_10}...[/yellow]")

        # Check how many words overlap
        overlap_count = 0
        for w in curr_words[-20:]:
            if w in nxt_words[:20]:
                overlap_count += 1

        console.print(f"  Overlap words  : [bold]~{overlap_count} words shared ✅[/bold]\n")
        verified += 1

        if verified >= 3:  # show first 3 pairs only
            console.print("[dim]  ... (showing first 3 pairs only)[/dim]\n")
            break


def print_summary(child_chunks: list[dict], parent_chunks: list[dict]):
    """Print final summary stats."""
    console.print("\n[bold yellow]📊 Chunking Summary[/bold yellow]\n")

    avg_child_tokens  = sum(c["token_count"] for c in child_chunks) / len(child_chunks)
    avg_parent_tokens = sum(p["token_count"] for p in parent_chunks) / len(parent_chunks)

    # Count children per parent
    parent_child_map: dict[str, int] = {}
    for c in child_chunks:
        parent_child_map[c["parent_id"]] = parent_child_map.get(c["parent_id"], 0) + 1

    avg_children_per_parent = sum(parent_child_map.values()) / len(parent_child_map)

    table = Table(show_lines=True, title="Final Stats")
    table.add_column("Metric",  style="white")
    table.add_column("Value",   style="cyan", justify="right")

    table.add_row("Total child chunks",             str(len(child_chunks)))
    table.add_row("Total parent chunks",            str(len(parent_chunks)))
    table.add_row("Avg child chunk size",           f"{avg_child_tokens:.1f} tokens")
    table.add_row("Avg parent chunk size",          f"{avg_parent_tokens:.1f} tokens")
    table.add_row("Avg children per parent",        f"{avg_children_per_parent:.1f}")
    table.add_row("Child chunk size config",        f"{CHILD_CHUNK_SIZE} tokens")
    table.add_row("Child overlap config",           f"{CHILD_CHUNK_OVERLAP} tokens")
    table.add_row("Parent chunk size config",       f"{PARENT_CHUNK_SIZE} tokens")

    console.print(table)


def main():
    console.rule("[bold cyan]Stage 1c — Chunker[/bold cyan]")

    # ── Load documents ───────────────────────────────────────────────
    data_dir = Path(__file__).parent.parent / "data"
    console.print(f"\n📁 Loading documents from: [green]{data_dir}[/green]")
    documents = load_all_documents(data_dir)
    if not documents:
        return

    # ── Print config ─────────────────────────────────────────────────
    print_chunk_config()

    # ── Build chunks ─────────────────────────────────────────────────
    console.print("[dim]Building child chunks...[/dim]")
    child_chunks = build_child_chunks(documents)

    console.print("[dim]Building parent chunks...[/dim]")
    parent_chunks = build_parent_chunks(documents)

    console.print("[dim]Linking children to parents...[/dim]\n")
    child_chunks = link_children_to_parents(child_chunks, parent_chunks)

    # ── Print everything ─────────────────────────────────────────────
    print_child_chunks(child_chunks)
    print_parent_chunks(parent_chunks)
    verify_overlap(child_chunks)
    print_summary(child_chunks, parent_chunks)

    console.print(f"\n[bold green]✅ Chunking complete![/bold green]")
    console.print(f"   Child chunks : [cyan]{len(child_chunks)}[/cyan] (will be embedded + indexed)")
    console.print(f"   Parent chunks: [cyan]{len(parent_chunks)}[/cyan] (will be stored for LLM context)")
    console.print(f"\n[dim]Next step → run: python ingestion/embedder.py[/dim]\n")

    return child_chunks, parent_chunks


if __name__ == "__main__":
    main()