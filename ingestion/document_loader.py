"""
Stage 1b — Document Loader
---------------------------
Reads all PDFs from the data/ folder and extracts raw text.
Prints exactly what was extracted so you can verify before chunking.

Usage:
    python ingestion/document_loader.py
"""

from pathlib import Path
from pypdf import PdfReader
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


def load_document(filepath: Path) -> dict:
    """Extract text from a single PDF and return structured data."""
    reader = PdfReader(str(filepath))

    pages = []
    total_chars = 0

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        pages.append({
            "page_number": page_num + 1,
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split())
        })
        total_chars += len(text)

    return {
        "filename": filepath.name,
        "filepath": str(filepath),
        "page_count": len(reader.pages),
        "pages": pages,
        "total_chars": total_chars,
        "total_words": sum(p["word_count"] for p in pages),
        "full_text": "\n\n".join(p["text"] for p in pages)
    }


def load_all_documents(data_dir: Path) -> list[dict]:
    """Load all PDFs from the data directory."""
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        console.print("[red]❌ No PDF files found in data/ folder.[/red]")
        console.print("[dim]Run: python ingestion/generate_docs.py first[/dim]")
        return []

    documents = []
    for filepath in pdf_files:
        with console.status(f"Loading [cyan]{filepath.name}[/cyan]..."):
            doc = load_document(filepath)
            documents.append(doc)
        console.print(f"  ✅ Loaded [cyan]{filepath.name}[/cyan]")

    return documents


def print_summary(documents: list[dict]):
    """Print a summary table of all loaded documents."""
    table = Table(title="📂 Loaded Documents Summary", show_lines=True)
    table.add_column("File",        style="cyan")
    table.add_column("Pages",       style="white",  justify="center")
    table.add_column("Words",       style="green",  justify="right")
    table.add_column("Characters",  style="green",  justify="right")

    total_words = 0
    total_chars = 0

    for doc in documents:
        table.add_row(
            doc["filename"],
            str(doc["page_count"]),
            str(doc["total_words"]),
            str(doc["total_chars"])
        )
        total_words += doc["total_words"]
        total_chars += doc["total_chars"]

    table.add_row(
        "[bold]TOTAL[/bold]",
        str(sum(d["page_count"] for d in documents)),
        f"[bold]{total_words}[/bold]",
        f"[bold]{total_chars}[/bold]"
    )

    console.print(table)


def print_text_preview(documents: list[dict]):
    """Print a preview of extracted text for each document."""
    console.print("\n[bold yellow]📖 Extracted Text Preview (first 300 chars per doc)[/bold yellow]\n")

    for doc in documents:
        preview = doc["full_text"][:300].replace("\n", " ")
        console.print(Panel(
            f"[dim]{preview}...[/dim]",
            title=f"[cyan]{doc['filename']}[/cyan]",
            border_style="blue"
        ))
        console.print()


def main():
    console.rule("[bold cyan]Stage 1b — Document Loader[/bold cyan]")

    data_dir = Path(__file__).parent.parent / "data"
    console.print(f"\n📁 Loading from: [green]{data_dir}[/green]\n")

    # ── Load all documents ───────────────────────────────────────────
    documents = load_all_documents(data_dir)
    if not documents:
        return

    # ── Print summary table ──────────────────────────────────────────
    console.print()
    print_summary(documents)

    # ── Print text previews ──────────────────────────────────────────
    print_text_preview(documents)

    # ── Final stats ──────────────────────────────────────────────────
    total_words = sum(d["total_words"] for d in documents)
    total_chars = sum(d["total_chars"] for d in documents)

    console.print(f"[bold green]✅ {len(documents)} documents loaded successfully[/bold green]")
    console.print(f"   Total words     : [green]{total_words}[/green]")
    console.print(f"   Total characters: [green]{total_chars}[/green]")
    console.print(f"\n[dim]Next step → run: python ingestion/chunker.py[/dim]\n")

    return documents


if __name__ == "__main__":
    main()