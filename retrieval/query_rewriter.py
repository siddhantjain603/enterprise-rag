"""
Stage 2b — Query Rewriter (HyDE)
----------------------------------
HyDE = Hypothetical Document Embeddings

Problem it solves:
  User queries are short (5-10 words).
  Policy documents are long (100-200 words per section).
  Embedding a short query and searching against long chunks
  creates a mismatch in vector space — poor retrieval results.

Solution:
  Use GPT-4o to generate a hypothetical "ideal answer" first.
  Embed THAT instead of the raw query.
  A fake-but-plausible answer looks much more like a real document
  in vector space → better retrieval matches.

Flow:
  Original query → GPT-4o → Hypothetical answer → embed → search

Usage:
    python retrieval/query_rewriter.py
"""

from pathlib import Path
from openai import AzureOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import sys
import time
import json

sys.path.append(str(Path(__file__).parent.parent))
import config

console = Console()

# ── Azure OpenAI client ──────────────────────────────────────────────
client = AzureOpenAI(
    azure_endpoint = config.AZURE_OPENAI_ENDPOINT,
    api_key        = config.AZURE_OPENAI_API_KEY,
    api_version    = config.AZURE_OPENAI_API_VERSION
)

# ── HyDE prompt ──────────────────────────────────────────────────────
HYDE_PROMPT = """You are an HR policy document assistant.

A user has asked a question about company policy. Your job is to write
a hypothetical passage that WOULD appear in an HR policy document
and would answer this question.

Rules:
- Write as if you are the policy document itself
- Use formal policy language
- Be specific with numbers, days, percentages where relevant
- Keep it to 3-5 sentences
- Do NOT say "I don't know" — always write a plausible policy passage
- Do NOT prefix with "According to policy" or similar — just write the passage directly

This hypothetical passage will be used for document retrieval only, not shown to the user."""


def rewrite_query(query: str) -> dict:
    """
    Takes a raw user query and generates a hypothetical document
    using HyDE technique. Returns both original and rewritten query
    along with metadata for verification.
    """
    console.rule("[bold cyan]Stage 2b — Query Rewriter (HyDE)[/bold cyan]")
    console.print(f"\n[bold]📥 Original query:[/bold] [yellow]\"{query}\"[/yellow]")
    console.print(f"[dim]Technique: HyDE — Hypothetical Document Embeddings[/dim]\n")

    # ── Why HyDE — log the reasoning ────────────────────────────────
    console.print("[dim]  → Why rewrite? Short query vs long document mismatch in vector space[/dim]")
    console.print(f"[dim]     Query length    : {len(query.split())} words[/dim]")
    console.print(f"[dim]     Avg chunk length: ~190 words[/dim]")
    console.print(f"[dim]     Gap             : {190 - len(query.split())} words — HyDE bridges this[/dim]\n")

    # ── Generate hypothetical document ───────────────────────────────
    console.print("[dim]  → Sending to GPT-4o to generate hypothetical policy passage...[/dim]")

    start = time.time()

    try:
        response = client.chat.completions.create(
            model=config.CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": HYDE_PROMPT},
                {"role": "user",   "content": f"Question: {query}"}
            ],
            max_tokens=200,
            temperature=0.3     # slight creativity to generate plausible passage
        )
    except Exception as e:
        console.print(f"  [red]✗ GPT-4o call failed: {str(e)[:80]}[/red]")
        console.print(f"  [yellow]⚠ Falling back to original query[/yellow]")
        return {
            "original_query"    : query,
            "rewritten_query"   : query,
            "hyde_passage"      : None,
            "fallback"          : True,
            "latency"           : 0
        }

    latency = round(time.time() - start, 3)
    hyde_passage = response.choices[0].message.content.strip()

    # ── Log the generated passage ────────────────────────────────────
    console.print(f"\n[bold yellow]📄 Hypothetical Document Generated ({latency}s):[/bold yellow]")
    console.print(Panel(
        f"[dim]{hyde_passage}[/dim]",
        title="HyDE Passage (used for embedding, not shown to user)",
        border_style="blue"
    ))

    # ── Stats comparison ─────────────────────────────────────────────
    original_words  = len(query.split())
    hyde_words      = len(hyde_passage.split())
    original_chars  = len(query)
    hyde_chars      = len(hyde_passage)

    console.print(f"\n[bold yellow]📊 Query vs HyDE Passage Comparison:[/bold yellow]\n")

    table = Table(show_lines=True)
    table.add_column("Metric",          style="white")
    table.add_column("Original Query",  style="cyan",   justify="right")
    table.add_column("HyDE Passage",    style="green",  justify="right")
    table.add_column("Improvement",     style="yellow", justify="right")

    table.add_row(
        "Word count",
        str(original_words),
        str(hyde_words),
        f"+{hyde_words - original_words} words"
    )
    table.add_row(
        "Character count",
        str(original_chars),
        str(hyde_chars),
        f"+{hyde_chars - original_chars} chars"
    )
    table.add_row(
        "Similarity to chunks",
        "Low (short query)",
        "High (document-like)",
        "↑ Better retrieval"
    )
    table.add_row(
        "Used for embedding",
        "❌ No",
        "✅ Yes",
        "—"
    )

    console.print(table)

    # ── What gets embedded ───────────────────────────────────────────
    console.print(f"\n[dim]  → What gets embedded and sent to Azure AI Search:[/dim]")
    console.print(f"  [green]HyDE passage[/green] (not the original query)")
    console.print(f"  [dim]  Original query is preserved separately for display and reranking[/dim]")

    console.print(Panel(
        f"[bold green]✅ Query rewriting complete[/bold green]\n"
        f"Original  : {query}\n"
        f"HyDE words: {hyde_words} words\n"
        f"Latency   : {latency}s\n"
        f"Next      : Embed HyDE passage → Hybrid Search",
        title="Rewriter Output",
        border_style="green"
    ))

    return {
        "original_query"    : query,
        "rewritten_query"   : hyde_passage,
        "hyde_passage"      : hyde_passage,
        "original_words"    : original_words,
        "hyde_words"        : hyde_words,
        "latency"           : latency,
        "fallback"          : False
    }


def run_test_suite():
    """Test HyDE rewriting on multiple queries."""

    test_queries = [
        "How many days of leave can I carry forward?",
        "What is the probationary period for new employees?",
        "What happens if I accept a gift from a vendor?",
    ]

    console.rule("[bold cyan]Stage 2b — Query Rewriter Test Suite[/bold cyan]")
    console.print(f"[dim]Testing HyDE on {len(test_queries)} queries...[/dim]\n")

    results = []
    for query in test_queries:
        result = rewrite_query(query)
        results.append(result)
        console.print()

    # ── Summary table ────────────────────────────────────────────────
    console.rule("[bold cyan]HyDE Rewriting Summary[/bold cyan]")

    table = Table(show_lines=True, title="Query Rewriting Results")
    table.add_column("Original Query",      style="cyan",   max_width=35)
    table.add_column("Original Words",      style="white",  justify="center")
    table.add_column("HyDE Words",          style="green",  justify="center")
    table.add_column("Word Expansion",      style="yellow", justify="center")
    table.add_column("Latency",             style="dim",    justify="center")
    table.add_column("Fallback?",           style="dim",    justify="center")

    for r in results:
        table.add_row(
            r["original_query"][:35] + "..." if len(r["original_query"]) > 35 else r["original_query"],
            str(r.get("original_words", "—")),
            str(r.get("hyde_words", "—")),
            f"+{r.get('hyde_words', 0) - r.get('original_words', 0)} words" if not r["fallback"] else "N/A",
            f"{r['latency']}s",
            "Yes ⚠️" if r["fallback"] else "No ✅"
        )

    console.print(table)
    console.print(f"\n[bold green]✅ HyDE rewriting complete for all {len(test_queries)} queries[/bold green]")
    console.print(f"[dim]Next step → run: python retrieval/hybrid_search.py[/dim]\n")

    return results


if __name__ == "__main__":
    run_test_suite()