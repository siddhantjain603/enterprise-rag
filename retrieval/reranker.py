"""
Stage 2d — Reranker
---------------------
Takes the top chunks from hybrid search (RRF output)
and re-scores them using Azure's Semantic Reranker.

Why reranker after RRF?
  RRF is fast but uses approximate scoring.
  The semantic reranker reads query + chunk TOGETHER
  like a human would — much more accurate relevance scoring.
  Tradeoff: slower, so we only run it on top 5 from RRF,
  not all chunks in the index.

Flow:
  RRF top 5 chunks → Semantic Reranker → re-scored + re-ordered
                                       → fetch parent chunks
                                       → send to LLM

Usage:
    python retrieval/reranker.py
"""

from pathlib import Path
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, QueryType, QueryCaptionType
from azure.core.credentials import AzureKeyCredential
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))
from ingestion.embedder import embed_text
from retrieval.query_rewriter import rewrite_query
import config

console = Console()

# ── Clients ───────────────────────────────────────────────────────────
credential = AzureKeyCredential(config.AZURE_SEARCH_API_KEY)

child_client = SearchClient(
    endpoint   = config.AZURE_SEARCH_ENDPOINT,
    index_name = config.AZURE_SEARCH_INDEX_NAME,
    credential = credential
)

parent_client = SearchClient(
    endpoint   = config.AZURE_SEARCH_ENDPOINT,
    index_name = config.AZURE_SEARCH_INDEX_NAME + "-parents",
    credential = credential
)

TOP_K_HYBRID   = 5     # chunks going INTO reranker
TOP_K_FINAL    = 3     # chunks coming OUT of reranker


def run_semantic_rerank(original_query: str, hyde_passage: str) -> list[dict]:
    """
    Run hybrid search + semantic reranking in a single Azure AI Search call.

    Azure's semantic reranker:
    - Takes the top N results from hybrid search
    - Runs a cross-encoder model that reads query + chunk together
    - Produces a @search.reranker_score (0.0 to 4.0)
    - Re-orders results by this new score
    """
    console.print(f"\n[dim]  → Embedding HyDE passage for vector component...[/dim]")
    start = time.time()
    query_vector = embed_text(hyde_passage)
    embed_latency = round(time.time() - start, 3)
    console.print(f"  [dim]  Embedding done ({embed_latency}s) — {len(query_vector)} dims[/dim]")

    console.print(f"\n[dim]  → Running hybrid search + semantic reranking in one call...[/dim]")
    console.print(f"  [dim]  query_type='semantic' activates the cross-encoder reranker[/dim]")
    console.print(f"  [dim]  semantic_configuration_name='rag-semantic-config' (set in indexer)[/dim]")

    start = time.time()

    vector_query = VectorizedQuery(
        vector              = query_vector,
        k_nearest_neighbors = TOP_K_HYBRID,
        fields              = "embedding"
    )

    results = child_client.search(
        search_text                 = original_query,   # BM25 on original query
        vector_queries              = [vector_query],   # vector on HyDE
        query_type                  = QueryType.SEMANTIC,              # ← activates reranker
        semantic_configuration_name = "rag-semantic-config",           # ← config from indexer
        query_caption               = QueryCaptionType.EXTRACTIVE,     # ← extracts key phrases
        select                      = ["chunk_id", "text", "doc_name", "parent_id", "token_count"],
        top                         = TOP_K_HYBRID
    )

    latency = round(time.time() - start, 3)

    chunks = []
    for r in results:
        chunks.append({
            "chunk_id"       : r["chunk_id"],
            "doc_name"       : r["doc_name"],
            "parent_id"      : r["parent_id"],
            "text"           : r["text"],
            "token_count"    : r["token_count"],
            "hybrid_score"   : round(r["@search.score"], 6),
            "reranker_score" : round(r["@search.reranker_score"], 6) if r.get("@search.reranker_score") else 0.0,
            "captions"       : [c.text for c in r.get("@search.captions", []) if c.text]
        })

    console.print(f"  [green]✓ Reranking done ({latency}s) — {len(chunks)} chunks scored[/green]")
    return chunks


def fetch_parent_chunks(reranked_chunks: list[dict]) -> list[dict]:
    """
    After reranking, fetch parent text for top chunks.
    Deduplicate parent IDs — multiple children can share a parent.
    """
    console.print(f"\n[dim]  → Fetching parent chunks for top {TOP_K_FINAL} reranked results...[/dim]")

    # Get unique parent IDs from top results only
    seen_parents = []
    for chunk in reranked_chunks[:TOP_K_FINAL]:
        pid = chunk["parent_id"]
        if pid not in seen_parents:
            seen_parents.append(pid)

    console.print(f"  [dim]  Unique parent IDs: {seen_parents}[/dim]")
    console.print(f"  [dim]  (deduplicated — multiple children can point to same parent)[/dim]")

    parents = []
    for pid in seen_parents:
        parent_doc = parent_client.get_document(
            key             = pid,
            selected_fields = ["parent_id", "text", "doc_name", "token_count"]
        )
        parents.append({
            "parent_id"  : parent_doc["parent_id"],
            "doc_name"   : parent_doc["doc_name"],
            "text"       : parent_doc["text"],
            "token_count": parent_doc["token_count"]
        })
        console.print(f"  [green]✓ Fetched[/green] [magenta]{pid}[/magenta] — {parent_doc['token_count']} tokens")

    return parents


def print_before_reranking(chunks: list[dict]):
    """Print chunks ordered by hybrid score (before reranking)."""
    console.print(f"\n[bold yellow]📋 Before Reranking — Hybrid Score Order[/bold yellow]\n")

    sorted_by_hybrid = sorted(chunks, key=lambda x: x["hybrid_score"], reverse=True)

    table = Table(show_lines=True, title="Before Reranking (Hybrid Score)")
    table.add_column("Rank",            style="cyan",   justify="center")
    table.add_column("Chunk ID",        style="cyan",   no_wrap=True)
    table.add_column("Document",        style="white",  no_wrap=True)
    table.add_column("Hybrid Score",    style="yellow", justify="right")
    table.add_column("Preview",         style="dim",    max_width=50)

    for i, chunk in enumerate(sorted_by_hybrid):
        table.add_row(
            str(i + 1),
            chunk["chunk_id"],
            chunk["doc_name"],
            str(chunk["hybrid_score"]),
            chunk["text"][:70] + "..."
        )

    console.print(table)


def print_after_reranking(chunks: list[dict]):
    """Print chunks ordered by reranker score (after reranking)."""
    console.print(f"\n[bold yellow]🎯 After Reranking — Semantic Reranker Score Order[/bold yellow]\n")

    sorted_by_reranker = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)

    table = Table(show_lines=True, title="After Reranking (Semantic Reranker Score)")
    table.add_column("Rank",                style="bold cyan",  justify="center")
    table.add_column("Chunk ID",            style="cyan",       no_wrap=True)
    table.add_column("Document",            style="white",      no_wrap=True)
    table.add_column("Reranker Score",      style="green",      justify="right")
    table.add_column("Preview",             style="dim",        max_width=50)

    for i, chunk in enumerate(sorted_by_reranker):
        table.add_row(
            str(i + 1),
            chunk["chunk_id"],
            chunk["doc_name"],
            str(chunk["reranker_score"]),
            chunk["text"][:70] + "..."
        )

    console.print(table)
    console.print(f"  [dim]Score type: semantic reranker (0.0 to 4.0 — higher = more relevant)[/dim]")

    return sorted_by_reranker


def print_rank_movement(chunks: list[dict]):
    """
    Show how each chunk's rank changed from hybrid to reranker.
    This is the key verification — did reranking improve ordering?
    """
    console.print(f"\n[bold yellow]📊 Rank Movement — Hybrid vs Reranker[/bold yellow]\n")

    hybrid_order   = sorted(chunks, key=lambda x: x["hybrid_score"],   reverse=True)
    reranker_order = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)

    hybrid_ranks   = {c["chunk_id"]: i + 1 for i, c in enumerate(hybrid_order)}
    reranker_ranks = {c["chunk_id"]: i + 1 for i, c in enumerate(reranker_order)}

    table = Table(show_lines=True, title="Rank Movement After Semantic Reranking")
    table.add_column("Chunk ID",        style="cyan",       no_wrap=True)
    table.add_column("Document",        style="white",      no_wrap=True)
    table.add_column("Hybrid Rank",     style="yellow",     justify="center")
    table.add_column("Reranker Rank",   style="bold green", justify="center")
    table.add_column("Movement",        style="magenta",    justify="center")
    table.add_column("Reranker Score",  style="green",      justify="right")

    for chunk in reranker_order:
        cid         = chunk["chunk_id"]
        h_rank      = hybrid_ranks[cid]
        r_rank      = reranker_ranks[cid]
        diff        = h_rank - r_rank

        if diff > 0:
            movement = f"⬆️  +{diff}"
        elif diff < 0:
            movement = f"⬇️  {diff}"
        else:
            movement = "➡️  same"

        table.add_row(
            cid,
            chunk["doc_name"],
            str(h_rank),
            str(r_rank),
            movement,
            str(chunk["reranker_score"])
        )

    console.print(table)


def print_parent_context(parents: list[dict]):
    """Print the final parent chunks that will be sent to LLM."""
    console.print(f"\n[bold yellow]🔶 Parent Chunks Fetched — This Goes to LLM[/bold yellow]\n")

    for i, parent in enumerate(parents):
        console.print(Panel(
            f"[dim]{parent['text']}[/dim]",
            title=f"[magenta]{parent['parent_id']}[/magenta] | "
                  f"[cyan]{parent['doc_name']}[/cyan] | "
                  f"[green]{parent['token_count']} tokens[/green]",
            border_style="magenta"
        ))
        console.print()


def run_reranker(query: str) -> dict:
    """Full reranking pipeline for a single query."""

    console.rule("[bold cyan]Stage 2d — Reranker[/bold cyan]")
    console.print(f"\n[bold]🔍 Query:[/bold] [yellow]\"{query}\"[/yellow]\n")

    # ── Step 1: HyDE rewrite ─────────────────────────────────────────
    console.print("[bold yellow]Step 1: HyDE Query Rewrite[/bold yellow]")
    rewrite_result = rewrite_query(query)
    hyde_passage   = rewrite_result["rewritten_query"]
    console.print()

    # ── Step 2: Hybrid + rerank in one call ──────────────────────────
    console.print("[bold yellow]Step 2: Hybrid Search + Semantic Reranking[/bold yellow]")
    reranked_chunks = run_semantic_rerank(query, hyde_passage)

    # ── Step 3: Show before/after ────────────────────────────────────
    print_before_reranking(reranked_chunks)
    final_order = print_after_reranking(reranked_chunks)
    print_rank_movement(reranked_chunks)

    # ── Step 4: Fetch parent chunks ───────────────────────────────────
    console.print("\n[bold yellow]Step 3: Fetch Parent Chunks[/bold yellow]")
    parents = fetch_parent_chunks(final_order)

    # ── Step 5: Show parent context ───────────────────────────────────
    print_parent_context(parents)

    # ── Summary ───────────────────────────────────────────────────────
    top_final    = final_order[:TOP_K_FINAL]
    total_tokens = sum(p["token_count"] for p in parents)

    console.print(Panel(
        f"[bold green]✅ Reranking Complete[/bold green]\n"
        f"Chunks into reranker  : {len(reranked_chunks)}\n"
        f"Chunks after reranker : {TOP_K_FINAL} selected\n"
        f"Top chunks            : {[c['chunk_id'] for c in top_final]}\n"
        f"Parent chunks fetched : {len(parents)}\n"
        f"Total context tokens  : {total_tokens}\n"
        f"Next                  : Send parent text to GPT-4o",
        title="Reranker Output",
        border_style="green"
    ))

    return {
        "query"          : query,
        "reranked_chunks": final_order,
        "parents"        : parents,
        "total_tokens"   : total_tokens
    }


if __name__ == "__main__":
    query = "How many days of leave can I carry forward?"
    run_reranker(query)