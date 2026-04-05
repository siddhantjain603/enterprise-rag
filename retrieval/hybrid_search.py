"""
Stage 2c — Hybrid Search
--------------------------
Runs three searches and shows you every score:
  1. Vector Search only      → cosine similarity scores
  2. Keyword Search only     → BM25 scores
  3. Hybrid Search (RRF)     → combined rank scores

Then shows a side-by-side rank comparison so you can see
exactly how RRF changes the ordering.

Flow:
  HyDE passage → embed → vector search  ──┐
  Original query          → keyword search ─┼→ RRF → unified ranked list
                                           ──┘

Usage:
    python retrieval/hybrid_search.py
"""

from pathlib import Path
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
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
credential    = AzureKeyCredential(config.AZURE_SEARCH_API_KEY)
search_client = SearchClient(
    endpoint   = config.AZURE_SEARCH_ENDPOINT,
    index_name = config.AZURE_SEARCH_INDEX_NAME,
    credential = credential
)

TOP_K       = 5     # how many results to retrieve from each search
RRF_K       = 60    # RRF smoothing constant (standard is 60)


def vector_search_only(hyde_passage: str, top_k: int = TOP_K) -> list[dict]:
    """
    Pure vector search — embed the HyDE passage and find
    nearest neighbors using cosine similarity.
    No keyword matching at all.
    """
    console.print(f"\n[dim]  → Embedding HyDE passage for vector search...[/dim]")
    start = time.time()
    query_vector = embed_text(hyde_passage)
    embed_latency = round(time.time() - start, 3)
    console.print(f"  [dim]Embedding done ({embed_latency}s) — vector has {len(query_vector)} dims[/dim]")

    console.print(f"  [dim]Running vector search (HNSW cosine similarity) top {top_k}...[/dim]")
    start = time.time()

    vector_query = VectorizedQuery(
        vector        = query_vector,
        k_nearest_neighbors = top_k,
        fields        = "embedding"
    )

    results = search_client.search(
        search_text   = None,       # ← None means pure vector, no keyword
        vector_queries= [vector_query],
        select        = ["chunk_id", "text", "doc_name", "parent_id", "token_count"],
        top           = top_k
    )

    search_latency = round(time.time() - start, 3)

    chunks = []
    for r in results:
        chunks.append({
            "chunk_id"  : r["chunk_id"],
            "doc_name"  : r["doc_name"],
            "parent_id" : r["parent_id"],
            "text"      : r["text"],
            "score"     : round(r["@search.score"], 6),
            "token_count": r["token_count"]
        })

    console.print(f"  [green]✓ Vector search done ({search_latency}s) — {len(chunks)} results[/green]")
    return chunks


def keyword_search_only(original_query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Pure BM25 keyword search — uses the ORIGINAL query (not HyDE).
    Good for exact term matches like policy names, numbers, IDs.
    """
    console.print(f"\n  [dim]→ Running keyword search (BM25) on original query top {top_k}...[/dim]")
    start = time.time()

    results = search_client.search(
        search_text   = original_query,   # ← plain text, BM25 scoring
        vector_queries= None,             # ← no vector
        select        = ["chunk_id", "text", "doc_name", "parent_id", "token_count"],
        top           = top_k
    )

    search_latency = round(time.time() - start, 3)

    chunks = []
    for r in results:
        chunks.append({
            "chunk_id"  : r["chunk_id"],
            "doc_name"  : r["doc_name"],
            "parent_id" : r["parent_id"],
            "text"      : r["text"],
            "score"     : round(r["@search.score"], 6),
            "token_count": r["token_count"]
        })

    console.print(f"  [green]✓ Keyword search done ({search_latency}s) — {len(chunks)} results[/green]")
    return chunks


def compute_rrf(
    vector_results  : list[dict],
    keyword_results : list[dict],
    k               : int = RRF_K
) -> list[dict]:
    """
    Reciprocal Rank Fusion — merges two ranked lists into one.

    Formula: RRF(d) = sum over lists of [ 1 / (k + rank(d)) ]

    Key insight: ignores raw scores entirely.
    Only the RANK of each document matters.
    A document appearing in BOTH lists gets a higher combined score.
    """
    console.print(f"\n  [dim]→ Computing RRF fusion (k={k})...[/dim]")
    console.print(f"  [dim]  Formula: RRF(chunk) = 1/(k + vector_rank) + 1/(k + keyword_rank)[/dim]")

    # Build rank maps: chunk_id → rank (1-indexed)
    vector_ranks  = {r["chunk_id"]: i + 1 for i, r in enumerate(vector_results)}
    keyword_ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(keyword_results)}

    # Collect all unique chunk IDs from both lists
    all_chunk_ids = set(vector_ranks.keys()) | set(keyword_ranks.keys())

    # Build a lookup for chunk metadata
    chunk_meta = {}
    for r in vector_results + keyword_results:
        chunk_meta[r["chunk_id"]] = r

    rrf_scores = []
    for chunk_id in all_chunk_ids:
        v_rank = vector_ranks.get(chunk_id, None)   # None if not in vector results
        k_rank = keyword_ranks.get(chunk_id, None)  # None if not in keyword results

        # RRF score — only add component if chunk appeared in that list
        rrf_score = 0.0
        if v_rank is not None:
            rrf_score += 1 / (k + v_rank)
        if k_rank is not None:
            rrf_score += 1 / (k + k_rank)

        meta = chunk_meta[chunk_id]
        rrf_scores.append({
            "chunk_id"      : chunk_id,
            "doc_name"      : meta["doc_name"],
            "parent_id"     : meta["parent_id"],
            "text"          : meta["text"],
            "token_count"   : meta["token_count"],
            "rrf_score"     : round(rrf_score, 6),
            "vector_rank"   : v_rank,
            "keyword_rank"  : k_rank,
            "in_both"       : v_rank is not None and k_rank is not None,
            # formula breakdown for logging
            "vector_contrib": round(1 / (k + v_rank), 6) if v_rank else 0.0,
            "keyword_contrib": round(1 / (k + k_rank), 6) if k_rank else 0.0,
        })

    # Sort by RRF score descending
    rrf_scores.sort(key=lambda x: x["rrf_score"], reverse=True)

    console.print(f"  [green]✓ RRF fusion complete — {len(rrf_scores)} unique chunks ranked[/green]")
    return rrf_scores


def print_vector_results(results: list[dict]):
    """Print vector search results with cosine scores."""
    console.print(f"\n[bold yellow]🔵 Vector Search Results (cosine similarity)[/bold yellow]\n")

    table = Table(show_lines=True, title="Vector Search — Top Results")
    table.add_column("Rank",        style="cyan",    justify="center")
    table.add_column("Chunk ID",    style="cyan",    no_wrap=True)
    table.add_column("Document",    style="white",   no_wrap=True)
    table.add_column("Score",       style="green",   justify="right")
    table.add_column("Tokens",      style="dim",     justify="center")
    table.add_column("Preview",     style="dim",     max_width=45)

    for i, r in enumerate(results):
        table.add_row(
            str(i + 1),
            r["chunk_id"],
            r["doc_name"],
            str(r["score"]),
            str(r["token_count"]),
            r["text"][:60] + "..."
        )

    console.print(table)
    console.print(f"  [dim]Score type: cosine similarity (0.0 to 1.0, higher = more similar)[/dim]")


def print_keyword_results(results: list[dict]):
    """Print keyword search results with BM25 scores."""
    console.print(f"\n[bold yellow]🟡 Keyword Search Results (BM25)[/bold yellow]\n")

    table = Table(show_lines=True, title="Keyword Search — Top Results")
    table.add_column("Rank",        style="cyan",    justify="center")
    table.add_column("Chunk ID",    style="cyan",    no_wrap=True)
    table.add_column("Document",    style="white",   no_wrap=True)
    table.add_column("Score",       style="yellow",  justify="right")
    table.add_column("Tokens",      style="dim",     justify="center")
    table.add_column("Preview",     style="dim",     max_width=45)

    for i, r in enumerate(results):
        table.add_row(
            str(i + 1),
            r["chunk_id"],
            r["doc_name"],
            str(r["score"]),
            str(r["token_count"]),
            r["text"][:60] + "..."
        )

    console.print(table)
    console.print(f"  [dim]Score type: BM25 (no upper bound — higher = more keyword overlap)[/dim]")
    console.print(f"  [dim]Note: BM25 and cosine scores are NOT comparable — different scales[/dim]")


def print_rrf_results(rrf_results: list[dict], top_k: int = TOP_K):
    """Print RRF fusion results with score breakdown."""
    console.print(f"\n[bold yellow]🟢 Hybrid Search Results (RRF Fusion)[/bold yellow]\n")

    table = Table(show_lines=True, title=f"RRF Fusion — Top {top_k} Results")
    table.add_column("RRF Rank",        style="bold cyan",  justify="center")
    table.add_column("Chunk ID",        style="cyan",       no_wrap=True)
    table.add_column("Document",        style="white",      no_wrap=True)
    table.add_column("RRF Score",       style="green",      justify="right")
    table.add_column("Vec Rank",        style="blue",       justify="center")
    table.add_column("KW Rank",         style="yellow",     justify="center")
    table.add_column("In Both?",        style="magenta",    justify="center")
    table.add_column("Preview",         style="dim",        max_width=35)

    for i, r in enumerate(rrf_results[:top_k]):
        in_both = "✅ Yes" if r["in_both"] else "❌ No"
        v_rank  = str(r["vector_rank"])  if r["vector_rank"]  else "—"
        k_rank  = str(r["keyword_rank"]) if r["keyword_rank"] else "—"

        table.add_row(
            str(i + 1),
            r["chunk_id"],
            r["doc_name"],
            str(r["rrf_score"]),
            v_rank,
            k_rank,
            in_both,
            r["text"][:50] + "..."
        )

    console.print(table)

    # Print RRF formula breakdown for top 3
    console.print(f"\n[bold yellow]🔢 RRF Formula Breakdown (top 3)[/bold yellow]\n")
    for i, r in enumerate(rrf_results[:3]):
        v_part = f"1/(60+{r['vector_rank']})={r['vector_contrib']}" if r["vector_rank"] else "not in vector results"
        k_part = f"1/(60+{r['keyword_rank']})={r['keyword_contrib']}" if r["keyword_rank"] else "not in keyword results"
        console.print(
            f"  [cyan]{r['chunk_id']}[/cyan]: "
            f"[blue]{v_part}[/blue] + "
            f"[yellow]{k_part}[/yellow] = "
            f"[bold green]{r['rrf_score']}[/bold green]"
        )


def print_rank_comparison(
    vector_results  : list[dict],
    keyword_results : list[dict],
    rrf_results     : list[dict],
    top_k           : int = TOP_K
):
    """
    Side-by-side comparison showing how ranks changed after RRF.
    This is the most important table — shows the value of fusion.
    """
    console.print(f"\n[bold yellow]📊 Rank Comparison — Before vs After RRF[/bold yellow]\n")

    # Build rank lookups
    v_rank_map = {r["chunk_id"]: i + 1 for i, r in enumerate(vector_results)}
    k_rank_map = {r["chunk_id"]: i + 1 for i, r in enumerate(keyword_results)}

    table = Table(show_lines=True, title="How Rankings Changed After RRF Fusion")
    table.add_column("Chunk ID",        style="cyan",       no_wrap=True)
    table.add_column("Document",        style="white",      no_wrap=True)
    table.add_column("Vector Rank",     style="blue",       justify="center")
    table.add_column("Keyword Rank",    style="yellow",     justify="center")
    table.add_column("RRF Rank",        style="bold green", justify="center")
    table.add_column("Movement",        style="magenta",    justify="center")

    for i, r in enumerate(rrf_results[:top_k]):
        rrf_rank = i + 1
        v_rank   = v_rank_map.get(r["chunk_id"], "—")
        k_rank   = k_rank_map.get(r["chunk_id"], "—")

        # Calculate movement from vector rank to RRF rank
        if isinstance(v_rank, int):
            diff = v_rank - rrf_rank
            if diff > 0:
                movement = f"⬆️  +{diff} (from vec)"
            elif diff < 0:
                movement = f"⬇️  {diff} (from vec)"
            else:
                movement = "➡️  same"
        else:
            movement = "🆕 new entry"

        table.add_row(
            r["chunk_id"],
            r["doc_name"],
            str(v_rank),
            str(k_rank),
            str(rrf_rank),
            movement
        )

    console.print(table)


def run_hybrid_search(query: str, use_hyde: bool = True) -> list[dict]:
    """
    Full hybrid search pipeline:
    1. Rewrite query using HyDE
    2. Vector search with HyDE embedding
    3. Keyword search with original query
    4. RRF fusion
    5. Print all results
    """
    console.rule("[bold cyan]Stage 2c — Hybrid Search[/bold cyan]")
    console.print(f"\n[bold]🔍 Query:[/bold] [yellow]\"{query}\"[/yellow]\n")

    # ── Step 1: HyDE rewrite ─────────────────────────────────────────
    console.print("[bold yellow]Step 1: Query Rewriting (HyDE)[/bold yellow]")
    rewrite_result  = rewrite_query(query)
    hyde_passage    = rewrite_result["rewritten_query"]
    original_query  = rewrite_result["original_query"]
    console.print()

    # ── Step 2: Vector search ────────────────────────────────────────
    console.print("[bold yellow]Step 2: Vector Search[/bold yellow]")
    vector_results = vector_search_only(hyde_passage, top_k=TOP_K)
    print_vector_results(vector_results)

    # ── Step 3: Keyword search ───────────────────────────────────────
    console.print("\n[bold yellow]Step 3: Keyword Search (BM25)[/bold yellow]")
    keyword_results = keyword_search_only(original_query, top_k=TOP_K)
    print_keyword_results(keyword_results)

    # ── Step 4: RRF fusion ───────────────────────────────────────────
    console.print("\n[bold yellow]Step 4: RRF Fusion[/bold yellow]")
    rrf_results = compute_rrf(vector_results, keyword_results)
    print_rrf_results(rrf_results, top_k=TOP_K)

    # ── Step 5: Rank comparison ──────────────────────────────────────
    print_rank_comparison(vector_results, keyword_results, rrf_results)

    # ── Final output ─────────────────────────────────────────────────
    top_results = rrf_results[:TOP_K]
    parent_ids  = list(dict.fromkeys(r["parent_id"] for r in top_results))

    console.print(Panel(
        f"[bold green]✅ Hybrid Search Complete[/bold green]\n"
        f"Vector results   : {len(vector_results)} chunks\n"
        f"Keyword results  : {len(keyword_results)} chunks\n"
        f"After RRF        : {len(rrf_results)} unique chunks ranked\n"
        f"Top {TOP_K} selected  : {[r['chunk_id'] for r in top_results]}\n"
        f"Parent IDs       : {parent_ids}\n"
        f"Next             : Reranker will re-score these {TOP_K} chunks",
        title="Hybrid Search Output",
        border_style="green"
    ))

    return top_results


if __name__ == "__main__":
    query = "How many days of leave can I carry forward?"
    run_hybrid_search(query)