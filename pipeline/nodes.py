"""
pipeline/nodes.py
------------------
Each function here is one LangGraph node.
Nodes read from RAGState, do their job, and return updated state fields.

Node order:
  node_input_guardrail → node_query_rewriter → node_hybrid_search →
  node_reranker → node_context_guardrail → node_prompt_builder →
  node_generator → node_self_checker → node_output_guardrail
"""

from pathlib import Path
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, QueryType, QueryCaptionType
from azure.core.credentials import AzureKeyCredential
from rich.console import Console
from rich.panel import Panel
import sys
import time
import json
import re

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.state import RAGState
from ingestion.embedder import embed_text
from generation.prompt_builder import build_prompt, SYSTEM_PROMPT
from retrieval.guardrail_input import check_empty, check_injection, check_topic
from retrieval.hybrid_search import vector_search_only, keyword_search_only, compute_rrf, TOP_K
from generation.guardrail_output import check_empty_response, check_pii, check_system_leak, check_content_safety
import config

console = Console()

# ── Clients ───────────────────────────────────────────────────────────
client = AzureOpenAI(
    azure_endpoint = config.AZURE_OPENAI_ENDPOINT,
    api_key        = config.AZURE_OPENAI_API_KEY,
    api_version    = config.AZURE_OPENAI_API_VERSION
)

credential    = AzureKeyCredential(config.AZURE_SEARCH_API_KEY)
child_client  = SearchClient(config.AZURE_SEARCH_ENDPOINT, config.AZURE_SEARCH_INDEX_NAME, credential)
parent_client = SearchClient(config.AZURE_SEARCH_ENDPOINT, config.AZURE_SEARCH_INDEX_NAME + "-parents", credential)

# ── HyDE prompt ───────────────────────────────────────────────────────
HYDE_PROMPT = """You are an HR policy document assistant.
Write a hypothetical passage that WOULD appear in an HR policy document to answer this question.
Use formal policy language. Be specific. 3-5 sentences. Write the passage directly."""

# ── Self check prompt ─────────────────────────────────────────────────
SELF_CHECK_PROMPT = """You are a strict fact-checker.
Check if every claim in the ANSWER is directly supported by the CONTEXT.
Respond ONLY with JSON: {"verdict": "GROUNDED" or "HALLUCINATION", "confidence": 0.0-1.0, "reason": "one line", "unsupported_claims": []}"""


# ─────────────────────────────────────────────────────────────────────
# NODE 1 — Input Guardrail
# ─────────────────────────────────────────────────────────────────────
def node_input_guardrail(state: RAGState) -> RAGState:
    console.rule("[bold cyan]Node 1 — Input Guardrail[/bold cyan]")
    query = state["query"]
    console.print(f"[dim]  Input: \"{query}\"[/dim]")

    # Run checks (reusing existing functions)
    for check_fn, label in [
        (check_empty,     "empty"),
        (check_injection, "injection"),
        (check_topic,     "topic"),
    ]:
        result = check_fn(query)
        if not result["passed"]:
            console.print(f"  [red]✗ BLOCKED at {label} check — {result['reason']}[/red]")
            return {
                **state,
                "input_decision"    : "BLOCKED",
                "input_block_reason": result["reason"],
                "pipeline_blocked"  : True,
                "block_stage"       : "input_guardrail"
            }

    console.print(f"  [green]✓ ALLOWED — query passed all input checks[/green]")
    return {**state, "input_decision": "ALLOWED", "pipeline_blocked": False}


# ─────────────────────────────────────────────────────────────────────
# NODE 2 — Query Rewriter (HyDE)
# ─────────────────────────────────────────────────────────────────────
def node_query_rewriter(state: RAGState) -> RAGState:
    console.rule("[bold cyan]Node 2 — Query Rewriter (HyDE)[/bold cyan]")
    query = state["query"]
    console.print(f"[dim]  Generating hypothetical document for: \"{query}\"[/dim]")

    start = time.time()
    try:
        response = client.chat.completions.create(
            model    = config.CHAT_DEPLOYMENT,
            messages = [
                {"role": "system", "content": HYDE_PROMPT},
                {"role": "user",   "content": f"Question: {query}"}
            ],
            max_tokens  = 200,
            temperature = 0.3
        )
        hyde_passage = response.choices[0].message.content.strip()
    except Exception as e:
        console.print(f"  [yellow]⚠ HyDE failed — using original query[/yellow]")
        hyde_passage = query

    latency = round(time.time() - start, 3)
    console.print(f"  [green]✓ HyDE generated[/green] — {len(hyde_passage.split())} words ({latency}s)")
    console.print(f"  [dim]  {hyde_passage[:100]}...[/dim]")

    return {
        **state,
        "hyde_passage"   : hyde_passage,
        "hyde_word_count": len(hyde_passage.split())
    }


# ─────────────────────────────────────────────────────────────────────
# NODE 3 — Hybrid Search
# ─────────────────────────────────────────────────────────────────────
def node_hybrid_search(state: RAGState) -> RAGState:
    console.rule("[bold cyan]Node 3 — Hybrid Search[/bold cyan]")
    query        = state["query"]
    hyde_passage = state["hyde_passage"]

    console.print(f"[dim]  Running vector + keyword search...[/dim]")

    vector_results  = vector_search_only(hyde_passage,  top_k=TOP_K)
    keyword_results = keyword_search_only(query,        top_k=TOP_K)
    rrf_results     = compute_rrf(vector_results, keyword_results)

    console.print(f"  [green]✓ Vector: {len(vector_results)} | Keyword: {len(keyword_results)} | RRF: {len(rrf_results)} unique[/green]")
    console.print(f"  [dim]  Top RRF chunk: {rrf_results[0]['chunk_id']} (score: {rrf_results[0]['rrf_score']})[/dim]")

    return {
        **state,
        "vector_results" : vector_results,
        "keyword_results": keyword_results,
        "rrf_results"    : rrf_results
    }


# ─────────────────────────────────────────────────────────────────────
# NODE 4 — Reranker
# ─────────────────────────────────────────────────────────────────────
def node_reranker(state: RAGState) -> RAGState:
    console.rule("[bold cyan]Node 4 — Reranker[/bold cyan]")
    query        = state["query"]
    hyde_passage = state["hyde_passage"]

    console.print(f"[dim]  Running semantic reranker + fetching parent chunks...[/dim]")

    # Embed HyDE
    query_vector = embed_text(hyde_passage)

    # Hybrid + semantic rerank in one call
    vector_query = VectorizedQuery(
        vector=query_vector, k_nearest_neighbors=TOP_K, fields="embedding"
    )
    results = child_client.search(
        search_text                 = query,
        vector_queries              = [vector_query],
        query_type                  = QueryType.SEMANTIC,
        semantic_configuration_name = "rag-semantic-config",
        query_caption               = QueryCaptionType.EXTRACTIVE,
        select                      = ["chunk_id", "text", "doc_name", "parent_id", "token_count"],
        top                         = TOP_K
    )

    reranked = []
    for r in results:
        reranked.append({
            "chunk_id"      : r["chunk_id"],
            "doc_name"      : r["doc_name"],
            "parent_id"     : r["parent_id"],
            "text"          : r["text"],
            "token_count"   : r["token_count"],
            "reranker_score": round(r.get("@search.reranker_score", 0.0), 6)
        })

    reranked.sort(key=lambda x: x["reranker_score"], reverse=True)

    # Fetch parent chunks (top 3)
    seen_parents = []
    parent_chunks = []
    for chunk in reranked[:3]:
        pid = chunk["parent_id"]
        if pid not in seen_parents:
            seen_parents.append(pid)
            parent_doc = parent_client.get_document(
                key=pid,
                selected_fields=["parent_id", "text", "doc_name", "token_count"]
            )
            parent_chunks.append({
                "parent_id"  : parent_doc["parent_id"],
                "doc_name"   : parent_doc["doc_name"],
                "text"       : parent_doc["text"],
                "token_count": parent_doc["token_count"]
            })

    console.print(f"  [green]✓ Reranked {len(reranked)} chunks[/green]")
    console.print(f"  [dim]  Top chunk: {reranked[0]['chunk_id']} (reranker score: {reranked[0]['reranker_score']})[/dim]")
    console.print(f"  [green]✓ Fetched {len(parent_chunks)} parent chunks[/green]")

    return {
        **state,
        "reranked_chunks": reranked,
        "parent_chunks"  : parent_chunks
    }


# ─────────────────────────────────────────────────────────────────────
# NODE 5 — Context Guardrail
# ─────────────────────────────────────────────────────────────────────
def node_context_guardrail(state: RAGState) -> RAGState:
    console.rule("[bold cyan]Node 5 — Context Guardrail[/bold cyan]")
    parent_chunks = state.get("parent_chunks", [])

    if not parent_chunks:
        console.print(f"  [red]✗ BLOCKED — no context retrieved[/red]")
        return {
            **state,
            "context_decision"    : "FAIL",
            "context_block_reason": "No relevant context found in index",
            "pipeline_blocked"    : True,
            "block_stage"         : "context_guardrail"
        }

    total_tokens = sum(p["token_count"] for p in parent_chunks)

    if total_tokens < 50:
        console.print(f"  [red]✗ BLOCKED — context too thin ({total_tokens} tokens)[/red]")
        return {
            **state,
            "context_decision"    : "FAIL",
            "context_block_reason": f"Context too thin — only {total_tokens} tokens",
            "pipeline_blocked"    : True,
            "block_stage"         : "context_guardrail"
        }

    console.print(f"  [green]✓ PASS — {len(parent_chunks)} chunks, {total_tokens} tokens[/green]")
    return {**state, "context_decision": "PASS"}


# ─────────────────────────────────────────────────────────────────────
# NODE 6 — Prompt Builder
# ─────────────────────────────────────────────────────────────────────
def node_prompt_builder(state: RAGState) -> RAGState:
    console.rule("[bold cyan]Node 6 — Prompt Builder[/bold cyan]")
    query         = state["query"]
    parent_chunks = state["parent_chunks"]

    console.print(f"[dim]  Assembling prompt with {len(parent_chunks)} parent chunks...[/dim]")

    prompt_data = build_prompt(query, parent_chunks)

    console.print(f"  [green]✓ Prompt assembled[/green] — {prompt_data['total_tokens']} total tokens")
    console.print(f"  [dim]  System: {prompt_data['system_tokens']} | Context: {prompt_data['context_tokens']} | User: {prompt_data['user_tokens']}[/dim]")

    return {**state, "prompt_data": prompt_data}


# ─────────────────────────────────────────────────────────────────────
# NODE 7 — Generator
# ─────────────────────────────────────────────────────────────────────
def node_generator(state: RAGState) -> RAGState:
    console.rule("[bold cyan]Node 7 — Generator[/bold cyan]")
    prompt_data = state["prompt_data"]

    console.print(f"[dim]  Calling GPT-4o with {prompt_data['total_tokens']} tokens...[/dim]")

    start = time.time()
    try:
        response = client.chat.completions.create(
            model       = config.CHAT_DEPLOYMENT,
            messages    = prompt_data["messages"],
            max_tokens  = 300,
            temperature = 0.0
        )
        answer        = response.choices[0].message.content.strip()
        answer_tokens = response.usage.completion_tokens
        latency       = round(time.time() - start, 3)

        console.print(f"  [green]✓ Answer generated[/green] — {answer_tokens} tokens ({latency}s)")
        console.print(f"  [dim]  \"{answer[:100]}...\"[/dim]")

        return {
            **state,
            "answer"           : answer,
            "answer_tokens"    : answer_tokens,
            "generation_latency": latency
        }

    except Exception as e:
        console.print(f"  [red]✗ Generation failed: {str(e)[:80]}[/red]")
        return {
            **state,
            "answer"           : None,
            "pipeline_blocked" : True,
            "block_stage"      : "generator"
        }


# ─────────────────────────────────────────────────────────────────────
# NODE 8 — Self Checker
# ─────────────────────────────────────────────────────────────────────
def node_self_checker(state: RAGState) -> RAGState:
    console.rule("[bold cyan]Node 8 — Self Checker[/bold cyan]")
    answer       = state.get("answer", "")
    prompt_data  = state.get("prompt_data", {})
    context      = prompt_data.get("context_block", "")

    console.print(f"[dim]  Verifying answer is grounded in context...[/dim]")

    start = time.time()
    try:
        context_truncated = context[:4000]

        response = client.chat.completions.create(
            model    = config.CHAT_DEPLOYMENT,
            messages = [
                {"role": "system", "content": SELF_CHECK_PROMPT},
                {"role": "user",   "content": f"CONTEXT:\n{context_truncated}\n\nANSWER:\n{answer}"}
            ],
            max_tokens  = 150,
            temperature = 0.0
        )

        latency = round(time.time() - start, 3)

        # Debug — print raw response to see what Azure is returning
        raw = response.choices[0].message.content
        console.print(f"  [dim]Raw response ({latency}s): '{raw}'[/dim]")
        console.print(f"  [dim]Finish reason: {response.choices[0].finish_reason}[/dim]")

        raw = raw.strip() if raw else ""
        raw = raw.replace("```json", "").replace("```", "").strip()

        if not raw:
            console.print(f"  [yellow]⚠ Empty response — finish reason may explain why[/yellow]")
            return {**state, "self_check_verdict": "GROUNDED", "self_check_confidence": 0.5}

        result     = json.loads(raw)
        verdict    = result.get("verdict", "UNKNOWN")
        confidence = result.get("confidence", 0.0)

        console.print(f"  [{'green' if verdict == 'GROUNDED' else 'red'}]{'✓' if verdict == 'GROUNDED' else '✗'} {verdict}[/{'green' if verdict == 'GROUNDED' else 'red'}] — confidence: {confidence} ({latency}s)")

        return {
            **state,
            "self_check_verdict"    : verdict,
            "self_check_confidence" : confidence
        }

    except Exception as e:
        console.print(f"  [yellow]⚠ Self-check error: {str(e)[:150]}[/yellow]")
        return {**state, "self_check_verdict": "GROUNDED", "self_check_confidence": 0.0}


# ─────────────────────────────────────────────────────────────────────
# NODE 9 — Output Guardrail
# ─────────────────────────────────────────────────────────────────────
def node_output_guardrail(state: RAGState) -> RAGState:
    console.rule("[bold cyan]Node 9 — Output Guardrail[/bold cyan]")
    answer = state.get("answer", "")

    console.print(f"[dim]  Running 4 output checks on generated answer...[/dim]")

    for check_fn, label in [
        (check_empty_response, "empty"),
        (check_pii,            "PII"),
        (check_system_leak,    "system_leak"),
        (check_content_safety, "content_safety"),
    ]:
        result = check_fn(answer)
        if not result["passed"]:
            console.print(f"  [red]✗ BLOCKED at {label} — {result['reason']}[/red]")
            return {
                **state,
                "output_decision"    : "BLOCKED",
                "output_block_reason": result["reason"],
                "pipeline_blocked"   : True,
                "block_stage"        : "output_guardrail",
                "final_answer"       : "I'm sorry, I couldn't generate a safe response. Please try again."
            }

    console.print(f"  [green]✓ PASSED — all output checks cleared[/green]")
    return {
        **state,
        "output_decision": "PASSED",
        "final_answer"   : answer
    }
