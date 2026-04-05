"""
pipeline/state.py
------------------
Defines the shared state object that flows through every LangGraph node.

Think of this as the "memory" of the pipeline —
each node reads from it and writes back to it.
The graph passes this state from node to node automatically.
"""

from typing import TypedDict, Optional


class RAGState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    Each node reads what it needs and adds its output.

    Flow:
    input_guardrail → query_rewriter → hybrid_search →
    reranker → context_guardrail → prompt_builder →
    generator → self_checker → output_guardrail → END
    """

    # ── Input ────────────────────────────────────────────────────────
    query               : str               # original user query

    # ── Guardrail 1 (input) ──────────────────────────────────────────
    input_decision      : Optional[str]     # ALLOWED or BLOCKED
    input_block_reason  : Optional[str]     # why it was blocked

    # ── Query rewriter ───────────────────────────────────────────────
    hyde_passage        : Optional[str]     # hypothetical document
    hyde_word_count     : Optional[int]     # words in HyDE passage

    # ── Hybrid search ────────────────────────────────────────────────
    vector_results      : Optional[list]    # top K from vector search
    keyword_results     : Optional[list]    # top K from BM25
    rrf_results         : Optional[list]    # merged RRF ranked list

    # ── Reranker ─────────────────────────────────────────────────────
    reranked_chunks     : Optional[list]    # chunks after semantic rerank
    parent_chunks       : Optional[list]    # fetched parent texts

    # ── Context guardrail ────────────────────────────────────────────
    context_decision    : Optional[str]     # PASS or FAIL
    context_block_reason: Optional[str]

    # ── Prompt builder ───────────────────────────────────────────────
    prompt_data         : Optional[dict]    # assembled prompt + token counts

    # ── Generator ────────────────────────────────────────────────────
    answer              : Optional[str]     # GPT-4o raw answer
    answer_tokens       : Optional[int]
    generation_latency  : Optional[float]

    # ── Self checker ─────────────────────────────────────────────────
    self_check_verdict  : Optional[str]     # GROUNDED or HALLUCINATION
    self_check_confidence: Optional[float]

    # ── Output guardrail ─────────────────────────────────────────────
    output_decision     : Optional[str]     # PASSED or BLOCKED
    output_block_reason : Optional[str]

    # ── Final ────────────────────────────────────────────────────────
    final_answer        : Optional[str]     # answer returned to user
    pipeline_blocked    : Optional[bool]    # True if blocked anywhere
    block_stage         : Optional[str]     # which stage blocked it
