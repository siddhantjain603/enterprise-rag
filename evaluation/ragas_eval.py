"""
evaluation/ragas_eval.py
-------------------------
Evaluates the full RAG pipeline using RAGAS metrics.

Runs 10 test questions through the LangGraph pipeline,
collects answers + contexts, then scores with RAGAS.

Metrics:
  Faithfulness      → is answer grounded in context? (0-1)
  Answer Relevance  → does answer address the question? (0-1)
  Context Precision → were retrieved chunks actually useful? (0-1)

Usage:
    python evaluation/ragas_eval.py
"""

from pathlib import Path
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.graph import run_pipeline
import config

console = Console()

# ── Test questions ────────────────────────────────────────────────────
TEST_QUESTIONS = [
    "How many days of annual leave are employees entitled to?",
    "What is the probationary period for new employees?",
    "How many days of leave can I carry forward to next year?",
    "What happens to unused leave beyond the carry-forward limit?",
    "How many sick leave days are employees entitled to per year?",
    "What is the maximum gift value an employee can accept from a vendor?",
    "How many days of bereavement leave is an employee entitled to?",
    "How far in advance must leave be applied for?",
    "How often are performance reviews conducted?",
    "What is the performance rating scale used by the company?",
]


def run_evaluation_dataset() -> list[dict]:
    """
    Run all test questions through the full LangGraph pipeline.
    Collect question, answer, and contexts for RAGAS.
    """
    console.rule("[bold cyan]Step 1 — Running Test Questions Through Pipeline[/bold cyan]")
    console.print(f"\n[dim]Running {len(TEST_QUESTIONS)} questions through full RAG pipeline...[/dim]\n")

    results = []

    for i, question in enumerate(TEST_QUESTIONS):
        console.print(f"\n[bold cyan]Question {i+1}/{len(TEST_QUESTIONS)}:[/bold cyan] [yellow]{question}[/yellow]")

        start   = time.time()
        state   = run_pipeline(question)
        latency = round(time.time() - start, 3)

        answer        = state.get("final_answer", "")
        parent_chunks = state.get("parent_chunks", [])
        contexts      = [p["text"] for p in parent_chunks] if parent_chunks else [""]
        was_blocked   = state.get("pipeline_blocked", False)

        if was_blocked or not answer:
            console.print(f"  [yellow]⚠ Pipeline blocked or no answer — skipping[/yellow]")
            continue

        console.print(f"  [green]✓ Answer collected[/green] — {len(answer.split())} words | {len(contexts)} context chunks | {latency}s")
        console.print(f"  [dim]  Answer: {answer[:80]}...[/dim]")

        results.append({
            "question"  : question,
            "answer"    : answer,
            "contexts"  : contexts,
            "latency"   : latency,
            "self_check": state.get("self_check_verdict", "UNKNOWN")
        })

    console.print(f"\n[bold green]✓ Collected {len(results)}/{len(TEST_QUESTIONS)} valid results[/bold green]\n")
    return results


def run_ragas_scoring(results: list[dict]):
    """
    Feed collected results into RAGAS for scoring.
    Uses Azure OpenAI for LLM and embeddings.
    """
    console.rule("[bold cyan]Step 2 — Running RAGAS Evaluation[/bold cyan]")
    console.print(f"\n[dim]Initialising RAGAS with Azure OpenAI...[/dim]")

    azure_llm = AzureChatOpenAI(
        azure_endpoint   = config.AZURE_OPENAI_ENDPOINT,
        azure_deployment = config.CHAT_DEPLOYMENT,
        api_version      = config.AZURE_OPENAI_API_VERSION,
        api_key          = config.AZURE_OPENAI_API_KEY,
        temperature      = 0.0
    )

    azure_embeddings = AzureOpenAIEmbeddings(
        azure_endpoint   = config.AZURE_OPENAI_ENDPOINT,
        azure_deployment = config.EMBEDDING_DEPLOYMENT,
        api_version      = config.AZURE_OPENAI_API_VERSION,
        api_key          = config.AZURE_OPENAI_API_KEY,
    )

    ragas_llm        = LangchainLLMWrapper(azure_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(azure_embeddings)

    console.print(f"  [green]✓ Azure OpenAI configured for RAGAS[/green]")
    console.print(f"  [dim]  LLM: {config.CHAT_DEPLOYMENT} | Embeddings: {config.EMBEDDING_DEPLOYMENT}[/dim]\n")

    dataset = Dataset.from_dict({
        "question" : [r["question"] for r in results],
        "answer"   : [r["answer"]   for r in results],
        "contexts" : [r["contexts"] for r in results],
        "reference": [r["answer"]   for r in results],
    })

    console.print(f"  [green]✓ Dataset built — {len(results)} rows[/green]\n")
    console.print(f"[bold yellow]Running RAGAS scoring (this may take 2-3 minutes)...[/bold yellow]\n")

    start = time.time()

    scores = evaluate(
        dataset    = dataset,
        metrics    = [faithfulness, answer_relevancy, context_precision],
        llm        = ragas_llm,
        embeddings = ragas_embeddings,
    )

    latency = round(time.time() - start, 3)
    console.print(f"\n  [green]✓ RAGAS scoring complete ({latency}s)[/green]\n")

    return scores, latency


def print_results(results: list[dict], scores, eval_latency: float):
    """Print full evaluation results — per question and aggregate."""

    console.rule("[bold cyan]Step 3 — Evaluation Results[/bold cyan]")

    # ── Per question pipeline results ─────────────────────────────────
    console.print(f"\n[bold yellow]📋 Per-Question Pipeline Results[/bold yellow]\n")

    table = Table(show_lines=True, title="Per-Question Pipeline Results")
    table.add_column("#",          style="cyan",   justify="center")
    table.add_column("Question",   style="white",  max_width=35)
    table.add_column("Answer",     style="dim",    max_width=35)
    table.add_column("Contexts",   style="green",  justify="center")
    table.add_column("Self-Check", style="yellow", justify="center")
    table.add_column("Latency",    style="dim",    justify="right")

    for i, r in enumerate(results):
        self_check_display = (
            "[green]GROUNDED[/green]"       if r["self_check"] == "GROUNDED"
            else "[red]HALLUCINATION[/red]" if r["self_check"] == "HALLUCINATION"
            else "[dim]UNKNOWN[/dim]"
        )
        table.add_row(
            str(i + 1),
            r["question"][:35] + "..." if len(r["question"]) > 35 else r["question"],
            r["answer"][:35]   + "..." if len(r["answer"])   > 35 else r["answer"],
            str(len(r["contexts"])),
            self_check_display,
            f"{r['latency']}s"
        )

    console.print(table)

    # ── Extract numeric scores safely ─────────────────────────────────
    scores_df  = scores.to_pandas()
    num_df     = scores_df.select_dtypes(include="number")
    avg_scores = num_df.mean().to_dict()

    faithfulness_score      = round(avg_scores.get("faithfulness",     0), 4)
    answer_relevancy_score  = round(avg_scores.get("answer_relevancy", 0), 4)
    context_precision_score = round(avg_scores.get("context_precision",0), 4)
    overall = round(
        (faithfulness_score + answer_relevancy_score + context_precision_score) / 3, 4
    )

    # ── Aggregate score table ─────────────────────────────────────────
    console.print(f"\n[bold yellow]📊 RAGAS Aggregate Scores[/bold yellow]\n")

    score_table = Table(show_lines=True, title="RAGAS Evaluation Scores")
    score_table.add_column("Metric",           style="white",      no_wrap=True)
    score_table.add_column("Score",            style="bold green", justify="center")
    score_table.add_column("Out of",           style="dim",        justify="center")
    score_table.add_column("What It Measures", style="dim",        max_width=45)

    score_table.add_row(
        "Faithfulness",
        str(faithfulness_score), "1.0",
        "Are all claims in the answer supported by context?"
    )
    score_table.add_row(
        "Answer Relevancy",
        str(answer_relevancy_score), "1.0",
        "Does the answer actually address the question?"
    )
    score_table.add_row(
        "Context Precision",
        str(context_precision_score), "1.0",
        "Were the retrieved chunks useful for the answer?"
    )
    score_table.add_row(
        "[bold]Overall Average[/bold]",
        f"[bold]{overall}[/bold]", "1.0",
        "Mean of all three metrics"
    )

    console.print(score_table)

    # ── Per question RAGAS scores ─────────────────────────────────────
    console.print(f"\n[bold yellow]📋 Per-Question RAGAS Scores[/bold yellow]\n")

    per_q_table = Table(show_lines=True, title="Per-Question RAGAS Breakdown")
    per_q_table.add_column("#",                 style="cyan",  justify="center")
    per_q_table.add_column("Question",          style="white", max_width=32)
    per_q_table.add_column("Faithfulness",      style="green", justify="center")
    per_q_table.add_column("Answer Relevancy",  style="green", justify="center")
    per_q_table.add_column("Context Precision", style="green", justify="center")

    for i, row in num_df.iterrows():
        per_q_table.add_row(
            str(i + 1),
            results[i]["question"][:32] + "..." if len(results[i]["question"]) > 32 else results[i]["question"],
            str(round(row.get("faithfulness",     0), 3)),
            str(round(row.get("answer_relevancy", 0), 3)),
            str(round(row.get("context_precision",0), 3)),
        )

    console.print(per_q_table)

    # ── Final summary panel ───────────────────────────────────────────
    grounded_count = sum(1 for r in results if r["self_check"] == "GROUNDED")

    console.print(Panel(
        f"[bold green]✅ Evaluation Complete[/bold green]\n\n"
        f"Questions tested      : {len(results)}/{len(TEST_QUESTIONS)}\n"
        f"Self-check GROUNDED   : {grounded_count}/{len(results)}\n"
        f"Evaluation latency    : {eval_latency}s\n\n"
        f"[bold]RAGAS Scores:[/bold]\n"
        f"  Faithfulness        : [bold green]{faithfulness_score}[/bold green]\n"
        f"  Answer Relevancy    : [bold green]{answer_relevancy_score}[/bold green]\n"
        f"  Context Precision   : [bold green]{context_precision_score}[/bold green]\n"
        f"  Overall Average     : [bold green]{overall}[/bold green]\n\n"
        f"[dim]Save these numbers to numbers.txt![/dim]",
        title="Final RAGAS Evaluation Summary",
        border_style="green"
    ))

    return {
        "faithfulness"      : faithfulness_score,
        "answer_relevancy"  : answer_relevancy_score,
        "context_precision" : context_precision_score,
        "overall"           : overall
    }


def main():
    console.rule("[bold cyan]🎯 RAGAS Evaluation — Enterprise RAG Pipeline[/bold cyan]")
    console.print(f"\n[dim]Pipeline: 9 LangGraph nodes | Metrics: Faithfulness, Answer Relevancy, Context Precision[/dim]\n")

    results = run_evaluation_dataset()

    if not results:
        console.print("[red]No results collected — cannot run RAGAS evaluation[/red]")
        return

    scores, eval_latency = run_ragas_scoring(results)
    final_scores = print_results(results, scores, eval_latency)

    return final_scores


if __name__ == "__main__":
    main()
