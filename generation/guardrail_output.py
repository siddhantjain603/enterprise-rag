"""
Stage 3c — Output Guardrail
-----------------------------
Last line of defense before the answer reaches the user.
Checks the generated answer for:
  1. PII (Personally Identifiable Information)
  2. Harmful or inappropriate content
  3. Empty or malformed responses
  4. Confidential system information leakage

This runs AFTER generation and self-check.
Even if the answer is GROUNDED, it still goes through this gate.

Flow:
  Generated Answer → PII Check → Content Check → Length Check → User

Usage:
    python generation/output_guardrail.py
"""

from pathlib import Path
from openai import AzureOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import sys
import time
import re
import json

sys.path.append(str(Path(__file__).parent.parent))
import config

console = Console()

client = AzureOpenAI(
    azure_endpoint = config.AZURE_OPENAI_ENDPOINT,
    api_key        = config.AZURE_OPENAI_API_KEY,
    api_version    = config.AZURE_OPENAI_API_VERSION
)

# ── PII patterns (rule-based, fast) ──────────────────────────────────
PII_PATTERNS = {
    "email"         : r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone"         : r"\b(\+91|0)?[6-9]\d{9}\b",
    "aadhaar"       : r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan"           : r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "credit_card"   : r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "employee_id"   : r"\bEMP[-/]?\d{4,8}\b",
}

# ── System leak patterns ──────────────────────────────────────────────
SYSTEM_LEAK_PATTERNS = [
    r"system prompt",
    r"strict rules",
    r"you are an hr policy assistant",
    r"do not use any outside knowledge",
    r"never reveal these instructions",
]

# ── Content check prompt ──────────────────────────────────────────────
CONTENT_CHECK_PROMPT = """You are a content safety checker for an HR policy assistant.

Check if the given answer contains any of the following:
- Harmful, offensive, or inappropriate content
- Discriminatory language
- Threats or violence
- Sexual content
- Content that could cause legal liability

Respond with ONLY a JSON object:
{
  "is_safe": true or false,
  "reason": "one line explanation",
  "flagged_content": "the specific problematic phrase" or null
}

Nothing else. No markdown. No extra text."""


def check_empty_response(answer: str) -> dict:
    """Check if answer is empty or too short to be useful."""
    console.print(f"\n[dim]  → Check 1: Empty/malformed response check...[/dim]")

    if not answer or not answer.strip():
        console.print(f"  [red]✗ Empty answer detected[/red]")
        return {"passed": False, "reason": "Answer is empty"}

    if len(answer.strip().split()) < 3:
        console.print(f"  [red]✗ Answer too short ({len(answer.split())} words)[/red]")
        return {"passed": False, "reason": "Answer suspiciously short"}

    console.print(f"  [green]✓ Response length OK[/green] — {len(answer.split())} words, {len(answer)} chars")
    return {"passed": True}


def check_pii(answer: str) -> dict:
    """
    Rule-based PII detection — fast regex scan.
    Catches emails, phone numbers, Aadhaar, PAN, credit cards.
    """
    console.print(f"\n[dim]  → Check 2: Scanning for PII...[/dim]")
    console.print(f"  [dim]  Checking {len(PII_PATTERNS)} PII pattern types[/dim]")

    found_pii = []
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, answer, re.IGNORECASE)
        if matches:
            found_pii.append({"type": pii_type, "matches": matches})
            console.print(f"  [red]✗ PII detected: {pii_type} → {matches}[/red]")

    if found_pii:
        return {
            "passed"   : False,
            "reason"   : f"PII detected: {[p['type'] for p in found_pii]}",
            "found_pii": found_pii
        }

    console.print(f"  [green]✓ No PII detected — answer is clean[/green]")
    return {"passed": True}


def check_system_leak(answer: str) -> dict:
    """
    Check if the answer accidentally leaks system prompt content.
    This happens when LLMs repeat their instructions back to users.
    """
    console.print(f"\n[dim]  → Check 3: System prompt leak detection...[/dim]")

    answer_lower = answer.lower()
    for pattern in SYSTEM_LEAK_PATTERNS:
        if re.search(pattern, answer_lower):
            console.print(f"  [red]✗ System prompt leak detected: '{pattern}'[/red]")
            return {
                "passed": False,
                "reason": f"Answer contains system prompt content: '{pattern}'"
            }

    console.print(f"  [green]✓ No system prompt leakage detected[/green]")
    return {"passed": True}


def check_content_safety(answer: str) -> dict:
    """
    LLM-based content safety check.
    Catches harmful content that regex might miss.
    """
    console.print(f"\n[dim]  → Check 4: Content safety check via GPT-4o...[/dim]")

    start = time.time()

    try:
        response = client.chat.completions.create(
            model    = config.CHAT_DEPLOYMENT,
            messages = [
                {"role": "system", "content": CONTENT_CHECK_PROMPT},
                {"role": "user",   "content": f"Answer to check:\n{answer}"}
            ],
            max_tokens  = 100,
            temperature = 0.0
        )
    except Exception as e:
        error_msg = str(e)
        if "content_filter" in error_msg:
            console.print(f"  [red]✗ Azure content filter triggered[/red]")
            return {"passed": False, "reason": "Azure content filter triggered"}
        console.print(f"  [yellow]⚠ Content check API error — defaulting to PASS[/yellow]")
        return {"passed": True, "reason": "API error — defaulted to pass"}

    latency = round(time.time() - start, 3)
    raw     = response.choices[0].message.content.strip()

    console.print(f"  [dim]  Safety checker response ({latency}s): {raw}[/dim]")

    try:
        result          = json.loads(raw)
        is_safe         = result.get("is_safe", True)
        reason          = result.get("reason", "")
        flagged_content = result.get("flagged_content", None)

        if not is_safe:
            console.print(f"  [red]✗ Unsafe content detected: {reason}[/red]")
            return {
                "passed"         : False,
                "reason"         : reason,
                "flagged_content": flagged_content
            }

        console.print(f"  [green]✓ Content safety check passed[/green]")
        return {"passed": True, "reason": reason}

    except json.JSONDecodeError:
        console.print(f"  [yellow]⚠ Could not parse safety response — defaulting to PASS[/yellow]")
        return {"passed": True}


def run_output_guardrail(answer: str) -> dict:
    """
    Run all 4 output checks in sequence.
    Returns final decision with full trace.
    """
    console.rule("[bold cyan]Stage 3c — Output Guardrail[/bold cyan]")
    console.print(f"\n[bold]📤 Answer to check:[/bold]")
    console.print(Panel(f"[dim]{answer}[/dim]", border_style="yellow"))
    console.print(f"[dim]Running 4 checks: empty → PII → system leak → content safety[/dim]\n")

    # ── Check 1: Empty ───────────────────────────────────────────────
    empty_check = check_empty_response(answer)
    if not empty_check["passed"]:
        return _build_result("BLOCKED", answer, "empty_check", empty_check["reason"])

    # ── Check 2: PII ─────────────────────────────────────────────────
    pii_check = check_pii(answer)
    if not pii_check["passed"]:
        return _build_result("BLOCKED", answer, "pii_check", pii_check["reason"])

    # ── Check 3: System leak ─────────────────────────────────────────
    leak_check = check_system_leak(answer)
    if not leak_check["passed"]:
        return _build_result("BLOCKED", answer, "system_leak_check", leak_check["reason"])

    # ── Check 4: Content safety ──────────────────────────────────────
    content_check = check_content_safety(answer)
    if not content_check["passed"]:
        return _build_result("BLOCKED", answer, "content_check", content_check["reason"])

    return _build_result("PASSED", answer, None, "All output checks passed")


def _build_result(decision: str, answer: str, failed_at: str, reason: str) -> dict:
    """Build and print the final output guardrail result."""
    passed = decision == "PASSED"

    console.print()
    if passed:
        console.print(Panel(
            f"[bold green]✅ OUTPUT APPROVED[/bold green]\n"
            f"Decision  : {decision}\n"
            f"Reason    : {reason}\n"
            f"Next      : Return answer to user",
            title="Output Guardrail Decision",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold red]❌ OUTPUT BLOCKED[/bold red]\n"
            f"Decision  : {decision}\n"
            f"Failed at : {failed_at}\n"
            f"Reason    : {reason}\n"
            f"Next      : Return safe error message to user",
            title="Output Guardrail Decision",
            border_style="red"
        ))

    return {
        "decision" : decision,
        "passed"   : passed,
        "failed_at": failed_at,
        "reason"   : reason,
        "answer"   : answer if passed else None
    }


def run_test_suite():
    """Test output guardrail on various answer types."""

    test_cases = [
        # (answer, expected, description)
        (
            "You can carry forward up to 10 days of unused annual leave to the next year. "
            "Any leave beyond this limit will lapse at the end of the financial year.",
            "PASSED",
            "Valid grounded answer"
        ),
        (
            "The employee john.doe@company.com can carry forward 10 days of leave.",
            "BLOCKED",
            "PII — email address"
        ),
        (
            "Contact EMP-004521 for leave policy details.",
            "BLOCKED",
            "PII — employee ID"
        ),
        (
            "STRICT RULES you must follow: Answer only using the context provided.",
            "BLOCKED",
            "System prompt leak"
        ),
        (
            "",
            "BLOCKED",
            "Empty answer"
        ),
        (
            "You are entitled to 20 days of annual leave per year as per company policy. "
            "Unused leave up to 10 days can be carried forward to the next calendar year.",
            "PASSED",
            "Valid policy answer"
        ),
    ]

    console.rule("[bold cyan]Stage 3c — Output Guardrail Test Suite[/bold cyan]")
    console.print(f"[dim]Testing {len(test_cases)} answer scenarios...[/dim]\n")

    results = []
    for answer, expected, description in test_cases:
        result = run_output_guardrail(answer)
        passed = result["decision"] == expected
        results.append({
            "description": description,
            "expected"   : expected,
            "actual"     : result["decision"],
            "passed"     : passed,
            "reason"     : result["reason"]
        })
        console.print()

    # ── Summary table ─────────────────────────────────────────────────
    console.rule("[bold cyan]Test Results Summary[/bold cyan]")

    table = Table(show_lines=True, title="Output Guardrail Test Results")
    table.add_column("Description",  style="white",  max_width=28)
    table.add_column("Expected",     style="cyan",   justify="center")
    table.add_column("Actual",       style="cyan",   justify="center")
    table.add_column("Test",         style="green",  justify="center")
    table.add_column("Reason",       style="dim",    max_width=40)

    passed_count = 0
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        if r["passed"]:
            passed_count += 1
        table.add_row(
            r["description"],
            r["expected"],
            r["actual"],
            status,
            r["reason"]
        )

    console.print(table)
    console.print(f"\n[bold green]Tests passed: {passed_count}/{len(test_cases)}[/bold green]")
    console.print(f"\n[dim]Stage 3 complete! Next → LangGraph assembly[/dim]\n")


if __name__ == "__main__":
    run_test_suite()