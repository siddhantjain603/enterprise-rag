"""
Stage 1a — Generate Placeholder PDFs
-------------------------------------
Creates 3 sample HR policy PDFs in the data/ folder.
Run this first to have documents to work with.

Usage:
    python ingestion/generate_docs.py
"""

import os
from pathlib import Path
from pypdf import PdfWriter
from pypdf.generic import NameObject
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

# ── Document content ─────────────────────────────────────────────────

DOCUMENTS = {
    "hr_policy.pdf": {
        "title": "Human Resources Policy Document",
        "pages": [
            {
                "heading": "1. Employment Terms",
                "body": (
                    "All employees are required to complete a probationary period of 90 days upon joining the organization. "
                    "During this period, performance will be reviewed by the line manager every 30 days. "
                    "Upon successful completion of probation, employees will receive a confirmation letter and become eligible for all company benefits. "
                    "The standard working hours are 9:00 AM to 6:00 PM, Monday to Friday, totaling 45 hours per week. "
                    "Flexible working arrangements may be approved by the HR department on a case-by-case basis subject to business needs. "
                    "Employees are expected to maintain a professional code of conduct at all times within and outside the workplace when representing the company."
                )
            },
            {
                "heading": "2. Leave Policy",
                "body": (
                    "Employees are entitled to 20 days of paid annual leave per calendar year. "
                    "Leave must be applied for at least 2 weeks in advance through the HR portal and is subject to manager approval. "
                    "Unused leave can be carried forward to the next year up to a maximum of 10 days. "
                    "Any leave beyond this carry-forward limit will lapse at the end of the financial year and will not be encashed. "
                    "Sick leave is granted up to 12 days per year and requires a medical certificate for absences exceeding 3 consecutive days. "
                    "Maternity leave is granted for 26 weeks as per statutory requirements, and paternity leave is granted for 5 working days. "
                    "Leave without pay may be granted at the discretion of the management for personal emergencies beyond the entitled leave balance."
                )
            },
            {
                "heading": "3. Performance Review",
                "body": (
                    "The company conducts formal performance reviews twice a year — in June and December. "
                    "Each review involves a self-assessment by the employee followed by a structured discussion with the line manager. "
                    "Performance ratings are on a scale of 1 to 5, where 5 represents exceptional performance exceeding all targets. "
                    "Employees rated 4 or above are eligible for an annual merit increment of up to 15 percent of their base salary. "
                    "Employees rated below 2 for two consecutive review cycles will be placed on a Performance Improvement Plan. "
                    "The Performance Improvement Plan lasts 60 days and outlines specific measurable goals the employee must achieve. "
                    "Promotion decisions are directly linked to performance ratings and recommendations from the line manager."
                )
            }
        ]
    },

    "leave_policy.pdf": {
        "title": "Leave Management Policy",
        "pages": [
            {
                "heading": "1. Types of Leave",
                "body": (
                    "The organization recognizes the following types of leave for all permanent employees. "
                    "Casual Leave of 8 days per year is granted for personal or family matters requiring short absences. "
                    "Earned Leave of 20 days per year accrues at the rate of 1.67 days per month of service. "
                    "Medical Leave of 12 days per year is provided for health-related absences and requires medical documentation for claims exceeding 3 days. "
                    "Compensatory Leave is granted to employees who work on designated public holidays or weekends at the request of management. "
                    "Bereavement Leave of 3 days is available in the event of the death of an immediate family member. "
                    "Study Leave of up to 5 days per year may be approved for employees pursuing approved professional certifications relevant to their role."
                )
            },
            {
                "heading": "2. Leave Application Process",
                "body": (
                    "All leave applications must be submitted through the official HR portal at least 5 working days in advance for planned absences. "
                    "Emergency leave must be reported to the manager via phone or email within 2 hours of the start of the working day. "
                    "Leave approval is at the discretion of the line manager based on team availability and project commitments. "
                    "Employees must hand over pending responsibilities to a designated colleague before proceeding on leave. "
                    "Unapproved absences will be marked as Loss of Pay and may result in disciplinary action if repeated. "
                    "Leave records are maintained in the HR system and employees can view their balance through the employee self-service portal. "
                    "Any disputes regarding leave approval must be escalated to the HR Business Partner within 3 working days."
                )
            },
            {
                "heading": "3. Public Holidays",
                "body": (
                    "The organization observes 12 public holidays per calendar year as notified by the HR department in January. "
                    "A holiday calendar is published at the beginning of each year and is accessible on the company intranet. "
                    "Employees required to work on public holidays will receive compensatory leave to be availed within 30 days. "
                    "Optional holidays of up to 2 days may be chosen by employees from a list of festivals provided by HR. "
                    "Employees working from locations outside the headquarters may follow the local state holiday calendar with prior HR approval. "
                    "Public holiday entitlement is pro-rated for employees who join or leave mid-year based on their date of joining or separation."
                )
            }
        ]
    },

    "code_of_conduct.pdf": {
        "title": "Code of Conduct and Ethics Policy",
        "pages": [
            {
                "heading": "1. Professional Conduct",
                "body": (
                    "All employees are expected to maintain the highest standards of integrity and professionalism in their interactions. "
                    "Employees must treat colleagues, clients, and vendors with respect and dignity regardless of their position or background. "
                    "Discrimination, harassment, or bullying of any kind will not be tolerated and will result in immediate disciplinary action. "
                    "Confidential business information must not be shared with unauthorized parties inside or outside the organization. "
                    "Employees must declare any conflicts of interest to their manager and the HR department before engaging in activities that may pose a conflict. "
                    "Use of company assets including hardware, software, and internet must be limited to business purposes only. "
                    "Any violation of the code of conduct must be reported to the Ethics Helpline, which ensures complete anonymity for the reporter."
                )
            },
            {
                "heading": "2. Data Privacy and Security",
                "body": (
                    "Employees handling customer or employee data must comply with the company's data protection policy and applicable regulations. "
                    "Personal data must not be stored on personal devices or shared via unofficial communication channels such as personal email. "
                    "All data transfers must be encrypted using company-approved tools and protocols as defined by the IT security team. "
                    "Employees must immediately report any suspected data breach or unauthorized access to the IT security team and their manager. "
                    "Access to sensitive data is granted on a need-to-know basis and must be reviewed every quarter by the data owner. "
                    "Employees leaving the organization must return all company data and surrender access credentials on or before the last working day. "
                    "Non-compliance with data privacy policies may result in legal liability for both the employee and the organization."
                )
            },
            {
                "heading": "3. Anti-Bribery and Corruption",
                "body": (
                    "The company maintains a zero-tolerance policy towards bribery, corruption, and facilitation payments of any kind. "
                    "Employees must not offer, accept, or solicit gifts, hospitality, or favors that may influence business decisions. "
                    "Gifts of nominal value not exceeding 500 rupees may be accepted during festivals but must be declared to the manager. "
                    "Political contributions on behalf of the company are strictly prohibited without prior written approval from the Board. "
                    "Employees involved in procurement or vendor management must follow the three-quote policy for all purchases above 50,000 rupees. "
                    "Any suspected instance of bribery must be reported to the Compliance Officer within 24 hours of becoming aware of it. "
                    "Whistleblowers reporting genuine concerns in good faith are protected from retaliation under the company's whistleblower protection policy."
                )
            }
        ]
    }
}


def create_pdf(filename: str, content: dict, output_dir: Path) -> dict:
    """Create a single PDF file and return its stats."""
    filepath = output_dir / filename

    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = styles["Title"]
    story.append(Paragraph(content["title"], title_style))
    story.append(Spacer(1, 0.5*cm))

    total_chars = len(content["title"])

    # Pages/sections
    for page in content["pages"]:
        heading_style = styles["Heading2"]
        body_style = styles["BodyText"]
        body_style.leading = 16

        story.append(Paragraph(page["heading"], heading_style))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(page["body"], body_style))
        story.append(Spacer(1, 0.5*cm))

        total_chars += len(page["heading"]) + len(page["body"])

    doc.build(story)

    file_size = filepath.stat().st_size
    word_count = sum(
        len(p["body"].split()) + len(p["heading"].split())
        for p in content["pages"]
    ) + len(content["title"].split())

    return {
        "filename": filename,
        "filepath": str(filepath),
        "pages": len(content["pages"]),
        "word_count": word_count,
        "char_count": total_chars,
        "file_size_kb": round(file_size / 1024, 2)
    }


def main():
    console.rule("[bold cyan]Stage 1a — Generating Placeholder PDFs[/bold cyan]")

    # ── Setup output dir ─────────────────────────────────────────────
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"\n📁 Output directory: [green]{output_dir}[/green]\n")

    # ── Generate each PDF ────────────────────────────────────────────
    stats = []
    for filename, content in DOCUMENTS.items():
        with console.status(f"Creating [cyan]{filename}[/cyan]..."):
            stat = create_pdf(filename, content, output_dir)
            stats.append(stat)
        console.print(f"  ✅ Created [cyan]{filename}[/cyan]")

    # ── Print summary table ──────────────────────────────────────────
    console.print()
    table = Table(title="📄 Generated Documents Summary", show_lines=True)
    table.add_column("File",           style="cyan",  no_wrap=True)
    table.add_column("Sections",       style="white", justify="center")
    table.add_column("Word Count",     style="green", justify="right")
    table.add_column("Char Count",     style="green", justify="right")
    table.add_column("Size (KB)",      style="yellow",justify="right")

    total_words = 0
    total_chars = 0
    for s in stats:
        table.add_row(
            s["filename"],
            str(s["pages"]),
            str(s["word_count"]),
            str(s["char_count"]),
            str(s["file_size_kb"])
        )
        total_words += s["word_count"]
        total_chars += s["char_count"]

    table.add_row(
        "[bold]TOTAL[/bold]",
        str(sum(s["pages"] for s in stats)),
        f"[bold]{total_words}[/bold]",
        f"[bold]{total_chars}[/bold]",
        f"[bold]{sum(s['file_size_kb'] for s in stats):.2f}[/bold]"
    )

    console.print(table)

    console.print(f"\n[bold green]✅ {len(DOCUMENTS)} PDFs created successfully in /data folder[/bold green]")
    console.print("[dim]Next step → run: python ingestion/document_loader.py[/dim]\n")


if __name__ == "__main__":
    main()