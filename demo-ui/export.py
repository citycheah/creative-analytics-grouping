"""Executive PDF Report Generator for Creative Theme & Telco Category Intelligence.

Generates a formatted PDF summary using ReportLab.
Includes executive metrics, portfolio breakdown, strategic recommendations,
and ranked theme performance tables.

Usage:
    python demo-ui/export.py --output demo-ui/data/report.pdf
"""

import io
import os
import sys
from typing import Optional

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import config as cfg
from common.logging import get_logger

logger = get_logger("export", stage="reporting")


def _build_minimal_pdf(title_str: str, lines_list: list) -> bytes:
    """Pure-Python standard PDF 1.4 generator when ReportLab is not installed."""
    def _esc(s: str) -> str:
        return s.encode("ascii", "replace").decode("ascii").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content_ops = ["BT", "/F1 15 Tf", "40 750 Td", f"({_esc(title_str)}) Tj", "/F1 9 Tf", "0 -22 Td"]
    y = 728
    for line in lines_list[:48]:
        if y < 50:
            break
        content_ops.append(f"({_esc(line[:96])}) Tj")
        content_ops.append("0 -13 Td")
        y -= 13
    content_ops.append("ET")
    stream_bytes = "\n".join(content_ops).encode("ascii")

    objs = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        f"4 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii") + stream_bytes + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for o in objs:
        offsets.append(out.tell())
        out.write(o)
    xref_pos = out.tell()
    out.write(f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode("ascii"))
    out.write(f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii"))
    return out.getvalue()


def generate_pdf_report(
    themes_summary_path: Optional[str] = None,
    executive_summary_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> bytes:
    """Generates PDF report and returns PDF byte stream."""
    ui_dir = os.path.join(repo_root, "demo-ui", "data")
    t_path = themes_summary_path or os.path.join(ui_dir, "themes_summary.csv")
    s_path = executive_summary_path or os.path.join(ui_dir, "executive_summary.md")

    themes = pd.read_csv(t_path) if os.path.exists(t_path) else pd.DataFrame()
    exec_summary_text = ""
    if os.path.exists(s_path):
        with open(s_path, "r", encoding="utf-8") as fh:
            exec_summary_text = fh.read()

    if not HAS_REPORTLAB:
        brand_name = getattr(cfg, "ADVERTISER_NAME", "Maxis")
        country_name = getattr(cfg, "COUNTRY", "Malaysia")
        summary_lines = [line.replace("**", "").replace("###", "").strip() for line in exec_summary_text.splitlines() if line.strip()]
        if not themes.empty:
            summary_lines.append("--- Ranked Creative Themes ---")
            valid = themes[themes["cluster_id"] != -1].sort_values("total_spend", ascending=False)
            for r in valid.head(12).itertuples():
                summary_lines.append(f"{r.theme_label}: ${r.total_spend:,.0f} ({r.pct_of_total_spend:.1f}%) | {r.creative_count} ads")
        pdf_bytes = _build_minimal_pdf(f"{brand_name} ({country_name}) - Creative Intelligence Briefing", summary_lines)
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
        return pdf_bytes

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#065F46"),  # Telco Emerald Deep Green
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
    )
    heading2_style = ParagraphStyle(
        "ReportH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    table_text = ParagraphStyle(
        "TableText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.white,
    )

    elements = []

    brand_name = getattr(cfg, "ADVERTISER_NAME", "Maxis")
    country_name = getattr(cfg, "COUNTRY", "Malaysia")
    elements.append(Paragraph(f"{brand_name} ({country_name}) — Creative Intelligence Briefing", title_style))
    elements.append(
        Paragraph(
            f"Multimodal AI Clustering, Telco Category SOV & FinOps Analysis &middot; Generated {pd.Timestamp.now().strftime('%B %d, %Y')}",
            subtitle_style,
        )
    )
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#059669"), spaceAfter=12))

    # KPI Summary Cards Table
    if not themes.empty:
        valid = themes[themes["cluster_id"] != -1]
        total_spend = themes["total_spend"].sum()
        total_imp = themes["total_impressions"].sum()
        total_creatives = themes["creative_count"].sum()
        num_themes = len(valid)

        kpi_data = [
            [
                Paragraph(f"<b>TOTAL ANALYZED SPEND</b><br/><font size=12 color='#065F46'><b>${total_spend:,.0f}</b></font>", body_style),
                Paragraph(f"<b>TOTAL IMPRESSIONS</b><br/><font size=12 color='#0F172A'><b>{total_imp:,.0f}</b></font>", body_style),
                Paragraph(f"<b>CREATIVE PLACEMENTS</b><br/><font size=12 color='#0F172A'><b>{total_creatives}</b></font>", body_style),
                Paragraph(f"<b>IDENTIFIED THEMES</b><br/><font size=12 color='#065F46'><b>{num_themes}</b></font>", body_style),
            ]
        ]
        kpi_table = Table(kpi_data, colWidths=[135, 135, 135, 135])
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        elements.append(kpi_table)
        elements.append(Spacer(1, 14))

    # Executive Briefing Section
    elements.append(Paragraph("Strategic Executive Briefing", heading2_style))
    if exec_summary_text:
        lines = exec_summary_text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            clean_line = line.replace("**", "")
            if line.startswith("###"):
                elements.append(Paragraph(f"<b>{clean_line.replace('###', '').strip()}</b>", heading2_style))
            elif line.startswith("-"):
                elements.append(Paragraph(f"&bull; {clean_line[1:].strip()}", body_style))
            elif line[0].isdigit() and line[1:3] in (". ", ") "):
                elements.append(Paragraph(f"<b>{clean_line[:2]}</b> {clean_line[2:].strip()}", body_style))
            else:
                elements.append(Paragraph(clean_line, body_style))
            elements.append(Spacer(1, 4))
    else:
        elements.append(Paragraph("Multi-modal semantic analysis categorized ad creatives into cohesive themes.", body_style))
    elements.append(Spacer(1, 10))

    # Ranked Themes Table
    if not themes.empty:
        elements.append(Paragraph("Ranked Theme Performance Breakdown", heading2_style))
        valid = themes[themes["cluster_id"] != -1].sort_values("total_spend", ascending=False)
        total_spend = themes["total_spend"].sum()

        table_rows = [
            [
                Paragraph("Theme Label", table_header),
                Paragraph("Spend ($)", table_header),
                Paragraph("Share (%)", table_header),
                Paragraph("Impressions", table_header),
                Paragraph("Raw / Renders", table_header),
            ]
        ]
        for r in valid.head(12).itertuples():
            pct = (100.0 * r.total_spend / total_spend) if total_spend > 0 else 0.0
            renders = getattr(r, "distinct_renders", r.creative_count)
            table_rows.append([
                Paragraph(str(r.theme_label)[:42], table_text),
                Paragraph(f"${r.total_spend:,.0f}", table_text),
                Paragraph(f"{pct:.1f}%", table_text),
                Paragraph(f"{r.total_impressions:,.0f}", table_text),
                Paragraph(f"{r.creative_count} / {renders}", table_text),
            ])

        theme_tbl = Table(table_rows, colWidths=[210, 80, 60, 105, 85])
        theme_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#065F46")),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        elements.append(theme_tbl)

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info(f"Saved PDF report to {output_path} ({len(pdf_bytes):,} bytes)")

    return pdf_bytes


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Executive PDF Report")
    parser.add_argument("--output", type=str, default=os.path.join(repo_root, "demo-ui", "data", "report.pdf"))
    args = parser.parse_args()

    generate_pdf_report(output_path=args.output)


if __name__ == "__main__":
    main()
