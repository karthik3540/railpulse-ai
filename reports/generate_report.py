"""
RailPulse AI — PDF Report Generator
Generates a downloadable A4 PDF maintenance report using ReportLab.
"""

import io
from datetime import datetime
from typing import List, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fusion_engine.fusion import DefectEvent


# ── Color Palette ──────────────────────────────────────────────────────────
DARK_BG = colors.HexColor("#0A1628")
HEADER_BG = colors.HexColor("#112240")
ACCENT = colors.HexColor("#64FFDA")
CRITICAL_COLOR = colors.HexColor("#FF5370")
WARNING_COLOR = colors.HexColor("#FFCB6B")
OK_COLOR = colors.HexColor("#C3E88D")
TEXT_WHITE = colors.HexColor("#CCD6F6")
TEXT_LIGHT = colors.HexColor("#8892B0")


def _severity_color(severity: str):
    if severity == "CRITICAL":
        return CRITICAL_COLOR
    elif severity == "WARNING":
        return WARNING_COLOR
    return OK_COLOR


def generate_report(
    track_events: List[DefectEvent],
    ohe_events: List[DefectEvent],
    lhs: Dict,
) -> bytes:
    """
    Generate a PDF maintenance report.

    Parameters
    ----------
    track_events : list of DefectEvent
    ohe_events : list of DefectEvent
    lhs : dict with keys: track_lhs, ohe_lhs, composite_lhs, priority

    Returns
    -------
    bytes : PDF file content
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=24,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )

    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1a1a2e"),
        spaceBefore=16,
        spaceAfter=10,
        borderWidth=1,
        borderColor=colors.HexColor("#64FFDA"),
        borderPadding=4,
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#333333"),
        spaceAfter=6,
    )

    elements = []

    # ── Title ──────────────────────────────────────────────────────────
    elements.append(Paragraph("🚂 RailPulse AI — Maintenance Report", title_style))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Delhi-Agra Railway Corridor | FAR AWAY Hackathon 2026",
        subtitle_style,
    ))
    elements.append(Spacer(1, 10))

    # ── Section 1: LHS Summary ─────────────────────────────────────────
    elements.append(Paragraph("Section 1: Likelihood of Health Score (LHS) Summary", heading_style))

    lhs_data = [
        ["Metric", "Value"],
        ["Track LHS", f"{lhs['track_lhs']:.2f}"],
        ["OHE LHS", f"{lhs['ohe_lhs']:.2f}"],
        ["Composite LHS", f"{lhs['composite_lhs']:.2f}"],
        ["Priority", lhs["priority"]],
    ]

    lhs_table = Table(lhs_data, colWidths=[200, 200])
    lhs_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f0f4f8")),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
    ]))
    elements.append(lhs_table)
    elements.append(Spacer(1, 20))

    # ── Section 2: Ranked Defect Events ────────────────────────────────
    elements.append(Paragraph("Section 2: Ranked Defect Events (by Risk Index)", heading_style))

    # Combine and sort all events
    all_events = list(track_events) + list(ohe_events)
    all_events.sort(key=lambda e: e.risk_index, reverse=True)

    event_header = ["#", "Type", "Defect Class", "Risk", "Confidence", "Severity", "Location"]
    event_data = [event_header]

    for i, event in enumerate(all_events, 1):
        event_data.append([
            str(i),
            event.asset_type.upper(),
            event.defect_class,
            f"{event.risk_index:.1f}",
            f"{event.confidence:.1f}%",
            event.severity,
            f"({event.lat:.3f}, {event.lon:.3f})",
        ])

    col_widths = [25, 45, 100, 40, 65, 60, 100]
    event_table = Table(event_data, colWidths=col_widths)

    # Build style commands
    table_style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
    ]

    # Color-code severity cells
    for i, event in enumerate(all_events, 1):
        if event.severity == "CRITICAL":
            table_style_cmds.append(("TEXTCOLOR", (5, i), (5, i), colors.HexColor("#d32f2f")))
            table_style_cmds.append(("FONTNAME", (5, i), (5, i), "Helvetica-Bold"))
        elif event.severity == "WARNING":
            table_style_cmds.append(("TEXTCOLOR", (5, i), (5, i), colors.HexColor("#f57c00")))
            table_style_cmds.append(("FONTNAME", (5, i), (5, i), "Helvetica-Bold"))

    event_table.setStyle(TableStyle(table_style_cmds))
    elements.append(event_table)
    elements.append(Spacer(1, 20))

    # ── Footer ─────────────────────────────────────────────────────────
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#888888"),
        alignment=TA_CENTER,
    )
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        "RailPulse AI v1.0 — Railway Predictive Maintenance System | FAR AWAY Hackathon 2026",
        footer_style,
    ))

    # ── Build PDF ──────────────────────────────────────────────────────
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
