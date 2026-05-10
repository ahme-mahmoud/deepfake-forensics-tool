"""
pdf_report.py
=============
PDF Forensic Report Generator

Produces a professional, court-admissible style PDF report including:
  - Case header with chain-of-custody
  - Per-module scores with visual bars
  - SHAP explainability findings
  - Evidence images (ELA, heatmaps, etc.)
  - Final verdict with recommendation

Uses ReportLab (lightweight, no external dependencies).
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

logger = logging.getLogger("pdf_report")

# ── Colour palette ─────────────────────────────────────────────────────────────
C_DARK    = colors.HexColor("#1a1f2e")
C_BLUE    = colors.HexColor("#1565c0")
C_GREEN   = colors.HexColor("#2e7d32")
C_ORANGE  = colors.HexColor("#e65100")
C_RED     = colors.HexColor("#b71c1c")
C_GRAY    = colors.HexColor("#546e7a")
C_LGRAY   = colors.HexColor("#eceff1")
C_WHITE   = colors.white
C_BLACK   = colors.black

W, H = A4   # 595.27 × 841.89 pts


def _score_colour(score: float) -> colors.HexColor:
    if score < 0.35: return C_GREEN
    if score < 0.60: return C_ORANGE
    return C_RED


def _score_label(score: float) -> str:
    if score < 0.20: return "AUTHENTIC"
    if score < 0.40: return "LIKELY AUTHENTIC"
    if score < 0.60: return "INCONCLUSIVE"
    if score < 0.80: return "LIKELY MANIPULATED"
    return "MANIPULATED"


def _bar_table(score: float, width: float = 120) -> Table:
    """Render a horizontal score bar as a 2-cell table."""
    filled  = int(score * width)
    empty   = width - filled
    col = _score_colour(score)
    data = [[""]]
    t = Table(data, colWidths=[filled + 0.01], rowHeights=[8])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), col),
        ("LINEABOVE",  (0,0), (-1,-1), 0.5, C_GRAY),
        ("LINEBELOW",  (0,0), (-1,-1), 0.5, C_GRAY),
        ("LINEBEFORE", (0,0), (-1,-1), 0.5, C_GRAY),
        ("LINEAFTER",  (0,0), (-1,-1), 0.5, C_GRAY),
    ]))
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Page header / footer
# ═══════════════════════════════════════════════════════════════════════════════

def _header_footer(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(C_DARK)
    canvas.rect(0, H - 28*mm, W, 28*mm, fill=1, stroke=0)
    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(15*mm, H - 14*mm, "AI-Powered Deepfake & Image Manipulation Detection System")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(W - 15*mm, H - 14*mm, f"FORENSIC REPORT  —  CONFIDENTIAL")

    # Footer bar
    canvas.setFillColor(C_LGRAY)
    canvas.rect(0, 0, W, 12*mm, fill=1, stroke=0)
    canvas.setFillColor(C_GRAY)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(15*mm, 4*mm,
                      f"Page {doc.page}  |  This report was generated automatically. "
                      "Human expert review is recommended for critical decisions.")
    canvas.restoreState()


# ═══════════════════════════════════════════════════════════════════════════════
# Styles
# ═══════════════════════════════════════════════════════════════════════════════

def _styles():
    base = getSampleStyleSheet()
    s = {
        "h1": ParagraphStyle("h1", parent=base["Heading1"],
                              fontSize=16, textColor=C_DARK, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                              fontSize=12, textColor=C_BLUE, spaceAfter=3),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontSize=9, textColor=C_BLACK, leading=13),
        "small": ParagraphStyle("small", parent=base["Normal"],
                                fontSize=8, textColor=C_GRAY),
        "mono": ParagraphStyle("mono", parent=base["Normal"],
                               fontName="Courier", fontSize=8,
                               textColor=C_DARK),
        "verdict": ParagraphStyle("verdict", parent=base["Normal"],
                                  fontSize=14, fontName="Helvetica-Bold",
                                  alignment=TA_CENTER),
        "caption": ParagraphStyle("caption", parent=base["Normal"],
                                  fontSize=7, textColor=C_GRAY,
                                  alignment=TA_CENTER),
    }
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Section builders
# ═══════════════════════════════════════════════════════════════════════════════

def _section_header(title: str, st) -> List:
    return [
        HRFlowable(width="100%", thickness=1.5, color=C_BLUE),
        Spacer(1, 2*mm),
        Paragraph(title, st["h2"]),
        Spacer(1, 2*mm),
    ]


def _coc_section(report: Dict, st) -> List:
    coc = report.get("chain_of_custody", {})
    rows = [
        ["Field", "Value"],
        ["Case ID",          report.get("case_id", "N/A")],
        ["Generated (UTC)",  report.get("generated_at", "N/A")],
        ["Investigator",     report.get("investigator_id", "N/A")],
        ["Image Path",       coc.get("image_path", "N/A")],
        ["SHA-256",          coc.get("sha256", "N/A")],
        ["File Size",        f"{coc.get('file_size','N/A')} bytes"],
        ["Format",           coc.get("image_format", "N/A")],
        ["Dimensions",       str(coc.get("image_size", "N/A"))],
    ]
    col_w = [45*mm, 120*mm]
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("BACKGROUND",    (0, 1), (-1, -1), C_LGRAY),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_GRAY),
        ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
        ("WORDWRAP",      (1, 1), (1, -1),  True),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return _section_header("Chain of Custody", st) + [t, Spacer(1, 4*mm)]


def _scores_section(report: Dict, st) -> List:
    mods = report.get("modules", {})
    final = report.get("final", {})

    key_map = {
        "Compression / ELA"      : ("compression_ela",        "ela_score"),
        "Splicing Detection"     : ("splicing_detection",      "splicing_score"),
        "AI-Generation"          : ("ai_generated_detection",  "ai_generated_score"),
        "Deepfake Detection"     : ("deepfake_detection",      "deepfake_score"),
    }

    rows = [["Module", "Signal Score", "ML Score", "Fused Score", "Verdict", "Bar"]]
    for display, (mod_key, _score_key) in key_map.items():
        m    = mods.get(mod_key, {})
        s    = m.get("score", 0)
        ml_s = m.get("ml_score", "—")
        sig_s= m.get("signal_score", s)
        ml_s_disp = f"{ml_s:.3f}" if isinstance(ml_s, float) else "—"
        rows.append([
            display,
            f"{sig_s:.3f}" if isinstance(sig_s, float) else "—",
            ml_s_disp,
            f"{s:.3f}",
            _score_label(s),
            "",   # bar placeholder
        ])

    col_w = [42*mm, 22*mm, 22*mm, 22*mm, 35*mm, 30*mm]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_GRAY),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    # Colour the verdict cells
    for i, (display, (mod_key, _)) in enumerate(key_map.items(), 1):
        m     = mods.get(mod_key, {})
        score = m.get("score", 0)
        c     = _score_colour(score)
        t.setStyle(TableStyle([
            ("TEXTCOLOR", (4, i), (4, i), c),
            ("FONTNAME",  (4, i), (4, i), "Helvetica-Bold"),
        ]))

    final_score = final.get("manipulation_probability", 0)
    fc = _score_colour(final_score)

    final_row_data = [
        ["FINAL MANIPULATION PROBABILITY",
         "", "", f"{final_score:.4f}",
         _score_label(final_score), ""]
    ]
    ft = Table(final_row_data, colWidths=col_w)
    ft.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",     (4, 0), (4, 0),  fc),
        ("ALIGN",         (1, 0), (-1, 0), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    return (
        _section_header("Detection Scores", st) +
        [t, Spacer(1, 2*mm), ft, Spacer(1, 5*mm)]
    )


def _shap_section(report: Dict, st) -> List:
    shap_data = report.get("shap_explanations", {})
    if not shap_data:
        return []

    elems = _section_header("Explainability (SHAP Analysis)", st)
    elems.append(Paragraph(
        "SHAP (SHapley Additive exPlanations) identifies which features drove "
        "each detection score. Features with positive SHAP values increase the "
        "manipulation probability; negative values decrease it.",
        st["body"]
    ))
    elems.append(Spacer(1, 3*mm))

    for module, exp in shap_data.items():
        if not exp.get("available"):
            continue
        elems.append(Paragraph(f"Module: {module.upper()}", st["h2"]))
        elems.append(Paragraph(exp.get("summary", ""), st["body"]))
        elems.append(Spacer(1, 2*mm))

        top = exp.get("top_features", [])[:6]
        if top:
            rows  = [["Feature Group", "SHAP Value", "Direction", "Description"]]
            for f in top:
                rows.append([
                    f["feature_group"],
                    f"{f['shap_value']:+.4f}",
                    f["direction"][:30],
                    f["description"][:45],
                ])
            t = Table(rows, colWidths=[30*mm, 22*mm, 55*mm, 60*mm])
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), C_BLUE),
                ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
                ("GRID",          (0, 0), (-1, -1), 0.3, C_GRAY),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            # Colour SHAP values
            for i, f in enumerate(top, 1):
                c = C_RED if f["shap_value"] > 0 else C_GREEN
                t.setStyle(TableStyle([("TEXTCOLOR", (1, i), (1, i), c)]))
            elems.append(t)
            elems.append(Spacer(1, 3*mm))

    return elems


def _evidence_section(report: Dict, st) -> List:
    ev_files = [f for f in report.get("evidence_files", []) if os.path.isfile(f)]
    if not ev_files:
        return []

    elems = _section_header("Visual Evidence", st)
    elems.append(Paragraph(
        "The following images were generated during analysis. "
        "Each is traceable to the SHA-256 hash in the chain-of-custody header.",
        st["body"]
    ))
    elems.append(Spacer(1, 3*mm))

    # Show up to 4 evidence images in a 2×2 grid
    img_w = 80*mm
    img_h = 55*mm
    shown = []
    for ef in ev_files[:4]:
        try:
            img = RLImage(ef, width=img_w, height=img_h, kind="proportional")
            cap = Path(ef).stem.split("_")[-1].replace("-", " ").title()
            shown.append([img, Paragraph(cap, st["caption"])])
        except Exception as e:
            logger.warning("Cannot embed %s: %s", ef, e)

    # Pack into rows of 2
    for i in range(0, len(shown), 2):
        row_items = shown[i:i+2]
        while len(row_items) < 2:
            row_items.append(["", ""])
        row_imgs = [[ri[0] for ri in row_items]]
        row_caps = [[ri[1] for ri in row_items]]
        ti = Table(row_imgs, colWidths=[img_w + 5*mm, img_w + 5*mm])
        tc = Table(row_caps, colWidths=[img_w + 5*mm, img_w + 5*mm])
        ti.setStyle(TableStyle([
            ("ALIGN",   (0,0), (-1,-1), "CENTER"),
            ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
        ]))
        tc.setStyle(TableStyle([
            ("ALIGN",   (0,0), (-1,-1), "CENTER"),
        ]))
        elems += [ti, tc, Spacer(1, 4*mm)]

    return elems


def _verdict_section(report: Dict, st) -> List:
    final = report.get("final", {})
    score = final.get("manipulation_probability", 0)
    label = final.get("label", "UNKNOWN")
    rec   = final.get("recommendation", "")
    fc    = _score_colour(score)

    verdict_text = f'<font color="{fc.hexval()}">{label}</font>'

    data = [[Paragraph(f"Final Score: {score:.4f}", st["verdict"]),
             Paragraph(verdict_text, st["verdict"])]]
    t = Table(data, colWidths=[90*mm, 80*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_LGRAY),
        ("BOX",        (0, 0), (-1, -1), 1.5, C_DARK),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    return (
        _section_header("Final Verdict", st) +
        [t, Spacer(1, 4*mm),
         Paragraph("Recommendation:", st["h2"]),
         Paragraph(rec, st["body"]),
         Spacer(1, 4*mm)]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def generate(report: Dict, output_path: str) -> str:
    """
    Generate a PDF forensic report from the master report dict.

    Parameters
    ----------
    report      : dict produced by report_generator.build_report()
    output_path : destination .pdf path

    Returns
    -------
    output_path (str)
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    st = _styles()

    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=32*mm,  bottomMargin=18*mm,
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="main"
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=frame,
                                       onPage=_header_footer)])

    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("Forensic Analysis Report", st["h1"]))
    story.append(Paragraph(
        f"Case: {report.get('case_id','N/A')}  |  "
        f"Tool: {report.get('tool','N/A')} v{report.get('version','N/A')}",
        st["small"]
    ))
    story.append(Spacer(1, 4*mm))

    # ── Sections ──────────────────────────────────────────────────────────────
    story += _coc_section(report, st)
    story += _scores_section(report, st)
    story += _verdict_section(report, st)
    story += _shap_section(report, st)
    story += _evidence_section(report, st)

    doc.build(story)
    logger.info("PDF report saved → %s", output_path)
    return output_path
