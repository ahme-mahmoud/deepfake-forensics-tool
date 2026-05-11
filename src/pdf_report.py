"""
pdf_report.py
=============
Forensic PDF Report Generator  —  v3.0

Generates a professional multi-page PDF from the report dict produced by
report_generator.build_report().

NEW schema keys used (NO legacy "modules" key):
    report["chain_of_custody"]
    report["module_scores"]       — {"ela": 0.3, "splicing": 0.6, ...}
    report["module_labels"]       — {"ela": "LIKELY AUTHENTIC", ...}
    report["ml_availability"]     — {"splicing": True, "ai_gen": False, ...}
    report["fusion_breakdown"]    — weighted contribution per module
    report["module_details"]      — rich per-module dicts for interpretation text
    report["final"]               — final verdict block
    report["evidence_files"]      — list of file paths
    report["shap_explanations"]   — always {} (SHAP disabled — handled gracefully)
    report["pdf_path"]            — set on the report dict after generation

Dependencies:
    reportlab  (pip install reportlab)

Usage:
    from pdf_report import generate
    pdf_path = generate(report, output_dir="reports")
    # report["pdf_path"] is also set automatically
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pdf_report")

# ── Score-colour thresholds ───────────────────────────────────────────────────
def _score_rgb(score: float):
    """Return an RGB tuple for a manipulation score."""
    if score < 0.25:
        return (0.10, 0.65, 0.25)   # green
    if score < 0.50:
        return (0.85, 0.65, 0.10)   # amber
    if score < 0.75:
        return (0.90, 0.40, 0.10)   # orange
    return (0.80, 0.10, 0.10)       # red


def _score_label_short(score: float) -> str:
    if score < 0.20: return "AUTHENTIC"
    if score < 0.40: return "LIKELY AUTHENTIC"
    if score < 0.60: return "INCONCLUSIVE"
    if score < 0.80: return "LIKELY MANIPULATED"
    return "MANIPULATED"


# ══════════════════════════════════════════════════════════════════════════════
# Internal builders  (all use reportlab primitives)
# ══════════════════════════════════════════════════════════════════════════════

def _build_pdf(report: Dict[str, Any], output_path: str) -> None:
    """Core PDF construction — called only after reportlab import succeeds."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    W, H   = A4
    MARGIN = 18 * mm
    styles = getSampleStyleSheet()

    # ── Custom paragraph styles ───────────────────────────────────────────────
    title_style = ParagraphStyle(
        "ForensicTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "ForensicH2",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#16213e"),
        spaceBefore=10,
        spaceAfter=4,
        borderPad=2,
    )
    body_style = ParagraphStyle(
        "ForensicBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        spaceAfter=3,
    )
    mono_style = ParagraphStyle(
        "ForensicMono",
        parent=styles["Code"],
        fontSize=8,
        leading=11,
    )
    warn_style = ParagraphStyle(
        "ForensicWarn",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#8B0000"),
        leading=13,
    )

    story: List = []

    def spacer(h: float = 4) -> None:
        story.append(Spacer(1, h * mm))

    def h2(text: str) -> None:
        story.append(Paragraph(text, h2_style))

    def body(text: str) -> None:
        story.append(Paragraph(text, body_style))

    def kv_table(rows: List[tuple], col_widths=None) -> None:
        if col_widths is None:
            col_widths = [65 * mm, W - 2 * MARGIN - 65 * mm]
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eaf6")),
            ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("LEADING",    (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)

    def score_bar_table(module_name: str, score: float,
                        ml_ok: bool, label: str) -> None:
        """Render one module score as a coloured progress bar row."""
        bar_pct    = max(0.02, min(score, 1.0))
        bar_w_full = W - 2 * MARGIN - 90 * mm
        bar_w_fill = bar_w_full * bar_pct
        r, g, b    = _score_rgb(score)
        fill_color = colors.Color(r, g, b)
        ml_tag     = "ML ✓" if ml_ok else "signal"

        bar_data = [[
            Paragraph(f"<b>{module_name}</b>", body_style),
            Paragraph(f"{score:.2f}", body_style),
            Paragraph(label, body_style),
            Paragraph(ml_tag, body_style),
        ]]
        bar_t = Table(
            bar_data,
            colWidths=[55 * mm, 18 * mm, 70 * mm, 20 * mm],
        )
        bar_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#e8eaf6")),
            ("BACKGROUND", (1, 0), (1, 0), fill_color),
            ("TEXTCOLOR",  (1, 0), (1, 0), colors.white),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(bar_t)
        spacer(1)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — Header + Chain of Custody + Module Score Summary
    # ══════════════════════════════════════════════════════════════════════════

    story.append(Paragraph("FORENSIC IMAGE ANALYSIS REPORT", title_style))
    story.append(Paragraph(
        f"{report.get('tool', 'DeepFake Forensics Tool')}  "
        f"v{report.get('version', '3.0')}",
        body_style,
    ))
    spacer(2)

    # ── Header metadata ────────────────────────────────────────────────────────
    h2("Report Information")
    kv_table([
        ("Generated (UTC)", report.get("generated_at", "N/A")),
        ("Case ID",         report.get("case_id", "N/A")),
        ("Investigator",    report.get("investigator_id", "AUTO")),
        ("Host",            report.get("hostname", "N/A")),
    ])
    spacer()

    # ── Chain of custody ──────────────────────────────────────────────────────
    h2("Chain of Custody")
    coc = report.get("chain_of_custody", {})
    file_size = coc.get("file_size", "N/A")
    if isinstance(file_size, int):
        file_size = f"{file_size:,} bytes"
    kv_table([
        ("Image Path",  str(coc.get("image_path",   "N/A"))),
        ("SHA-256",     str(coc.get("sha256",        "N/A"))),
        ("File Size",   file_size),
        ("Format",      str(coc.get("image_format",  "N/A"))),
        ("Colour Mode", str(coc.get("image_mode",    "N/A"))),
        ("Dimensions",  str(coc.get("image_size",    "N/A"))),
    ])
    spacer()

    # ── Module score summary ───────────────────────────────────────────────────
    h2("Module Score Summary")
    module_scores  = report.get("module_scores",  {})
    module_labels  = report.get("module_labels",  {})
    ml_avail       = report.get("ml_availability", {})

    _display_names = {
        "ela"      : "Compression / ELA",
        "splicing" : "Splicing Detection",
        "ai_gen"   : "AI-Generation Detection",
        "deepfake" : "Deepfake Detection",
    }
    for key in ("ela", "splicing", "ai_gen", "deepfake"):
        score = module_scores.get(key, 0.0)
        label = module_labels.get(key, _score_label_short(score))
        ml_ok = ml_avail.get(key, False)
        score_bar_table(_display_names.get(key, key), score, ml_ok, label)
    spacer()

    # ── Final verdict block ────────────────────────────────────────────────────
    h2("Final Verdict")
    final      = report.get("final", {})
    fin_score  = final.get("manipulation_probability", 0.0)
    fin_label  = final.get("label", _score_label_short(fin_score))
    fin_conf   = final.get("confidence", "N/A")
    fin_dom    = final.get("dominant_module", "N/A")
    fin_above  = final.get("modules_above_50pct", 0)
    fin_rec    = final.get("recommendation", "")

    r, g, b    = _score_rgb(fin_score)
    verdict_color = colors.Color(r, g, b)

    verdict_data = [[
        Paragraph("<b>Manipulation Probability</b>", body_style),
        Paragraph(f"<b>{fin_score:.4f}</b>", body_style),
    ], [
        Paragraph("Assessment", body_style),
        Paragraph(fin_label, body_style),
    ], [
        Paragraph("Confidence", body_style),
        Paragraph(fin_conf, body_style),
    ], [
        Paragraph("Dominant Module", body_style),
        Paragraph(fin_dom.upper(), body_style),
    ], [
        Paragraph("Modules > 50%", body_style),
        Paragraph(str(fin_above), body_style),
    ]]

    vt = Table(verdict_data, colWidths=[65 * mm, W - 2 * MARGIN - 65 * mm])
    vt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), verdict_color),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#e8eaf6")),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
    ]))
    story.append(vt)
    spacer(2)

    if fin_rec:
        story.append(Paragraph(fin_rec, warn_style if fin_score >= 0.55 else body_style))
    spacer()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — Per-Module Detail + Fusion Breakdown
    # ══════════════════════════════════════════════════════════════════════════

    story.append(PageBreak())
    story.append(Paragraph("Detailed Module Analysis", title_style))
    spacer(2)

    details = report.get("module_details", {})

    _detail_map = [
        ("compression_ela",       "1. Compression / ELA"),
        ("splicing_detection",    "2. Splicing Detection"),
        ("ai_generated_detection","3. AI-Generation Detection"),
        ("deepfake_detection",    "4. Deepfake Detection"),
    ]

    for detail_key, display_name in _detail_map:
        m = details.get(detail_key, {})
        if not m:
            continue
        h2(display_name)
        ml_tag = "TRAINED ✓" if m.get("ml_available") else "Signal-only (no ML model)"
        rows = [
            ("ML Status",    ml_tag),
            ("Fused Score",  f"{m.get('score', 0.0):.4f}"),
            ("Signal Score", f"{m.get('signal_score', 0.0):.4f}"),
        ]
        if m.get("ml_score") is not None:
            rows.append(("ML Score", f"{m['ml_score']:.4f}"))
        rows.append(("Verdict", m.get("label", "N/A")))
        if m.get("confidence"):
            rows.append(("Confidence", str(m["confidence"])))
        if "suspicious_regions" in m:
            rows.append(("Suspicious Regions", str(m["suspicious_regions"])))
        kv_table(rows)
        interpretation = m.get("interpretation", "")
        if interpretation:
            body(interpretation)
        spacer(2)

    # ── Score fusion breakdown ─────────────────────────────────────────────────
    h2("Score Fusion Breakdown")
    breakdown = report.get("fusion_breakdown", {})
    if breakdown:
        bd_rows  = [["Module", "Raw Score", "Weighted Contribution"]]
        for mod, contrib in breakdown.items():
            raw = module_scores.get(mod, 0.0)
            bd_rows.append([mod, f"{raw:.4f}", f"{contrib:.4f}"])
        bd_t = Table(
            bd_rows,
            colWidths=[60 * mm, 50 * mm, 60 * mm],
        )
        bd_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f5f5f5"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ]))
        story.append(bd_t)
    else:
        body("Fusion breakdown not available.")
    spacer()

    # ── ML availability summary ────────────────────────────────────────────────
    h2("ML Model Availability")
    ml_rows = [["Module", "Status"]]
    for mod, available in ml_avail.items():
        status = "TRAINED ✓" if available else "Not trained — signal only"
        ml_rows.append([_display_names.get(mod, mod), status])
    ml_t = Table(ml_rows, colWidths=[80 * mm, W - 2 * MARGIN - 80 * mm])
    ml_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
    ]))
    story.append(ml_t)
    spacer()

    # ── SHAP — graceful disabled notice ───────────────────────────────────────
    shap_data = report.get("shap_explanations", {})
    if not shap_data:
        h2("Feature Explainability (SHAP)")
        body(
            "SHAP explainability is currently disabled. Feature importance "
            "analysis will be available once the ensemble classifiers expose "
            "a compatible tree-based interface."
        )
        spacer()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — Evidence Gallery
    # ══════════════════════════════════════════════════════════════════════════

    evidence_files = report.get("evidence_files", [])
    image_files    = [
        f for f in evidence_files
        if str(f).lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        and os.path.isfile(str(f))
    ]

    if image_files:
        story.append(PageBreak())
        story.append(Paragraph("Evidence Gallery", title_style))
        spacer(2)

        for ef in image_files:
            try:
                img_w  = W - 2 * MARGIN
                rl_img = RLImage(ef, width=img_w, height=img_w * 0.45,
                                 kind="proportional")
                caption = Paragraph(
                    f"<i>{Path(ef).name}</i>", body_style
                )
                story.append(rl_img)
                story.append(caption)
                spacer(3)
            except Exception as exc:
                logger.warning("Could not embed evidence image %s: %s", ef, exc)
                body(f"[Image could not be embedded: {Path(ef).name}]")

    # ── Footer / end-of-report notice ─────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        f"END OF REPORT  —  {report.get('tool', 'Forensic Tool')}  "
        f"v{report.get('version', '3.0')}  "
        f"—  Case: {report.get('case_id', 'N/A')}",
        ParagraphStyle("footer", parent=body_style,
                       textColor=colors.HexColor("#555555"), fontSize=8),
    ))

    # ── Build PDF ──────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title=f"Forensic Report — {report.get('case_id', 'N/A')}",
        author=report.get("investigator_id", "AUTO"),
    )
    doc.build(story)
    logger.info("PDF report generated → %s", output_path)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate(
    report:     Dict[str, Any],
    output_dir: str = "reports",
    filename:   Optional[str] = None,
) -> Optional[str]:
    """
    Generate a PDF forensic report from a report dict.

    Parameters
    ----------
    report     : dict produced by report_generator.build_report()
    output_dir : directory to write the PDF into
    filename   : override the auto-generated filename (optional)

    Returns
    -------
    str path to the generated PDF, or None if reportlab is not installed.
    Also sets report["pdf_path"] = <path> for downstream convenience.
    """
    try:
        import reportlab  # noqa: F401 — just check availability
    except ImportError:
        logger.warning(
            "reportlab not installed — PDF generation skipped.  "
            "Install with: pip install reportlab"
        )
        report["pdf_path"] = None
        return None

    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        case_id  = report.get("case_id", "UNKNOWN")
        filename = f"forensic_report_{case_id}.pdf"

    output_path = os.path.join(output_dir, filename)

    try:
        _build_pdf(report, output_path)
        report["pdf_path"] = output_path
        return output_path
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc, exc_info=True)
        report["pdf_path"] = None
        return None
