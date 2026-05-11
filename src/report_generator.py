"""
report_generator.py  —  v3.0
==============================
Forensic Evidence Report Generator

Compatible with the NEW architecture:
    src/ai_gen_module.py
    src/splicing_module.py
    src/deepfake_module.py
    src/compression_analysis.py
    src/score_fusion.py

Report schema (NEW — matches dashboard.py and pdf_report.py expectations):
    report["module_scores"]     — {"ela": 0.3, "splicing": 0.6, ...}
    report["module_labels"]     — {"ela": "LIKELY AUTHENTIC", ...}
    report["ml_availability"]   — {"splicing": True, "ai_gen": False, ...}
    report["fusion_breakdown"]  — weighted_breakdown dict from score_fusion
    report["final"]             — final verdict block
    report["evidence_files"]    — list of evidence file paths
    report["pdf_path"]          — set externally by pdf_report after generation
    report["shap_explanations"] — always {} (SHAP disabled)

Output formats:
    - Human-readable .txt  (always generated)
    - Structured  .json    (always generated)

Legacy keys REMOVED:
    report["modules"]  — replaced by module_scores / module_labels / ml_availability
"""

import json
import logging
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from score_fusion import compute_final_score, fuse_module_score

logger = logging.getLogger("report_generator")

TOOL_VERSION = "3.0.0"
TOOL_NAME    = "DeepFake Forensics Tool"


# ══════════════════════════════════════════════════════════════════════════════
# Interpretation helpers  (public API — safe to import from other modules)
# ══════════════════════════════════════════════════════════════════════════════

def _score_label(score: float) -> str:
    if score < 0.20: return "AUTHENTIC (very low suspicion)"
    if score < 0.40: return "LIKELY AUTHENTIC (low suspicion)"
    if score < 0.60: return "INCONCLUSIVE (moderate suspicion)"
    if score < 0.80: return "LIKELY MANIPULATED (high suspicion)"
    return "MANIPULATED (very high suspicion)"


def _score_bar(score: float, width: int = 30) -> str:
    filled = int(round(score * width))
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:.2f}"


def _interpret_ela(score: float) -> str:
    if score < 0.25:
        return ("ELA residuals are low and spatially uniform, consistent with "
                "a singly-compressed JPEG with no post-processing.")
    if score < 0.50:
        return ("Moderate ELA residuals detected. Some regions compress "
                "differently — possible light editing or format conversion artefacts.")
    if score < 0.75:
        return ("Significant ELA anomalies. Multiple regions show compression "
                "inconsistencies typical of spliced or locally-edited content.")
    return ("Strong ELA evidence of manipulation. Large bright regions in the "
            "ELA heat-map indicate a different compression history — hallmark "
            "of image splicing or object insertion.")


def _interpret_splicing(signal_score: float, ml_score: Optional[float]) -> str:
    if ml_score is not None and ml_score > 0.60:
        return ("ML ensemble detected significant splicing/tampering indicators: "
                "ELA inconsistencies across JPEG quality levels, JPEG ghost "
                "artifacts, and/or anomalous block-level noise or color variance.")
    if signal_score > 0.50:
        return ("Heuristic analysis found block-level ELA variance and possible "
                "copy-move patterns. ML confirmation unavailable or inconclusive "
                "— manual review recommended.")
    return ("No significant splicing indicators. ELA distribution, block noise "
            "levels, and copy-move analysis are within normal bounds.")


def _interpret_ai(signal_score: float, ml_score: Optional[float]) -> str:
    if ml_score is not None and ml_score > 0.60:
        return ("ML ensemble detected AI-generation signatures: spectral frequency "
                "decay inconsistent with camera optics, low noise residual (no "
                "sensor noise), and anomalous texture regularity characteristic "
                "of GAN or diffusion-model output.")
    if signal_score > 0.50:
        return ("Heuristic frequency/noise analysis flagged possible AI generation. "
                "ML confirmation inconclusive — consider manual spectral inspection.")
    return ("No strong AI-generation indicators. Spectral, noise, and texture "
            "profiles are consistent with a photographic origin.")


def _interpret_deepfake(score: float, ml_score: Optional[float],
                        label: str = "") -> str:
    if "ERROR" in label.upper():
        return "Deepfake analysis encountered an error. Results may be unreliable."
    if ml_score is not None and ml_score > 0.60:
        return ("ML ensemble detected deepfake characteristics: face geometry "
                "inconsistencies, unnatural blending boundaries, and/or skin-tone "
                "color distribution typical of GAN face-swapping.")
    if score > 0.50:
        return ("Moderate deepfake signals detected. Confidence is limited without "
                "a fully trained ML model — recommend manual review.")
    return ("No significant deepfake indicators. Facial geometry and color "
            "distribution appear natural and internally consistent.")


def _overall_recommendation(final_score: float) -> str:
    if final_score < 0.30:
        return ("RECOMMENDATION: Image appears AUTHENTIC. No further forensic "
                "investigation is warranted based on automated analysis. Human "
                "review is advised before a final conclusion.")
    if final_score < 0.55:
        return ("RECOMMENDATION: INCONCLUSIVE. Some anomalies were detected but "
                "are insufficient to conclude deliberate manipulation. Manual "
                "examination by a certified digital forensics examiner is strongly "
                "recommended.")
    return ("RECOMMENDATION: Image is LIKELY MANIPULATED. Multiple independent "
            "forensic indicators support this conclusion. Do NOT accept this image "
            "as authentic evidence without certified forensic review. Preserve the "
            "original file and this report as part of the chain of custody.")


# ══════════════════════════════════════════════════════════════════════════════
# Core report builder
# ══════════════════════════════════════════════════════════════════════════════

def build_report(
    image_path:      str,
    ela_result:      Dict[str, Any],
    splicing_result: Dict[str, Any],
    ai_result:       Dict[str, Any],
    deepfake_result: Dict[str, Any],
    investigator_id: str = "AUTO",
    case_id:         Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assemble the master forensic report from individual module results.

    Each module result (splicing_result, ai_result, deepfake_result) is
    expected to contain at minimum:
        score        — fused/final score for this module  [0, 1]
        ml_score     — ML ensemble probability (float or None)
        signal_score — heuristic-only score (float)
        ml_available — bool

    Returns a report dict using the NEW schema:
        module_scores, module_labels, ml_availability,
        fusion_breakdown, final, evidence_files, shap_explanations
    """
    now    = datetime.now(timezone.utc)
    sha256 = ela_result.get("sha256", "UNKNOWN")

    # ── Pull fused scores ──────────────────────────────────────────────────────
    ela_score      = float(ela_result.get("score",
                           ela_result.get("ela_score", 0.0)))
    splicing_score = float(splicing_result.get("score",
                           splicing_result.get("fused_score",
                           splicing_result.get("probability_splicing", 0.0))))
    ai_score       = float(ai_result.get("score",
                           ai_result.get("fused_score",
                           ai_result.get("probability_ai_generated", 0.0))))
    deepfake_score = float(deepfake_result.get("score",
                           deepfake_result.get("fused_score",
                           deepfake_result.get("probability_deepfake", 0.0))))

    # ── Pull ML scores (None when model not loaded) ───────────────────────────
    spl_ml  = splicing_result.get("ml_score")
    ai_ml   = ai_result.get("ml_score")
    dfk_ml  = deepfake_result.get("ml_score")

    # ── Pull signal scores ────────────────────────────────────────────────────
    spl_sig = float(splicing_result.get("signal_score", splicing_score))
    ai_sig  = float(ai_result.get("signal_score",  ai_score))
    dfk_sig = float(deepfake_result.get("signal_score", deepfake_score))

    # ── ML availability flags ─────────────────────────────────────────────────
    spl_ml_ok = bool(splicing_result.get("ml_available", spl_ml is not None))
    ai_ml_ok  = bool(ai_result.get("ml_available",       ai_ml  is not None))
    dfk_ml_ok = bool(deepfake_result.get("ml_available",  dfk_ml is not None))

    # ── NEW: module_scores dict — authoritative input to score_fusion ─────────
    module_scores_raw = {
        "ela"      : ela_score,
        "splicing" : splicing_score,
        "ai_gen"   : ai_score,
        "deepfake" : deepfake_score,
    }

    # ── NEW: module_labels dict ───────────────────────────────────────────────
    module_labels = {mod: _score_label(s) for mod, s in module_scores_raw.items()}

    # ── NEW: ml_availability dict ─────────────────────────────────────────────
    ml_availability = {
        "ela"      : False,        # ELA is always signal-only
        "splicing" : spl_ml_ok,
        "ai_gen"   : ai_ml_ok,
        "deepfake" : dfk_ml_ok,
    }

    # ── Final score via score_fusion (authoritative) ──────────────────────────
    fusion_result    = compute_final_score(module_scores_raw)
    final_score      = fusion_result["final_score"]
    fusion_breakdown = fusion_result["weighted_breakdown"]   # per-module weighted contributions
    dominant_module  = fusion_result["dominant_module"]
    confidence_level = fusion_result["confidence"]

    # ── Evidence file inventory ───────────────────────────────────────────────
    evidence_files: List[str] = []
    _evidence_keys = (
        "ela_image_path", "panel_image_path",
        "edge_heatmap_path", "copymove_img_path",
        "spectrum_path", "noise_path", "face_annotated_path",
    )
    for key in _evidence_keys:
        for result in (ela_result, splicing_result, ai_result, deepfake_result):
            val = result.get(key)
            if val and os.path.isfile(str(val)):
                evidence_files.append(str(val))

    # ── Module version strings (best-effort) ──────────────────────────────────
    module_versions = {
        "compression_analysis" : ela_result.get("version", "1.0"),
        "splicing_module"      : splicing_result.get("version", "2.0"),
        "ai_gen_module"        : ai_result.get("version", "3.0"),
        "deepfake_module"      : deepfake_result.get("version", "2.0"),
        "score_fusion"         : "1.0",
    }

    # ── Per-module detail blocks (for text/PDF rendering) ────────────────────
    # Kept as a sub-key inside the report for rendering convenience.
    # NOT the old top-level "modules" key — stored as "module_details".
    module_details = {
        "compression_ela": {
            "score"             : ela_score,
            "signal_score"      : ela_score,
            "ml_score"          : None,
            "ml_available"      : False,
            "label"             : module_labels["ela"],
            "suspicious_regions": len(ela_result.get("suspicious_regions", [])),
            "interpretation"    : _interpret_ela(ela_score),
        },
        "splicing_detection": {
            "score"        : splicing_score,
            "signal_score" : spl_sig,
            "ml_score"     : spl_ml,
            "ml_available" : spl_ml_ok,
            "label"        : module_labels["splicing"],
            "is_spliced"   : splicing_result.get("is_spliced", splicing_score > 0.55),
            "confidence"   : splicing_result.get("confidence"),
            "interpretation": _interpret_splicing(spl_sig, spl_ml),
        },
        "ai_generated_detection": {
            "score"        : ai_score,
            "signal_score" : ai_sig,
            "ml_score"     : ai_ml,
            "ml_available" : ai_ml_ok,
            "label"        : module_labels["ai_gen"],
            "is_ai"        : ai_result.get("is_ai_generated", ai_score > 0.65),
            "confidence"   : ai_result.get("confidence"),
            "interpretation": _interpret_ai(ai_sig, ai_ml),
        },
        "deepfake_detection": {
            "score"        : deepfake_score,
            "signal_score" : dfk_sig,
            "ml_score"     : dfk_ml,
            "ml_available" : dfk_ml_ok,
            "label"        : module_labels["deepfake"],
            "df_label"     : deepfake_result.get("label", ""),
            "confidence"   : deepfake_result.get("confidence"),
            "interpretation": _interpret_deepfake(
                deepfake_score, dfk_ml,
                deepfake_result.get("label", "")),
        },
    }

    # ── Assemble final report dict ─────────────────────────────────────────────
    report: Dict[str, Any] = {
        # ── Header ────────────────────────────────────────────────────────────
        "tool"            : TOOL_NAME,
        "version"         : TOOL_VERSION,
        "generated_at"    : now.isoformat(),
        "investigator_id" : investigator_id,
        "case_id"         : case_id or f"CASE-{now.strftime('%Y%m%d-%H%M%S')}",
        "hostname"        : socket.gethostname(),
        "platform"        : platform.platform(),
        "module_versions" : module_versions,

        # ── Chain of custody ──────────────────────────────────────────────────
        "chain_of_custody": {
            "image_path"  : str(Path(image_path).resolve()),
            "sha256"      : sha256,
            "file_size"   : ela_result.get("metadata", {}).get("file_bytes", "N/A"),
            "image_format": ela_result.get("metadata", {}).get("format",     "N/A"),
            "image_mode"  : ela_result.get("metadata", {}).get("mode",       "N/A"),
            "image_size"  : ela_result.get("metadata", {}).get("size_px",    "N/A"),
        },

        # ── NEW flat score keys (used by dashboard.py and pdf_report.py) ─────
        "module_scores"   : module_scores_raw,
        "module_labels"   : module_labels,
        "ml_availability" : ml_availability,
        "fusion_breakdown": fusion_breakdown,

        # ── Per-module rich detail (used by text renderer & PDF) ──────────────
        "module_details"  : module_details,

        # ── Final verdict ──────────────────────────────────────────────────────
        "final": {
            "manipulation_probability" : final_score,
            "label"                    : _score_label(final_score),
            "confidence"               : confidence_level.upper(),
            "dominant_module"          : dominant_module,
            "modules_above_50pct"      : fusion_result.get("modules_above_50pct", 0),
            "recommendation"           : _overall_recommendation(final_score),
        },

        # ── Evidence inventory ─────────────────────────────────────────────────
        "evidence_files"    : evidence_files,

        # ── SHAP disabled ─────────────────────────────────────────────────────
        "shap_explanations" : {},

        # ── pdf_path filled in later by pdf_report.generate() ─────────────────
        "pdf_path"          : None,
    }

    return report


# ══════════════════════════════════════════════════════════════════════════════
# Text report renderer
# ══════════════════════════════════════════════════════════════════════════════

def render_text_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    DIV  = "═" * 72
    DIV2 = "─" * 72

    def h1(title: str) -> None:
        lines.append(f"\n{DIV}"); lines.append(f"  {title}"); lines.append(DIV)

    def h2(title: str) -> None:
        lines.append(f"\n{DIV2}"); lines.append(f"  {title}"); lines.append(DIV2)

    def kv(key: str, value: Any, indent: int = 2) -> None:
        lines.append(f"{' ' * indent}{key:<30}: {value}")

    # ── Title ─────────────────────────────────────────────────────────────────
    h1(f"  {report['tool']}  v{report['version']}  — FORENSIC ANALYSIS REPORT")
    kv("Generated (UTC)", report["generated_at"])
    kv("Case ID",         report["case_id"])
    kv("Investigator ID", report["investigator_id"])
    kv("Host",            report["hostname"])

    # ── Module versions ───────────────────────────────────────────────────────
    versions = report.get("module_versions", {})
    if versions:
        h2("MODULE VERSIONS")
        for mod, ver in versions.items():
            kv(mod, ver)

    # ── Chain of custody ──────────────────────────────────────────────────────
    h2("CHAIN OF CUSTODY")
    coc = report["chain_of_custody"]
    kv("Image Path",  coc["image_path"])
    kv("SHA-256",     coc["sha256"])
    kv("File Size",
       f"{coc['file_size']:,} bytes" if isinstance(coc["file_size"], int)
       else coc["file_size"])
    kv("Format",      coc["image_format"])
    kv("Colour Mode", coc["image_mode"])
    kv("Dimensions",  str(coc["image_size"]))

    # ── Module analysis results — read from module_details ───────────────────
    h2("MODULE ANALYSIS RESULTS")
    details = report.get("module_details", {})

    def module_block(display_name: str, key: str) -> None:
        m = details.get(key, {})
        if not m:
            lines.append(f"\n  ► {display_name}  [no data]")
            return
        ml_tag = "ML✓" if m.get("ml_available") else "signal-only"
        lines.append(f"\n  ► {display_name}  [{ml_tag}]")
        lines.append(f"    Fused Score  : {_score_bar(m.get('score', 0.0))}")
        if m.get("signal_score") is not None:
            lines.append(f"    Signal Score : {m['signal_score']:.4f}")
        if m.get("ml_score") is not None:
            lines.append(f"    ML Score     : {m['ml_score']:.4f}")
        lines.append(f"    Verdict      : {m.get('label', 'N/A')}")
        if m.get("confidence"):
            lines.append(f"    Confidence   : {m['confidence']}")
        lines.append(f"    Analysis     : {m.get('interpretation', '')}")

    module_block("1. Compression / ELA",       "compression_ela")
    module_block("2. Splicing Detection",       "splicing_detection")
    module_block("3. AI-Generation Detection",  "ai_generated_detection")
    module_block("4. Deepfake Detection",       "deepfake_detection")

    # ── Score fusion breakdown ────────────────────────────────────────────────
    h2("SCORE FUSION BREAKDOWN")
    breakdown = report.get("fusion_breakdown", {})
    for mod, contrib in breakdown.items():
        raw = report.get("module_scores", {}).get(mod, 0.0)
        lines.append(f"  {mod:<12}  raw={raw:.4f}  weighted_contrib={contrib:.4f}")

    # ── Final verdict ──────────────────────────────────────────────────────────
    h2("FINAL VERDICT")
    final = report["final"]
    lines.append(f"\n  MANIPULATION PROBABILITY: "
                 f"{_score_bar(final['manipulation_probability'], 40)}")
    lines.append(f"  ASSESSMENT              : {final['label']}")
    lines.append(f"  CONFIDENCE              : {final.get('confidence', 'N/A')}")
    lines.append(f"  DOMINANT MODULE         : {final.get('dominant_module', 'N/A')}")
    lines.append(f"  MODULES ABOVE 50%       : {final.get('modules_above_50pct', 0)}")
    lines.append(f"\n  {final['recommendation']}")

    # ── ML availability ────────────────────────────────────────────────────────
    h2("ML MODEL AVAILABILITY")
    ml_avail = report.get("ml_availability", {})
    for mod, available in ml_avail.items():
        status = "TRAINED ✓" if available else "NOT TRAINED — signal only"
        kv(mod, status)

    # ── Evidence inventory ─────────────────────────────────────────────────────
    h2("EVIDENCE FILE INVENTORY")
    if report.get("evidence_files"):
        for i, ef in enumerate(report["evidence_files"], 1):
            lines.append(f"  [{i:02d}] {ef}")
    else:
        lines.append("  No evidence files saved.")

    lines.append(f"\n{DIV}")
    lines.append(f"  END OF REPORT — {report['tool']}  v{report['version']}")
    lines.append(DIV + "\n")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Save helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_report(
    report:     Dict[str, Any],
    output_dir: str  = "reports",
    save_json:  bool = True,
    save_txt:   bool = True,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stem  = f"forensic_report_{report.get('case_id', 'UNKNOWN')}"
    paths: Dict[str, str] = {}

    if save_json:
        p = os.path.join(output_dir, f"{stem}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        paths["json"] = p
        logger.info("JSON report → %s", p)

    if save_txt:
        p = os.path.join(output_dir, f"{stem}.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(render_text_report(report))
        paths["txt"] = p
        logger.info("Text report → %s", p)

    return paths


def print_report(report: Dict[str, Any]) -> None:
    """Print a formatted text report to stdout."""
    print(render_text_report(report))
