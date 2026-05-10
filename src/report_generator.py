"""
report_generator.py  —  v2.0
==============================
Forensic Evidence Report Generator

Compatible with app.py v4.0 (ai_gen_module_v3, splicing_module_v2,
deepfake_classifier_v2).

Changes from v1:
    - Module score keys updated to match new module outputs
      (probability_splicing, probability_ai_generated, etc.)
    - ml_score / signal_score / fused_score now surfaced per module
    - Module versions block rendered in report
    - ml_availability shown per module
    - _score_label / _overall_recommendation unchanged (safe to import)

Output formats:
    - Human-readable .txt  (always generated)
    - Structured  .json    (always generated)
"""

import json
import logging
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("report_generator")

TOOL_VERSION = "2.0.0"
TOOL_NAME    = "DeepFake Forensics Tool"


# ══════════════════════════════════════════════════════════════════════════════
# Interpretation helpers  (unchanged API — safe to import from other modules)
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
        return ("ML ensemble (GBM + ExtraTrees + SVM) detected significant "
                "splicing/tampering indicators: ELA inconsistencies across "
                "JPEG quality levels, JPEG ghost artifacts, and/or anomalous "
                "block-level noise or color variance.")
    if signal_score > 0.50:
        return ("Heuristic analysis found block-level ELA variance and "
                "possible copy-move patterns. ML confirmation unavailable "
                "or inconclusive — manual review recommended.")
    return ("No significant splicing indicators. ELA distribution, block "
            "noise levels, and copy-move analysis are within normal bounds.")


def _interpret_ai(signal_score: float, ml_score: Optional[float]) -> str:
    if ml_score is not None and ml_score > 0.60:
        return ("ML ensemble detected AI-generation signatures: spectral "
                "frequency decay inconsistent with camera optics, low noise "
                "residual (no sensor noise), and anomalous texture regularity "
                "characteristic of GAN or diffusion-model output.")
    if signal_score > 0.50:
        return ("Heuristic frequency/noise analysis flagged possible AI "
                "generation. ML confirmation inconclusive — consider manual "
                "spectral inspection.")
    return ("No strong AI-generation indicators. Spectral, noise, and texture "
            "profiles are consistent with a photographic origin.")


def _interpret_deepfake(score: float, ml_score: Optional[float],
                        label: str = "") -> str:
    if "ERROR" in label.upper():
        return "Deepfake analysis encountered an error. Results may be unreliable."
    if ml_score is not None and ml_score > 0.60:
        return ("ML ensemble detected deepfake characteristics: face geometry "
                "inconsistencies, unnatural blending boundaries, and/or "
                "skin-tone color distribution typical of GAN face-swapping.")
    if score > 0.50:
        return ("Moderate deepfake signals detected. Confidence is limited "
                "without a fully trained ML model — recommend manual review.")
    return ("No significant deepfake indicators. Facial geometry and color "
            "distribution appear natural and internally consistent.")


def _overall_recommendation(final_score: float) -> str:
    if final_score < 0.30:
        return ("RECOMMENDATION: Image appears AUTHENTIC. No further forensic "
                "investigation is warranted based on automated analysis. Human "
                "review is advised before a final conclusion.")
    if final_score < 0.55:
        return ("RECOMMENDATION: INCONCLUSIVE. Some anomalies were detected "
                "but are insufficient to conclude deliberate manipulation. "
                "Manual examination by a certified digital forensics examiner "
                "is strongly recommended.")
    return ("RECOMMENDATION: Image is LIKELY MANIPULATED. Multiple independent "
            "forensic indicators support this conclusion. Do NOT accept this "
            "image as authentic evidence without certified forensic review. "
            "Preserve the original file and this report as part of the chain "
            "of custody.")


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

    Compatible with app.py v4.0 wrapper output:
        splicing_result / ai_result / deepfake_result each contain:
            score        — fused final score for this module
            ml_score     — ML ensemble probability (float or None)
            signal_score — heuristic-only score

    Returns
    -------
    Nested dict — serialisable to JSON, formattable as text, renderable as PDF.
    """
    now    = datetime.now(timezone.utc)
    sha256 = ela_result.get("sha256", "UNKNOWN")

    # ── Pull scores (prefer 'score' key set by app.py wrappers) ──────────────
    ela_score      = float(ela_result.get("score",
                           ela_result.get("ela_score", 0.0)))
    splicing_score = float(splicing_result.get("score",
                           splicing_result.get("probability_splicing", 0.0)))
    ai_score       = float(ai_result.get("score",
                           ai_result.get("probability_ai_generated", 0.0)))
    deepfake_score = float(deepfake_result.get("score",
                           deepfake_result.get("probability_deepfake", 0.0)))

    # ── ML scores (may be None if models not trained) ─────────────────────────
    spl_ml  = splicing_result.get("ml_score")
    ai_ml   = ai_result.get("ml_score")
    dfk_ml  = deepfake_result.get("ml_score")

    spl_sig = float(splicing_result.get("signal_score", splicing_score))
    ai_sig  = float(ai_result.get("signal_score",  ai_score))
    dfk_sig = float(deepfake_result.get("signal_score", deepfake_score))

    # ── ML availability flags ─────────────────────────────────────────────────
    spl_ml_ok  = splicing_result.get("ml_available", spl_ml is not None)
    ai_ml_ok   = ai_result.get("ml_available",       ai_ml  is not None)
    dfk_ml_ok  = deepfake_result.get("ml_available",  dfk_ml is not None)

    # ── Final score (computed by app.py via score_fusion; fallback = average) ─
    final_score = round(
        float(sum([ela_score, splicing_score, ai_score, deepfake_score]) / 4), 4
    )

    # ── Evidence file inventory ───────────────────────────────────────────────
    evidence_files: List[str] = []
    for key in ("ela_image_path", "panel_image_path",
                "edge_heatmap_path", "copymove_img_path",
                "spectrum_path", "noise_path", "face_annotated_path"):
        for result in (ela_result, splicing_result, ai_result, deepfake_result):
            val = result.get(key)
            if val and os.path.isfile(str(val)):
                evidence_files.append(str(val))

    report = {
        # ── Header ────────────────────────────────────────────────────────────
        "tool"            : TOOL_NAME,
        "version"         : TOOL_VERSION,
        "generated_at"    : now.isoformat(),
        "investigator_id" : investigator_id,
        "case_id"         : case_id or f"CASE-{now.strftime('%Y%m%d-%H%M%S')}",
        "hostname"        : socket.gethostname(),
        "platform"        : platform.platform(),

        # ── Chain of custody ──────────────────────────────────────────────────
        "chain_of_custody": {
            "image_path"  : str(Path(image_path).resolve()),
            "sha256"      : sha256,
            "file_size"   : ela_result.get("metadata", {}).get("file_bytes", "N/A"),
            "image_format": ela_result.get("metadata", {}).get("format",     "N/A"),
            "image_mode"  : ela_result.get("metadata", {}).get("mode",       "N/A"),
            "image_size"  : ela_result.get("metadata", {}).get("size_px",    "N/A"),
        },

        # ── Module results ────────────────────────────────────────────────────
        "modules": {
            "compression_ela": {
                "score"             : ela_score,
                "signal_score"      : ela_score,
                "ml_score"          : None,
                "ml_available"      : False,
                "label"             : _score_label(ela_score),
                "suspicious_regions": len(ela_result.get("suspicious_regions", [])),
                "interpretation"    : _interpret_ela(ela_score),
            },
            "splicing_detection": {
                "score"        : splicing_score,
                "signal_score" : spl_sig,
                "ml_score"     : spl_ml,
                "ml_available" : spl_ml_ok,
                "label"        : _score_label(splicing_score),
                # v2 raw keys (pass through if present)
                "is_spliced"   : splicing_result.get("is_spliced", splicing_score > 0.55),
                "confidence"   : splicing_result.get("confidence"),
                "interpretation": _interpret_splicing(spl_sig, spl_ml),
            },
            "ai_generated_detection": {
                "score"        : ai_score,
                "signal_score" : ai_sig,
                "ml_score"     : ai_ml,
                "ml_available" : ai_ml_ok,
                "label"        : _score_label(ai_score),
                "is_ai"        : ai_result.get("is_ai_generated", ai_score > 0.65),
                "confidence"   : ai_result.get("confidence"),
                "interpretation": _interpret_ai(ai_sig, ai_ml),
            },
            "deepfake_detection": {
                "score"        : deepfake_score,
                "signal_score" : dfk_sig,
                "ml_score"     : dfk_ml,
                "ml_available" : dfk_ml_ok,
                "label"        : _score_label(deepfake_score),
                "df_label"     : deepfake_result.get("label", ""),
                "confidence"   : deepfake_result.get("confidence"),
                "interpretation": _interpret_deepfake(
                    deepfake_score, dfk_ml,
                    deepfake_result.get("label", "")),
            },
        },

        # ── Final verdict ──────────────────────────────────────────────────────
        "final": {
            "manipulation_probability" : final_score,
            "label"                    : _score_label(final_score),
            "confidence"               : "HIGH" if abs(final_score - 0.5) > 0.25 else
                                         "MEDIUM" if abs(final_score - 0.5) > 0.10 else
                                         "LOW",
            "dominant_module"          : max(
                {"ela": ela_score, "splicing": splicing_score,
                 "ai_gen": ai_score, "deepfake": deepfake_score}.items(),
                key=lambda x: x[1]
            )[0],
            "recommendation"           : _overall_recommendation(final_score),
        },

        # ── Evidence inventory ─────────────────────────────────────────────────
        "evidence_files": evidence_files,
    }

    return report


# ══════════════════════════════════════════════════════════════════════════════
# Text report renderer
# ══════════════════════════════════════════════════════════════════════════════

def render_text_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    DIV  = "═" * 72
    DIV2 = "─" * 72

    def h1(title):
        lines.append(f"\n{DIV}"); lines.append(f"  {title}"); lines.append(DIV)

    def h2(title):
        lines.append(f"\n{DIV2}"); lines.append(f"  {title}"); lines.append(DIV2)

    def kv(key, value, indent=2):
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
    kv("File Size",   f"{coc['file_size']:,} bytes"
                      if isinstance(coc['file_size'], int) else coc['file_size'])
    kv("Format",      coc["image_format"])
    kv("Colour Mode", coc["image_mode"])
    kv("Dimensions",  str(coc["image_size"]))

    # ── Module results ─────────────────────────────────────────────────────────
    h2("MODULE ANALYSIS RESULTS")
    mods = report["modules"]

    def module_block(display_name: str, m: Dict) -> None:
        ml_tag  = "ML✓" if m.get("ml_available") else "signal-only"
        lines.append(f"\n  ► {display_name}  [{ml_tag}]")
        lines.append(f"    Fused Score  : {_score_bar(m['score'])}")
        if m.get("signal_score") is not None:
            lines.append(f"    Signal Score : {m['signal_score']:.4f}")
        if m.get("ml_score") is not None:
            lines.append(f"    ML Score     : {m['ml_score']:.4f}")
        lines.append(f"    Verdict      : {m['label']}")
        if m.get("confidence"):
            lines.append(f"    Confidence   : {m['confidence']}")
        lines.append(f"    Analysis     : {m['interpretation']}")

    module_block("1. Compression / ELA",       mods["compression_ela"])
    module_block("2. Splicing Detection",       mods["splicing_detection"])
    module_block("3. AI-Generation Detection",  mods["ai_generated_detection"])
    module_block("4. Deepfake Detection",       mods["deepfake_detection"])

    # ── Final verdict ──────────────────────────────────────────────────────────
    h2("FINAL VERDICT")
    final = report["final"]
    lines.append(f"\n  MANIPULATION PROBABILITY: "
                 f"{_score_bar(final['manipulation_probability'], 40)}")
    lines.append(f"  ASSESSMENT              : {final['label']}")
    lines.append(f"  CONFIDENCE              : {final.get('confidence','N/A')}")
    lines.append(f"  DOMINANT MODULE         : {final.get('dominant_module','N/A')}")
    lines.append(f"\n  {final['recommendation']}")

    # ── ML availability ────────────────────────────────────────────────────────
    h2("ML MODEL AVAILABILITY")
    for mod in ("splicing_detection", "ai_generated_detection", "deepfake_detection"):
        m = mods.get(mod, {})
        status = "TRAINED ✓" if m.get("ml_available") else "NOT TRAINED — signal only"
        kv(mod, status)

    # ── Evidence inventory ─────────────────────────────────────────────────────
    h2("EVIDENCE FILE INVENTORY")
    if report["evidence_files"]:
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
    print(render_text_report(report))