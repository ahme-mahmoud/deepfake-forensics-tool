"""
report_generator.py
===================
Module 5 — Forensic Evidence Report Generator

Produces a comprehensive, investigator-ready forensic report containing:
    • Case metadata (timestamp, investigator ID, tool version)
    • Chain-of-custody (SHA-256, file metadata)
    • Per-module scores with plain-English interpretation
    • Final manipulation probability with confidence label
    • Evidence file inventory
    • Textual analysis and recommendations

Output formats:
    • Human-readable .txt  (always generated)
    • Structured  .json    (always generated — machine-readable for pipelines)
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

TOOL_VERSION = "1.0.0"
TOOL_NAME    = "DeepFake Forensics Tool"


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

def _score_label(score: float) -> str:
    """Convert a [0,1] score to an investigator-friendly label."""
    if score < 0.20:
        return "AUTHENTIC (very low suspicion)"
    elif score < 0.40:
        return "LIKELY AUTHENTIC (low suspicion)"
    elif score < 0.60:
        return "INCONCLUSIVE (moderate suspicion)"
    elif score < 0.80:
        return "LIKELY MANIPULATED (high suspicion)"
    else:
        return "MANIPULATED (very high suspicion)"


def _score_bar(score: float, width: int = 30) -> str:
    """Return an ASCII progress bar for the given score."""
    filled = int(round(score * width))
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.2f}"


def _interpret_ela(score: float) -> str:
    if score < 0.25:
        return ("ELA residuals are low and spatially uniform, consistent with "
                "a singly-compressed JPEG with no post-processing.")
    elif score < 0.50:
        return ("Moderate ELA residuals detected.  Some regions compress "
                "differently from the rest — possible light editing or "
                "format conversion artefacts.")
    elif score < 0.75:
        return ("Significant ELA anomalies.  Multiple regions show "
                "compression inconsistencies typical of spliced or "
                "locally-edited content.")
    else:
        return ("Strong ELA evidence of manipulation.  Large, bright regions "
                "in the ELA heat-map indicate that substantial portions of "
                "the image have a different compression history — hallmark "
                "of image splicing or object insertion.")


def _interpret_splicing(edge: float, light: float, cm: float) -> str:
    parts = []
    if edge > 0.50:
        parts.append("block-level edge density is highly irregular")
    if light > 0.50:
        parts.append("gradient orientation diverges sharply between regions")
    if cm > 0.30:
        parts.append("copy-move duplicated blocks were detected")
    if not parts:
        return ("No significant splicing indicators.  Edge distribution, "
                "lighting direction, and copy-move analysis are within "
                "normal bounds.")
    return ("Splicing indicators: " + "; ".join(parts) + ".  "
            "These findings collectively suggest at least one region was "
            "inserted from a different source image.")


def _interpret_ai(freq: float, noise: float, texture: float) -> str:
    parts = []
    if freq > 0.50:
        parts.append("spectral analysis reveals GAN checkerboard artifacts")
    if noise > 0.50:
        parts.append("noise residual autocorrelation is inconsistent with "
                     "camera sensor characteristics")
    if texture > 0.50:
        parts.append("texture co-occurrence entropy is abnormally low")
    if not parts:
        return ("No strong indicators of AI generation.  Spectral, noise, "
                "and texture profiles are consistent with a photographic origin.")
    return ("AI-generation indicators: " + "; ".join(parts) + ".  "
            "These patterns are characteristic of GAN or diffusion-model output.")


def _interpret_deepfake(faces: int, lm: float, blend: float, colour: float) -> str:
    if faces == 0:
        return ("No human faces were detected in the image.  "
                "Deepfake analysis was not applicable.")
    parts = []
    if lm > 0.40:
        parts.append("facial landmark geometry is inconsistent")
    if blend > 0.40:
        parts.append("blending boundary detected at face perimeter")
    if colour > 0.40:
        parts.append("skin-tone colour distribution is spatially irregular")
    if not parts:
        return (f"{faces} face(s) detected.  Facial geometry, blending "
                "boundary, and colour distribution appear natural.")
    return (f"{faces} face(s) detected.  Deepfake indicators: "
            + "; ".join(parts) + ".  "
            "These findings are consistent with GAN-based face-swapping.")


def _overall_recommendation(final_score: float) -> str:
    if final_score < 0.30:
        return ("RECOMMENDATION: Image appears AUTHENTIC.  No further "
                "forensic investigation is warranted based on automated "
                "analysis.  Human review is advised before final conclusion.")
    elif final_score < 0.55:
        return ("RECOMMENDATION: INCONCLUSIVE.  Some anomalies were detected "
                "but are insufficient to conclude deliberate manipulation.  "
                "Manual examination by a certified digital forensics examiner "
                "is strongly recommended.")
    else:
        return ("RECOMMENDATION: Image is LIKELY MANIPULATED.  Multiple "
                "independent forensic indicators support this conclusion.  "
                "Do NOT accept this image as authentic evidence without "
                "certified forensic review.  Preserve the original file and "
                "this report as part of the chain of custody.")


# ---------------------------------------------------------------------------
# Core report builder
# ---------------------------------------------------------------------------

def build_report(
    image_path: str,
    ela_result:      Dict[str, Any],
    splicing_result: Dict[str, Any],
    ai_result:       Dict[str, Any],
    deepfake_result: Dict[str, Any],
    investigator_id: str = "AUTO",
    case_id:         Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assemble the master forensic report dictionary from individual module
    results.

    Parameters
    ----------
    image_path       : Path to the investigated image.
    ela_result       : Output from compression_analysis.analyze()
    splicing_result  : Output from splicing_detector.analyze()
    ai_result        : Output from ai_generated_detector.analyze()
    deepfake_result  : Output from deepfake_detector.analyze()
    investigator_id  : Identifier of the analyst / system running the tool.
    case_id          : Optional case reference number.

    Returns
    -------
    A nested dict that can be serialised to JSON or formatted as text.
    """
    now    = datetime.now(timezone.utc)
    sha256 = ela_result.get("sha256", "UNKNOWN")

    # ── Module scores ─────────────────────────────────────────────────────────
    ela_score      = ela_result.get("ela_score",          0.0)
    splicing_score = splicing_result.get("splicing_score", 0.0)
    ai_score       = ai_result.get("ai_generated_score",  0.0)
    deepfake_score = deepfake_result.get("deepfake_score", 0.0)

    # ── Final manipulation probability (equal-weight average) ─────────────────
    scores = [ela_score, splicing_score, ai_score, deepfake_score]
    final_score = round(float(sum(scores) / len(scores)), 4)

    # ── Evidence file inventory ───────────────────────────────────────────────
    evidence_files: List[str] = []
    for key in ("ela_image_path", "panel_image_path",
                "edge_heatmap_path", "copymove_img_path",
                "spectrum_path", "noise_path",
                "face_annotated_path"):
        for result in (ela_result, splicing_result, ai_result, deepfake_result):
            val = result.get(key)
            if val and os.path.isfile(str(val)):
                evidence_files.append(str(val))

    report = {
        # ─── Header ────────────────────────────────────────────────────────────
        "tool"            : TOOL_NAME,
        "version"         : TOOL_VERSION,
        "generated_at"    : now.isoformat(),
        "investigator_id" : investigator_id,
        "case_id"         : case_id or f"CASE-{now.strftime('%Y%m%d-%H%M%S')}",
        "hostname"        : socket.gethostname(),
        "platform"        : platform.platform(),

        # ─── Chain of custody ──────────────────────────────────────────────────
        "chain_of_custody": {
            "image_path"  : str(Path(image_path).resolve()),
            "sha256"      : sha256,
            "file_size"   : ela_result.get("metadata", {}).get("file_bytes", "N/A"),
            "image_format": ela_result.get("metadata", {}).get("format",     "N/A"),
            "image_mode"  : ela_result.get("metadata", {}).get("mode",       "N/A"),
            "image_size"  : ela_result.get("metadata", {}).get("size_px",    "N/A"),
        },

        # ─── Module results ────────────────────────────────────────────────────
        "modules": {
            "compression_ela": {
                "score"              : ela_score,
                "label"              : _score_label(ela_score),
                "suspicious_regions" : len(ela_result.get("suspicious_regions", [])),
                "interpretation"     : _interpret_ela(ela_score),
            },
            "splicing_detection": {
                "score"             : splicing_score,
                "label"             : _score_label(splicing_score),
                "edge_score"        : splicing_result.get("edge_score",      0.0),
                "lighting_score"    : splicing_result.get("lighting_score",  0.0),
                "copy_move_score"   : splicing_result.get("copy_move_score", 0.0),
                "match_pairs_count" : splicing_result.get("match_pairs_count", 0),
                "interpretation"    : _interpret_splicing(
                    splicing_result.get("edge_score",      0.0),
                    splicing_result.get("lighting_score",  0.0),
                    splicing_result.get("copy_move_score", 0.0),
                ),
            },
            "ai_generated_detection": {
                "score"           : ai_score,
                "label"           : _score_label(ai_score),
                "frequency_score" : ai_result.get("frequency_score", 0.0),
                "noise_score"     : ai_result.get("noise_score",     0.0),
                "texture_score"   : ai_result.get("texture_score",   0.0),
                "interpretation"  : _interpret_ai(
                    ai_result.get("frequency_score", 0.0),
                    ai_result.get("noise_score",     0.0),
                    ai_result.get("texture_score",   0.0),
                ),
            },
            "deepfake_detection": {
                "score"           : deepfake_score,
                "label"           : _score_label(deepfake_score),
                "faces_detected"  : deepfake_result.get("faces_detected",  0),
                "landmark_score"  : deepfake_result.get("landmark_score",  0.0),
                "blending_score"  : deepfake_result.get("blending_score",  0.0),
                "colour_score"    : deepfake_result.get("colour_score",    0.0),
                "eye_glint_score" : deepfake_result.get("eye_glint_score", 0.0),
                "interpretation"  : _interpret_deepfake(
                    deepfake_result.get("faces_detected", 0),
                    deepfake_result.get("landmark_score", 0.0),
                    deepfake_result.get("blending_score", 0.0),
                    deepfake_result.get("colour_score",   0.0),
                ),
            },
        },

        # ─── Final verdict ─────────────────────────────────────────────────────
        "final": {
            "manipulation_probability" : final_score,
            "label"                    : _score_label(final_score),
            "recommendation"           : _overall_recommendation(final_score),
        },

        # ─── Evidence inventory ────────────────────────────────────────────────
        "evidence_files": evidence_files,
    }

    return report


# ---------------------------------------------------------------------------
# Text report renderer
# ---------------------------------------------------------------------------

def render_text_report(report: Dict[str, Any]) -> str:
    """Convert the report dict to a formatted plaintext string."""
    lines: List[str] = []
    DIV  = "═" * 72
    DIV2 = "─" * 72

    def h1(title: str) -> None:
        lines.append(f"\n{DIV}")
        lines.append(f"  {title}")
        lines.append(DIV)

    def h2(title: str) -> None:
        lines.append(f"\n{DIV2}")
        lines.append(f"  {title}")
        lines.append(DIV2)

    def kv(key: str, value: Any, indent: int = 2) -> None:
        pad = " " * indent
        lines.append(f"{pad}{key:<30}: {value}")

    # ── Title banner ──────────────────────────────────────────────────────────
    h1(f"  {report['tool']}  v{report['version']}  — FORENSIC ANALYSIS REPORT")
    kv("Generated (UTC)", report["generated_at"])
    kv("Case ID",         report["case_id"])
    kv("Investigator ID", report["investigator_id"])
    kv("Host",            report["hostname"])

    # ── Chain of custody ──────────────────────────────────────────────────────
    h2("CHAIN OF CUSTODY")
    coc = report["chain_of_custody"]
    kv("Image Path",   coc["image_path"])
    kv("SHA-256",      coc["sha256"])
    kv("File Size",    f"{coc['file_size']:,} bytes" if isinstance(coc['file_size'], int) else coc['file_size'])
    kv("Format",       coc["image_format"])
    kv("Colour Mode",  coc["image_mode"])
    kv("Dimensions",   str(coc["image_size"]))

    # ── Module results ─────────────────────────────────────────────────────────
    h2("MODULE ANALYSIS RESULTS")
    mods = report["modules"]

    def module_block(name: str, m: Dict) -> None:
        lines.append(f"\n  ► {name}")
        lines.append(f"    Score  : {_score_bar(m['score'])}")
        lines.append(f"    Verdict: {m['label']}")
        # Sub-scores
        for key, val in m.items():
            if key.endswith("_score") and key != "score":
                lines.append(f"    {key:<28}: {val:.4f}")
        lines.append(f"    Analysis: {m['interpretation']}")

    module_block("1. Compression / ELA",       mods["compression_ela"])
    module_block("2. Splicing Detection",       mods["splicing_detection"])
    module_block("3. AI-Generation Detection",  mods["ai_generated_detection"])
    module_block("4. Deepfake Detection",       mods["deepfake_detection"])

    # ── Final verdict ──────────────────────────────────────────────────────────
    h2("FINAL VERDICT")
    final = report["final"]
    lines.append(f"\n  MANIPULATION PROBABILITY: {_score_bar(final['manipulation_probability'], 40)}")
    lines.append(f"  ASSESSMENT              : {final['label']}")
    lines.append(f"\n  {final['recommendation']}")

    # ── Evidence inventory ─────────────────────────────────────────────────────
    h2("EVIDENCE FILE INVENTORY")
    if report["evidence_files"]:
        for i, ef in enumerate(report["evidence_files"], 1):
            lines.append(f"  [{i:02d}] {ef}")
    else:
        lines.append("  No evidence files saved.")

    lines.append(f"\n{DIV}")
    lines.append("  END OF REPORT — " + report["tool"])
    lines.append(DIV + "\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_report(
    report: Dict[str, Any],
    output_dir: str = "reports",
    save_json: bool = True,
    save_txt:  bool = True,
) -> Dict[str, str]:
    """
    Save the report to disk in JSON and/or text format.

    Returns
    -------
    Dict mapping "json" and "txt" to their output paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    case_id  = report.get("case_id", "UNKNOWN")
    stem     = f"forensic_report_{case_id}"

    paths: Dict[str, str] = {}

    if save_json:
        json_path = os.path.join(output_dir, f"{stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        paths["json"] = json_path
        logger.info("JSON report → %s", json_path)

    if save_txt:
        txt_path = os.path.join(output_dir, f"{stem}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(render_text_report(report))
        paths["txt"] = txt_path
        logger.info("Text report → %s", txt_path)

    return paths


# ---------------------------------------------------------------------------
# Convenience: print to console
# ---------------------------------------------------------------------------

def print_report(report: Dict[str, Any]) -> None:
    """Print the formatted text report to stdout."""
    print(render_text_report(report))
