"""
app.py  —  Hybrid AI Forensics Pipeline  v3.1
=============================================
Full orchestration: Signal Processing + ML + SHAP + PDF + Forensic Report

USAGE:
    python app.py <image_path> [--case-id CASE-001]

PIPELINE:
    1. Feature Extraction     (HOG, LBP, FFT, ELA, Color, DCT, Noise)
    2. Compression Analysis   (ELA — Error Level Analysis)
    3. AI-Generated Detection (ensemble classifier + signal heuristics)
    4. Splicing Detection     (ensemble classifier + signal heuristics)
    5. Deepfake Detection     (ensemble classifier + signal heuristics)
    6. Score Fusion           (signal + ML → hybrid score per module)
    7. SHAP Explainability    (WHY was it flagged?)
    8. Forensic Report        (JSON + TXT + PDF)

MODELS (models/):
    ai_gen_ensemble.pkl   ai_gen_scaler.pkl
    splicing_ensemble.pkl splicing_scaler.pkl
    deepfake_model_v2.pkl
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
_SRC  = _ROOT / "src"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SRC))

# ── Core forensic modules (current names only) ────────────────────────────────
from src import compression_analysis
from src import ai_gen_module     as ai_gen_mod
from src import splicing_module   as splicing_mod
from src import deepfake_module   as deepfake_mod
from src import report_generator
from src import shap_explainer
from src import pdf_report

from src.feature_extractor import extract as extract_features
from src.score_fusion      import fuse_module_score, compute_final_score

# ── Model directory & required files ──────────────────────────────────────────
MODELS_DIR = _ROOT / "models"

_MODEL_FILES = [
    MODELS_DIR / "ai_gen_ensemble.pkl",
    MODELS_DIR / "ai_gen_scaler.pkl",
    MODELS_DIR / "splicing_ensemble.pkl",
    MODELS_DIR / "splicing_scaler.pkl",
    MODELS_DIR / "deepfake_model_v2.pkl",
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   AI-Powered Deepfake & Image Manipulation Detection Tool v3.1   ║
║   Hybrid: Signal Processing + ML + SHAP + PDF Evidence Report    ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Model existence check
# ═══════════════════════════════════════════════════════════════════════════════

def _models_exist() -> bool:
    """Return True only when every required trained model file is present."""
    return all(p.exists() for p in _MODEL_FILES)


def _check_models_or_exit() -> None:
    """Log model status; exit with a clear message if any model is missing."""
    missing = [p.name for p in _MODEL_FILES if not p.exists()]
    if missing:
        logger.error(
            "Missing trained model file(s) in %s: %s\n"
            "Train the models first, then re-run the pipeline.",
            MODELS_DIR, ", ".join(missing),
        )
        sys.exit(1)
    logger.info("All trained models verified in %s", MODELS_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-module wrapper helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _run_ela(image_path: str, ela_quality: int, ela_scale: int,
             evidence_dir: str, save_evidence: bool) -> dict:
    """Run ELA compression analysis. Returns dict with 'ela_score'."""
    try:
        result = compression_analysis.analyze(
            image_path,
            quality=ela_quality,
            scale=ela_scale,
            evidence_dir=evidence_dir,
            save_evidence=save_evidence,
        )
        logger.info("ELA score: %.4f", result.get("ela_score", 0.0))
        return result
    except Exception as exc:
        logger.error("ELA analysis failed: %s", exc)
        return {"ela_score": 0.0, "sha256": "ERROR", "error": str(exc)}


def _run_ai_gen(image_path: str, evidence_dir: str, save_evidence: bool) -> dict:
    """
    Run AI-generated image detection.

    Expected keys from ai_gen_module.predict():
        probability_ai_generated  – float [0, 1]
        is_ai_generated           – bool
        ml_score                  – float | None
        signal_score              – float
    """
    try:
        result = ai_gen_mod.predict(
            image_path,
            evidence_dir=evidence_dir,
            save_evidence=save_evidence,
        )
        prob   = float(result.get("probability_ai_generated", 0.0))
        ml_sc  = result.get("ml_score")
        sig_sc = float(result.get("signal_score", prob))
        logger.info(
            "AI-gen  | prob=%.4f  ml=%-7s  signal=%.4f  label=%s",
            prob,
            f"{ml_sc:.4f}" if ml_sc is not None else "N/A",
            sig_sc,
            "AI-GENERATED" if result.get("is_ai_generated") else "authentic",
        )
        return {**result, "_prob": prob, "_ml": ml_sc, "_sig": sig_sc}
    except Exception as exc:
        logger.error("AI-gen detection failed: %s", exc)
        return {
            "probability_ai_generated": 0.0,
            "is_ai_generated": False,
            "ml_score": None,
            "signal_score": 0.0,
            "_prob": 0.0, "_ml": None, "_sig": 0.0,
            "error": str(exc),
        }


def _run_splicing(image_path: str, evidence_dir: str, save_evidence: bool) -> dict:
    """
    Run image splicing / tampering detection.

    Expected keys from splicing_module.predict():
        probability_splicing  – float [0, 1]
        is_spliced            – bool
        ml_score              – float | None
        signal_score          – float
    """
    try:
        result = splicing_mod.predict(
            image_path,
            evidence_dir=evidence_dir,
            save_evidence=save_evidence,
        )
        prob   = float(result.get("probability_splicing", 0.0))
        ml_sc  = result.get("ml_score")
        sig_sc = float(result.get("signal_score", prob))
        logger.info(
            "Splicing| prob=%.4f  ml=%-7s  signal=%.4f  label=%s",
            prob,
            f"{ml_sc:.4f}" if ml_sc is not None else "N/A",
            sig_sc,
            "SPLICED" if result.get("is_spliced") else "intact",
        )
        return {**result, "_prob": prob, "_ml": ml_sc, "_sig": sig_sc}
    except Exception as exc:
        logger.error("Splicing detection failed: %s", exc)
        return {
            "probability_splicing": 0.0,
            "is_spliced": False,
            "ml_score": None,
            "signal_score": 0.0,
            "_prob": 0.0, "_ml": None, "_sig": 0.0,
            "error": str(exc),
        }


def _run_deepfake(image_path: str, evidence_dir: str, save_evidence: bool) -> dict:
    """
    Run deepfake face-manipulation detection.

    Expected keys from deepfake_module.predict():
        probability_deepfake  – float [0, 1]
        label                 – str  (e.g. "DEEPFAKE" / "REAL")
        confidence            – str  (e.g. "HIGH" / "MEDIUM" / "LOW")
    """
    try:
        result = deepfake_mod.predict(
            image_path,
            evidence_dir=evidence_dir,
            save_evidence=save_evidence,
        )
        prob   = float(result.get("probability_deepfake", 0.0))
        ml_sc  = result.get("ml_score",     prob)
        sig_sc = result.get("signal_score", prob)
        logger.info(
            "Deepfake| prob=%.4f  label=%s  confidence=%s",
            prob,
            result.get("label",      "UNKNOWN"),
            result.get("confidence", "UNKNOWN"),
        )
        return {
            **result,
            "_prob": prob,
            "_ml":   float(ml_sc)  if ml_sc  is not None else prob,
            "_sig":  float(sig_sc) if sig_sc is not None else prob,
        }
    except Exception as exc:
        logger.error("Deepfake detection failed: %s", exc)
        return {
            "probability_deepfake": 0.0,
            "label": "ERROR",
            "confidence": "NONE",
            "_prob": 0.0, "_ml": 0.0, "_sig": 0.0,
            "error": str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Core Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    image_path      : str,
    ela_quality     : int  = 95,
    ela_scale       : int  = 15,
    out_dir         : str  = "reports",
    evidence_dir    : str  = "reports/evidence",
    save_evidence   : bool = True,
    case_id         : str  = None,
    investigator_id : str  = "AUTO",
) -> dict:
    """
    Execute the full hybrid forensic pipeline.

    Returns the master report dict (compatible with report_generator).
    """
    print(BANNER)
    logger.info("Image: %s", image_path)
    t0 = time.perf_counter()

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    if not os.path.isfile(image_path):
        logger.error("File not found: %s", image_path)
        sys.exit(1)

    _check_models_or_exit()          # Abort cleanly if any .pkl is missing
    os.makedirs(evidence_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Feature Extraction
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [1/7] Feature Extraction ━━━━")
    feat_vec = None
    try:
        features = extract_features(image_path)
        feat_vec = features["combined"]
        logger.info("Feature vector: %d dims", len(feat_vec))
    except Exception as exc:
        logger.error("Feature extraction failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Compression / ELA Analysis
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [2/7] Compression / ELA Analysis ━━━━")
    ela_result = _run_ela(image_path, ela_quality, ela_scale,
                          evidence_dir, save_evidence)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — AI-Generated Image Detection
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [3/7] AI-Generated Detection ━━━━")
    ai_result = _run_ai_gen(image_path, evidence_dir, save_evidence)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Splicing / Tampering Detection
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [4/7] Splicing Detection ━━━━")
    splicing_result = _run_splicing(image_path, evidence_dir, save_evidence)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Deepfake Detection
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [5/7] Deepfake Detection ━━━━")
    deepfake_result = _run_deepfake(image_path, evidence_dir, save_evidence)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Score Fusion
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [6/7] Score Fusion ━━━━")

    ela_fused      = fuse_module_score(ela_result.get("ela_score", 0.0), None)
    ai_fused       = fuse_module_score(ai_result["_sig"],       ai_result["_ml"])
    splicing_fused = fuse_module_score(splicing_result["_sig"], splicing_result["_ml"])
    deepfake_fused = fuse_module_score(deepfake_result["_sig"], deepfake_result["_ml"])

    module_final_scores = {
        "ela"      : ela_fused["fused_score"],
        "ai_gen"   : ai_fused["fused_score"],
        "splicing" : splicing_fused["fused_score"],
        "deepfake" : deepfake_fused["fused_score"],
    }
    fusion_result = compute_final_score(module_final_scores)

    logger.info("Module fused scores → %s", module_final_scores)
    logger.info(
        "FINAL PROBABILITY: %.4f  |  confidence=%s  |  dominant=%s",
        fusion_result["final_score"],
        fusion_result["confidence"],
        fusion_result.get("dominant_module", "N/A"),
    )

    # ── Annotate result dicts with fused scores for report_generator ───────────
    ela_result["score"]               = ela_fused["fused_score"]

    ai_result["score"]                = ai_fused["fused_score"]
    ai_result["ml_score"]             = ai_fused.get("ml_score")
    ai_result["signal_score"]         = ai_fused["signal_score"]
    # Normalise key that report_generator may expect
    ai_result["ai_generated_score"]   = ai_result.get(
        "probability_ai_generated", ai_fused["fused_score"])

    splicing_result["score"]          = splicing_fused["fused_score"]
    splicing_result["ml_score"]       = splicing_fused.get("ml_score")
    splicing_result["signal_score"]   = splicing_fused["signal_score"]
    splicing_result["splicing_score"] = splicing_result.get(
        "probability_splicing", splicing_fused["fused_score"])

    deepfake_result["score"]          = deepfake_fused["fused_score"]
    deepfake_result["ml_score"]       = deepfake_fused.get("ml_score")
    deepfake_result["signal_score"]   = deepfake_fused["signal_score"]
    deepfake_result["deepfake_score"] = deepfake_result.get(
        "probability_deepfake", deepfake_fused["fused_score"])

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6b — SHAP Explainability
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [6b] SHAP Explainability ━━━━")
    shap_results: dict = {}
    if feat_vec is not None:
        for mod in ("ai_gen", "splicing", "deepfake"):
            try:
                shap_results[mod] = shap_explainer.explain(mod, feat_vec)
                if shap_results[mod].get("available"):
                    logger.info("SHAP %s: dominant_domain=%s",
                                mod, shap_results[mod].get("dominant_domain"))
            except Exception as exc:
                logger.warning("SHAP failed for %s: %s", mod, exc)
                shap_results[mod] = {"available": False, "summary": str(exc)}
    else:
        logger.warning("Feature vector unavailable — SHAP skipped")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — Build Master Report Dict & Save
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [7/7] Building & Saving Report ━━━━")

    report = report_generator.build_report(
        image_path      = image_path,
        ela_result      = ela_result,
        splicing_result = splicing_result,
        ai_result       = ai_result,
        deepfake_result = deepfake_result,
        investigator_id = investigator_id,
        case_id         = case_id,
    )

    # Override final verdict with hybrid fused values
    report["final"]["manipulation_probability"] = fusion_result["final_score"]
    report["final"]["confidence"]               = fusion_result["confidence"]
    report["final"]["dominant_module"]          = fusion_result.get("dominant_module")
    report["final"]["label"] = report_generator._score_label(
        fusion_result["final_score"])
    report["final"]["recommendation"] = report_generator._overall_recommendation(
        fusion_result["final_score"])

    # Attach supplementary data
    report["shap_explanations"]  = shap_results
    report["fusion_breakdown"]   = fusion_result
    report["feature_vector_dim"] = int(len(feat_vec)) if feat_vec is not None else 0
    report["elapsed_seconds"]    = round(time.perf_counter() - t0, 2)

    # ── Persist reports ────────────────────────────────────────────────────────
    if save_evidence:
        paths = report_generator.save_report(report, output_dir=out_dir)
        logger.info("JSON → %s", paths.get("json"))
        logger.info("TXT  → %s", paths.get("txt"))

        try:
            pdf_path = os.path.join(
                out_dir, f"forensic_report_{report['case_id']}.pdf")
            pdf_report.generate(report, pdf_path)
            report["pdf_path"] = pdf_path
            logger.info("PDF  → %s", pdf_path)
        except Exception as exc:
            logger.warning("PDF generation failed: %s", exc)
            report["pdf_path"] = None

    logger.info("Pipeline complete in %.2fs", report["elapsed_seconds"])
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid AI Forensics Pipeline v3.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("image_path",
                   help="Path to the image to analyse")
    p.add_argument("--quality",      type=int, default=95,
                   help="ELA JPEG quality (default 95)")
    p.add_argument("--scale",        type=int, default=15,
                   help="ELA amplification scale (default 15)")
    p.add_argument("--out-dir",      default="reports",
                   help="Output directory for reports (default: reports/)")
    p.add_argument("--no-save",      action="store_true",
                   help="Disable report saving to disk")
    p.add_argument("--json-only",    action="store_true",
                   help="Print full JSON report to stdout only")
    p.add_argument("--case-id",      default=None,
                   help="Override auto-generated case ID")
    p.add_argument("--investigator", default="AUTO",
                   help="Investigator ID embedded in the report")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    report = run_pipeline(
        image_path      = args.image_path,
        ela_quality     = args.quality,
        ela_scale       = args.scale,
        out_dir         = args.out_dir,
        evidence_dir    = os.path.join(args.out_dir, "evidence"),
        save_evidence   = not args.no_save,
        case_id         = args.case_id,
        investigator_id = args.investigator,
    )

    if args.json_only:
        print(json.dumps(report, indent=2, default=str))
    else:
        report_generator.print_report(report)

    final_prob = report["final"]["manipulation_probability"]
    sys.exit(1 if final_prob >= 0.55 else 0)


if __name__ == "__main__":
    main()
