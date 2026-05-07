"""
app.py  —  Hybrid AI Forensics Pipeline  v3.0
=============================================
Full orchestration: Signal Processing + ML + SHAP + PDF + Dashboard

USAGE:
    python app.py <image_path> [--case-id CASE-001] [--no-train]

PIPELINE:
    1. Feature Extraction     (HOG, LBP, FFT, ELA, Color, DCT, Noise)
    2. Signal-Processing      (ELA, Splicing, AI-Gen, Deepfake heuristics)
    3. ML Classifiers         (SVM + Random Forest per module)
    4. Score Fusion           (signal + ML → hybrid score per module)
    5. SHAP Explainability    (WHY was it flagged?)
    6. Evidence Visualization (ELA panels, heatmaps, FFT)
    7. Forensic Report        (JSON + TXT + PDF)
    8. Dashboard Output       (Streamlit)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).parent))

import compression_analysis
import splicing_detector
import ai_generated_detector
import deepfake_detector
import report_generator
from feature_extractor import extract as extract_features
from ml_classifier import get_classifier, MODELS_DIR
from score_fusion import fuse_module_score, compute_final_score
import shap_explainer
import pdf_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   AI-Powered Deepfake & Image Manipulation Detection Tool v3.0   ║
║   Hybrid: Signal Processing + ML + SHAP + PDF Evidence Report    ║
╚══════════════════════════════════════════════════════════════════╝
"""


def _models_exist() -> bool:
    for m in ("splicing", "ai_gen", "deepfake"):
        if not (MODELS_DIR / f"{m}_rf.pkl").exists():
            return False
    return True


def auto_train_if_needed(n_samples: int = 200) -> None:
    """Auto-train ML models if not yet trained."""
    if _models_exist():
        logger.info("ML models found — skipping auto-train")
        return
    logger.info("No trained models found → auto-training with %d synthetic samples", n_samples)
    from train_models import train_all
    train_all(n_samples=n_samples)
    from ml_classifier import reload_all
    reload_all()
    shap_explainer.invalidate_cache()
    logger.info("Auto-training complete")


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
    auto_train      : bool = True,
) -> dict:
    """
    Execute the full hybrid forensic pipeline.

    Returns the master report dict (compatible with dashboard + report_generator).
    """
    print(BANNER)
    logger.info("Image: %s", image_path)
    t0 = time.perf_counter()

    if not os.path.isfile(image_path):
        logger.error("File not found: %s", image_path)
        sys.exit(1)

    # ── Auto-train ML models if missing ───────────────────────────────────────
    if auto_train:
        auto_train_if_needed()

    os.makedirs(evidence_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Feature Extraction
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [1/7] Feature Extraction ━━━━")
    try:
        features = extract_features(image_path)
        feat_vec = features["combined"]
        logger.info("Feature vector: %d dims", len(feat_vec))
    except Exception as e:
        logger.error("Feature extraction failed: %s", e)
        feat_vec = None

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Signal-Processing Forensics
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [2/7] Signal-Processing Forensics ━━━━")
    sp_results = {}

    try:
        sp_results["ela"] = compression_analysis.analyze(
            image_path, quality=ela_quality, scale=ela_scale,
            evidence_dir=evidence_dir, save_evidence=save_evidence)
    except Exception as e:
        logger.error("ELA failed: %s", e)
        sp_results["ela"] = {"ela_score": 0.0, "sha256": "ERROR"}

    try:
        sp_results["splicing"] = splicing_detector.analyze(
            image_path, evidence_dir=evidence_dir, save_evidence=save_evidence)
    except Exception as e:
        logger.error("Splicing failed: %s", e)
        sp_results["splicing"] = {"splicing_score": 0.0}

    try:
        sp_results["ai"] = ai_generated_detector.analyze(
            image_path, evidence_dir=evidence_dir, save_evidence=save_evidence)
    except Exception as e:
        logger.error("AI-gen failed: %s", e)
        sp_results["ai"] = {"ai_generated_score": 0.0}

    try:
        sp_results["deepfake"] = deepfake_detector.analyze(
            image_path, evidence_dir=evidence_dir, save_evidence=save_evidence)
    except Exception as e:
        logger.error("Deepfake failed: %s", e)
        sp_results["deepfake"] = {"deepfake_score": 0.0}

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — ML Classifier Scoring
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [3/7] ML Classifier Scoring ━━━━")
    ml_scores = {}

    if feat_vec is not None:
        for mod in ("splicing", "ai_gen", "deepfake"):
            try:
                clf = get_classifier(mod)
                detail = clf.predict_detail(feat_vec)
                ml_scores[mod] = detail
                if detail["trained"]:
                    logger.info("ML %s: RF=%.3f SVM=%.3f → %.3f",
                                mod,
                                detail.get("rf_score", 0),
                                detail.get("svm_score", 0),
                                detail["score"])
                else:
                    logger.info("ML %s: not trained — using signal-processing only", mod)
            except Exception as e:
                logger.warning("ML scoring failed for %s: %s", mod, e)
                ml_scores[mod] = {"trained": False, "score": None}
    else:
        logger.warning("No feature vector — ML scoring skipped")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Score Fusion
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [4/7] Score Fusion ━━━━")

    ela_sig    = sp_results["ela"].get("ela_score", 0.0)
    spl_sig    = sp_results["splicing"].get("splicing_score", 0.0)
    ai_sig     = sp_results["ai"].get("ai_generated_score", 0.0)
    dfk_sig    = sp_results["deepfake"].get("deepfake_score", 0.0)

    ela_fused  = fuse_module_score(ela_sig, None)
    spl_fused  = fuse_module_score(spl_sig, ml_scores.get("splicing",{}).get("score"))
    ai_fused   = fuse_module_score(ai_sig,  ml_scores.get("ai_gen",  {}).get("score"))
    dfk_fused  = fuse_module_score(dfk_sig, ml_scores.get("deepfake",{}).get("score"))

    module_final_scores = {
        "ela"      : ela_fused["fused_score"],
        "splicing" : spl_fused["fused_score"],
        "ai_gen"   : ai_fused["fused_score"],
        "deepfake" : dfk_fused["fused_score"],
    }
    fusion_result = compute_final_score(module_final_scores)
    logger.info("Fused scores: %s", module_final_scores)
    logger.info("FINAL PROBABILITY: %.4f  confidence=%s",
                fusion_result["final_score"], fusion_result["confidence"])

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — SHAP Explainability
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [5/7] SHAP Explainability ━━━━")
    shap_results = {}
    if feat_vec is not None:
        for mod in ("splicing", "ai_gen", "deepfake"):
            try:
                shap_results[mod] = shap_explainer.explain(mod, feat_vec)
                if shap_results[mod]["available"]:
                    logger.info("SHAP %s: dominant=%s",
                                mod, shap_results[mod].get("dominant_domain"))
            except Exception as e:
                logger.warning("SHAP failed for %s: %s", mod, e)
                shap_results[mod] = {"available": False, "summary": str(e)}

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Build Master Report Dict
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [6/7] Building Report ━━━━")

    # Enrich module results with fusion data
    sp_results["ela"]["score"]     = ela_fused["fused_score"]
    sp_results["splicing"]["score"] = spl_fused["fused_score"]
    sp_results["splicing"]["ml_score"]     = spl_fused.get("ml_score")
    sp_results["splicing"]["signal_score"] = spl_fused["signal_score"]
    sp_results["ai"]["score"]       = ai_fused["fused_score"]
    sp_results["ai"]["ml_score"]    = ai_fused.get("ml_score")
    sp_results["ai"]["signal_score"]= ai_fused["signal_score"]
    sp_results["deepfake"]["score"]        = dfk_fused["fused_score"]
    sp_results["deepfake"]["ml_score"]     = dfk_fused.get("ml_score")
    sp_results["deepfake"]["signal_score"] = dfk_fused["signal_score"]

    report = report_generator.build_report(
        image_path       = image_path,
        ela_result       = sp_results["ela"],
        splicing_result  = sp_results["splicing"],
        ai_result        = sp_results["ai"],
        deepfake_result  = sp_results["deepfake"],
        investigator_id  = investigator_id,
        case_id          = case_id,
    )

    # Override final score with fused value
    report["final"]["manipulation_probability"] = fusion_result["final_score"]
    report["final"]["confidence"]               = fusion_result["confidence"]
    report["final"]["dominant_module"]          = fusion_result["dominant_module"]
    report["final"]["label"] = report_generator._score_label(fusion_result["final_score"])
    report["final"]["recommendation"] = report_generator._overall_recommendation(
        fusion_result["final_score"])

    # Attach extra data
    report["shap_explanations"]   = shap_results
    report["ml_scores"]           = {k: v for k, v in ml_scores.items()
                                      if isinstance(v, dict)}
    report["fusion_breakdown"]    = fusion_result
    report["feature_vector_dim"]  = int(len(feat_vec)) if feat_vec is not None else 0
    report["elapsed_seconds"]     = round(time.perf_counter() - t0, 2)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — Save Reports (JSON + TXT + PDF)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━ [7/7] Saving Reports ━━━━")
    if save_evidence:
        paths = report_generator.save_report(report, output_dir=out_dir)
        logger.info("JSON → %s", paths.get("json"))
        logger.info("TXT  → %s", paths.get("txt"))

        # PDF
        try:
            pdf_path = os.path.join(out_dir, f"forensic_report_{report['case_id']}.pdf")
            pdf_report.generate(report, pdf_path)
            report["pdf_path"] = pdf_path
            logger.info("PDF  → %s", pdf_path)
        except Exception as e:
            logger.warning("PDF generation failed: %s", e)
            report["pdf_path"] = None

    logger.info("Pipeline complete in %.2fs", report["elapsed_seconds"])
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Hybrid Forensics Pipeline v3.0")
    p.add_argument("image_path")
    p.add_argument("--quality",       type=int, default=95)
    p.add_argument("--scale",         type=int, default=15)
    p.add_argument("--out-dir",       default="reports")
    p.add_argument("--no-save",       action="store_true")
    p.add_argument("--json-only",     action="store_true")
    p.add_argument("--case-id",       default=None)
    p.add_argument("--investigator",  default="AUTO")
    p.add_argument("--no-train",      action="store_true",
                   help="Skip auto-training even if models missing")
    return p.parse_args()


def main():
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
        auto_train      = not args.no_train,
    )

    if args.json_only:
        print(json.dumps(report, indent=2, default=str))
    else:
        report_generator.print_report(report)

    final = report["final"]["manipulation_probability"]
    sys.exit(1 if final >= 0.55 else 0)


if __name__ == "__main__":
    main()
