"""
app.py
======
DeepFake Forensics Tool — Master Orchestration Pipeline

USAGE:
    python app.py <image_path> [options]

OPTIONS:
    --quality   INT   ELA JPEG re-save quality  (default: 95)
    --scale     INT   ELA amplification factor  (default: 15)
    --out-dir   PATH  Report + evidence output directory (default: reports/)
    --no-save         Skip saving evidence images to disk
    --json-only       Print JSON report to stdout instead of text
    --case-id   STR   Custom case reference number

EXAMPLE:
    python app.py data/test/sample.jpg --case-id CASE-001
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is on the Python path so modules import cleanly
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(_SRC_DIR))

import compression_analysis
import splicing_detector
import ai_generated_detector
import deepfake_detector
import report_generator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("app")

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║      AI-Powered Deepfake & Image Manipulation Detection Tool     ║
║                  Digital Forensics System v1.0.0                 ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    image_path: str,
    ela_quality:    int  = 95,
    ela_scale:      int  = 15,
    out_dir:        str  = "reports",
    evidence_dir:   str  = "reports/evidence",
    save_evidence:  bool = True,
    case_id:        str  = None,
    investigator_id: str = "AUTO",
) -> dict:
    """
    Execute the full forensic pipeline and return the master report dict.

    Modules run in this order:
        1. Compression / ELA
        2. Splicing Detection
        3. AI-Generated Detection
        4. Deepfake Detection
        5. Report Generation
    """
    print(BANNER)
    logger.info("Starting forensic analysis on: %s", image_path)
    t_start = time.perf_counter()

    if not os.path.isfile(image_path):
        logger.error("File not found: %s", image_path)
        sys.exit(1)

    results = {}

    # ── Module 1: Compression / ELA ──────────────────────────────────────────
    logger.info("━━━━ [1/4] Compression Analysis (ELA) ━━━━")
    try:
        results["ela"] = compression_analysis.analyze(
            image_path,
            quality=ela_quality,
            scale=ela_scale,
            evidence_dir=evidence_dir,
            save_evidence=save_evidence,
        )
        logger.info("ELA score: %.4f", results["ela"]["ela_score"])
    except Exception as exc:
        logger.error("ELA failed: %s", exc)
        results["ela"] = {"ela_score": 0.0, "sha256": "ERROR", "error": str(exc)}

    # ── Module 2: Splicing Detection ─────────────────────────────────────────
    logger.info("━━━━ [2/4] Splicing Detection ━━━━")
    try:
        results["splicing"] = splicing_detector.analyze(
            image_path,
            evidence_dir=evidence_dir,
            save_evidence=save_evidence,
        )
        logger.info("Splicing score: %.4f", results["splicing"]["splicing_score"])
    except Exception as exc:
        logger.error("Splicing detection failed: %s", exc)
        results["splicing"] = {"splicing_score": 0.0, "error": str(exc)}

    # ── Module 3: AI-Generated Detection ─────────────────────────────────────
    logger.info("━━━━ [3/4] AI-Generated Image Detection ━━━━")
    try:
        results["ai"] = ai_generated_detector.analyze(
            image_path,
            evidence_dir=evidence_dir,
            save_evidence=save_evidence,
        )
        logger.info("AI-generated score: %.4f", results["ai"]["ai_generated_score"])
    except Exception as exc:
        logger.error("AI detection failed: %s", exc)
        results["ai"] = {"ai_generated_score": 0.0, "error": str(exc)}

    # ── Module 4: Deepfake Detection ──────────────────────────────────────────
    logger.info("━━━━ [4/4] Deepfake Detection ━━━━")
    try:
        results["deepfake"] = deepfake_detector.analyze(
            image_path,
            evidence_dir=evidence_dir,
            save_evidence=save_evidence,
        )
        logger.info("Deepfake score: %.4f", results["deepfake"]["deepfake_score"])
    except Exception as exc:
        logger.error("Deepfake detection failed: %s", exc)
        results["deepfake"] = {"deepfake_score": 0.0, "error": str(exc)}

    # ── Module 5: Report Generation ───────────────────────────────────────────
    logger.info("━━━━ [5/5] Generating Forensic Report ━━━━")
    report = report_generator.build_report(
        image_path       = image_path,
        ela_result       = results["ela"],
        splicing_result  = results["splicing"],
        ai_result        = results["ai"],
        deepfake_result  = results["deepfake"],
        investigator_id  = investigator_id,
        case_id          = case_id,
    )

    elapsed = time.perf_counter() - t_start
    report["elapsed_seconds"] = round(elapsed, 2)
    logger.info("Pipeline completed in %.2f s", elapsed)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="AI-Powered Deepfake & Image Manipulation Detection Tool",
    )
    parser.add_argument("image_path",           help="Path to the image to investigate")
    parser.add_argument("--quality",  type=int, default=95,        help="ELA JPEG quality (default: 95)")
    parser.add_argument("--scale",    type=int, default=15,        help="ELA scale factor (default: 15)")
    parser.add_argument("--out-dir",  default="reports",           help="Output directory (default: reports/)")
    parser.add_argument("--no-save",  action="store_true",         help="Do not save evidence images")
    parser.add_argument("--json-only",action="store_true",         help="Print JSON report to stdout")
    parser.add_argument("--case-id",  default=None,                help="Case reference number")
    parser.add_argument("--investigator", default="AUTO",          help="Investigator ID")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    evidence_dir = os.path.join(args.out_dir, "evidence")

    report = run_pipeline(
        image_path      = args.image_path,
        ela_quality     = args.quality,
        ela_scale       = args.scale,
        out_dir         = args.out_dir,
        evidence_dir    = evidence_dir,
        save_evidence   = not args.no_save,
        case_id         = args.case_id,
        investigator_id = args.investigator,
    )

    # Save to disk
    if not args.no_save:
        paths = report_generator.save_report(report, output_dir=args.out_dir)
        logger.info("Report saved → JSON: %s", paths.get("json"))
        logger.info("Report saved → TXT : %s", paths.get("txt"))

    # Output
    if args.json_only:
        print(json.dumps(report, indent=2, default=str))
    else:
        report_generator.print_report(report)

    # Exit code: 0 = clean, 1 = manipulated
    final_score = report["final"]["manipulation_probability"]
    sys.exit(1 if final_score >= 0.55 else 0)


if __name__ == "__main__":
    main()
