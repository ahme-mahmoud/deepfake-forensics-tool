"""
train_models.py
===============
ML Training Pipeline

Generates synthetic training data from real + manipulated images,
then trains SVM + Random Forest classifiers for each forensic module.

USAGE:
    python src/train_models.py
    python src/train_models.py --data-dir data/ --samples 500

DATASET SUPPORT (when real datasets are available):
    CIFAKE        → label=0 real, label=1 fake  → trains ai_gen module
    FaceForensics → label=0 real, label=1 fake  → trains deepfake module
    Real + spliced images                        → trains splicing module

WITHOUT REAL DATASETS:
    Automatically generates synthetic samples with forensically meaningful
    differences so the models learn correct feature relationships.

OUTPUT:
    models/splicing_rf.pkl,  models/splicing_svm.pkl,  models/splicing_scaler.pkl
    models/ai_gen_rf.pkl,    models/ai_gen_svm.pkl,    models/ai_gen_scaler.pkl
    models/deepfake_rf.pkl,  models/deepfake_svm.pkl,  models/deepfake_scaler.pkl
    models/training_report.json
"""

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from feature_extractor import extract
from ml_classifier import ForensicClassifier, MODELS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_models")


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Data Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _save_temp(arr: np.ndarray, quality: int = 90) -> str:
    """Save an RGB array to a temp JPEG and return the path."""
    import tempfile
    path = tempfile.mktemp(suffix=".jpg")
    Image.fromarray(arr.astype(np.uint8)).save(path, format="JPEG", quality=quality)
    return path


def _natural_image(size: int = 128) -> np.ndarray:
    """Synthetic natural photograph: gradient + noise + texture."""
    arr = np.zeros((size, size, 3), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            arr[y, x] = [
                255 * x / size,
                255 * y / size,
                128 + 40 * np.sin(y * 0.1) * np.cos(x * 0.1),
            ]
    noise = np.random.normal(0, 8, arr.shape)
    return np.clip(arr + noise, 0, 255).astype(np.uint8)


def _spliced_image(size: int = 128) -> np.ndarray:
    """
    Synthetic spliced image: patch from different compression history.
    The patch was saved at a different JPEG quality before being pasted,
    creating the double-compression artifact that ELA detects.
    """
    # Background
    bg = _natural_image(size)

    # Patch: different statistical properties (from a "different source")
    patch_h, patch_w = size // 3, size // 3
    patch = np.zeros((patch_h, patch_w, 3), dtype=np.float32)
    for y in range(patch_h):
        for x in range(patch_w):
            patch[y, x] = [
                200 - 150 * x / patch_w,
                50  + 150 * y / patch_h,
                180,
            ]
    patch_noise = np.random.normal(0, 5, patch.shape)
    patch = np.clip(patch + patch_noise, 0, 255).astype(np.uint8)

    # Pre-compress patch at different quality (simulates external source)
    buf = io.BytesIO()
    Image.fromarray(patch).save(buf, format="JPEG", quality=60)
    buf.seek(0)
    patch = np.array(Image.open(buf).convert("RGB"))

    # Paste into background
    y0, x0 = size // 4, size // 4
    bg[y0:y0+patch_h, x0:x0+patch_w] = patch
    return bg


def _ai_generated_image(size: int = 128) -> np.ndarray:
    """
    Synthetic GAN-like image: over-smooth + periodic spectral artifacts.
    Real GANs are smoother at high frequencies and have checkerboard peaks.
    """
    # Over-smooth gradient (simulates GAN over-smoothing)
    arr = np.zeros((size, size, 3), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            arr[y, x] = [
                128 + 60 * np.sin(y * np.pi / size),
                128 + 60 * np.cos(x * np.pi / size),
                180 + 30 * np.sin((x + y) * np.pi / size),
            ]

    # Apply strong blur (GAN over-smoothing)
    arr_uint8 = arr.astype(np.uint8)
    pil = Image.fromarray(arr_uint8)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=2))
    blurred = np.array(pil).astype(np.float32)

    # Add very low noise (camera has more)
    noise = np.random.normal(0, 1.5, blurred.shape)
    return np.clip(blurred + noise, 0, 255).astype(np.uint8)


def _deepfake_image(size: int = 128) -> np.ndarray:
    """
    Synthetic deepfake image: face region with colour inconsistency +
    blending boundary artifacts.
    """
    # Base face-like gradient
    arr = np.zeros((size, size, 3), dtype=np.float32)
    cx, cy, r = size // 2, size // 2, size // 3

    y_idx = np.arange(size).reshape(-1, 1)
    x_idx = np.arange(size).reshape(1, -1)
    dist  = np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2)

    # Background
    arr[:, :, 0] = 80
    arr[:, :, 1] = 100
    arr[:, :, 2] = 180

    # Face region (different colour distribution — the "swap")
    face_mask = dist < r
    arr[face_mask, 0] = 200 + np.random.normal(0, 15, face_mask.sum())
    arr[face_mask, 1] = 160 + np.random.normal(0, 20, face_mask.sum())
    arr[face_mask, 2] = 140 + np.random.normal(0, 15, face_mask.sum())

    # Hard blending boundary (deepfake artefact)
    boundary = (dist > r - 3) & (dist < r + 3)
    arr[boundary] = [255, 50, 50]  # Sharp edge

    noise = np.random.normal(0, 4, arr.shape)
    return np.clip(arr + noise, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction for a list of images
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_from_paths(paths: List[str]) -> np.ndarray:
    feats = []
    for p in paths:
        try:
            f = extract(p)
            feats.append(f["combined"])
        except Exception as e:
            logger.warning("Skipping %s: %s", p, e)
    return np.array(feats, dtype=np.float32) if feats else np.empty((0, 0))


def _generate_dataset(
    n_real: int,
    n_fake: int,
    fake_fn,
    real_dir: str = None,
    fake_dir: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) training data.
    Uses real image files from real_dir/fake_dir if available,
    otherwise falls back to synthetic generators.
    """
    real_paths, fake_paths = [], []

    # Try loading from disk first
    if real_dir and os.path.isdir(real_dir):
        for fn in Path(real_dir).glob("*.jpg"):
            real_paths.append(str(fn))
        for fn in Path(real_dir).glob("*.png"):
            real_paths.append(str(fn))

    if fake_dir and os.path.isdir(fake_dir):
        for fn in Path(fake_dir).glob("*.jpg"):
            fake_paths.append(str(fn))
        for fn in Path(fake_dir).glob("*.png"):
            fake_paths.append(str(fn))

    logger.info("Disk images → real: %d  fake: %d", len(real_paths), len(fake_paths))

    # Synthetic fill-up
    tmp_files = []

    if len(real_paths) < n_real:
        needed = n_real - len(real_paths)
        logger.info("Generating %d synthetic real images", needed)
        for _ in range(needed):
            img  = _natural_image()
            path = _save_temp(img, quality=92)
            real_paths.append(path)
            tmp_files.append(path)

    if len(fake_paths) < n_fake:
        needed = n_fake - len(fake_paths)
        logger.info("Generating %d synthetic fake images", needed)
        for _ in range(needed):
            img  = fake_fn()
            path = _save_temp(img, quality=92)
            fake_paths.append(path)
            tmp_files.append(path)

    # Extract features
    logger.info("Extracting features for %d real + %d fake images",
                len(real_paths[:n_real]), len(fake_paths[:n_fake]))
    X_real = _extract_from_paths(real_paths[:n_real])
    X_fake = _extract_from_paths(fake_paths[:n_fake])

    # Cleanup temp files
    for p in tmp_files:
        try: os.remove(p)
        except: pass

    if len(X_real) == 0 or len(X_fake) == 0:
        raise RuntimeError("Feature extraction produced empty arrays")

    X = np.vstack([X_real, X_fake])
    y = np.array([0] * len(X_real) + [1] * len(X_fake), dtype=int)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _evaluate(clf: ForensicClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Run full evaluation on a held-out test set."""
    X_s   = clf.scaler.transform(X_test)
    y_pred = clf.rf.predict(X_s)
    y_prob = clf.rf.predict_proba(X_s)[:, 1]

    threshold = 0.5
    y_bin     = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y_test, y_bin)
    prec = precision_score(y_test, y_bin, zero_division=0)
    rec  = recall_score(y_test, y_bin, zero_division=0)
    f1   = f1_score(y_test, y_bin, zero_division=0)
    cm   = confusion_matrix(y_test, y_bin).tolist()

    logger.info("Accuracy=%.3f  Precision=%.3f  Recall=%.3f  F1=%.3f",
                acc, prec, rec, f1)

    return {
        "accuracy"  : round(acc,  4),
        "precision" : round(prec, 4),
        "recall"    : round(rec,  4),
        "f1_score"  : round(f1,   4),
        "confusion_matrix": cm,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main training routine
# ═══════════════════════════════════════════════════════════════════════════════

def train_all(n_samples: int = 300, data_dir: str = None) -> Dict:
    """
    Train all three forensic modules and return a metrics report.

    Parameters
    ----------
    n_samples : int   Number of real + fake samples per module
    data_dir  : str   Root data directory (expects real/ and fake/ subdirs)
    """
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    report = {}

    real_dir = os.path.join(data_dir, "real") if data_dir else None
    fake_dir = os.path.join(data_dir, "fake") if data_dir else None

    modules = [
        ("splicing", _spliced_image,     "Splicing Detection"),
        ("ai_gen",   _ai_generated_image, "AI-Generation Detection"),
        ("deepfake", _deepfake_image,     "Deepfake Detection"),
    ]

    for module, fake_fn, display_name in modules:
        logger.info("=" * 55)
        logger.info("Training module: %s", display_name)
        logger.info("=" * 55)

        # Build dataset
        X, y = _generate_dataset(
            n_real=n_samples,
            n_fake=n_samples,
            fake_fn=fake_fn,
            real_dir=real_dir,
            fake_dir=fake_dir,
        )

        logger.info("Dataset shape: X=%s  y=%s  (real=%d  fake=%d)",
                    X.shape, y.shape, (y==0).sum(), (y==1).sum())

        # Train / test split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42)

        # Train
        clf = ForensicClassifier(module)
        clf.fit(X_tr, y_tr)

        # Evaluate
        metrics = _evaluate(clf, X_te, y_te)
        report[module] = {"display_name": display_name, **metrics}

        logger.info("Module %s → F1=%.3f  Acc=%.3f",
                    module, metrics["f1_score"], metrics["accuracy"])

        # Save models
        clf.save()

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train forensic ML models")
    parser.add_argument("--samples",  type=int, default=300,
                        help="Samples per class per module (default: 300)")
    parser.add_argument("--data-dir", default=str(_ROOT / "data"),
                        help="Root data directory (expects real/ and fake/)")
    args = parser.parse_args()

    logger.info("Starting training: %d samples/class  data=%s",
                args.samples, args.data_dir)

    report = train_all(n_samples=args.samples, data_dir=args.data_dir)

    # Save training report
    out = _ROOT / "models" / "training_report.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    for mod, m in report.items():
        print(f"  {m['display_name']:<30} "
              f"Acc={m['accuracy']:.3f}  F1={m['f1_score']:.3f}")
    print(f"\n  Report saved → {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
