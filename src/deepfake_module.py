"""
deepfake_classifier.py
==========================

"""

import os
import cv2
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple

from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE = 128   # resize target before feature extraction


def _hog_features(gray: np.ndarray) -> np.ndarray:
    """HOG captures face geometry. ~2916 dims."""
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )


def _lbp_features(gray: np.ndarray) -> np.ndarray:
    """
    Local Binary Pattern — captures micro-texture.
    Deepfake skin often has unnatural smoothness → low LBP variance.
    26 dims.
    """
    lbp = local_binary_pattern(gray, P=24, R=3, method="uniform")
    hist, _ = np.histogram(lbp, bins=26, range=(0, 26), density=True)
    return hist


def _color_histogram(bgr: np.ndarray) -> np.ndarray:
    """
    YCbCr color histograms — better than RGB for detecting
    color-space inconsistencies left by GAN generators.
    192 dims (64 bins × 3 channels).
    """
    ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hists = []
    for ch in range(3):
        h, _ = np.histogram(
            ycbcr[:, :, ch], bins=64, range=(0, 256), density=True
        )
        hists.append(h)
    return np.concatenate(hists)


def _dct_features(gray: np.ndarray) -> np.ndarray:
    """
    DCT (frequency domain) — GAN-generated images leave
    characteristic frequency artifacts invisible in pixel space.
    8 dims.
    """
    dct = cv2.dct(gray.astype(np.float32))
    dct_abs = np.abs(dct)
    return np.array([
        dct_abs.mean(),
        dct_abs.std(),
        dct_abs.max(),
        np.percentile(dct_abs, 10),
        np.percentile(dct_abs, 25),
        np.percentile(dct_abs, 75),
        np.percentile(dct_abs, 90),
        float(np.sum(dct_abs > dct_abs.mean() + 2 * dct_abs.std())),
    ], dtype=np.float32)


def _glcm_features(gray: np.ndarray) -> np.ndarray:
    """
    Gray-Level Co-occurrence Matrix — captures repetitive texture
    patterns that GANs often over-smooth.
    20 dims.
    """
    distances = [1, 3]
    angles    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
    glcm = graycomatrix(
        gray_uint8, distances=distances, angles=angles,
        levels=256, symmetric=True, normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    feats = []
    for p in props:
        vals = graycoprops(glcm, p).flatten()
        feats.extend([vals.mean(), vals.std(), vals.min(), vals.max()])
    return np.array(feats, dtype=np.float32)


def _gradient_features(gray: np.ndarray) -> np.ndarray:
    """
    Gradient magnitude statistics.
    Deepfake faces show unusual sharpness patterns at blend boundaries.
    8 dims.
    """
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gm = cv2.magnitude(gx, gy)
    return np.array([
        gm.mean(), gm.std(), gm.max(),
        np.percentile(gm, 25), np.percentile(gm, 75),
        float(np.sum(gm > gm.mean() + 2 * gm.std())),  # high-gradient pixels
        float(np.sum(gm < 1.0)),                         # flat regions
        float(np.var(gm)),
    ], dtype=np.float32)


def _noise_features(gray: np.ndarray) -> np.ndarray:
    """
    Residual noise analysis.
    GANs produce characteristic noise patterns.
    6 dims.
    """
    blur   = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    noise  = gray.astype(np.float32) - blur
    return np.array([
        noise.mean(), noise.std(), noise.max(),
        np.percentile(np.abs(noise), 75),
        np.percentile(np.abs(noise), 95),
        float(np.sum(np.abs(noise) > 10)),   # strong noise pixel count
    ], dtype=np.float32)


def _face_region_stats(gray: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """
    Compare face-region stats vs image background.
    Deepfakes often have color/texture discontinuity at face boundary.
    16 dims.
    """
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    rx, ry = w // 4, h // 4

    face_mask = np.zeros_like(gray)
    cv2.ellipse(face_mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    bg_mask = cv2.bitwise_not(face_mask)

    feats = []
    for ch in range(3):
        channel = bgr[:, :, ch].astype(np.float32)
        face_px = channel[face_mask > 0]
        bg_px   = channel[bg_mask > 0]
        feats.extend([
            face_px.mean() - bg_px.mean(),   # mean difference
            face_px.std()  - bg_px.std(),    # std difference
            np.percentile(face_px, 75) - np.percentile(bg_px, 75),
            np.percentile(face_px, 25) - np.percentile(bg_px, 25),
        ])
    return np.array(feats, dtype=np.float32)


def extract_features(image_path: str) -> np.ndarray:
    """
    Full feature extraction pipeline.
    Returns a single flat numpy array.
    Total dims: ~3200+
    """
    bgr  = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    bgr  = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    feature_parts = [
        _hog_features(gray),          # ~2916
        _lbp_features(gray),          #    26
        _color_histogram(bgr),        #   192
        _dct_features(gray),          #     8
        _glcm_features(gray),         #    20
        _gradient_features(gray),     #     8
        _noise_features(gray),        #     6
        _face_region_stats(gray, bgr),#    16
    ]

    combined = np.concatenate(feature_parts).astype(np.float32)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Model Definition
# ─────────────────────────────────────────────────────────────────────────────

def build_model() -> Pipeline:
    """
    VotingClassifier ensemble:
      - GradientBoosting  (best single learner for tabular/feature data)
      - ExtraTrees        (high variance, good diversity)
      - SVM               (strong on high-dim feature vectors)

    Wrapped in a Pipeline with StandardScaler.
    Uses soft voting → outputs calibrated probabilities.
    """
    gb  = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    et  = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    svm = CalibratedClassifierCV(
        SVC(kernel="rbf", C=10, gamma="scale", probability=False),
        cv=3, method="isotonic",
    )

    ensemble = VotingClassifier(
        estimators=[("gb", gb), ("et", et), ("svm", svm)],
        voting="soft",
        weights=[2, 1, 1],   # GradientBoosting gets double weight
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("scaler",   StandardScaler()),
        ("ensemble", ensemble),
    ])
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Training
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_root: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expects:
        data_root/
            real/   ← genuine images
            fake/   ← deepfake images

    Returns X (n_samples, n_features), y (n_samples,)
    """
    X, y = [], []
    for label, folder in enumerate(["real", "fake"]):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Missing folder: {folder_path}")
        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"  [{folder}] {len(files)} images")
        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                feat = extract_features(fpath)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"  SKIP {fname}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train(
    data_root: str = "data",
    model_path: str = "deepfake_model_v2.pkl",
    test_size: float = 0.20,
) -> Pipeline:
    """
    Train and save the model.
    """
    print("=" * 60)
    print("Deepfake Classifier v2 — Training")
    print("=" * 60)

    print("\n[1/4] Loading dataset ...")
    X, y = load_dataset(data_root)
    print(f"  Total samples: {len(X)}  |  Features: {X.shape[1]}")
    print(f"  Real: {(y==0).sum()}  |  Fake: {(y==1).sum()}")

    print("\n[2/4] Splitting train / test ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    print("\n[3/4] Fitting ensemble ...")
    model = build_model()
    model.fit(X_train, y_train)

    print("\n[4/4] Evaluation ...")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    print(f"  ROC-AUC: {auc:.4f}")

    joblib.dump(model, model_path)
    print(f"\nModel saved → {model_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Inference
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: Pipeline | None = None
_MODEL_PATH = Path(__file__).parent.parent / "models" / "deepfake_model_v2.pkl"


def predict(
    image_path: str,
    evidence_dir=None,
    save_evidence: bool = True,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Predict whether an image is a deepfake.

    Returns:
        probability_deepfake : float  (0 = real, 1 = fake)
        label                : str    ("REAL" or "FAKE")
        confidence           : float  (0 = unsure, 1 = certain)
        explanation          : str
    """
    global _model_cache

    # Load model once
    if _model_cache is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at '{_MODEL_PATH}'. Run train() first."
            )
        _model_cache = joblib.load(_MODEL_PATH)

    feat  = extract_features(image_path).reshape(1, -1)
    proba = float(_model_cache.predict_proba(feat)[0, 1])
    label = "FAKE" if proba > threshold else "REAL"
    conf  = round(abs(proba - 0.5) * 2, 4)   # 0 = unsure, 1 = certain

    return {
        "probability_deepfake": round(proba, 4),
        "label":                label,
        "confidence":           conf,
        "explanation": (
            f"Ensemble (GradBoost+ExtraTrees+SVM) | "
            f"score={proba:.3f} | threshold={threshold} | {label}"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Train :  python deepfake_classifier_v2.py train <data_folder>")
        print("  Predict: python deepfake_classifier_v2.py predict <image_path>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "train":
        root = sys.argv[2] if len(sys.argv) > 2 else "data"
        train(data_root=root)

    elif cmd == "predict":
        if len(sys.argv) < 3:
            print("Provide image path.")
            sys.exit(1)
        result = predict(sys.argv[2])
        print(json.dumps(result, indent=2))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
