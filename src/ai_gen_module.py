"""
ai_gen_module.py — MODULE 1: AI-Generation Detector  
==========================================================

DATASET:
    CIFAKE — https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
    Structure:
        data/cifake/REAL/*.jpg
        data/cifake/FAKE/*.jpg

SAVED MODELS:
    models/ai_gen_scaler.pkl
    models/ai_gen_ensemble.pkl

OUTPUT:
    probability_ai_generated ∈ [0.0, 1.0]
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

logger = logging.getLogger("ai_gen_module")

_ROOT      = Path(__file__).parent.parent
MODELS_DIR = _ROOT / "models"

# ── Feature config ─────────────────────────────────────────────────────────────
IMG_SIZE    = (128, 128)
FFT_BINS    = 32
LBP_RADIUS  = 3
LBP_POINTS  = 24
LBP_BINS    = 26        # uniform: n_points + 2
BLOCK_SIZE  = 32

# ── Inference config ───────────────────────────────────────────────────────────
SIGNAL_CAP         = 0.70   # hard cap on heuristic score
FUSION_W_ML        = 0.85   # ML ensemble weight
FUSION_W_SIGNAL    = 0.15   # signal heuristic weight
DECISION_THRESHOLD = 0.65   # final classification threshold


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def _fft_features(gray: np.ndarray) -> np.ndarray:
    """
    Radial FFT profile + slope + peak ratio + frequency symmetry.
    GAN checkerboard patterns show up as isolated high-freq peaks.
    Diffusion models show unnaturally flat high-freq decay.
    Total: FFT_BINS + 4 = 36 dims
    """
    fft = np.fft.fft2(gray.astype(np.float32))
    mag = np.log1p(np.abs(np.fft.fftshift(fft)))
    h, w   = mag.shape
    cy, cx = h // 2, w // 2

    y_idx = np.arange(h).reshape(-1, 1)
    x_idx = np.arange(w).reshape(1, -1)
    r_map = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    max_r = min(cy, cx)

    edges   = np.linspace(0, max_r, FFT_BINS + 1)
    profile = np.zeros(FFT_BINS, dtype=np.float32)
    for i in range(FFT_BINS):
        mask = (r_map >= edges[i]) & (r_map < edges[i + 1])
        profile[i] = float(mag[mask].mean()) if mask.any() else 0.0

    norm_profile = profile / (profile.max() + 1e-8)

    # slope of frequency decay (AI: flatter → slope closer to 0)
    slope = float(np.polyfit(np.arange(FFT_BINS), norm_profile, 1)[0])

    # high-frequency anomalous peaks (GAN checkerboard)
    dc_zone    = r_map < max_r // 10
    hf_vals    = mag[~dc_zone]
    mu, sd     = hf_vals.mean(), hf_vals.std()
    peak_ratio = float((hf_vals > mu + 4 * sd).sum()) / max(hf_vals.size, 1)

    # quadrant symmetry — diffusion images are unnaturally symmetric
    q1 = mag[:cy, :cx].mean()
    q2 = mag[:cy, cx:].mean()
    q3 = mag[cy:, :cx].mean()
    q4 = mag[cy:, cx:].mean()
    quad_std = float(np.std([q1, q2, q3, q4]))

    return np.concatenate(
        [norm_profile, [slope, peak_ratio, quad_std,
                        float(norm_profile[FFT_BINS // 2:].mean())]]
    ).astype(np.float32)


def _dct_block_features(gray: np.ndarray) -> np.ndarray:
    """
    Block-DCT coefficient statistics.
    JPEG compression + GAN generation both leave distinct DCT signatures.
    Total: 8 dims
    """
    dct = cv2.dct(gray.astype(np.float32))
    abs_dct = np.abs(dct)

    h, w = gray.shape
    # AC coefficients only (skip DC at [0,0])
    ac = abs_dct.copy()
    ac[0, 0] = 0.0

    high_freq_mean = float(ac[h // 2:, w // 2:].mean())
    low_freq_mean  = float(ac[: h // 4, : w // 4].mean())
    ratio_hl       = high_freq_mean / (low_freq_mean + 1e-8)

    return np.array(
        [
            abs_dct.mean(),
            abs_dct.std(),
            np.percentile(abs_dct, 75),
            np.percentile(abs_dct, 95),
            high_freq_mean,
            low_freq_mean,
            ratio_hl,
            float(np.log1p(ac.sum())),
        ],
        dtype=np.float32,
    )


def _noise_features(gray: np.ndarray) -> np.ndarray:
    """
    Sensor noise analysis.
    Real cameras have structured noise; GANs have almost none or synthetic patterns.
    Total: 7 dims
    """
    smooth   = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    residual = gray.astype(np.float32) - smooth

    std_val  = float(residual.std())
    flat     = residual.flatten()
    autocorr = float(np.corrcoef(flat[:-1], flat[1:])[0, 1]) if len(flat) > 1 else 0.0

    hist, _ = np.histogram(residual, bins=64, range=(-30, 30))
    p       = hist.astype(np.float64) + 1e-9
    p      /= p.sum()
    entropy = float(-np.sum(p * np.log(p))) / np.log(64)

    lap        = cv2.Laplacian(gray, cv2.CV_64F)
    smoothness = float(np.log1p(lap.var()))

    # Median Absolute Deviation of noise (robust noise estimator)
    mad = float(np.median(np.abs(residual - np.median(residual))))

    return np.array(
        [std_val, autocorr, entropy, smoothness, mad,
         float(residual.max()), float(residual.min())],
        dtype=np.float32,
    )


def _multiscale_sharpness(gray: np.ndarray) -> np.ndarray:
    """
    Laplacian at 3 scales + block uniformity.
    Diffusion images have suspicious sharpness uniformity across all scales.
    Total: 10 dims
    """
    feats = []
    img = gray.copy()
    for scale in range(3):
        lap = cv2.Laplacian(img, cv2.CV_64F)
        feats.extend([float(np.log1p(lap.var())), float(lap.mean())])
        img = cv2.pyrDown(img)  # halve resolution each scale

    # block-level coefficient of variation
    blocks = []
    lap_full = cv2.Laplacian(gray, cv2.CV_64F)
    for y in range(0, gray.shape[0] - BLOCK_SIZE, BLOCK_SIZE):
        for x in range(0, gray.shape[1] - BLOCK_SIZE, BLOCK_SIZE):
            blocks.append(lap_full[y: y + BLOCK_SIZE, x: x + BLOCK_SIZE].var())

    if blocks:
        arr = np.array(blocks, dtype=np.float32)
        mu  = arr.mean() + 1e-8
        feats.extend([float(arr.std() / mu), float(arr.min() / (arr.max() + 1e-8)),
                      float(np.log1p(mu)), float(arr.std())])
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)


def _lbp_features(gray: np.ndarray) -> np.ndarray:
    """
    Local Binary Pattern histogram — micro-texture fingerprint.
    AI skin: too regular → low LBP entropy.
    Total: 26 dims
    """
    lbp  = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS,
                           range=(0, LBP_POINTS + 2))
    hist = hist.astype(np.float32)
    return hist / (hist.sum() + 1e-8)


def _glcm_features(gray: np.ndarray) -> np.ndarray:
    """
    Gray-Level Co-occurrence Matrix properties.
    GANs tend to over-smooth → high homogeneity, low contrast.
    Total: 20 dims
    """
    gray_u8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
    glcm = graycomatrix(
        gray_u8,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    feats = []
    for p in props:
        vals = graycoprops(glcm, p).flatten()
        feats.extend([vals.mean(), vals.std(), vals.min(), vals.max()])
    return np.array(feats, dtype=np.float32)


def _color_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    YCbCr stats + HSV saturation + chroma noise.
    AI images often have unnaturally uniform chroma.
    Total: 14 dims
    """
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    hsv   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    sat = hsv[:, :, 1]
    hue = hsv[:, :, 0]

    # chroma noise: residual in Cb/Cr channels
    cb_blur   = cv2.GaussianBlur(ycbcr[:, :, 2], (5, 5), 0)
    cr_blur   = cv2.GaussianBlur(ycbcr[:, :, 1], (5, 5), 0)
    cb_noise  = float((ycbcr[:, :, 2] - cb_blur).std())
    cr_noise  = float((ycbcr[:, :, 1] - cr_blur).std())

    hue_hist, _ = np.histogram(hue, bins=18, range=(0, 180))
    p_hue = hue_hist.astype(np.float64) + 1e-9
    p_hue /= p_hue.sum()
    hue_entropy = float(-np.sum(p_hue * np.log(p_hue)))

    return np.array(
        [
            float(img_rgb[:, :, 0].std()),
            float(img_rgb[:, :, 1].std()),
            float(img_rgb[:, :, 2].std()),
            float(sat.mean()), float(sat.std()),
            hue_entropy,
            cb_noise, cr_noise,
            float(ycbcr[:, :, 0].mean()), float(ycbcr[:, :, 0].std()),
            float(ycbcr[:, :, 1].std()),  float(ycbcr[:, :, 2].std()),
            float(sat.max() - sat.min()),
            float(np.percentile(sat, 95) - np.percentile(sat, 5)),
        ],
        dtype=np.float32,
    )


def _gradient_regularity(gray: np.ndarray) -> np.ndarray:
    """
    Gradient angle entropy per block + magnitude stats.
    AI images have overly uniform gradient fields.
    Total: 6 dims
    """
    gx     = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy     = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angles = np.arctan2(gy, gx)
    mag    = cv2.magnitude(gx, gy)

    block_entropies = []
    for y in range(0, gray.shape[0] - BLOCK_SIZE, BLOCK_SIZE):
        for x in range(0, gray.shape[1] - BLOCK_SIZE, BLOCK_SIZE):
            block = angles[y: y + BLOCK_SIZE, x: x + BLOCK_SIZE]
            hist, _ = np.histogram(block, bins=9, range=(-np.pi, np.pi))
            p = hist.astype(np.float64) + 1e-9
            p /= p.sum()
            block_entropies.append(-np.sum(p * np.log(p)))

    if not block_entropies:
        return np.zeros(6, dtype=np.float32)

    arr = np.array(block_entropies)
    mu  = arr.mean() + 1e-8
    return np.array(
        [mu, arr.std(), arr.std() / mu,
         float(mag.mean()), float(mag.std()),
         float(np.percentile(mag, 90))],
        dtype=np.float32,
    )


def extract_features(image_path: str) -> np.ndarray:
    """
    Full feature extraction pipeline v3.

    Breakdown (total ~161 dims):
        FFT profile + extras  : 36
        DCT block stats       :  8
        Noise residual        :  7
        Multi-scale sharpness : 10
        LBP histogram         : 26
        GLCM texture          : 20
        Color / chroma        : 14
        Gradient regularity   :  6
                               ———
                               ~127  (exact dims depend on image size / blocks)

    Returns L2-normalised float32 vector.
    """
    pil  = Image.open(image_path).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    rgb  = np.array(pil)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    vec = np.concatenate([
        _fft_features(gray),
        _dct_block_features(gray),
        _noise_features(gray),
        _multiscale_sharpness(gray),
        _lbp_features(gray),
        _glcm_features(gray),
        _color_features(rgb),
        _gradient_regularity(gray),
    ])
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Signal Score (heuristic fallback, v3)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_signal_score(feat: np.ndarray) -> float:
    """
    Heuristic fallback when ML models not available.

    v3 changes vs v2:
      - Uses 5 indicators instead of 3
      - Each capped individually before combining
      - Overall cap = SIGNAL_CAP (0.70)

    Indicators (all: high value → more likely AI):
      fft_slope     : flat decay → AI over-smoothing
      noise_std     : low noise  → no camera sensor noise
      peak_ratio    : high peaks → GAN checkerboard
      autocorr      : high autocorr → structured (non-random) noise
      quad_std      : low quad_std → unnatural frequency symmetry
    """
    # feature indices (see _fft_features and _noise_features)
    fft_slope  = float(feat[FFT_BINS])          # index 32
    peak_ratio = float(feat[FFT_BINS + 1])      # index 33
    quad_std   = float(feat[FFT_BINS + 2])      # index 34  (quad symmetry)
    noise_std  = float(feat[FFT_BINS + 4 + 0])  # first noise feature
    autocorr   = float(feat[FFT_BINS + 4 + 1])  # second noise feature

    slope_s  = float(np.clip(1.0 + fft_slope / 0.03, 0.0, 1.0))
    noise_s  = float(np.clip(1.0 - noise_std / 8.0,  0.0, 1.0))
    peak_s   = float(np.clip(peak_ratio * 200.0,      0.0, 1.0))
    autocorr_s = float(np.clip((autocorr - 0.3) / 0.5, 0.0, 1.0))
    sym_s    = float(np.clip(1.0 - quad_std / 0.5,   0.0, 1.0))

    raw = (0.30 * slope_s +
           0.30 * noise_s +
           0.15 * peak_s  +
           0.15 * autocorr_s +
           0.10 * sym_s)

    return float(min(raw, SIGNAL_CAP))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Training
# ══════════════════════════════════════════════════════════════════════════════

def load_cifake_dataset(
    data_dir: str, max_per_class: int = 5000
) -> Tuple[np.ndarray, np.ndarray]:
    real_dir = Path(data_dir) / "REAL"
    fake_dir = Path(data_dir) / "FAKE"

    real_paths: List[Path] = []
    fake_paths: List[Path] = []

    for folder, store in [(real_dir, real_paths), (fake_dir, fake_paths)]:
        if folder.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                store += list(folder.glob(ext))
            logger.info("%s: %d images", folder.name, len(store))
        else:
            logger.warning("Not found: %s", folder)

    if not real_paths or not fake_paths:
        logger.warning("Dataset missing — falling back to synthetic data")
        return _generate_synthetic(n=max_per_class)

    np.random.shuffle(real_paths)
    np.random.shuffle(fake_paths)

    X_real = _extract_batch(real_paths[:max_per_class], "REAL")
    X_fake = _extract_batch(fake_paths[:max_per_class], "FAKE")

    X = np.vstack([X_real, X_fake])
    y = np.array([0] * len(X_real) + [1] * len(X_fake), dtype=int)
    return X, y


def _extract_batch(paths: List[Path], label: str) -> np.ndarray:
    feats = []
    for i, p in enumerate(paths):
        try:
            feats.append(extract_features(str(p)))
            if (i + 1) % 500 == 0:
                logger.info("  [%s] %d / %d", label, i + 1, len(paths))
        except Exception as e:
            logger.debug("Skip %s: %s", p, e)
    logger.info("  [%s] Done — %d features", label, len(feats))
    return np.array(feats, dtype=np.float32)


def _generate_synthetic(n: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic fallback when CIFAKE not available (for quick smoke-tests)."""
    import tempfile
    from PIL import ImageFilter

    logger.info("Generating %d synthetic samples per class", n)
    real_feats: List[np.ndarray] = []
    fake_feats: List[np.ndarray] = []
    tmp_files:  List[str]        = []
    size = 128

    for _ in range(n):
        # Real: natural gradient + camera-like noise
        arr = np.zeros((size, size, 3), dtype=np.float32)
        for y in range(size):
            for x in range(size):
                arr[y, x] = [255 * x / size, 255 * y / size,
                             128 + 30 * np.sin(y * 0.15) * np.cos(x * 0.15)]
        arr = np.clip(arr + np.random.normal(0, 9, arr.shape), 0, 255).astype(np.uint8)
        tmp = tempfile.mktemp(suffix=".jpg")
        Image.fromarray(arr).save(tmp, format="JPEG", quality=93)
        tmp_files.append(tmp)
        try:
            real_feats.append(extract_features(tmp))
        except Exception:
            pass

        # Fake: smooth sinusoidal + minimal noise (GAN-like)
        arr2 = np.zeros((size, size, 3), dtype=np.float32)
        for y in range(size):
            for x in range(size):
                arr2[y, x] = [128 + 60 * np.sin(y * np.pi / size),
                              128 + 60 * np.cos(x * np.pi / size),
                              180 + 25 * np.sin((x + y) * np.pi / size)]
        arr2 = np.clip(
            np.array(Image.fromarray(arr2.astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(2)
            )).astype(np.float32) + np.random.normal(0, 1.0, (size, size, 3)),
            0, 255,
        ).astype(np.uint8)
        tmp2 = tempfile.mktemp(suffix=".jpg")
        Image.fromarray(arr2).save(tmp2, format="JPEG", quality=93)
        tmp_files.append(tmp2)
        try:
            fake_feats.append(extract_features(tmp2))
        except Exception:
            pass

    for f in tmp_files:
        try:
            os.remove(f)
        except OSError:
            pass

    X = np.vstack([np.array(real_feats), np.array(fake_feats)])
    y = np.array([0] * len(real_feats) + [1] * len(fake_feats), dtype=int)
    return X, y


def build_ensemble() -> VotingClassifier:
    """
    v3 ensemble: GradientBoosting + ExtraTrees + calibrated SVM.

    Why GBM instead of plain RF?
      - GBM fits residuals iteratively → much better on tabular features
      - ExtraTrees adds diversity (random splits vs best splits)
      - SVM with RBF stays strong on high-dimensional normalised vectors

    Weights: GBM=2, ET=1, SVM=1  (GBM double weight — strongest learner)
    """
    gbm = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )
    et = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    svm_cal = CalibratedClassifierCV(
        SVC(kernel="rbf", C=10.0, gamma="scale",
            class_weight="balanced", probability=False),
        cv=5, method="isotonic",
    )
    return VotingClassifier(
        estimators=[("gbm", gbm), ("et", et), ("svm", svm_cal)],
        voting="soft",
        weights=[2, 1, 1],
        n_jobs=-1,
    )


def train(
    data_dir:       str   = None,
    max_per_class:  int   = 5000,
    test_size:      float = 0.20,
) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("MODULE 1 v3: AI-Generation Detector — Training")
    logger.info("=" * 60)

    cifake_dir = data_dir or str(_ROOT / "data" / "cifake")

    print("[1/4] Loading dataset ...")
    X, y = load_cifake_dataset(cifake_dir, max_per_class)
    print(f"  Samples: {len(X)}  |  Features: {X.shape[1]}")
    print(f"  Real={( y==0).sum()}  Fake={(y==1).sum()}")

    print("[2/4] Train/test split ...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    print("[3/4] Fitting ensemble (GBM + ExtraTrees + SVM) ...")
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    ensemble = build_ensemble()
    ensemble.fit(X_tr_s, y_tr)

    print("[4/4] Evaluating ...")
    metrics = _evaluate(ensemble, X_te_s, y_te)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(scaler,   open(MODELS_DIR / "ai_gen_scaler.pkl",   "wb"))
    pickle.dump(ensemble, open(MODELS_DIR / "ai_gen_ensemble.pkl", "wb"))
    print(f"Models saved → {MODELS_DIR}")
    return metrics


def _evaluate(ensemble: VotingClassifier,
              X_te_s:   np.ndarray,
              y_te:     np.ndarray) -> Dict[str, Any]:
    y_prob = ensemble.predict_proba(X_te_s)[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    auc    = roc_auc_score(y_te, y_prob)

    report = {
        "accuracy"        : round(accuracy_score(y_te, y_pred),               4),
        "precision"       : round(precision_score(y_te, y_pred, zero_division=0), 4),
        "recall"          : round(recall_score(y_te, y_pred, zero_division=0), 4),
        "f1_score"        : round(f1_score(y_te, y_pred, zero_division=0),    4),
        "roc_auc"         : round(auc,                                         4),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "threshold"       : DECISION_THRESHOLD,
    }
    print(classification_report(y_te, y_pred, target_names=["Real", "Fake"]))
    print(f"  ROC-AUC : {auc:.4f}")
    logger.info("AI-Gen v3 → Acc=%.3f  F1=%.3f  AUC=%.3f",
                report["accuracy"], report["f1_score"], auc)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Inference
# ══════════════════════════════════════════════════════════════════════════════

_scaler:   Optional[StandardScaler]   = None
_ensemble: Optional[VotingClassifier] = None
_loaded:   bool = False


def _load_models() -> bool:
    global _scaler, _ensemble, _loaded
    if _loaded:
        return True
    sp = MODELS_DIR / "ai_gen_scaler.pkl"
    ep = MODELS_DIR / "ai_gen_ensemble.pkl"
    if not (sp.exists() and ep.exists()):
        return False
    try:
        _scaler   = pickle.load(open(sp, "rb"))
        _ensemble = pickle.load(open(ep, "rb"))
        _loaded   = True
        logger.info("AI-Gen v3 models loaded ✓")
        return True
    except Exception as e:
        logger.warning("Failed to load models: %s", e)
        return False


def predict(
    image_path: str,
    evidence_dir=None,
    save_evidence: bool = True,
    threshold: float = DECISION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Predict P(AI-generated) for a single image.

    Fusion:
        combined = ML_ensemble * 0.85 + signal_heuristic * 0.15
        is_ai_generated = combined >= 0.65

    Returns
    -------
    {
        probability_ai_generated : float   [0, 1]
        is_ai_generated          : bool
        ml_score                 : float   (ensemble probability)
        signal_score             : float   (heuristic, capped at 0.70)
        ml_available             : bool
        confidence               : float   [0, 1]  (distance from 0.5)
        threshold                : float
        features_used            : str
    }
    """
    try:
        feat = extract_features(image_path)
    except Exception as e:
        logger.error("Feature extraction failed: %s", e)
        return {
            "probability_ai_generated": 0.5,
            "is_ai_generated": False,
            "ml_score": 0.5, "signal_score": 0.5,
            "ml_available": False, "confidence": 0.0,
            "threshold": threshold,
            "features_used": "error",
        }

    signal_score = _compute_signal_score(feat)

    if not _load_models():
        logger.warning("Models not trained — signal score only")
        return {
            "probability_ai_generated": round(signal_score, 4),
            "is_ai_generated": signal_score >= threshold,
            "ml_score": signal_score,
            "signal_score": round(signal_score, 4),
            "ml_available": False,
            "confidence": round(abs(signal_score - 0.5) * 2, 4),
            "threshold": threshold,
            "features_used": "signal_only",
        }

    feat_s  = _scaler.transform(feat.reshape(1, -1))
    ml_prob = float(_ensemble.predict_proba(feat_s)[0][1])

    combined = round(
        ml_prob      * FUSION_W_ML     +
        signal_score * FUSION_W_SIGNAL,
        4,
    )

    logger.info(
        "AI-Gen predict: ML=%.4f  signal=%.4f  →  %.4f  (thr=%.2f)",
        ml_prob, signal_score, combined, threshold,
    )

    return {
        "probability_ai_generated": combined,
        "is_ai_generated"         : combined >= threshold,
        "ml_score"                : round(ml_prob,      4),
        "signal_score"            : round(signal_score, 4),
        "ml_available"            : True,
        "confidence"              : round(abs(combined - 0.5) * 2, 4),
        "threshold"               : threshold,
        "features_used"           : "ml+signal",
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Train  : python ai_gen_module_v3.py train [data_dir]")
        print("  Predict: python ai_gen_module_v3.py predict <image_path>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "train":
        root = sys.argv[2] if len(sys.argv) > 2 else None
        metrics = train(data_dir=root)
        print(json.dumps(metrics, indent=2))

    elif cmd == "predict":
        if len(sys.argv) < 3:
            print("Provide image path.")
            sys.exit(1)
        result = predict(sys.argv[2])
        print(json.dumps(result, indent=2))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
