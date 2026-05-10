"""
splicing_module.py — MODULE 3: Image Splicing / Tampering Detector  (v2)
=========================================================================
IMPROVEMENTS OVER v1:
   
DATASET:
    CASIA TIDE — https://forensics.idealtest.org
    Structure:
        data/casia/authentic/*.jpg   (or .tif)
        data/casia/tampered/*.jpg

SAVED MODELS:
    models/splicing_scaler.pkl
    models/splicing_ensemble.pkl

OUTPUT:
    probability_splicing ∈ [0.0, 1.0]
"""

import io
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

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

logger = logging.getLogger("splicing_module")

_ROOT      = Path(__file__).parent.parent.parent
MODELS_DIR = _ROOT / "models"

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE         = (256, 256)
ELA_QUALITIES    = [92, 75, 60]   # 3 levels — catches double-compression at any quality
BLOCK_SIZE       = 32
BLOCK_STRIDE     = 16
DCT_BLOCK        = 8
N_DCT_COEFFS     = 25
COPY_MOVE_THRESH = 0.98

# ── Inference config ───────────────────────────────────────────────────────────
FUSION_W_ML        = 0.80
FUSION_W_SIGNAL    = 0.20
DECISION_THRESHOLD = 0.55   # splicing is subtle → lower threshold than deepfake


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def _ela_single(pil_img: Image.Image, quality: int) -> np.ndarray:
    """Run ELA at one quality level and return the residual map."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    diff = np.abs(
        np.array(pil_img).astype(np.float32) -
        np.array(recompressed).astype(np.float32)
    ).mean(axis=2)
    return diff


def _ela_features(pil_img: Image.Image) -> np.ndarray:
    """
    Multi-quality ELA residual statistics.

    Using 3 quality levels catches spliced regions regardless of
    what quality the original image was saved at.

    Per quality: [mean, std, max, p90, high_ratio, block_std, block_max] = 7
    Total: 7 × 3 = 21 features

    + gradient of the ELA map (boundary detection): 4 features
    Grand total: 25 features
    """
    feats = []
    ela_maps = []

    for q in ELA_QUALITIES:
        diff = _ela_single(pil_img, q)
        ela_maps.append(diff)
        mx   = diff.max()
        p90  = float(np.percentile(diff, 90))
        hr   = float((diff > 0.10 * mx).mean()) if mx > 0 else 0.0

        # block-level variance map
        h, w     = diff.shape
        block_vals = []
        for y in range(0, h - BLOCK_SIZE, BLOCK_SIZE):
            for x in range(0, w - BLOCK_SIZE, BLOCK_SIZE):
                block_vals.append(diff[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE].mean())

        ba     = np.array(block_vals, dtype=np.float32) if block_vals else np.array([0.0])
        feats.extend([diff.mean(), diff.std(), mx, p90, hr,
                      float(ba.std()), float(ba.max())])

    # ELA gradient: edges of the splice region show up as sharp
    # gradients in the ELA map at the primary quality
    ela_primary = ela_maps[0].astype(np.float32)
    gx = cv2.Sobel(ela_primary, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ela_primary, cv2.CV_32F, 0, 1, ksize=3)
    gm = cv2.magnitude(gx, gy)
    feats.extend([
        float(gm.mean()), float(gm.std()),
        float(np.percentile(gm, 90)),
        float((gm > gm.mean() + 2 * gm.std()).mean()),   # sharp-edge pixel fraction
    ])

    return np.array(feats, dtype=np.float32)


def _dct_block_features(gray: np.ndarray) -> np.ndarray:
    """
    JPEG block DCT coefficient statistics.
    Spliced patches inserted at a different quality → different AC energy.
    Total: 9 features
    """
    gray_f      = gray.astype(np.float32)
    ac_energies = []
    dc_values   = []

    for y in range(0, gray.shape[0] - DCT_BLOCK, DCT_BLOCK):
        for x in range(0, gray.shape[1] - DCT_BLOCK, DCT_BLOCK):
            block = gray_f[y:y + DCT_BLOCK, x:x + DCT_BLOCK]
            dct   = cv2.dct(block)
            dc_values.append(float(dct[0, 0]))
            ac = dct.flatten()[1:]
            ac_energies.append(float((ac ** 2).mean()))

    if not ac_energies:
        return np.zeros(9, dtype=np.float32)

    ac  = np.array(ac_energies, dtype=np.float32)
    dc  = np.array(dc_values,   dtype=np.float32)
    std = ac.std() + 1e-8
    m3  = ((ac - ac.mean()) ** 3).mean()
    m4  = ((ac - ac.mean()) ** 4).mean()
    kurt = float(np.clip(m4 / (std ** 4) - 3.0, -5, 5))
    skew = float(np.clip(m3 / (std ** 3),       -5, 5))

    return np.array([
        ac.mean(), ac.std(), ac.max(),
        float(np.percentile(ac, 75)),
        dc.var(), dc.std(),
        kurt, skew,
        float((ac > ac.mean() + 2 * ac.std()).mean()),  # high-AC block fraction
    ], dtype=np.float32)


def _jpeg_ghost_features(pil_img: Image.Image) -> np.ndarray:
    """
    JPEG ghost detection (double-compression inconsistency).

    A genuine image re-saved at quality Q shows uniform ELA across
    the whole image. A spliced patch that was originally saved at
    a DIFFERENT quality creates a visible "ghost" — its ELA residual
    is systematically different from the background.

    We compare ELA residuals across quality pairs to measure inconsistency.
    Total: 6 features
    """
    maps = {q: _ela_single(pil_img, q) for q in ELA_QUALITIES}

    # Cross-quality inconsistency
    q_pairs = [(ELA_QUALITIES[0], ELA_QUALITIES[1]),
               (ELA_QUALITIES[0], ELA_QUALITIES[2]),
               (ELA_QUALITIES[1], ELA_QUALITIES[2])]

    feats = []
    for qa, qb in q_pairs:
        diff = np.abs(maps[qa] - maps[qb])
        feats.extend([float(diff.mean()), float(diff.std())])

    return np.array(feats, dtype=np.float32)


def _jpeg_blocking_features(gray: np.ndarray) -> np.ndarray:
    """
    JPEG blocking artifact boundary discontinuities.
    Spliced regions have different quantization → different block patterns.
    Total: 6 features
    """
    h_diffs = []
    for y in range(DCT_BLOCK, gray.shape[0] - DCT_BLOCK, DCT_BLOCK):
        row_above = gray[y - 1, :].astype(np.float32)
        row_below = gray[y,     :].astype(np.float32)
        h_diffs.append(float(np.abs(row_above - row_below).mean()))

    v_diffs = []
    for x in range(DCT_BLOCK, gray.shape[1] - DCT_BLOCK, DCT_BLOCK):
        col_left  = gray[:, x - 1].astype(np.float32)
        col_right = gray[:, x    ].astype(np.float32)
        v_diffs.append(float(np.abs(col_left - col_right).mean()))

    h = np.array(h_diffs) if h_diffs else np.array([0.0])
    v = np.array(v_diffs) if v_diffs else np.array([0.0])

    return np.array([
        h.mean(), h.std(),
        v.mean(), v.std(),
        float(np.clip((h.mean() + 1e-6) / (v.mean() + 1e-6), 0, 5)),
        float(h.max() - v.max()),   # horizontal vs vertical blocking imbalance
    ], dtype=np.float32)


def _edge_density_features(gray: np.ndarray) -> np.ndarray:
    """
    Block-level Canny edge density.
    Spliced regions often have anomalous edge density vs surroundings.
    Total: 6 features
    """
    edges = cv2.Canny(gray, 50, 150)
    blocks = []

    for y in range(0, gray.shape[0] - BLOCK_SIZE, BLOCK_SIZE):
        for x in range(0, gray.shape[1] - BLOCK_SIZE, BLOCK_SIZE):
            b = edges[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            blocks.append(float(b.mean()) / 255.0)

    if not blocks:
        return np.zeros(6, dtype=np.float32)

    arr = np.array(blocks)
    mu, sigma = arr.mean(), arr.std()
    cv         = sigma / (mu + 1e-8)
    outlier_f  = float((np.abs(arr - mu) > 2 * sigma).mean()) if sigma > 1e-6 else 0.0

    return np.array([
        mu, sigma, cv, outlier_f, arr.max(),
        float(np.percentile(arr, 75) - np.percentile(arr, 25)),  # IQR
    ], dtype=np.float32)


def _noise_variance_features(gray: np.ndarray) -> np.ndarray:
    """
    Per-block noise level estimation (normalized).
    Spliced regions have different noise std from background.
    Total: 6 features
    """
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    resid  = gray.astype(np.float32) - smooth
    blocks = []

    for y in range(0, gray.shape[0] - BLOCK_SIZE, BLOCK_SIZE):
        for x in range(0, gray.shape[1] - BLOCK_SIZE, BLOCK_SIZE):
            b = resid[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            blocks.append(float(b.std()))

    if not blocks:
        return np.zeros(6, dtype=np.float32)

    arr = np.array(blocks)
    mu, sigma = arr.mean(), arr.std()
    cv         = sigma / (mu + 1e-8)
    outlier_f  = float((np.abs(arr - mu) > 2 * sigma).mean()) if sigma > 1e-6 else 0.0

    return np.array([
        mu, sigma, cv, outlier_f,
        float(arr.max() - arr.min()),
        float(np.percentile(arr, 90)),
    ], dtype=np.float32)


def _color_inconsistency_features(pil_img: Image.Image) -> np.ndarray:
    """
    Block-level color inconsistency in Lab color space.

    Lab is perceptually uniform — ΔE between adjacent blocks
    shows where a foreign patch was inserted.
    Total: 6 features
    """
    rgb  = np.array(pil_img).astype(np.uint8)
    lab  = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    h, w = lab.shape[:2]

    block_means: List[np.ndarray] = []
    for y in range(0, h - BLOCK_SIZE, BLOCK_SIZE):
        for x in range(0, w - BLOCK_SIZE, BLOCK_SIZE):
            b = lab[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            block_means.append(b.mean(axis=(0, 1)))  # (L, a, b)

    if len(block_means) < 2:
        return np.zeros(6, dtype=np.float32)

    bm  = np.array(block_means, dtype=np.float32)   # (N, 3)
    delta_e: List[float] = []
    for i in range(len(bm) - 1):
        de = float(np.linalg.norm(bm[i] - bm[i + 1]))
        delta_e.append(de)

    de_arr = np.array(delta_e)
    mu, sigma = de_arr.mean(), de_arr.std()

    return np.array([
        mu, sigma,
        float(de_arr.max()),
        float(sigma / (mu + 1e-8)),                             # CV
        float((de_arr > mu + 2 * sigma).mean()),                # outlier fraction
        float(np.percentile(de_arr, 95) - np.percentile(de_arr, 5)),  # range
    ], dtype=np.float32)


def _copy_move_features(gray: np.ndarray, max_blocks: int = 1500) -> np.ndarray:
    """
    DCT block-matching copy-move detection.
    Detects when a region of the image was copy-pasted from elsewhere.
    Total: 4 features
    """
    feats, positions = [], []

    for y in range(0, gray.shape[0] - DCT_BLOCK, BLOCK_STRIDE):
        for x in range(0, gray.shape[1] - DCT_BLOCK, BLOCK_STRIDE):
            block = gray[y:y + DCT_BLOCK, x:x + DCT_BLOCK].astype(np.float32)
            dct   = cv2.dct(block).flatten()[:N_DCT_COEFFS]
            norm  = np.linalg.norm(dct)
            feats.append(dct / norm if norm > 0 else dct)
            positions.append((y, x))

    if not feats:
        return np.zeros(4, dtype=np.float32)

    fa = np.array(feats)
    if len(fa) > max_blocks:
        idx      = np.random.choice(len(fa), max_blocks, replace=False)
        fa       = fa[idx]
        positions = [positions[i] for i in idx]

    sort_idx   = np.lexsort(fa.T)
    sorted_fa  = fa[sort_idx]
    sorted_pos = [positions[i] for i in sort_idx]

    matches, sims = 0, []
    for i in range(len(sorted_fa) - 1):
        sim = float(np.dot(sorted_fa[i], sorted_fa[i + 1]))
        if sim < COPY_MOVE_THRESH:
            continue
        (y1, x1), (y2, x2) = sorted_pos[i], sorted_pos[i + 1]
        if np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) >= 2 * DCT_BLOCK:
            matches += 1
            sims.append(sim)

    total = len(fa)
    mr    = matches / total if total > 0 else 0.0
    ms    = float(np.mean(sims)) if sims else 0.0
    mx_s  = float(np.max(sims))  if sims else 0.0

    return np.array([mr, float(matches), ms, mx_s], dtype=np.float32)


def extract_features(image_path: str) -> np.ndarray:
    """
    Full splicing feature vector (v2).

    Feature breakdown:
        Multi-quality ELA + gradient  : 25
        JPEG ghost (cross-quality)    :  6
        DCT block stats               :  9
        JPEG blocking artifacts       :  6
        Edge density map              :  6
        Noise variance map            :  6
        Color inconsistency (Lab ΔE)  :  6
        Copy-move DCT matching        :  4
                                       ——
                                       68 features

    Returns L2-normalised float32 vector.
    """
    pil  = Image.open(image_path).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)

    vec = np.concatenate([
        _ela_features(pil),
        _jpeg_ghost_features(pil),
        _dct_block_features(gray),
        _jpeg_blocking_features(gray),
        _edge_density_features(gray),
        _noise_variance_features(gray),
        _color_inconsistency_features(pil),
        _copy_move_features(gray),
    ])
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Signal Score (heuristic fallback, v2)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_signal_score(feat: np.ndarray) -> float:
    """
    Heuristic fallback when ML models not available.

    v2: uses multi-quality ELA variance + JPEG ghost inconsistency
    instead of just raw ELA mean/std.

    Indicators:
        ela_block_std  : high variance across blocks → splice boundary
        ela_high_ratio : high-ELA pixel fraction → large spliced area
        ghost_mean     : cross-quality ELA disagreement → double compression
    """
    # ELA stats from quality=92 (first 7 features)
    ela_block_std  = float(np.clip(feat[5] / 10.0, 0.0, 1.0))   # feat[5] = block_std @ q92
    ela_high_ratio = float(np.clip(feat[4] * 3.0,  0.0, 1.0))   # feat[4] = high_ratio @ q92

    # JPEG ghost: indices 21–26 (cross-quality differences)
    ghost_mean = float(np.clip(feat[21] / 5.0, 0.0, 1.0)) if len(feat) > 21 else 0.0

    raw = 0.40 * ela_block_std + 0.30 * ela_high_ratio + 0.30 * ghost_mean
    return float(np.clip(raw, 0.0, 0.80))   # cap at 0.80


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Training
# ══════════════════════════════════════════════════════════════════════════════

def _load_casia_dataset(
    data_dir: str, max_per_class: int = 3000
) -> Tuple[np.ndarray, np.ndarray]:
    """Load CASIA TIDE or compatible splice dataset."""
    auth_dir = tamp_dir = None

    for name in ("authentic", "Authentic", "real", "REAL", "Au"):
        d = Path(data_dir) / name
        if d.exists():
            auth_dir = d
            break

    for name in ("tampered", "Tampered", "fake", "FAKE", "Tp", "spliced"):
        d = Path(data_dir) / name
        if d.exists():
            tamp_dir = d
            break

    auth_paths: List[Path] = []
    tamp_paths: List[Path] = []

    for folder, store in [(auth_dir, auth_paths), (tamp_dir, tamp_paths)]:
        if folder is not None:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp"):
                store += list(folder.glob(ext))
            logger.info("%s: %d images", folder.name, len(store))
        else:
            logger.warning("Folder not found in %s", data_dir)

    if not auth_paths or not tamp_paths:
        logger.warning("CASIA not found — using synthetic fallback")
        return _generate_synthetic(n=max_per_class)

    np.random.shuffle(auth_paths)
    np.random.shuffle(tamp_paths)

    X_auth = _extract_batch(auth_paths[:max_per_class], "AUTHENTIC")
    X_tamp = _extract_batch(tamp_paths[:max_per_class], "TAMPERED")

    X = np.vstack([X_auth, X_tamp])
    y = np.array([0] * len(X_auth) + [1] * len(X_tamp), dtype=int)
    return X, y


def _extract_batch(paths: List[Path], label: str) -> np.ndarray:
    feats = []
    for i, p in enumerate(paths):
        try:
            feats.append(extract_features(str(p)))
            if (i + 1) % 300 == 0:
                logger.info("  [%s] %d / %d", label, i + 1, len(paths))
        except Exception as e:
            logger.debug("Skip %s: %s", p, e)
    logger.info("  [%s] Done — %d features", label, len(feats))
    return np.array(feats, dtype=np.float32)


def _generate_synthetic(n: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic fallback when CASIA is not available."""
    import tempfile
    logger.info("Generating %d synthetic authentic + %d spliced images", n, n)
    auth_f: List[np.ndarray] = []
    spl_f:  List[np.ndarray] = []
    tmp_files: List[str]     = []
    size = 256

    for _ in range(n):
        # Authentic: natural gradient + uniform noise
        arr = np.zeros((size, size, 3), dtype=np.float32)
        for y in range(size):
            for x in range(size):
                arr[y, x] = [255 * x / size, 255 * y / size,
                             128 + 30 * np.sin(y * 0.1) * np.cos(x * 0.1)]
        arr = np.clip(arr + np.random.normal(0, 7, arr.shape), 0, 255).astype(np.uint8)
        p = tempfile.mktemp(suffix=".jpg")
        Image.fromarray(arr).save(p, format="JPEG", quality=92)
        tmp_files.append(p)
        try:
            auth_f.append(extract_features(p))
        except Exception:
            pass

        # Tampered: patch saved at different JPEG quality inserted into background
        bg    = arr.copy()
        ph, pw = size // 3, size // 3
        patch  = np.full((ph, pw, 3), [200, 80, 60], dtype=np.uint8)
        patch += np.random.randint(-15, 15, patch.shape).astype(np.uint8)

        # Re-save patch at low quality to simulate double compression
        buf = io.BytesIO()
        Image.fromarray(patch).save(buf, format="JPEG", quality=55)
        buf.seek(0)
        patch = np.array(Image.open(buf).convert("RGB"))

        y0, x0 = size // 4, size // 4
        bg[y0:y0 + ph, x0:x0 + pw] = patch

        p2 = tempfile.mktemp(suffix=".jpg")
        Image.fromarray(bg).save(p2, format="JPEG", quality=92)
        tmp_files.append(p2)
        try:
            spl_f.append(extract_features(p2))
        except Exception:
            pass

    for t in tmp_files:
        try:
            os.remove(t)
        except OSError:
            pass

    X = np.vstack([np.array(auth_f), np.array(spl_f)])
    y = np.array([0] * len(auth_f) + [1] * len(spl_f), dtype=int)
    return X, y


def build_ensemble() -> VotingClassifier:
    """
    v2 ensemble: GradientBoosting + ExtraTrees + calibrated SVM.

    Same architecture as ai_gen_module v3 for consistency.
    GBM gets double weight — strongest learner on structured forensic features.
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
    data_dir:      str   = None,
    max_per_class: int   = 3000,
    test_size:     float = 0.20,
) -> Dict[str, Any]:
    """Train and save the splicing detector."""
    logger.info("=" * 60)
    logger.info("MODULE 3 v2: Splicing Detector — Training")
    logger.info("=" * 60)

    casia_dir = data_dir or str(_ROOT / "data" / "casia")

    print("[1/4] Loading dataset ...")
    X, y = _load_casia_dataset(casia_dir, max_per_class)
    print(f"  Samples: {len(X)}  |  Features: {X.shape[1]}")
    print(f"  Authentic={( y==0).sum()}  Tampered={(y==1).sum()}")

    print("[2/4] Train/test split ...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    print("[3/4] Fitting ensemble (GBM + ExtraTrees + SVM) ...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    ensemble = build_ensemble()
    ensemble.fit(X_tr_s, y_tr)

    print("[4/4] Evaluating ...")
    metrics = _evaluate(ensemble, X_te_s, y_te)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(scaler,   open(MODELS_DIR / "splicing_scaler.pkl",   "wb"))
    pickle.dump(ensemble, open(MODELS_DIR / "splicing_ensemble.pkl", "wb"))
    print(f"Models saved → {MODELS_DIR}")
    return metrics


def _evaluate(
    ensemble: VotingClassifier,
    X_te_s:   np.ndarray,
    y_te:     np.ndarray,
) -> Dict[str, Any]:
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
    print(classification_report(y_te, y_pred, target_names=["Authentic", "Tampered"]))
    print(f"  ROC-AUC : {auc:.4f}")
    logger.info("Splicing v2 → Acc=%.3f  F1=%.3f  AUC=%.3f",
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
    sp = MODELS_DIR / "splicing_scaler.pkl"
    ep = MODELS_DIR / "splicing_ensemble.pkl"
    if not (sp.exists() and ep.exists()):
        return False
    try:
        _scaler   = pickle.load(open(sp, "rb"))
        _ensemble = pickle.load(open(ep, "rb"))
        _loaded   = True
        logger.info("Splicing v2 models loaded ✓")
        return True
    except Exception as e:
        logger.warning("Failed to load models: %s", e)
        return False


def predict(image_path: str) -> Dict[str, Any]:
    """
    Predict P(spliced/tampered) for a single image.

    Fusion:
        combined = ML_ensemble * 0.80 + signal_heuristic * 0.20
        is_spliced = combined >= 0.55

    Returns
    -------
    {
        probability_splicing : float   [0, 1]
        is_spliced           : bool
        ml_score             : float
        signal_score         : float
        ml_available         : bool
        confidence           : float   [0, 1]
        threshold            : float
    }
    """
    try:
        feat = extract_features(image_path)
    except Exception as e:
        logger.error("Feature extraction failed: %s", e)
        return {
            "probability_splicing": 0.0,
            "is_spliced": False,
            "ml_score": 0.0, "signal_score": 0.0,
            "ml_available": False, "confidence": 0.0,
            "threshold": DECISION_THRESHOLD,
            "error": str(e),
        }

    signal_score = _compute_signal_score(feat)

    if not _load_models():
        logger.warning("Models not trained — signal score only")
        return {
            "probability_splicing": round(signal_score, 4),
            "is_spliced": signal_score >= DECISION_THRESHOLD,
            "ml_score": signal_score,
            "signal_score": round(signal_score, 4),
            "ml_available": False,
            "confidence": round(abs(signal_score - 0.5) * 2, 4),
            "threshold": DECISION_THRESHOLD,
        }

    feat_s  = _scaler.transform(feat.reshape(1, -1))
    ml_prob = float(_ensemble.predict_proba(feat_s)[0][1])

    combined = round(
        ml_prob      * FUSION_W_ML     +
        signal_score * FUSION_W_SIGNAL,
        4,
    )

    logger.info(
        "Splicing predict: ML=%.4f  signal=%.4f  →  %.4f  (thr=%.2f)",
        ml_prob, signal_score, combined, DECISION_THRESHOLD,
    )

    return {
        "probability_splicing": combined,
        "is_spliced"          : combined >= DECISION_THRESHOLD,
        "ml_score"            : round(ml_prob,      4),
        "signal_score"        : round(signal_score, 4),
        "ml_available"        : True,
        "confidence"          : round(abs(combined - 0.5) * 2, 4),
        "threshold"           : DECISION_THRESHOLD,
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
        print("  Train  : python splicing_module_v2.py train [data_dir]")
        print("  Predict: python splicing_module_v2.py predict <image_path>")
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
