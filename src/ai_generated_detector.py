"""
ai_generated_detector.py  ·  v2.0  — CNN + Signal-Processing Hybrid
======================================================================
    • Added a CNN-based scorer using TensorFlow/Keras when available.
    • CNN input: 128×128 RGB  →  single probability output [0,1].
    • Pre-trained model loaded from  models/ai_cnn.h5  if it exists.
    • If no model file exists, a lightweight placeholder is auto-created
      and saved (random weights, but shows the full integration path).
    • CNN toggle:  analyze(..., use_cnn=True/False)
    • Final score = weighted blend(CNN, frequency, noise, texture).
    • All v1 signal-processing sub-detectors unchanged.

DEPENDENCY MATRIX:
    tensorflow installed + models/ai_cnn.h5 exists → real CNN   ★
    tensorflow installed, no .h5                   → placeholder CNN ✓
    tensorflow NOT installed                        → signal-only   ✓
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from src.feature_extractor import extract as extract_features
from compression_analysis import compute_sha256

def _safe_entropy(arr: np.ndarray) -> float:
    """Numpy-based Shannon entropy (avoids scipy/torch compatibility issues)."""
    p = arr.astype(np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p /= p.sum()
    return float(-np.sum(p * np.log(p)))

logger = logging.getLogger("ai_generated_detector")

# ── ML Classifier (SVM + Random Forest) ──────────────────────────────────────
try:
    import ml_classifier as _ml
    ML_OK = True
    logger.info("ML classifier module loaded (SVM + Random Forest)")
except ImportError:
    ML_OK = False
    _ml   = None
    logger.warning("ml_classifier not found → ML scoring disabled")

# ── Optional TensorFlow (graceful degradation) ────────────────────────────────
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TF info logs
    import tensorflow as tf
    TF_OK = True
    logger.info("TensorFlow %s available", tf.__version__)
except ImportError:
    TF_OK = False
    logger.warning("TensorFlow not installed → CNN disabled. "
                   "Install: pip install tensorflow")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).parent.parent
MODEL_PATH = str(_ROOT / "models" / "ai_cnn.h5")
CNN_INPUT  = (128, 128)   # width × height fed to the CNN


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CNN Model Management
# ═══════════════════════════════════════════════════════════════════════════════

def _build_placeholder_cnn() -> "tf.keras.Model":
    """
    Lightweight MobileNet-style placeholder CNN.

    Architecture (fast, ~500K params):
        Input 128×128×3
        → Conv(32) → BN → ReLU → MaxPool
        → Conv(64) → BN → ReLU → MaxPool
        → Conv(128)→ BN → ReLU → GlobalAvgPool
        → Dense(64) → ReLU → Dropout(0.3)
        → Dense(1)  → Sigmoid

    With random weights this outputs values near 0.5.
    Replace with trained weights for real performance.

    Training target:
        label=0 → real photograph
        label=1 → AI-generated image
    Suggested datasets: CIFAKE, FaceForensics++, Stable Diffusion vs COCO
    """
    inp = tf.keras.Input(shape=(*CNN_INPUT, 3), name="image_input")
    x   = inp

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x   = tf.keras.layers.GlobalAveragePooling2D()(x)
    x   = tf.keras.layers.Dense(64, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="ai_prob")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="ai_detector_cnn")
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


_cnn_model = None   # lazy-loaded singleton

def _load_cnn() -> Optional["tf.keras.Model"]:
    """
    Load or create the CNN model (lazy singleton).
    Order of preference:
        1. Load  models/ai_cnn.h5  (pre-trained, best accuracy)
        2. Build placeholder (random weights, shows integration path)
        3. Return None if TF not available
    """
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model
    if not TF_OK:
        return None

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if os.path.isfile(MODEL_PATH):
        logger.info("Loading CNN from %s", MODEL_PATH)
        _cnn_model = tf.keras.models.load_model(MODEL_PATH)
    else:
        logger.warning("No trained model at %s — using placeholder CNN. "
                       "Train and save a model there for real accuracy.", MODEL_PATH)
        _cnn_model = _build_placeholder_cnn()
        # Save placeholder so future runs load faster
        try:
            _cnn_model.save(MODEL_PATH)
            logger.info("Placeholder CNN saved → %s", MODEL_PATH)
        except Exception as e:
            logger.warning("Could not save placeholder: %s", e)

    return _cnn_model


def cnn_score(image_path: str) -> float:
    """
    Run the CNN and return P(AI-generated) in [0, 1].

    Pre-processing:
        • Resize to CNN_INPUT (128×128)
        • Scale pixels to [0, 1]
        • Add batch dimension

    Returns 0.5 (neutral) when TF is unavailable.
    """
    model = _load_cnn()
    if model is None:
        logger.info("CNN unavailable → returning neutral 0.5")
        return 0.5

    img = Image.open(image_path).convert("RGB").resize(CNN_INPUT)
    arr = np.array(img, dtype=np.float32) / 255.0      # [0,1] normalisation
    arr = np.expand_dims(arr, axis=0)                  # (1, 128, 128, 3)

    prob = float(model.predict(arr, verbose=0)[0][0])
    logger.info("CNN score: %.4f", prob)
    return round(float(np.clip(prob, 0.0, 1.0)), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Signal-Processing Sub-detectors (v1, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_fft_spectrum(gray: np.ndarray) -> np.ndarray:
    fft       = np.fft.fft2(gray.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    return np.log1p(np.abs(fft_shift))


def _radial_profile(spec: np.ndarray, n=32) -> np.ndarray:
    h,w   = spec.shape
    cy,cx = h//2, w//2
    y_idx = np.arange(h).reshape(-1,1)
    x_idx = np.arange(w).reshape(1,-1)
    r_map = np.sqrt((y_idx-cy)**2 + (x_idx-cx)**2)
    edges = np.linspace(0, min(cy,cx), n+1)
    prof  = np.zeros(n)
    for i in range(n):
        m = (r_map>=edges[i]) & (r_map<edges[i+1])
        prof[i] = spec[m].mean() if m.any() else 0
    return prof


def _grid_peak_count(spec: np.ndarray, sigma=4.0) -> int:
    h,w   = spec.shape
    cy,cx = h//2, w//2
    y_idx = np.arange(h).reshape(-1,1)
    x_idx = np.arange(w).reshape(1,-1)
    r_map = np.sqrt((y_idx-cy)**2 + (x_idx-cx)**2)
    dc    = r_map < min(cy,cx)//10
    hf    = spec[~dc]
    mu,sd = hf.mean(), hf.std()
    return int((hf > mu+sigma*sd).sum()) if sd > 1e-6 else 0


def detect_frequency_artifacts(gray: np.ndarray) -> float:
    spec  = _compute_fft_spectrum(gray)
    peaks = _grid_peak_count(spec)
    prof  = _radial_profile(spec)
    pk_s  = float(np.clip(peaks/500.0, 0, 1))
    nm    = prof/prof.max() if prof.max()>0 else prof
    slope = float(np.polyfit(np.arange(len(nm)), nm, 1)[0])
    dc_s  = float(np.clip(1.0+slope/0.02, 0, 1))
    score = round(0.5*pk_s + 0.5*dc_s, 4)
    logger.info("Frequency score: %.4f (peaks=%d slope=%.4f)", score, peaks, slope)
    return score


def detect_noise_pattern(gray: np.ndarray) -> float:
    smooth   = cv2.GaussianBlur(gray.astype(np.float32),(5,5),0)
    residual = gray.astype(np.float32) - smooth
    ns       = float(np.clip(1.0-residual.std()/6.0, 0, 1))
    flat     = residual.flatten()
    ac       = float(np.corrcoef(flat[:-1],flat[1:])[0,1]) if len(flat)>1 else 0
    ac_s     = float(np.clip((ac+1.0)/2.0, 0, 1))
    hist,_   = np.histogram(residual, bins=64, range=(-30,30))
    ent_s    = float(np.clip(1.0-_safe_entropy(hist+1e-9)/np.log(64), 0, 1))
    score    = round(0.30*ns + 0.40*ac_s + 0.30*ent_s, 4)
    logger.info("Noise score: %.4f", score)
    return score


def detect_texture_regularity(gray: np.ndarray) -> float:
    levels = 64
    q      = (gray.astype(np.float32)/255.0*(levels-1)).astype(np.int32).clip(0,levels-1)
    glcm   = np.zeros((levels,levels), dtype=np.float64)
    np.add.at(glcm, (q[:,:-1].flatten(), q[:,1:].flatten()), 1)
    glcm  /= (glcm.sum()+1e-9)
    ent    = float(_safe_entropy(glcm.flatten()+1e-9))
    score  = round(float(np.clip((5.5-ent)/3.5, 0, 1)), 4)
    logger.info("Texture score: %.4f (GLCM entropy=%.4f)", score, ent)
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Visualisation helpers (v1, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _vis_spectrum(gray):
    spec = _compute_fft_spectrum(gray)
    n    = cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(n, cv2.COLORMAP_MAGMA)

def _vis_noise(gray):
    smooth   = cv2.GaussianBlur(gray.astype(np.float32),(5,5),0)
    residual = gray.astype(np.float32) - smooth
    amp      = np.clip(np.abs(residual)*10, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(amp, cv2.COLORMAP_HOT)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Public Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(image_path: str,
            evidence_dir: str  = "reports/evidence",
            save_evidence: bool = True,
            use_cnn: bool       = True,
            use_ml: bool        = True) -> Dict[str,Any]:
    """
    Run AI-generation detection (signal-processing + optional CNN).

    Parameters
    ----------
    use_cnn : bool
        Toggle CNN scoring. Set False to use signal-processing only
        (faster, no TF dependency).

    Returns
    -------
    {
        sha256, frequency_score, noise_score, texture_score,
        cnn_score,           ← NEW (0.5 if CNN disabled/unavailable)
        cnn_available,       ← NEW bool
        ai_generated_score,  ← weighted blend of all four
        spectrum_path, noise_path
    }
    """
    image_path = str(Path(image_path).resolve())
    sha256     = compute_sha256(image_path)

    original   = Image.open(image_path).convert("RGB")
    orig_array = np.array(original)
    gray       = cv2.cvtColor(orig_array, cv2.COLOR_RGB2GRAY)

    logger.info("ai_generated_detector v2 | CNN=%s | %s",
                TF_OK and use_cnn, image_path)

    # ── Signal-processing scores (always run) ─────────────────────────────────
    freq_score    = detect_frequency_artifacts(gray)
    noise_score   = detect_noise_pattern(gray)
    texture_score = detect_texture_regularity(gray)

    # ── CNN score (optional) ──────────────────────────────────────────────────
    cnn_available = TF_OK and use_cnn
    cnn_s = cnn_score(image_path) if cnn_available else 0.5

    # ── ML Model score: SVM + Random Forest ensemble ──────────────────────────
    ml_available = ML_OK and use_ml

    if ml_available:
        clf = _ml.get_classifier("ai_gen")

        features = extract_features(image_path)

        result = clf.predict_proba(features)

        ml_ensemble = result["ensemble_score"]
        svm_s       = result["svm_score"]
        rf_s        = result["rf_score"]

        ml_label = "FAKE" if ml_ensemble > 0.5 else "REAL"

        logger.info(
            "ML score: SVM=%.4f  RF=%.4f  Ensemble=%.4f  → %s",
            svm_s, rf_s, ml_ensemble, ml_label
        )
    else:
        ml_ensemble = 0.5
        svm_s       = 0.5
        rf_s        = 0.5
        ml_label    = "N/A"

    # ── Weighted combination ──────────────────────────────────────────────────
    # Priority: ML Models > CNN > Signal Processing

    if ml_available:
        ai_score = round(
            0.40 * ml_ensemble +
            0.25 * freq_score +
            0.20 * noise_score +
            0.15 * texture_score,
            4
        )

    elif cnn_available:
        ai_score = round(
            0.40 * cnn_s +
            0.30 * freq_score +
            0.20 * noise_score +
            0.10 * texture_score,
            4
        )

    else:
        ai_score = round(
            0.45 * freq_score +
            0.30 * noise_score +
            0.25 * texture_score,
            4
        )

    logger.info("AI-generated score v3: %.4f", ai_score)

    # ── Save evidence ─────────────────────────────────────────────────────────
    spectrum_path = noise_path = None

    if save_evidence:
        os.makedirs(evidence_dir, exist_ok=True)

        stem = Path(image_path).stem
        h12  = sha256[:12]

        sp = os.path.join(
            evidence_dir,
            f"{stem}_{h12}_fft_spectrum.jpg"
        )

        np_ = os.path.join(
            evidence_dir,
            f"{stem}_{h12}_noise_residual.jpg"
        )

        cv2.imwrite(sp, _vis_spectrum(gray))
        cv2.imwrite(np_, _vis_noise(gray))

        spectrum_path = sp
        noise_path    = np_

    return {
        "sha256": sha256,

        "frequency_score": freq_score,
        "noise_score": noise_score,
        "texture_score": texture_score,

        "cnn_score": cnn_s,
        "cnn_available": cnn_available,

        "ml_svm_score": svm_s,
        "ml_rf_score": rf_s,
        "ml_ensemble_score": ml_ensemble,
        "ml_available": ml_available,
        "ml_label": ml_label,

        "ai_generated_score": ai_score,

        "spectrum_path": spectrum_path,
        "noise_path": noise_path,
    }
