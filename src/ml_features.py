"""
ml_features.py
==============
Shared Feature Extraction Pipeline for all ML Models.

WHY THIS FILE EXISTS:
    Both the deepfake classifier and AI-generated classifier need to turn
    raw images into numerical feature vectors before an ML model can learn
    from them or make predictions.  This module centralises that logic so
    both classifiers use IDENTICAL features — making the system consistent
    and easy to maintain.

FEATURE VECTOR STRUCTURE  (total ≈ 556 dimensions):
    ┌──────────────────────────────────────────────┬────────┐
    │ Feature Group                                │  Dims  │
    ├──────────────────────────────────────────────┼────────┤
    │ HOG  (texture + edges)                       │  ~324  │
    │ FFT radial profile  (frequency fingerprint)  │   32   │
    │ ELA statistics  (compression history)        │    5   │
    │ Colour histogram YCbCr  (colour profile)     │   96   │
    │ Statistical moments  (global image stats)    │   12   │
    │ Local Binary Pattern histogram  (micro-tex.) │   59   │
    └──────────────────────────────────────────────┴────────┘

All features are L2-normalised before being returned so that the
downstream SVM / RandomForest / Voting classifiers receive well-
conditioned input regardless of image resolution or content.
"""

import io
import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from skimage.feature import hog, local_binary_pattern

logger = logging.getLogger("ml_features")

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SIZE    = (128, 128)    # All images resized here before feature extraction
HOG_PIXELS     = (16, 16)     # Pixels per cell  →  8×8 cells for 128-px image
HOG_CELLS      = (2, 2)       # Cells per block
ELA_QUALITY    = 92            # JPEG re-save quality for ELA feature
ELA_SCALE      = 15            # Amplification
LBP_RADIUS     = 3
LBP_POINTS     = 24           # 8 * radius


# ═══════════════════════════════════════════════════════════════════════════════
# Individual feature extractors
# ═══════════════════════════════════════════════════════════════════════════════

def _feat_hog(gray: np.ndarray) -> np.ndarray:
    """
    Histogram of Oriented Gradients.
    Captures texture, edge patterns, and shape — very discriminative
    between real photos and AI-generated images (different structure).
    """
    feat = hog(
        gray,
        pixels_per_cell=HOG_PIXELS,
        cells_per_block=HOG_CELLS,
        feature_vector=True,
        channel_axis=None,
    )
    return feat.astype(np.float32)


def _feat_fft(gray: np.ndarray) -> np.ndarray:
    """
    Radial FFT energy profile (32 bins).
    GAN images have characteristic frequency fingerprints —
    the model learns to distinguish these from real-photo spectra.
    """
    fft     = np.fft.fft2(gray.astype(np.float32))
    shifted = np.fft.fftshift(fft)
    mag     = np.log1p(np.abs(shifted))

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y_idx  = np.arange(h).reshape(-1, 1)
    x_idx  = np.arange(w).reshape(1, -1)
    r_map  = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    n_bins = 32
    max_r  = min(cy, cx)
    edges  = np.linspace(0, max_r, n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (r_map >= edges[i]) & (r_map < edges[i + 1])
        profile[i] = float(mag[mask].mean()) if mask.any() else 0.0
    return profile


def _feat_ela(pil_img: Image.Image, gray: np.ndarray) -> np.ndarray:
    """
    5 ELA statistics:  mean, std, max, % high-error pixels, entropy of ELA.
    Captures JPEG compression-history fingerprint — key for splicing/editing.
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(pil_img, recompressed)
    diff_arr = np.array(diff, dtype=np.float32).mean(axis=2)   # to grayscale

    ela_mean = float(diff_arr.mean())
    ela_std  = float(diff_arr.std())
    ela_max  = float(diff_arr.max()) if diff_arr.max() > 0 else 1.0
    high_ratio = float((diff_arr > 0.1 * ela_max).mean())

    # Entropy of ELA histogram
    hist, _ = np.histogram(diff_arr, bins=32, range=(0, ela_max + 1e-6))
    p = hist.astype(np.float64) + 1e-9
    p /= p.sum()
    ela_entropy = float(-np.sum(p * np.log(p)))

    return np.array([ela_mean, ela_std, ela_max, high_ratio, ela_entropy],
                    dtype=np.float32)


def _feat_colour(pil_img: Image.Image) -> np.ndarray:
    """
    YCbCr colour histogram (32 bins × 3 channels = 96 dims).
    Captures skin-tone / colour distribution differences between
    real photos and AI-generated/manipulated images.
    """
    bgr   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(ycbcr[:, :, ch], bins=32, range=(0, 256))
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats)


def _feat_stats(gray: np.ndarray) -> np.ndarray:
    """
    12 global statistical moments:
        mean, std, skewness, kurtosis  (per each of 4 quadrants compressed)
    Captures tonal distribution properties that differ between real and fake.
    """
    h, w = gray.shape
    quadrants = [
        gray[:h//2, :w//2], gray[:h//2, w//2:],
        gray[h//2:, :w//2], gray[h//2:, w//2:],
    ]
    feats = []
    for q in quadrants:
        q_f  = q.astype(np.float64)
        mean = q_f.mean()
        std  = q_f.std() + 1e-9
        # Skewness (3rd moment)
        skew = float(((q_f - mean) ** 3).mean() / (std ** 3))
        feats.append(float(mean))
        feats.append(float(std))
        feats.append(float(np.clip(skew, -5, 5)))
    return np.array(feats, dtype=np.float32)


def _feat_lbp(gray: np.ndarray) -> np.ndarray:
    """
    Local Binary Pattern histogram (59 uniform bins).
    Captures micro-texture patterns — AI models often produce
    unnaturally regular micro-textures that LBP can detect.
    """
    lbp  = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp, bins=LBP_POINTS + 3,
                           range=(0, LBP_POINTS + 3))
    return hist.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def extract(image_path: str) -> np.ndarray:
    """
    Extract the full feature vector for one image.

    Parameters
    ----------
    image_path : str   path to any image file (JPEG, PNG, BMP …)

    Returns
    -------
    feature_vector : np.ndarray  shape (N,)  dtype float32  L2-normalised
    """
    # ── Load + resize ─────────────────────────────────────────────────────────
    pil_img = Image.open(image_path).convert("RGB").resize(TARGET_SIZE)
    gray    = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)

    # ── Compute each feature group ────────────────────────────────────────────
    f_hog    = _feat_hog(gray)
    f_fft    = _feat_fft(gray)
    f_ela    = _feat_ela(pil_img, gray)
    f_colour = _feat_colour(pil_img)
    f_stats  = _feat_stats(gray)
    f_lbp    = _feat_lbp(gray)

    # ── Concatenate all features ──────────────────────────────────────────────
    vec = np.concatenate([f_hog, f_fft, f_ela, f_colour, f_stats, f_lbp])

    # ── L2-normalise so scale differences don't bias the classifier ───────────
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    logger.debug("Feature vector shape: %s", vec.shape)
    return vec.astype(np.float32)


def extract_from_array(pil_img: Image.Image) -> np.ndarray:
    """
    Same as extract() but accepts a PIL Image directly
    (used during training when the image is already in memory).
    """
    pil_img = pil_img.resize(TARGET_SIZE)
    gray    = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)

    f_hog    = _feat_hog(gray)
    f_fft    = _feat_fft(gray)
    f_ela    = _feat_ela(pil_img, gray)
    f_colour = _feat_colour(pil_img)
    f_stats  = _feat_stats(gray)
    f_lbp    = _feat_lbp(gray)

    vec  = np.concatenate([f_hog, f_fft, f_ela, f_colour, f_stats, f_lbp])
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).astype(np.float32)
