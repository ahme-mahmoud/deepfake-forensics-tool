"""
feature_extractor.py
====================
Unified Forensic Feature Extraction Engine

Extracts a rich multi-domain feature vector from any input image.
This single vector feeds ALL downstream ML classifiers.

Feature domains:
  HOG    — shape / gradient structure (manipulated regions break continuity)
  LBP    — local texture micro-patterns (GAN outputs differ from cameras)
  FFT    — frequency spectrum profile (GAN checkerboard + over-smoothing)
  ELA    — JPEG compression residuals (splice regions have different history)
  COLOR  — per-channel moments + YCbCr histograms (lighting inconsistency)
  DCT    — JPEG block coefficient variance (blocking artifact anomalies)
  NOISE  — sensor-noise residual statistics (GAN / deepfake fingerprints)

Final vector dimension: ~1180 float32 values
All extractors operate on a 128×128 resized copy for speed & consistency.
"""

import io
import logging
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops
from skimage.feature import hog, local_binary_pattern

logger = logging.getLogger("feature_extractor")

# ── Global config ──────────────────────────────────────────────────────────────
TARGET_SIZE         = (128, 128)
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS    = 9
LBP_RADIUS          = 3
LBP_N_POINTS        = 24
LBP_N_BINS          = 64
FFT_BINS            = 32
COLOR_BINS          = 32
ELA_QUALITY         = 92
DCT_BLOCK           = 8


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HOG Features
# ═══════════════════════════════════════════════════════════════════════════════

def _hog_features(gray: np.ndarray) -> np.ndarray:
    """
    Histogram of Oriented Gradients — captures edge/gradient structure.
    Spliced regions break gradient continuity at boundaries.
    """
    feat = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        feature_vector=True,
        transform_sqrt=True,
    )
    return feat.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LBP Texture Features
# ═══════════════════════════════════════════════════════════════════════════════

def _lbp_features(gray: np.ndarray) -> np.ndarray:
    """
    Local Binary Patterns — captures micro-texture.
    GAN images have characteristically different LBP distributions
    compared to real camera images.
    """
    lbp  = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_N_BINS,
                           range=(0, LBP_N_POINTS + 2))
    hist = hist.astype(np.float32)
    tot  = hist.sum()
    return hist / tot if tot > 0 else hist


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FFT Frequency Features
# ═══════════════════════════════════════════════════════════════════════════════

def _fft_features(gray: np.ndarray) -> np.ndarray:
    """
    Radial frequency energy profile from 2D FFT.
    GAN / diffusion images:
      - have periodic checkerboard peaks (transposed-conv artifacts)
      - over-smooth high frequencies (flatter decay slope)
    Returns: FFT_BINS radial bins + 2 summary stats (peak_count, slope)
    """
    fft    = np.fft.fft2(gray.astype(np.float32))
    mag    = np.log1p(np.abs(np.fft.fftshift(fft)))

    h, w   = mag.shape
    cy, cx = h // 2, w // 2
    y_idx  = np.arange(h).reshape(-1, 1)
    x_idx  = np.arange(w).reshape(1, -1)
    r_map  = np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2)
    max_r  = min(cy, cx)

    # Radial energy profile
    edges   = np.linspace(0, max_r, FFT_BINS + 1)
    profile = np.zeros(FFT_BINS, dtype=np.float32)
    for i in range(FFT_BINS):
        mask = (r_map >= edges[i]) & (r_map < edges[i + 1])
        profile[i] = mag[mask].mean() if mask.any() else 0.0

    mx = profile.max()
    if mx > 0:
        profile = profile / mx

    # Summary: slope of linear fit (flatter = more AI-like)
    slope = float(np.polyfit(np.arange(FFT_BINS), profile, 1)[0])

    # Isolated peak count (GAN checkerboard peaks)
    dc_mask  = r_map < max_r // 10
    hf_vals  = mag[~dc_mask]
    mu, sd   = hf_vals.mean(), hf_vals.std()
    pk_count = float((hf_vals > mu + 4 * sd).sum()) / max(hf_vals.size, 1)

    return np.append(profile, [slope, pk_count]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ELA Compression Features
# ═══════════════════════════════════════════════════════════════════════════════

def _ela_features(pil_img: Image.Image) -> np.ndarray:
    """
    Error Level Analysis residual statistics.
    Re-save at known quality; measure pixel-wise differences.
    Spliced regions have a different compression history → higher residuals.
    Returns: [mean, std, max, p90, high_pixel_ratio] (5 values)
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    recomp = Image.open(buf).convert("RGB")

    diff   = np.array(ImageChops.difference(pil_img, recomp)).astype(np.float32)
    gray_d = diff.mean(axis=2)

    mx   = gray_d.max()
    high = (gray_d > mx * 0.10).mean() if mx > 0 else 0.0

    return np.array([
        gray_d.mean(),
        gray_d.std(),
        mx,
        float(np.percentile(gray_d, 90)),
        high,
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Colour Statistical Features
# ═══════════════════════════════════════════════════════════════════════════════

def _color_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Per-channel statistical moments (mean, std, skewness, kurtosis) for RGB
    plus normalised histograms over YCbCr channels.
    Lighting inconsistency in spliced images and colour over-smoothing in
    GAN outputs are captured here.
    """
    feats = []

    # RGB moments
    for c in range(3):
        ch  = img_rgb[:, :, c].astype(np.float64)
        mu  = ch.mean()
        std = ch.std() + 1e-8
        m3  = ((ch - mu)**3).mean()
        m4  = ((ch - mu)**4).mean()
        feats += [mu, std, m3 / std**3, m4 / std**4 - 3.0]

    # YCbCr histograms
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    for c in range(3):
        h, _ = np.histogram(ycbcr[:, :, c], bins=COLOR_BINS, range=(0, 256))
        h    = h.astype(np.float32)
        tot  = h.sum()
        feats.extend((h / tot if tot > 0 else h).tolist())

    return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DCT Block Features
# ═══════════════════════════════════════════════════════════════════════════════

def _dct_features(gray: np.ndarray) -> np.ndarray:
    """
    JPEG block DCT coefficient statistics.
    Real images have characteristic blocking patterns.
    AI-generated images and spliced regions often show anomalies
    in the AC coefficient distribution.
    Returns: [mean_ac_energy, std_ac_energy, dc_var, ac_kurtosis] (4 values)
    """
    h, w         = gray.shape
    gray_f       = gray.astype(np.float32)
    ac_energies  = []
    dc_values    = []

    for y in range(0, h - DCT_BLOCK, DCT_BLOCK):
        for x in range(0, w - DCT_BLOCK, DCT_BLOCK):
            block = gray_f[y:y+DCT_BLOCK, x:x+DCT_BLOCK]
            dct   = cv2.dct(block)
            dc_values.append(float(dct[0, 0]))
            ac = dct.flatten()[1:]
            ac_energies.append(float((ac**2).mean()))

    if not ac_energies:
        return np.zeros(4, dtype=np.float32)

    ac_arr   = np.array(ac_energies, dtype=np.float32)
    dc_arr   = np.array(dc_values,   dtype=np.float32)
    std      = ac_arr.std() + 1e-8
    m4       = ((ac_arr - ac_arr.mean())**4).mean()
    kurtosis = m4 / std**4 - 3.0

    return np.array([
        ac_arr.mean(),
        ac_arr.std(),
        dc_arr.var(),
        kurtosis,
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Noise Residual Features
# ═══════════════════════════════════════════════════════════════════════════════

def _noise_features(gray: np.ndarray) -> np.ndarray:
    """
    Camera sensor noise residual statistics.
    Obtained by subtracting a Gaussian-smoothed version of the image.
    Real cameras have PRNU noise; GAN images have different noise profiles.
    Returns: [std, autocorr, entropy_norm, smoothness] (4 values)
    """
    smooth   = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    residual = gray.astype(np.float32) - smooth

    std = float(residual.std())

    flat = residual.flatten()
    autocorr = float(np.corrcoef(flat[:-1], flat[1:])[0, 1]) if len(flat) > 1 else 0.0

    hist, _   = np.histogram(residual, bins=64, range=(-30, 30))
    p         = hist.astype(np.float64) + 1e-9
    p        /= p.sum()
    entropy   = float(-np.sum(p * np.log(p)))
    max_ent   = float(np.log(64))
    ent_norm  = entropy / max_ent if max_ent > 0 else 0.0

    # Laplacian-based smoothness (low = over-smooth = AI-like)
    lap        = cv2.Laplacian(gray, cv2.CV_64F)
    smoothness = float(lap.var())

    return np.array([std, autocorr, ent_norm, smoothness], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def extract(image_path: str) -> Dict[str, np.ndarray]:
    """
    Extract ALL forensic features from an image.

    Returns a dict of named feature arrays AND a combined feature vector:
    {
        "hog":      np.ndarray,   # HOG gradient features
        "lbp":      np.ndarray,   # LBP texture histogram
        "fft":      np.ndarray,   # FFT radial profile + stats
        "ela":      np.ndarray,   # ELA compression statistics
        "color":    np.ndarray,   # Colour moments + histograms
        "dct":      np.ndarray,   # DCT block coefficient stats
        "noise":    np.ndarray,   # Noise residual statistics
        "combined": np.ndarray,   # All features concatenated (the ML input)
    }
    """
    pil_img  = Image.open(image_path).convert("RGB")
    pil_img  = pil_img.resize(TARGET_SIZE, Image.LANCZOS)
    img_rgb  = np.array(pil_img)
    gray     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    hog_f   = _hog_features(gray)
    lbp_f   = _lbp_features(gray)
    fft_f   = _fft_features(gray)
    ela_f   = _ela_features(pil_img)
    color_f = _color_features(img_rgb)
    dct_f   = _dct_features(gray)
    noise_f = _noise_features(gray)

    combined = np.concatenate([hog_f, lbp_f, fft_f, ela_f, color_f, dct_f, noise_f])

    logger.debug("Feature dims: HOG=%d LBP=%d FFT=%d ELA=%d COLOR=%d DCT=%d NOISE=%d TOTAL=%d",
                 len(hog_f), len(lbp_f), len(fft_f), len(ela_f),
                 len(color_f), len(dct_f), len(noise_f), len(combined))

    return {
        "hog":      hog_f,
        "lbp":      lbp_f,
        "fft":      fft_f,
        "ela":      ela_f,
        "color":    color_f,
        "dct":      dct_f,
        "noise":    noise_f,
        "combined": combined,
    }


def feature_names() -> list:
    """Return a list of feature group names with their sizes."""
    dummy = np.zeros((128, 128), dtype=np.uint8)
    dummy_pil = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
    return [
        f"hog_{i}"   for i in range(len(_hog_features(dummy)))
    ] + [
        f"lbp_{i}"   for i in range(LBP_N_BINS)
    ] + [
        f"fft_{i}"   for i in range(FFT_BINS + 2)
    ] + [
        "ela_mean", "ela_std", "ela_max", "ela_p90", "ela_high_ratio"
    ] + [
        f"color_{i}" for i in range(12 + 3 * COLOR_BINS)
    ] + [
        "dct_mean_ac", "dct_std_ac", "dct_dc_var", "dct_ac_kurtosis"
    ] + [
        "noise_std", "noise_autocorr", "noise_entropy", "noise_smoothness"
    ]
