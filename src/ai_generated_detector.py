"""
ai_generated_detector.py
========================
Module 3 — AI-Generated / GAN Image Detection

FORENSIC LOGIC:
    GAN-generated (and diffusion-generated) images leave characteristic
    fingerprints that are *absent* in real photographs:

    1. FREQUENCY DOMAIN ARTIFACTS
       GANs over-smooth certain frequency bands and introduce periodic grid
       patterns from transposed-convolution "checkerboard" artifacts.
       We analyse the 2-D FFT magnitude spectrum for:
           (a) Radial energy distribution (real photos decay faster at high-freq)
           (b) Grid/periodic peaks in the spectrum (GAN checkerboard)

    2. NOISE PATTERN ANALYSIS
       Real cameras introduce sensor noise with specific autocorrelation
       properties (PRNU — Photo Response Non-Uniformity).  GAN outputs have
       systematically different, smoother noise profiles.

    3. CO-OCCURRENCE MATRIX (texture regularity)
       AI-generated images are statistically too "perfect" — their pixel
       co-occurrence matrices show lower entropy than natural photographs
       of comparable content.

Each sub-score is normalised to [0, 1].
Module score = weighted combination.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.stats import entropy as scipy_entropy

from compression_analysis import compute_sha256

logger = logging.getLogger("ai_generated_detector")


# ===========================================================================
# Sub-detector 1 — Frequency Domain Analysis
# ===========================================================================

def _compute_fft_spectrum(gray: np.ndarray) -> np.ndarray:
    """Return the log-magnitude FFT spectrum centred at DC."""
    fft      = np.fft.fft2(gray.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    log_mag   = np.log1p(magnitude)   # log(1+x) avoids log(0)
    return log_mag


def _radial_energy_profile(spectrum: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """
    Compute the average spectral energy as a function of radial frequency.
    Returns a 1-D array of length n_bins (DC=bin 0, Nyquist=bin n_bins-1).
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)

    y_idx, x_idx = np.ogrid[:h, :w]
    r_map = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    bin_edges = np.linspace(0, max_r, n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        mask = (r_map >= bin_edges[i]) & (r_map < bin_edges[i + 1])
        vals = spectrum[mask]
        profile[i] = vals.mean() if vals.size > 0 else 0.0
    return profile


def _detect_grid_peaks(spectrum: np.ndarray, threshold_sigma: float = 4.0) -> int:
    """
    Count anomalous peaks in the FFT spectrum that suggest periodic GAN artifacts.
    We blank out the central DC region (low-frequency content) and look for
    isolated spikes in the high-frequency area.
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    # Blank DC region (central 10 %)
    mask = np.ones_like(spectrum, dtype=bool)
    r_dc = min(cy, cx) // 10
    y_idx, x_idx = np.ogrid[:h, :w]
    dc_zone = (y_idx - cy) ** 2 + (x_idx - cx) ** 2 < r_dc ** 2
    mask[dc_zone] = False

    high_freq = spectrum[mask]
    mu, sigma = high_freq.mean(), high_freq.std()
    if sigma < 1e-6:
        return 0

    peaks = int((high_freq > mu + threshold_sigma * sigma).sum())
    return peaks


def detect_frequency_artifacts(gray: np.ndarray) -> float:
    """
    Detect GAN frequency artifacts.

    Score intuition
    ---------------
    • Real photos: radial energy decays smoothly; few isolated peaks.
    • GAN images:  checkerboard → periodic peaks; over-smoothing changes
                   the decay slope at mid-frequencies.

    Returns a score in [0, 1].
    """
    spectrum = _compute_fft_spectrum(gray)
    profile  = _radial_energy_profile(spectrum)
    peaks    = _detect_grid_peaks(spectrum)

    # ── Peak anomaly score ──────────────────────────────────────────────────
    # Empirically GAN images have 2–30× more isolated peaks than real photos
    peak_score = float(np.clip(peaks / 500.0, 0.0, 1.0))

    # ── Radial decay score ──────────────────────────────────────────────────
    # Fit a linear regression to the log-profile; GAN images are flatter
    if profile.max() > 0:
        norm_profile = profile / profile.max()
    else:
        norm_profile = profile

    x = np.arange(len(norm_profile), dtype=np.float64)
    # slope of linear fit: more negative = faster decay (more natural)
    slope = float(np.polyfit(x, norm_profile, 1)[0])
    # GAN slope is closer to 0 (flat).  Real: slope ≈ -0.025 to -0.04
    decay_score = float(np.clip(1.0 + slope / 0.02, 0.0, 1.0))

    score = 0.50 * peak_score + 0.50 * decay_score
    logger.info("Frequency artifact score: %.4f  (peaks=%d, slope=%.4f)",
                score, peaks, slope)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


# ===========================================================================
# Sub-detector 2 — Noise Residual Analysis
# ===========================================================================

def _extract_noise_residual(gray: np.ndarray) -> np.ndarray:
    """
    Subtract a Gaussian-smoothed version of the image to isolate sensor noise.
    The residual captures high-frequency noise patterns.
    """
    smoothed = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    residual = gray.astype(np.float32) - smoothed
    return residual


def detect_noise_pattern(gray: np.ndarray) -> float:
    """
    Analyse noise residual autocorrelation.

    GAN/diffusion outputs have different noise profiles than real cameras.
    We measure:
        (a) Noise standard deviation  — GAN tends to be lower (over-smoothed)
        (b) Autocorrelation at lag-1  — real PRNU shows specific correlation
        (c) Entropy of the residual   — AI images are often less entropic

    Returns a score in [0, 1].
    """
    residual = _extract_noise_residual(gray)

    # Standard deviation: very low STD → over-smoothed → GAN-like
    noise_std = float(residual.std())
    # Typical real-camera noise STD ~ 3-8; GAN images often < 2
    std_score = float(np.clip(1.0 - noise_std / 6.0, 0.0, 1.0))

    # Autocorrelation at lag 1 (horizontal)
    flat = residual.flatten()
    if len(flat) > 1:
        autocorr = float(np.corrcoef(flat[:-1], flat[1:])[0, 1])
    else:
        autocorr = 0.0
    # High positive autocorrelation → structured noise → AI-generated
    autocorr_score = float(np.clip((autocorr + 1.0) / 2.0, 0.0, 1.0))

    # Entropy of residual histogram
    hist, _ = np.histogram(residual, bins=64, range=(-30, 30))
    ent = float(scipy_entropy(hist + 1e-9))   # avoid log(0)
    max_ent = float(np.log(64))
    # Low entropy → more regular noise → AI
    ent_score = float(np.clip(1.0 - ent / max_ent, 0.0, 1.0))

    score = 0.30 * std_score + 0.40 * autocorr_score + 0.30 * ent_score
    logger.info("Noise pattern score: %.4f  (std=%.2f, autocorr=%.4f, entropy=%.4f)",
                score, noise_std, autocorr, ent)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


# ===========================================================================
# Sub-detector 3 — Co-occurrence Matrix Regularity
# ===========================================================================

def _glcm_entropy(gray: np.ndarray, levels: int = 64, step: int = 1) -> float:
    """
    Compute a simplified Gray-Level Co-occurrence Matrix (GLCM) entropy.
    We quantise to *levels* gray levels and compute horizontal co-occurrences.
    """
    # Quantise
    q = (gray.astype(np.float32) / 255.0 * (levels - 1)).astype(np.int32)
    q = np.clip(q, 0, levels - 1)

    # Horizontal co-occurrence (offset = (0, step))
    left  = q[:, :-step].flatten()
    right = q[:, step:].flatten()

    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (left, right), 1)
    glcm /= (glcm.sum() + 1e-9)

    ent = float(scipy_entropy(glcm.flatten() + 1e-9))
    return ent


def detect_texture_regularity(gray: np.ndarray) -> float:
    """
    AI-generated images tend to be "too regular" — their GLCM entropy is
    lower than that of natural photographs with comparable content.

    Heuristic thresholds are learned empirically.

    Returns a score in [0, 1]  (higher → more likely AI-generated).
    """
    ent = _glcm_entropy(gray)
    # Empirical observation: natural photos have GLCM entropy ~ 3.5–5.5
    # AI-generated images tend to cluster around 2.0–3.5
    # We linearly map [5.5, 2.0] → [0, 1]
    score = float(np.clip((5.5 - ent) / 3.5, 0.0, 1.0))
    logger.info("Texture regularity score: %.4f  (GLCM entropy=%.4f)", score, ent)
    return round(score, 4)


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def visualise_spectrum(gray: np.ndarray) -> np.ndarray:
    """Return a uint8 BGR image of the log-magnitude FFT spectrum."""
    spectrum = _compute_fft_spectrum(gray)
    norm = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
    norm_uint8 = norm.astype(np.uint8)
    coloured = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_MAGMA)
    return coloured


def visualise_noise_residual(gray: np.ndarray) -> np.ndarray:
    """Return a uint8 BGR amplified noise residual image."""
    residual = _extract_noise_residual(gray)
    amplified = np.clip(np.abs(residual) * 10, 0, 255).astype(np.uint8)
    coloured = cv2.applyColorMap(amplified, cv2.COLORMAP_HOT)
    return coloured


# ===========================================================================
# Public entry point
# ===========================================================================

def analyze(
    image_path: str,
    evidence_dir: str = "reports/evidence",
    save_evidence: bool = True,
) -> Dict[str, Any]:
    """
    Run all three AI-generation sub-detectors.

    Returns
    -------
    {
        sha256,
        frequency_score, noise_score, texture_score,
        ai_generated_score,   ← weighted combination
        evidence paths …
    }
    """
    image_path = str(Path(image_path).resolve())
    sha256     = compute_sha256(image_path)

    original   = Image.open(image_path).convert("RGB")
    orig_array = np.array(original)
    gray       = cv2.cvtColor(orig_array, cv2.COLOR_RGB2GRAY)

    logger.info("Running AI-generation detection on: %s", image_path)

    freq_score    = detect_frequency_artifacts(gray)
    noise_score   = detect_noise_pattern(gray)
    texture_score = detect_texture_regularity(gray)

    ai_score = round(
        0.45 * freq_score + 0.30 * noise_score + 0.25 * texture_score, 4
    )
    logger.info("AI-generated score: %.4f", ai_score)

    spectrum_path = None
    noise_path    = None

    if save_evidence:
        os.makedirs(evidence_dir, exist_ok=True)
        stem       = Path(image_path).stem
        short_hash = sha256[:12]

        spec_img  = visualise_spectrum(gray)
        spec_out  = os.path.join(evidence_dir, f"{stem}_{short_hash}_fft_spectrum.jpg")
        cv2.imwrite(spec_out, spec_img)
        spectrum_path = spec_out

        noise_img = visualise_noise_residual(gray)
        noise_out = os.path.join(evidence_dir, f"{stem}_{short_hash}_noise_residual.jpg")
        cv2.imwrite(noise_out, noise_img)
        noise_path = noise_out

        logger.info("Spectrum  → %s", spec_out)
        logger.info("Noise map → %s", noise_out)

    return {
        "sha256"          : sha256,
        "frequency_score" : freq_score,
        "noise_score"     : noise_score,
        "texture_score"   : texture_score,
        "ai_generated_score": ai_score,
        "spectrum_path"   : spectrum_path,
        "noise_path"      : noise_path,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python ai_generated_detector.py <image_path>")
        sys.exit(1)
    print(json.dumps(analyze(sys.argv[1]), indent=2, default=str))
