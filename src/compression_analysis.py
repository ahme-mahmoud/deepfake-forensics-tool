"""
compression_analysis.py
=======================
Module 1 — Error Level Analysis (ELA) & JPEG Compression Forensics

PURPOSE:
    JPEG images lose quality every time they are re-saved.  If a region of
    an image was pasted from an *external* source and then the composite was
    saved as JPEG, the pasted region will have a *different* compression
    history than the background.  ELA exposes these differences as bright
    areas in an amplified difference image.

FORENSIC LOGIC:
    1.  Re-compress the input image at a known quality level (e.g. Q=95).
    2.  Compute the absolute pixel-wise difference between the original and
        the re-compressed version.
    3.  Amplify the difference (×scale) so subtle inconsistencies become
        visible.
    4.  Derive a scalar "manipulation score" from the ELA image statistics.

CHAIN OF CUSTODY:
    All inputs are hashed (SHA-256) before processing.  The ELA output image
    is saved to reports/evidence/ with the source hash embedded in the
    filename so the artefact can always be traced back to the original file.
"""

import io
import os
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("compression_analysis")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ELA_QUALITY: int = 95          # Re-save quality for ELA
DEFAULT_ELA_SCALE: int = 15           # Amplification factor
DEFAULT_EVIDENCE_DIR: str = "reports/evidence"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def compute_sha256(image_path: str) -> str:
    """Return the SHA-256 hex digest of a file — chain-of-custody anchor."""
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    """Re-encode a PIL image to JPEG in memory and return raw bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# Core ELA implementation
# ---------------------------------------------------------------------------

def run_ela(
    image_path: str,
    quality: int = DEFAULT_ELA_QUALITY,
    scale: int = DEFAULT_ELA_SCALE,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform Error Level Analysis on *image_path*.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the input image (JPEG, PNG, BMP …).
    quality : int
        JPEG re-save quality (1–95).  Lower = larger differences for authentic
        images; the sweet-spot for forensics is 90–95.
    scale : int
        Amplification multiplier applied to the difference image.

    Returns
    -------
    original_array  : np.ndarray  — Original image as uint8 RGB array.
    ela_array       : np.ndarray  — Amplified ELA image as uint8 RGB array.
    ela_score       : float       — Normalised manipulation score in [0, 1].
    """
    logger.info("Running ELA on: %s  (quality=%d, scale=%d)", image_path, quality, scale)

    # ── 1. Load ─────────────────────────────────────────────────────────────
    original = Image.open(image_path).convert("RGB")
    original_array = np.array(original)

    # ── 2. Re-compress to JPEG at target quality ────────────────────────────
    jpeg_bytes = _to_jpeg_bytes(original, quality)
    recompressed = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

    # ── 3. Pixel-wise absolute difference ───────────────────────────────────
    diff = ImageChops.difference(original, recompressed)
    diff_array = np.array(diff, dtype=np.float32)

    # ── 4. Amplify ──────────────────────────────────────────────────────────
    amplified = np.clip(diff_array * scale, 0, 255).astype(np.uint8)
    ela_image = Image.fromarray(amplified)

    # ── 5. Enhance contrast so subtle regions are easier to read ────────────
    ela_image = ImageEnhance.Contrast(ela_image).enhance(1.5)
    ela_array = np.array(ela_image)

    # ── 6. Compute forensic score ────────────────────────────────────────────
    ela_score = _compute_ela_score(diff_array, original_array)
    logger.info("ELA score: %.4f", ela_score)

    return original_array, ela_array, ela_score


def _compute_ela_score(diff_array: np.ndarray, original_array: np.ndarray) -> float:
    """
    Derive a normalised [0, 1] manipulation probability from ELA statistics.

    Intuition
    ---------
    • A pristine, unmodified JPEG that has been saved once will have small,
      *uniform* ELA residuals — most of the image compresses similarly.
    • A composited image will have *high-variance* ELA residuals: pasted
      regions stand out strongly against the uniform background.

    We therefore combine:
        (a) Mean ELA intensity        — overall energy level
        (b) Std-dev of ELA intensity  — spatial variance (key indicator)
        (c) High-error pixel ratio    — fraction of pixels above threshold
    """
    gray_diff = diff_array.mean(axis=2)          # collapse channels → 2-D

    mean_ela  = float(gray_diff.mean())
    std_ela   = float(gray_diff.std())
    max_ela   = float(gray_diff.max()) if gray_diff.max() > 0 else 1.0

    # Fraction of pixels above 10 % of the per-image maximum
    threshold = 0.10 * max_ela
    high_ratio = float((gray_diff > threshold).mean())

    # Normalise each component to [0, 1]
    norm_mean  = min(mean_ela / 25.0, 1.0)   # empirically: >25 is suspicious
    norm_std   = min(std_ela  / 20.0, 1.0)   # high std = non-uniform = tampered
    norm_ratio = min(high_ratio / 0.30, 1.0) # >30 % high-error pixels → suspect

    # Weighted combination (std carries most discriminative power)
    score = 0.25 * norm_mean + 0.50 * norm_std + 0.25 * norm_ratio
    return round(float(np.clip(score, 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# Region analysis — locate the most suspicious zones
# ---------------------------------------------------------------------------

def detect_suspicious_regions(
    ela_array: np.ndarray,
    top_n: int = 5,
    min_area: int = 500,
) -> list:
    """
    Find the top-N contiguous bright regions in the ELA image.

    Returns a list of dicts with keys:
        bbox   — (x, y, w, h) bounding box
        area   — pixel count
        mean   — mean ELA intensity inside the region
    """
    gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)

    # Otsu threshold to isolate high-error regions
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to merge nearby pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi_mean = float(gray[y:y+h, x:x+w].mean())
        regions.append({"bbox": (x, y, w, h), "area": int(area), "mean": roi_mean})

    # Sort by mean intensity (brightest = most suspicious)
    regions.sort(key=lambda r: r["mean"], reverse=True)
    return regions[:top_n]


def annotate_image(
    original_array: np.ndarray,
    ela_array: np.ndarray,
    regions: list,
) -> np.ndarray:
    """
    Return a side-by-side forensic panel:
        Left  — original image with bounding boxes on suspicious regions
        Right — ELA heat-map

    The panel is returned as a uint8 BGR array (OpenCV convention) so it
    can be saved directly with cv2.imwrite().
    """
    orig_bgr = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)
    ela_bgr  = cv2.cvtColor(ela_array,      cv2.COLOR_RGB2BGR)

    # Draw bounding boxes on original
    annotated = orig_bgr.copy()
    for i, region in enumerate(regions, 1):
        x, y, w, h = region["bbox"]
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)
        label = f"R{i} ({region['mean']:.1f})"
        cv2.putText(annotated, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Add header banners
    def _add_banner(img: np.ndarray, text: str) -> np.ndarray:
        banner = np.zeros((32, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, text, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
        return np.vstack([banner, img])

    annotated  = _add_banner(annotated, "ORIGINAL  (suspicious regions boxed)")
    ela_banner = _add_banner(ela_bgr,   "ELA HEAT-MAP  (bright = manipulated)")

    # Resize to common height if dimensions differ (shouldn't, but defensive)
    h_target = max(annotated.shape[0], ela_banner.shape[0])
    if annotated.shape[0]  != h_target:
        annotated  = cv2.resize(annotated,  (annotated.shape[1],  h_target))
    if ela_banner.shape[0] != h_target:
        ela_banner = cv2.resize(ela_banner, (ela_banner.shape[1], h_target))

    panel = np.hstack([annotated, ela_banner])
    return panel


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze(
    image_path: str,
    quality: int = DEFAULT_ELA_QUALITY,
    scale: int = DEFAULT_ELA_SCALE,
    evidence_dir: str = DEFAULT_EVIDENCE_DIR,
    save_evidence: bool = True,
) -> Dict[str, Any]:
    """
    Full ELA forensic analysis pipeline.

    Parameters
    ----------
    image_path   : path to the image to investigate.
    quality      : JPEG re-save quality for ELA.
    scale        : ELA amplification factor.
    evidence_dir : directory where ELA output images are saved.
    save_evidence: if False, skip disk writes (useful in unit tests).

    Returns
    -------
    A dict with:
        sha256            — file hash (chain of custody)
        ela_score         — normalised manipulation probability [0, 1]
        suspicious_regions— list of region dicts
        ela_image_path    — path of the saved ELA evidence image (or None)
        panel_image_path  — path of the saved annotated panel (or None)
        metadata          — dict of image metadata
    """
    image_path = str(Path(image_path).resolve())
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ── Chain of custody ─────────────────────────────────────────────────────
    sha256 = compute_sha256(image_path)
    logger.info("SHA-256: %s", sha256)

    # ── ELA ──────────────────────────────────────────────────────────────────
    original_array, ela_array, ela_score = run_ela(image_path, quality, scale)

    # ── Region detection ─────────────────────────────────────────────────────
    regions = detect_suspicious_regions(ela_array)
    logger.info("Suspicious regions found: %d", len(regions))

    # ── Image metadata ────────────────────────────────────────────────────────
    pil_img = Image.open(image_path)
    metadata = {
        "format"    : pil_img.format or "UNKNOWN",
        "mode"      : pil_img.mode,
        "size_px"   : pil_img.size,           # (width, height)
        "file_bytes": os.path.getsize(image_path),
    }

    ela_image_path  = None
    panel_image_path = None

    if save_evidence:
        os.makedirs(evidence_dir, exist_ok=True)
        stem = Path(image_path).stem
        short_hash = sha256[:12]

        # Save ELA image
        ela_out = os.path.join(evidence_dir, f"{stem}_{short_hash}_ela.jpg")
        ela_pil = Image.fromarray(ela_array)
        ela_pil.save(ela_out, quality=95)
        ela_image_path = ela_out
        logger.info("ELA evidence saved → %s", ela_out)

        # Save annotated panel
        panel = annotate_image(original_array, ela_array, regions)
        panel_out = os.path.join(evidence_dir, f"{stem}_{short_hash}_panel.jpg")
        cv2.imwrite(panel_out, panel)
        panel_image_path = panel_out
        logger.info("Panel evidence saved → %s", panel_out)

    return {
        "sha256"             : sha256,
        "ela_score"          : ela_score,
        "suspicious_regions" : regions,
        "ela_image_path"     : ela_image_path,
        "panel_image_path"   : panel_image_path,
        "metadata"           : metadata,
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python compression_analysis.py <image_path>")
        sys.exit(1)

    result = analyze(sys.argv[1])
    print(json.dumps(result, indent=2, default=str))

