"""
splicing_detector.py
====================
Module 2 — Image Splicing & Tampering Detection

FORENSIC LOGIC:
    A composited image often betrays itself through three tell-tale signals:

    1. EDGE INCONSISTENCY
       Natural photographs have smooth, continuous edge transitions.
       When a region is pasted in from another source the paste boundary
       produces an abrupt, unnaturally sharp edge — detectable via Canny
       edge density analysis across image sub-blocks.

    2. LIGHTING INCONSISTENCY
       Different source images were lit differently.  We use gradient
       orientation histograms (similar to SIFT/HOG) per sub-block to
       detect blocks whose dominant light direction deviates sharply from
       the image majority.

    3. COPY-MOVE DETECTION (block-matching)
       A forger often duplicates a region within the same image to cover
       or clone an area.  We detect this by dividing the image into
       overlapping blocks, computing per-block DCT features, and finding
       near-duplicate block pairs that are spatially separated.

Each sub-detector returns a score in [0, 1].
The module score = weighted combination of the three.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image

from compression_analysis import compute_sha256

logger = logging.getLogger("splicing_detector")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLOCK_SIZE: int = 32          # Size of image blocks (pixels)
BLOCK_STRIDE: int = 16        # Stride for overlapping blocks (copy-move)
DCT_COEFF_COUNT: int = 25     # Number of DCT coefficients per block
COPY_MOVE_THRESHOLD: float = 0.98   # Cosine similarity → likely copy-move


# ===========================================================================
# Sub-detector 1 — Edge Inconsistency
# ===========================================================================

def _edge_density_per_block(gray: np.ndarray, block_size: int) -> np.ndarray:
    """
    Divide image into non-overlapping blocks and compute edge density in each.
    Returns a 2-D array of density values (fraction of edge pixels per block).
    """
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    rows = h // block_size
    cols = w // block_size

    density = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * block_size, (r + 1) * block_size
            x0, x1 = c * block_size, (c + 1) * block_size
            block = edges[y0:y1, x0:x1]
            density[r, c] = block.mean() / 255.0
    return density


def detect_edge_inconsistency(gray: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Score edge inconsistency.

    Intuition: pristine images have smooth variation in edge density across
    the image.  A pasted region often has a dramatically different density
    from its surroundings.  We flag blocks that deviate > 2 std from the
    mean and compute the fraction of such blocks as the score.

    Returns
    -------
    score       : float in [0, 1]
    density_map : 2-D np.ndarray (for visualisation)
    """
    density = _edge_density_per_block(gray, BLOCK_SIZE)
    if density.size == 0:
        return 0.0, density

    mu, sigma = density.mean(), density.std()
    if sigma < 1e-6:
        return 0.0, density

    z_scores = np.abs((density - mu) / sigma)
    anomalous_ratio = float((z_scores > 2.0).mean())

    # The more anomalous blocks, the more likely splicing
    score = float(np.clip(anomalous_ratio / 0.20, 0.0, 1.0))  # >20 % → score=1
    logger.info("Edge inconsistency score: %.4f  (anomalous blocks: %.1f %%)",
                score, anomalous_ratio * 100)
    return round(score, 4), density


# ===========================================================================
# Sub-detector 2 — Lighting Inconsistency
# ===========================================================================

def _gradient_orientation_histogram(gray: np.ndarray, bins: int = 9) -> np.ndarray:
    """Compute dominant gradient orientation histogram for a grayscale patch."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angles = np.arctan2(gy, gx)   # in [-π, π]
    hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
    total = hist.sum()
    return hist / total if total > 0 else hist.astype(np.float32)


def detect_lighting_inconsistency(gray: np.ndarray) -> float:
    """
    Compare gradient-orientation histograms across image blocks.

    If all blocks come from the same scene they should have similar dominant
    lighting directions.  We compute the chi-squared divergence of each
    block histogram from the global histogram and flag outlier blocks.

    Returns
    -------
    score : float in [0, 1]
    """
    h, w = gray.shape
    rows = h // BLOCK_SIZE
    cols = w // BLOCK_SIZE

    block_hists: List[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * BLOCK_SIZE, (r + 1) * BLOCK_SIZE
            x0, x1 = c * BLOCK_SIZE, (c + 1) * BLOCK_SIZE
            block_hist = _gradient_orientation_histogram(gray[y0:y1, x0:x1])
            block_hists.append(block_hist)

    if len(block_hists) < 4:
        return 0.0

    block_hists_arr = np.array(block_hists)
    global_hist = block_hists_arr.mean(axis=0)
    global_hist += 1e-9   # avoid division by zero

    chi2_distances = []
    for bh in block_hists_arr:
        chi2 = float(((bh - global_hist) ** 2 / global_hist).sum())
        chi2_distances.append(chi2)

    chi2_arr = np.array(chi2_distances)
    mu, sigma = chi2_arr.mean(), chi2_arr.std()
    if sigma < 1e-6:
        return 0.0

    outlier_fraction = float((chi2_arr > mu + 2 * sigma).mean())
    score = float(np.clip(outlier_fraction / 0.15, 0.0, 1.0))  # >15 % → score=1
    logger.info("Lighting inconsistency score: %.4f", score)
    return round(score, 4)


# ===========================================================================
# Sub-detector 3 — Copy-Move Detection
# ===========================================================================

def _dct_feature(block: np.ndarray, n_coeff: int) -> np.ndarray:
    """
    Compute a compact DCT-based feature vector for an image block.
    We take the top-left n_coeff coefficients of the 2-D DCT (zig-zag order).
    """
    block_f = block.astype(np.float32)
    dct = cv2.dct(block_f)
    # Flatten and take first n_coeff elements (low-frequency content)
    flat = dct.flatten()[:n_coeff]
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 0 else flat


def detect_copy_move(gray: np.ndarray, max_blocks: int = 2000) -> Tuple[float, List]:
    """
    Block-matching copy-move detection using DCT features.

    Algorithm
    ---------
    1.  Extract overlapping BLOCK_SIZE×BLOCK_SIZE blocks with stride BLOCK_STRIDE.
    2.  Compute a DCT feature vector per block.
    3.  Sort blocks lexicographically by their feature vectors.
    4.  Consecutive neighbours in the sorted list are the most similar pairs.
    5.  A pair is a copy-move candidate if:
            - cosine similarity ≥ COPY_MOVE_THRESHOLD
            - spatial distance ≥ 2 × BLOCK_SIZE  (not adjacent)

    Returns
    -------
    score        : float in [0, 1]
    match_pairs  : list of ((y1,x1),(y2,x2)) coordinate pairs
    """
    h, w = gray.shape
    features = []
    positions = []

    for y in range(0, h - BLOCK_SIZE, BLOCK_STRIDE):
        for x in range(0, w - BLOCK_SIZE, BLOCK_STRIDE):
            block = gray[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            feat = _dct_feature(block, DCT_COEFF_COUNT)
            features.append(feat)
            positions.append((y, x))

    if len(features) == 0:
        return 0.0, []

    features_arr = np.array(features)

    # Sub-sample if too many blocks (performance guard)
    if len(features_arr) > max_blocks:
        idx = np.random.choice(len(features_arr), max_blocks, replace=False)
        features_arr = features_arr[idx]
        positions    = [positions[i] for i in idx]

    # Lexicographic sort to bring similar vectors adjacent
    sort_idx = np.lexsort(features_arr.T)
    sorted_feats = features_arr[sort_idx]
    sorted_pos   = [positions[i] for i in sort_idx]

    match_pairs: List = []
    for i in range(len(sorted_feats) - 1):
        v1 = sorted_feats[i]
        v2 = sorted_feats[i + 1]
        cosine_sim = float(np.dot(v1, v2))   # both already L2-normalised

        if cosine_sim < COPY_MOVE_THRESHOLD:
            continue

        (y1, x1), (y2, x2) = sorted_pos[i], sorted_pos[i + 1]
        spatial_dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        if spatial_dist >= 2 * BLOCK_SIZE:
            match_pairs.append(((y1, x1), (y2, x2)))

    total_blocks = len(features_arr)
    match_ratio  = len(match_pairs) / total_blocks if total_blocks > 0 else 0
    score = float(np.clip(match_ratio / 0.05, 0.0, 1.0))  # >5 % pairs → score=1
    logger.info("Copy-move score: %.4f  (matches: %d / %d blocks)",
                score, len(match_pairs), total_blocks)
    return round(score, 4), match_pairs


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def annotate_copy_move(
    original_array: np.ndarray,
    match_pairs: List,
    max_lines: int = 50,
) -> np.ndarray:
    """Draw lines connecting copy-move block pairs on the original image."""
    vis = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)
    for (y1, x1), (y2, x2) in match_pairs[:max_lines]:
        cx1, cy1 = x1 + BLOCK_SIZE // 2, y1 + BLOCK_SIZE // 2
        cx2, cy2 = x2 + BLOCK_SIZE // 2, y2 + BLOCK_SIZE // 2
        cv2.line(vis, (cx1, cy1), (cx2, cy2), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(vis, (x1, y1), (x1 + BLOCK_SIZE, y1 + BLOCK_SIZE), (0, 255, 0), 1)
        cv2.rectangle(vis, (x2, y2), (x2 + BLOCK_SIZE, y2 + BLOCK_SIZE), (0, 255,   0), 1)
    return vis


def build_edge_heatmap(density_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Convert block-level density map to a full-resolution BGR heat-map."""
    density_uint8 = (density_map * 255).astype(np.uint8)
    upscaled = cv2.resize(density_uint8, target_size, interpolation=cv2.INTER_NEAREST)
    heatmap  = cv2.applyColorMap(upscaled, cv2.COLORMAP_JET)
    return heatmap


# ===========================================================================
# Public entry point
# ===========================================================================

def analyze(
    image_path: str,
    evidence_dir: str = "reports/evidence",
    save_evidence: bool = True,
) -> Dict[str, Any]:
    """
    Run all three splicing sub-detectors and return a combined result dict.

    Returns
    -------
    {
        sha256, edge_score, lighting_score, copy_move_score,
        splicing_score,  (weighted combination)
        match_pairs, evidence paths …
    }
    """
    image_path = str(Path(image_path).resolve())
    sha256 = compute_sha256(image_path)

    original = Image.open(image_path).convert("RGB")
    original_array = np.array(original)
    gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)

    logger.info("Running splicing detection on: %s", image_path)

    # ── Sub-detectors ────────────────────────────────────────────────────────
    edge_score, density_map = detect_edge_inconsistency(gray)
    lighting_score          = detect_lighting_inconsistency(gray)
    copy_move_score, pairs  = detect_copy_move(gray)

    # ── Weighted combination ─────────────────────────────────────────────────
    # Lighting and edge are most reliable for splicing; copy-move is bonus
    splicing_score = round(
        0.40 * edge_score + 0.35 * lighting_score + 0.25 * copy_move_score, 4
    )
    logger.info("Splicing score: %.4f", splicing_score)

    edge_heatmap_path  = None
    copymove_img_path  = None

    if save_evidence:
        os.makedirs(evidence_dir, exist_ok=True)
        stem       = Path(image_path).stem
        short_hash = sha256[:12]

        # Edge density heat-map
        w, h = original.size
        heatmap = build_edge_heatmap(density_map, (w, h))
        edge_out = os.path.join(evidence_dir, f"{stem}_{short_hash}_edge_heatmap.jpg")
        cv2.imwrite(edge_out, heatmap)
        edge_heatmap_path = edge_out
        logger.info("Edge heat-map → %s", edge_out)

        # Copy-move annotation
        cm_img = annotate_copy_move(original_array, pairs)
        cm_out = os.path.join(evidence_dir, f"{stem}_{short_hash}_copymove.jpg")
        cv2.imwrite(cm_out, cm_img)
        copymove_img_path = cm_out
        logger.info("Copy-move annotation → %s", cm_out)

    return {
        "sha256"            : sha256,
        "edge_score"        : edge_score,
        "lighting_score"    : lighting_score,
        "copy_move_score"   : copy_move_score,
        "splicing_score"    : splicing_score,
        "match_pairs_count" : len(pairs),
        "edge_heatmap_path" : edge_heatmap_path,
        "copymove_img_path" : copymove_img_path,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python splicing_detector.py <image_path>")
        sys.exit(1)
    print(json.dumps(analyze(sys.argv[1]), indent=2, default=str))
