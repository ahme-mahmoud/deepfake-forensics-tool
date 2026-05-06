"""
deepfake_detector.py
====================
Module 4 — Deepfake Face Detection

FORENSIC LOGIC:
    Deepfake generation pipelines have three characteristic weaknesses that
    can be exploited without a trained neural-network classifier:

    1. FACIAL LANDMARK GEOMETRY INCONSISTENCIES
       Deepfake faces often suffer from subtle warping around the jaw,
       eyes, and nose.  We use OpenCV's Haar/LBP face detector to locate
       the face and then analyse the symmetry, aspect ratio, and convex-hull
       regularity of detected facial features.

    2. BLENDING BOUNDARY DETECTION
       Every deepfake must blend the synthesised face onto the background.
       The blend boundary creates a narrow band of unusual colour/texture
       transition that we detect by looking at a ring around the face oval
       for strong local contrast combined with gradient discontinuities.

    3. COLOUR SPACE INCONSISTENCY
       Deepfake generators often fail to perfectly replicate skin tone
       across the entire face.  We analyse the Cb and Cr channels of the
       YCbCr colour space inside the face region for spatial uniformity.
       A suspiciously non-uniform Cb/Cr distribution (high entropy AND high
       spatial variance) suggests compositing.

    4. REFLECTION ASYMMETRY (eye-glint check)
       Real eyes have corneal reflections.  Poorly synthesised eyes
       sometimes have missing or asymmetric highlights.  We measure
       the intensity and symmetry of bright specular regions inside the
       detected eye bounding areas.

NOTE ON FACENET:
    Full FaceNet embedding comparison requires a pre-trained model which
    we do not ship (too large).  The module architecture is prepared for it:
    `FaceEmbeddingChecker` is stubbed out and returns a neutral score of 0.5
    so the rest of the pipeline is unaffected.  Integrate your own FaceNet
    weights by implementing `_embed()`.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.stats import entropy as scipy_entropy

from compression_analysis import compute_sha256

logger = logging.getLogger("deepfake_detector")

# ---------------------------------------------------------------------------
# Haar Cascade paths (bundled with OpenCV)
# ---------------------------------------------------------------------------
_CASCADE_DIR = cv2.data.haarcascades
FACE_CASCADE_PATH = os.path.join(_CASCADE_DIR, "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH  = os.path.join(_CASCADE_DIR, "haarcascade_eye.xml")

_face_cascade: Optional[cv2.CascadeClassifier] = None
_eye_cascade:  Optional[cv2.CascadeClassifier] = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    return _face_cascade


def _get_eye_cascade() -> cv2.CascadeClassifier:
    global _eye_cascade
    if _eye_cascade is None:
        _eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    return _eye_cascade


# ===========================================================================
# Face detection helper
# ===========================================================================

def detect_faces(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect frontal faces using Haar cascades.
    Returns list of (x, y, w, h) bounding boxes, sorted by area descending.
    """
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if not isinstance(faces, np.ndarray) or len(faces) == 0:
        return []
    faces_list = [tuple(f) for f in faces]
    faces_list.sort(key=lambda f: f[2] * f[3], reverse=True)
    return faces_list   # type: ignore[return-value]


# ===========================================================================
# Sub-detector 1 — Facial Landmark Geometry
# ===========================================================================

def detect_landmark_inconsistency(
    gray: np.ndarray,
    face_boxes: List[Tuple],
) -> float:
    """
    Analyse facial geometry for signs of warping or deformation.

    We measure:
        (a) Face aspect ratio deviation from typical human proportions
        (b) Eye symmetry: are the two eyes at the same vertical level?
        (c) Number of eyes detected inside each face (deepfakes sometimes
            produce 1 or 3 eyes due to blending artifacts)

    Returns a score in [0, 1].
    """
    if not face_boxes:
        logger.info("No faces detected — landmark score: 0.0 (cannot assess)")
        return 0.0

    eye_cascade = _get_eye_cascade()
    suspicion_scores = []

    for (fx, fy, fw, fh) in face_boxes:
        score = 0.0

        # ── Aspect ratio ──────────────────────────────────────────────────────
        aspect = fw / max(fh, 1)
        # Typical frontal face: width/height ≈ 0.70–0.90
        if aspect < 0.60 or aspect > 1.10:
            score += 0.3

        # ── Eye detection within face ROI ─────────────────────────────────────
        face_roi = gray[fy:fy+fh, fx:fx+fw]
        # Only look in top 60 % of face for eyes
        eye_roi  = face_roi[:int(fh * 0.60), :]
        eyes = eye_cascade.detectMultiScale(
            eye_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )

        if not isinstance(eyes, np.ndarray):
            eye_count = 0
        else:
            eye_count = len(eyes)

        if eye_count == 0:
            score += 0.3   # Missing eyes — suspicious
        elif eye_count == 1:
            score += 0.2   # Asymmetric detection — mildly suspicious
        elif eye_count >= 3:
            score += 0.4   # Phantom eye — strongly suspicious

        # ── Eye vertical symmetry ─────────────────────────────────────────────
        if isinstance(eyes, np.ndarray) and len(eyes) == 2:
            ey1, ey2 = int(eyes[0][1]), int(eyes[1][1])
            v_diff = abs(ey1 - ey2) / max(fh, 1)
            if v_diff > 0.12:
                score += 0.2   # eyes not at same height

        suspicion_scores.append(min(score, 1.0))

    final = float(np.mean(suspicion_scores))
    logger.info("Landmark inconsistency score: %.4f (faces=%d)", final, len(face_boxes))
    return round(float(np.clip(final, 0.0, 1.0)), 4)


# ===========================================================================
# Sub-detector 2 — Blending Boundary Detection
# ===========================================================================

def detect_blending_boundary(
    bgr: np.ndarray,
    face_boxes: List[Tuple],
    ring_width: int = 12,
) -> float:
    """
    Look for blending seams at the face perimeter.

    For each detected face we create an elliptical mask just inside the
    bounding box (the "inner face") and a slightly larger ellipse ("outer ring").
    We compare the gradient magnitude in the ring to the face interior.
    A deep-fake blend produces an unusually high gradient at this transition.

    Returns a score in [0, 1].
    """
    if not face_boxes:
        return 0.0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grad_mag = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )

    scores = []
    h_img, w_img = gray.shape

    for (fx, fy, fw, fh) in face_boxes:
        cx, cy = fx + fw // 2, fy + fh // 2
        rx_inner, ry_inner = fw // 2 - ring_width, fh // 2 - ring_width
        rx_outer, ry_outer = fw // 2 + ring_width, fh // 2 + ring_width

        # Masks
        mask_inner = np.zeros_like(gray)
        mask_outer = np.zeros_like(gray)
        cv2.ellipse(mask_inner, (cx, cy), (max(rx_inner, 1), max(ry_inner, 1)),
                    0, 0, 360, 255, -1)
        cv2.ellipse(mask_outer, (cx, cy), (min(rx_outer, w_img//2), min(ry_outer, h_img//2)),
                    0, 0, 360, 255, -1)
        ring_mask = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))

        inner_grad = grad_mag[mask_inner > 0].mean() if mask_inner.any() else 0
        ring_grad  = grad_mag[ring_mask  > 0].mean() if ring_mask.any()  else 0

        if inner_grad < 1e-3:
            scores.append(0.0)
            continue

        # Ratio of ring gradient to inner gradient
        ratio = ring_grad / inner_grad
        # Authentically, the face edge always has higher gradients than interior.
        # Deepfake blending creates an *extra* spike: ratio > 2.5 is suspicious.
        face_score = float(np.clip((ratio - 1.5) / 2.0, 0.0, 1.0))
        scores.append(face_score)

    final = float(np.mean(scores)) if scores else 0.0
    logger.info("Blending boundary score: %.4f", final)
    return round(float(np.clip(final, 0.0, 1.0)), 4)


# ===========================================================================
# Sub-detector 3 — Colour Space Inconsistency (YCbCr)
# ===========================================================================

def detect_colour_inconsistency(
    bgr: np.ndarray,
    face_boxes: List[Tuple],
) -> float:
    """
    Measure Cb/Cr spatial non-uniformity inside detected face regions.

    Deepfakes often fail to properly harmonise skin tone.  We use:
        (a) Local block-wise Cb and Cr standard deviation
        (b) Entropy of Cb and Cr histograms

    A high combined score suggests colour inconsistency consistent with
    face swapping.

    Returns a score in [0, 1].
    """
    if not face_boxes:
        return 0.0

    ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cb_ch = ycbcr[:, :, 2].astype(np.float32)   # OpenCV YCrCb: index 2 = Cb
    cr_ch = ycbcr[:, :, 1].astype(np.float32)   # index 1 = Cr

    scores = []
    for (fx, fy, fw, fh) in face_boxes:
        cb_roi = cb_ch[fy:fy+fh, fx:fx+fw]
        cr_roi = cr_ch[fy:fy+fh, fx:fx+fw]

        if cb_roi.size == 0:
            continue

        # Spatial variance within face region
        cb_std = float(cb_roi.std())
        cr_std = float(cr_roi.std())

        # Histogram entropy of Cb/Cr  (low entropy = over-uniform = deepfake)
        cb_hist, _ = np.histogram(cb_roi, bins=32, range=(0, 255))
        cr_hist, _ = np.histogram(cr_roi, bins=32, range=(0, 255))
        cb_ent = float(scipy_entropy(cb_hist + 1e-9))
        cr_ent = float(scipy_entropy(cr_hist + 1e-9))
        max_ent = np.log(32)

        # High std + low entropy = suspicious (inconsistent but not natural)
        std_score = float(np.clip((cb_std + cr_std) / 40.0, 0.0, 1.0))
        ent_score = float(np.clip(1.0 - (cb_ent + cr_ent) / (2 * max_ent), 0.0, 1.0))
        face_score = 0.50 * std_score + 0.50 * ent_score
        scores.append(face_score)

    final = float(np.mean(scores)) if scores else 0.0
    logger.info("Colour inconsistency score: %.4f", final)
    return round(float(np.clip(final, 0.0, 1.0)), 4)


# ===========================================================================
# Sub-detector 4 — Eye Glint Asymmetry (stub + real implementation)
# ===========================================================================

def detect_eye_glint_asymmetry(gray: np.ndarray, face_boxes: List[Tuple]) -> float:
    """
    Measure specular highlight asymmetry in detected eye regions.
    Real eyes have consistent corneal reflections; deepfakes often don't.
    Returns a score in [0, 1].
    """
    if not face_boxes:
        return 0.0

    eye_cascade = _get_eye_cascade()
    scores = []

    for (fx, fy, fw, fh) in face_boxes:
        face_roi   = gray[fy:fy+fh, fx:fx+fw]
        eye_region = face_roi[:int(fh * 0.55), :]
        eyes = eye_cascade.detectMultiScale(
            eye_region, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
        )
        if not isinstance(eyes, np.ndarray) or len(eyes) != 2:
            scores.append(0.1)   # Cannot assess — neutral score
            continue

        glint_means = []
        for (ex, ey, ew, eh) in eyes:
            eye_patch = eye_region[ey:ey+eh, ex:ex+ew]
            # Top 5 % brightest pixels as proxy for glint
            threshold = np.percentile(eye_patch, 95)
            glint_mean = float(eye_patch[eye_patch >= threshold].mean())
            glint_means.append(glint_mean)

        if len(glint_means) == 2:
            asymmetry = abs(glint_means[0] - glint_means[1]) / (
                max(glint_means[0], glint_means[1]) + 1e-6
            )
            # >30 % asymmetry is suspicious
            score = float(np.clip(asymmetry / 0.30, 0.0, 1.0))
            scores.append(score)

    final = float(np.mean(scores)) if scores else 0.0
    logger.info("Eye glint asymmetry score: %.4f", final)
    return round(float(np.clip(final, 0.0, 1.0)), 4)


# ===========================================================================
# FaceNet stub — integrate your own weights here
# ===========================================================================

class FaceEmbeddingChecker:
    """
    Placeholder for FaceNet-based embedding consistency check.

    To activate:
        1.  Install facenet-pytorch:  pip install facenet-pytorch
        2.  Replace `_embed()` with real FaceNet inference.
        3.  Remove the stub score return in `compare()`.
    """

    def __init__(self):
        self._model = None   # Load your model here

    def _embed(self, face_patch: np.ndarray) -> Optional[np.ndarray]:
        """Return a 512-d L2-normalised embedding vector, or None on failure."""
        # ─── STUB ─────────────────────────────────────────────────────────────
        return None

    def compare(self, face_patches: List[np.ndarray]) -> float:
        """
        Measure embedding distance consistency across multiple face crops.
        Returns a score in [0, 1] (higher = more inconsistent = more suspicious).
        Returns 0.5 (neutral) if the model is not loaded.
        """
        embeddings = [self._embed(p) for p in face_patches if self._embed(p) is not None]
        if len(embeddings) < 2:
            return 0.5   # Neutral / undetermined

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                similarities.append(sim)

        mean_sim = float(np.mean(similarities))
        # Low similarity across face crops → inconsistency → deepfake
        score = float(np.clip(1.0 - mean_sim, 0.0, 1.0))
        return round(score, 4)


# ===========================================================================
# Visualisation
# ===========================================================================

def annotate_faces(
    original_array: np.ndarray,
    face_boxes: List[Tuple],
    score: float,
) -> np.ndarray:
    """Draw face detections and overall deepfake score on the image."""
    vis = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)
    colour = (0, 0, 255) if score > 0.5 else (0, 200, 0)
    for i, (fx, fy, fw, fh) in enumerate(face_boxes, 1):
        cv2.rectangle(vis, (fx, fy), (fx+fw, fy+fh), colour, 2)
        cv2.putText(vis, f"Face {i}", (fx, fy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    label = f"Deepfake Score: {score:.2f}"
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
    return vis


# ===========================================================================
# Public entry point
# ===========================================================================

def analyze(
    image_path: str,
    evidence_dir: str = "reports/evidence",
    save_evidence: bool = True,
) -> Dict[str, Any]:
    """
    Run all deepfake sub-detectors.

    Returns
    -------
    {
        sha256, faces_detected, landmark_score, blending_score,
        colour_score, eye_glint_score, embedding_score,
        deepfake_score   ← weighted combination
    }
    """
    image_path = str(Path(image_path).resolve())
    sha256 = compute_sha256(image_path)

    original   = Image.open(image_path).convert("RGB")
    orig_array = np.array(original)
    bgr        = cv2.cvtColor(orig_array, cv2.COLOR_RGB2BGR)
    gray       = cv2.cvtColor(orig_array, cv2.COLOR_RGB2GRAY)

    logger.info("Running deepfake detection on: %s", image_path)

    face_boxes = detect_faces(gray)
    logger.info("Faces detected: %d", len(face_boxes))

    if not face_boxes:
        logger.warning("No faces found — deepfake score set to 0.0")
        return {
            "sha256"          : sha256,
            "faces_detected"  : 0,
            "landmark_score"  : 0.0,
            "blending_score"  : 0.0,
            "colour_score"    : 0.0,
            "eye_glint_score" : 0.0,
            "embedding_score" : 0.5,
            "deepfake_score"  : 0.0,
            "face_annotated_path": None,
        }

    landmark_score  = detect_landmark_inconsistency(gray, face_boxes)
    blending_score  = detect_blending_boundary(bgr, face_boxes)
    colour_score    = detect_colour_inconsistency(bgr, face_boxes)
    eye_glint_score = detect_eye_glint_asymmetry(gray, face_boxes)

    # FaceNet (stub — returns 0.5)
    embedding_score = FaceEmbeddingChecker().compare([])

    deepfake_score = round(
        0.30 * landmark_score  +
        0.25 * blending_score  +
        0.20 * colour_score    +
        0.15 * eye_glint_score +
        0.10 * embedding_score,
        4,
    )
    logger.info("Deepfake score: %.4f", deepfake_score)

    face_annotated_path = None
    if save_evidence:
        os.makedirs(evidence_dir, exist_ok=True)
        stem = Path(image_path).stem
        short_hash = sha256[:12]
        vis = annotate_faces(orig_array, face_boxes, deepfake_score)
        out = os.path.join(evidence_dir, f"{stem}_{short_hash}_deepfake_annotated.jpg")
        cv2.imwrite(out, vis)
        face_annotated_path = out
        logger.info("Annotated deepfake image → %s", out)

    return {
        "sha256"              : sha256,
        "faces_detected"      : len(face_boxes),
        "landmark_score"      : landmark_score,
        "blending_score"      : blending_score,
        "colour_score"        : colour_score,
        "eye_glint_score"     : eye_glint_score,
        "embedding_score"     : embedding_score,
        "deepfake_score"      : deepfake_score,
        "face_annotated_path" : face_annotated_path,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python deepfake_detector.py <image_path>")
        sys.exit(1)
    print(json.dumps(analyze(sys.argv[1]), indent=2, default=str))
