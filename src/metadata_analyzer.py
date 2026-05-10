"""
metadata_analyzer.py
====================
Module 5 — Metadata & Provenance Analysis

Detects AI-generated images by analyzing metadata signals
that modern generative models (Midjourney, DALL-E, Flux, Gemini) fail to fake:

1. EXIF Absence    — real cameras always embed EXIF; AI images often don't
2. Software Tags   — EXIF Software field sometimes reveals generation tools
3. PNG Text Chunks — Stable Diffusion / ComfyUI embed prompts in PNG metadata
4. Color Profile   — real cameras embed ICC profiles; AI images often skip this
5. Thumbnail Consistency — EXIF thumbnail vs full image mismatch
6. File Format     — PNG with no JPEG history defeats ELA (weak forensics signal)

This module specifically targets the gap in classical forensics:
classical ELA/FFT fails on modern AI → metadata fills the gap.
"""

import io
import logging
import os
import struct
import zlib
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("metadata_analyzer")

# ── Known AI software identifiers ────────────────────────────────────────────
AI_SOFTWARE_KEYWORDS = [
    "stable diffusion", "midjourney", "dall-e", "dalle",
    "diffusion", "comfyui", "automatic1111", "invokeai",
    "novelai", "flux", "gemini", "firefly", "generative",
    "ai generated", "bing image", "adobe firefly",
    "leonardo", "runway", "ideogram",
]

# ── Known camera EXIF fields ──────────────────────────────────────────────────
CAMERA_EXIF_TAGS = {
    "Make", "Model", "ExposureTime", "FNumber", "ISOSpeedRatings",
    "FocalLength", "LensModel", "GPSInfo", "Flash", "WhiteBalance",
    "MeteringMode", "ShutterSpeedValue", "ApertureValue",
}


def _read_exif(pil_img: Image.Image) -> Optional[dict]:
    """Extract EXIF data safely."""
    try:
        from PIL.ExifTags import TAGS
        raw = pil_img._getexif()
        if raw is None:
            return None
        return {TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        return None


def _read_png_metadata(image_path: str) -> dict:
    """
    Read PNG tEXt / iTXt / zTXt chunks.
    Stable Diffusion embeds full prompts + model names in these chunks.
    """
    metadata = {}
    try:
        with open(image_path, "rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return metadata

            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                length = struct.unpack(">I", header[:4])[0]
                chunk_type = header[4:8].decode("ascii", errors="replace")
                data = f.read(length)
                f.read(4)  # CRC

                if chunk_type == "tEXt":
                    try:
                        key, val = data.split(b"\x00", 1)
                        metadata[key.decode()] = val.decode("latin-1")
                    except Exception:
                        pass

                elif chunk_type == "iTXt":
                    try:
                        parts = data.split(b"\x00")
                        if len(parts) >= 2:
                            key = parts[0].decode()
                            val = parts[-1].decode("utf-8", errors="replace")
                            metadata[key] = val
                    except Exception:
                        pass

                elif chunk_type == "zTXt":
                    try:
                        key, rest = data.split(b"\x00", 1)
                        decompressed = zlib.decompress(rest[1:])
                        metadata[key.decode()] = decompressed.decode("latin-1")
                    except Exception:
                        pass

                elif chunk_type == "IEND":
                    break
    except Exception as e:
        logger.debug("PNG metadata read error: %s", e)
    return metadata


def _score_exif_absence(exif: Optional[dict], file_format: str) -> float:
    """
    Score the ABSENCE of camera-like EXIF fields.
    Real cameras always write Make/Model/ISO/FocalLength etc.
    AI-generated images typically have none of these.
    """
    if file_format == "PNG":
        # PNG rarely has full EXIF — not strongly indicative alone
        return 0.30

    if exif is None:
        # JPEG with NO EXIF is very suspicious
        return 0.80

    # Check how many camera-specific fields are present
    exif_keys      = set(str(k) for k in exif.keys())
    camera_present = len(CAMERA_EXIF_TAGS & exif_keys)

    if camera_present >= 5:
        return 0.05   # Lots of camera data → likely real
    elif camera_present >= 2:
        return 0.25
    elif camera_present == 0:
        return 0.70   # No camera fields → suspicious
    return 0.40


def _score_software_tag(exif: Optional[dict], png_meta: dict) -> tuple:
    """
    Check Software / tool fields for known AI generation keywords.
    Returns (score [0,1], detected_software str)
    """
    candidates = []

    if exif:
        for field in ("Software", "ProcessingSoftware", "HostComputer", "Artist",
                      "ImageDescription", "UserComment", "Copyright"):
            val = exif.get(field, "")
            if isinstance(val, bytes):
                try: val = val.decode("utf-8", errors="replace")
                except: val = ""
            candidates.append(str(val).lower())

    for key, val in png_meta.items():
        candidates.append(str(val).lower())
        candidates.append(key.lower())

    combined = " ".join(candidates)
    for keyword in AI_SOFTWARE_KEYWORDS:
        if keyword in combined:
            return 1.0, keyword.title()

    return 0.0, "none"


def _score_color_profile(pil_img: Image.Image) -> float:
    """
    Real cameras embed ICC color profiles.
    AI generators often skip this, or embed generic sRGB without camera data.
    """
    try:
        icc = pil_img.info.get("icc_profile")
        if icc is None:
            return 0.40   # No profile → slightly suspicious
        if len(icc) < 200:
            return 0.20   # Very short profile → generic/minimal
        return 0.05       # Full ICC profile → likely real camera
    except Exception:
        return 0.30


def _score_png_ai_metadata(png_meta: dict) -> tuple:
    """
    Check if PNG contains Stable Diffusion / ComfyUI metadata.
    These tools embed prompts, seeds, models directly in PNG chunks.
    Returns (score, description)
    """
    ai_keys = {"parameters", "prompt", "negative_prompt", "steps",
                "sampler", "cfg scale", "seed", "model hash",
                "model", "workflow", "comfyui"}
    found = []
    for key in png_meta:
        if key.lower() in ai_keys or any(k in key.lower() for k in ai_keys):
            found.append(key)

    if found:
        return 1.0, f"AI metadata keys: {found[:3]}"
    return 0.0, "none"


def _score_file_format_risk(pil_img: Image.Image, image_path: str) -> float:
    """
    PNG format without JPEG history means ELA cannot work.
    Screenshots destroy forensic artifacts.
    AI images shared as PNG are harder to detect classically.
    This is a CONTEXT flag, not a direct AI indicator.
    """
    ext = Path(image_path).suffix.lower()
    fmt = pil_img.format or ""

    if ext in (".png", ) or fmt == "PNG":
        # PNG: ELA blind spot. Moderate suspicion flag.
        return 0.35
    return 0.05


def _score_thumbnail_mismatch(pil_img: Image.Image, exif: Optional[dict]) -> float:
    """
    Check EXIF thumbnail vs full image consistency.
    Doctored images sometimes have mismatched thumbnails.
    """
    if exif is None:
        return 0.0

    try:
        from PIL import Image as PILImage
        thumb_data = None
        for tag, val in exif.items():
            if "thumbnail" in str(tag).lower() and isinstance(val, bytes):
                thumb_data = val
                break

        if thumb_data is None:
            return 0.0

        thumb = PILImage.open(io.BytesIO(thumb_data)).convert("RGB")
        full_small = pil_img.resize((thumb.width, thumb.height),
                                    PILImage.LANCZOS)
        diff = np.abs(np.array(thumb).astype(float) -
                      np.array(full_small).astype(float)).mean()

        # Large diff → thumbnail mismatch → suspicious
        score = float(np.clip(diff / 30.0, 0.0, 1.0))
        return round(score, 4)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(image_path: str) -> Dict[str, Any]:
    """
    Full metadata forensic analysis.

    Returns
    -------
    {
        metadata_score        : float [0,1] — overall AI-generation probability
        exif_absence_score    : float
        software_tag_score    : float
        ai_metadata_score     : float
        color_profile_score   : float
        format_risk_score     : float
        thumbnail_score       : float
        detected_software     : str
        has_exif              : bool
        exif_camera_fields    : int
        png_metadata_keys     : list
        file_format           : str
        explanation           : str
        flags                 : list  — list of triggered red flags
    }
    """
    image_path = str(Path(image_path).resolve())
    pil_img    = Image.open(image_path)
    file_fmt   = (pil_img.format or Path(image_path).suffix.upper().lstrip("."))

    # ── Extract metadata ──────────────────────────────────────────────────────
    exif     = _read_exif(pil_img)
    png_meta = _read_png_metadata(image_path) if file_fmt in ("PNG", ".PNG") else {}

    exif_keys      = set(str(k) for k in exif.keys()) if exif else set()
    camera_present = len(CAMERA_EXIF_TAGS & exif_keys)

    # ── Individual scores ─────────────────────────────────────────────────────
    exif_score           = _score_exif_absence(exif, file_fmt)
    soft_score, soft_det = _score_software_tag(exif, png_meta)
    ai_meta_score, ai_desc = _score_png_ai_metadata(png_meta)
    color_score          = _score_color_profile(pil_img)
    fmt_score            = _score_file_format_risk(pil_img, image_path)
    thumb_score          = _score_thumbnail_mismatch(pil_img, exif)

    # ── Weighted final score ──────────────────────────────────────────────────
    # Software/AI metadata tags are definitive → high weight
    metadata_score = round(
        0.35 * soft_score    +
        0.30 * ai_meta_score +
        0.20 * exif_score    +
        0.10 * color_score   +
        0.05 * thumb_score,
        4
    )

    # ── Red flags list ────────────────────────────────────────────────────────
    flags = []
    if exif is None and file_fmt in ("JPEG", "JPG"):
        flags.append("No EXIF in JPEG — highly suspicious")
    if soft_score >= 1.0:
        flags.append(f"AI software detected in metadata: {soft_det}")
    if ai_meta_score >= 1.0:
        flags.append(f"AI generation parameters found in PNG: {ai_desc}")
    if camera_present == 0 and exif is not None:
        flags.append("EXIF present but NO camera fields (Make/Model/ISO)")
    if color_score > 0.35:
        flags.append("No ICC color profile — unusual for real cameras")
    if file_fmt == "PNG" and not png_meta:
        flags.append("PNG with no metadata — ELA analysis not applicable")
    if thumb_score > 0.5:
        flags.append("EXIF thumbnail does not match image content")

    # ── Explanation ───────────────────────────────────────────────────────────
    if metadata_score > 0.7:
        explanation = "Strong metadata evidence of AI generation."
    elif metadata_score > 0.4:
        explanation = "Partial metadata indicators — inconclusive."
    elif flags:
        explanation = f"Minor metadata anomalies: {flags[0]}"
    else:
        explanation = "Metadata is consistent with a real photograph."

    logger.info("Metadata score: %.4f  flags: %d  software: %s",
                metadata_score, len(flags), soft_det)

    return {
        "metadata_score"      : metadata_score,
        "exif_absence_score"  : round(exif_score, 4),
        "software_tag_score"  : round(soft_score, 4),
        "ai_metadata_score"   : round(ai_meta_score, 4),
        "color_profile_score" : round(color_score, 4),
        "format_risk_score"   : round(fmt_score, 4),
        "thumbnail_score"     : round(thumb_score, 4),
        "detected_software"   : soft_det,
        "has_exif"            : exif is not None,
        "exif_camera_fields"  : camera_present,
        "png_metadata_keys"   : list(png_meta.keys())[:10],
        "file_format"         : file_fmt,
        "explanation"         : explanation,
        "flags"               : flags,
    }


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python metadata_analyzer.py <image_path>")
        sys.exit(1)
    print(json.dumps(analyze(sys.argv[1]), indent=2, default=str))
