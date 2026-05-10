"""
shap_explainer.py
=================
Explainability Layer — SHAP Feature Importance

Uses SHAP TreeExplainer on the Random Forest models to explain
WHY a specific image was flagged as manipulated.

Output per module:
  - Top-N most important features with their SHAP values
  - Plain-English explanation of each top feature
  - Overall explainability summary

This layer is critical for forensic credibility — investigators
need to know WHICH features triggered the detection, not just a score.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import shap

from ml_classifier import ForensicClassifier

logger = logging.getLogger("shap_explainer")

# Feature group descriptions for forensic investigators
FEATURE_DESCRIPTIONS = {
    "hog"    : "Edge/gradient structure anomaly",
    "lbp"    : "Micro-texture pattern inconsistency",
    "fft"    : "Frequency spectrum artifact (GAN checkerboard or over-smoothing)",
    "ela"    : "JPEG compression history mismatch (splice indicator)",
    "color"  : "Colour distribution statistical anomaly",
    "dct"    : "JPEG block DCT coefficient irregularity",
    "noise"  : "Sensor noise pattern deviation",
}


def _feature_group(idx: int, feature_sizes: Dict[str, int]) -> str:
    """Return the feature group name for a given feature index."""
    cursor = 0
    for group, size in feature_sizes.items():
        if cursor <= idx < cursor + size:
            return group
        cursor += size
    return "unknown"


def _get_feature_sizes() -> Dict[str, int]:
    """
    Compute exact feature group sizes by running extractor on a blank image.
    This guarantees sizes always match what was actually trained on.
    """
    import tempfile, os
    import numpy as np
    from PIL import Image

    # Create a minimal blank image for measurement
    tmp = tempfile.mktemp(suffix=".jpg")
    Image.fromarray(np.zeros((128,128,3),dtype=np.uint8)).save(tmp,quality=90)
    try:
        from feature_extractor import (
            _hog_features, _lbp_features, _fft_features,
            _ela_features, _color_features, _dct_features, _noise_features
        )
        gray   = np.zeros((128,128), dtype=np.uint8)
        pil    = Image.fromarray(np.zeros((128,128,3),dtype=np.uint8))
        rgb    = np.zeros((128,128,3), dtype=np.uint8)
        sizes = {
            "hog"  : len(_hog_features(gray)),
            "lbp"  : len(_lbp_features(gray)),
            "fft"  : len(_fft_features(gray)),
            "ela"  : len(_ela_features(pil)),
            "color": len(_color_features(rgb)),
            "dct"  : len(_dct_features(gray)),
            "noise": len(_noise_features(gray)),
        }
    finally:
        try: os.remove(tmp)
        except: pass
    return sizes


class SHAPExplainer:
    """
    Wraps SHAP TreeExplainer for the RandomForest forensic classifiers.
    One instance per module.
    """

    def __init__(self, clf: ForensicClassifier):
        self.clf          = clf
        self.explainer    = None
        self.feat_sizes   = _get_feature_sizes()
        self._ready       = False

        if clf.trained and clf.rf is not None:
            try:
                self.explainer = shap.TreeExplainer(
                    clf.rf,
                    feature_perturbation="tree_path_dependent"
                )
                self._ready = True
                logger.info("SHAP explainer ready for module: %s", clf.module)
            except Exception as e:
                logger.warning("SHAP init failed for %s: %s", clf.module, e)

    @property
    def ready(self) -> bool:
        return self._ready

    def explain(self, x: np.ndarray, top_n: int = 10) -> Dict:
        """
        Compute SHAP values for one feature vector and return
        a forensic explanation.

        Parameters
        ----------
        x     : 1-D feature vector (raw, before scaling)
        top_n : Number of top features to highlight

        Returns
        -------
        {
            "available"       : bool,
            "top_features"    : [{name, shap_value, raw_value, description}],
            "dominant_domain" : str,
            "summary"         : str,
            "shap_values"     : np.ndarray  (full vector)
        }
        """
        if not self._ready:
            return {
                "available"     : False,
                "summary"       : "SHAP explainability not available (model not trained).",
                "top_features"  : [],
                "dominant_domain": "N/A",
                "shap_values"   : None,
            }

        # Scale input (SHAP operates on scaled space)
        x_s    = self.clf.scaler.transform(x.reshape(1, -1))
        values = self.explainer.shap_values(x_s)

        # Handle different SHAP output formats:
        # New sklearn/shap: ndarray of shape (1, n_features, n_classes)
        # Old format: list of [class0_arr, class1_arr]
        if isinstance(values, np.ndarray) and values.ndim == 3:
            sv = values[0, :, 1]       # sample 0, all features, class 1 (manipulated)
        elif isinstance(values, list) and len(values) == 2:
            sv = np.array(values[1]).flatten()
        else:
            sv = np.array(values).flatten()

        if len(sv) == 0:
            return {"available": False, "summary": "SHAP returned empty values.",
                    "top_features": [], "dominant_domain": "N/A", "shap_values": None}

        # Top-N features by absolute SHAP value
        abs_sv   = np.abs(sv)
        top_idx  = np.argsort(abs_sv)[::-1][:top_n]

        top_features = []
        for idx in top_idx:
            group = _feature_group(idx, self.feat_sizes)
            top_features.append({
                "feature_index" : int(idx),
                "feature_group" : group,
                "shap_value"    : round(float(sv[idx]), 5),
                "raw_value"     : round(float(x[idx]), 5),
                "direction"     : "↑ increases manipulation score"
                                  if sv[idx] > 0
                                  else "↓ decreases manipulation score",
                "description"   : FEATURE_DESCRIPTIONS.get(group, group),
            })

        # Dominant group = group with highest total |SHAP| contribution
        group_totals: Dict[str, float] = {}
        for g in FEATURE_DESCRIPTIONS:
            # sum of |SHAP| for all features in this group
            start  = sum(v for k, v in self.feat_sizes.items()
                         if list(self.feat_sizes.keys()).index(k)
                         < list(self.feat_sizes.keys()).index(g))
            end    = start + self.feat_sizes.get(g, 0)
            if end <= len(abs_sv):
                group_totals[g] = float(abs_sv[start:end].sum())

        dominant = max(group_totals, key=group_totals.get) if group_totals else "N/A"

        # Plain-English forensic summary
        summary = _build_summary(self.clf.module, top_features, dominant)

        return {
            "available"       : True,
            "top_features"    : top_features,
            "dominant_domain" : dominant,
            "dominant_description": FEATURE_DESCRIPTIONS.get(dominant, dominant),
            "summary"         : summary,
            "shap_values"     : sv,
        }


def _build_summary(module: str, top_features: List[Dict], dominant: str) -> str:
    """Generate a plain-English forensic explanation string."""
    module_names = {
        "splicing" : "image splicing",
        "ai_gen"   : "AI-generated content",
        "deepfake" : "deepfake manipulation",
    }
    target = module_names.get(module, module)

    if not top_features:
        return f"No significant features identified for {target} detection."

    # Pick top 3 contributing groups
    seen, groups = set(), []
    for f in top_features:
        g = f["feature_group"]
        if g not in seen:
            seen.add(g)
            groups.append(g)
        if len(groups) >= 3:
            break

    group_phrases = [FEATURE_DESCRIPTIONS.get(g, g).lower() for g in groups]

    summary = (
        f"The {target} score is primarily driven by: "
        + "; ".join(group_phrases[:3]) + ". "
        f"The dominant forensic signal comes from the '{dominant}' feature domain, "
        f"which is consistent with known {target} patterns."
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience function used by the main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

_explainer_cache: Dict[str, SHAPExplainer] = {}

def explain(module: str, feature_vector: np.ndarray, top_n: int = 8) -> Dict:
    """
    Get SHAP explanation for one feature vector.
    Caches the explainer object across calls (expensive to build).
    """
    global _explainer_cache
    if module not in _explainer_cache:
        from ml_classifier import get_classifier
        clf = get_classifier(module)
        _explainer_cache[module] = SHAPExplainer(clf)

    return _explainer_cache[module].explain(feature_vector, top_n=top_n)


def invalidate_cache() -> None:
    """Clear explainer cache (call after retraining)."""
    global _explainer_cache
    _explainer_cache = {}
