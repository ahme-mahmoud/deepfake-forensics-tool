"""
score_fusion.py
===============
Hybrid Score Fusion Engine

Combines signal-processing forensic scores with ML classifier scores
into a single, calibrated manipulation probability for each module.

Architecture:
    Signal-Processing Score  (always available)
         +
    ML Classifier Score      (available when models are trained)
         =
    Hybrid Module Score

Then:
    Final Manipulation Probability = Weighted average of module scores

Weights are configurable. Default weights reflect module reliability.
"""

from typing import Dict, Optional

import numpy as np

# ── Default fusion weights (ML vs Signal-Processing per module) ────────────────
# When ML models are available, they carry more weight.
# Without ML, full weight falls on signal-processing.
ML_WEIGHT_IF_TRAINED    = 0.60
SIGNAL_WEIGHT_IF_TRAINED = 0.40

# Final score weights per module
MODULE_WEIGHTS = {
    "ela"      : 0.20,   # Compression / ELA
    "splicing" : 0.25,   # Splicing detection
    "ai_gen"   : 0.30,   # AI-generation detection (highest weight)
    "deepfake" : 0.25,   # Deepfake detection
}


def fuse_module_score(
    signal_score: float,
    ml_score: Optional[float],
    ml_weight: float = ML_WEIGHT_IF_TRAINED,
) -> Dict:
    """
    Combine signal-processing and ML scores for one module.

    Parameters
    ----------
    signal_score : float  Score from forensic signal-processing [0,1]
    ml_score     : float | None  Score from ML classifier (None if not trained)
    ml_weight    : float  Weight for ML score when available

    Returns
    -------
    {
        "signal_score" : float,
        "ml_score"     : float | None,
        "fused_score"  : float,   ← the authoritative module score
        "ml_used"      : bool,
    }
    """
    signal_score = float(np.clip(signal_score, 0.0, 1.0))

    if ml_score is None:
        # No ML available — use signal-processing only
        return {
            "signal_score": round(signal_score, 4),
            "ml_score"    : None,
            "fused_score" : round(signal_score, 4),
            "ml_used"     : False,
        }

    ml_score = float(np.clip(ml_score, 0.0, 1.0))
    sig_weight = 1.0 - ml_weight
    fused = ml_weight * ml_score + sig_weight * signal_score

    return {
        "signal_score": round(signal_score, 4),
        "ml_score"    : round(ml_score, 4),
        "fused_score" : round(float(np.clip(fused, 0.0, 1.0)), 4),
        "ml_used"     : True,
    }


def compute_final_score(module_scores: Dict[str, float]) -> Dict:
    """
    Compute the final manipulation probability from all module scores.

    Parameters
    ----------
    module_scores : {"ela": 0.3, "splicing": 0.6, "ai_gen": 0.4, "deepfake": 0.7}

    Returns
    -------
    {
        "final_score"        : float,
        "weighted_breakdown" : {module: weighted_contribution},
        "dominant_module"    : str,
        "confidence"         : str,  ← "high" / "medium" / "low"
    }
    """
    weights  = MODULE_WEIGHTS
    total_w  = 0.0
    weighted = 0.0
    breakdown = {}

    for mod, score in module_scores.items():
        w = weights.get(mod, 0.25)
        contribution = w * float(np.clip(score, 0, 1))
        breakdown[mod] = round(contribution, 4)
        weighted  += contribution
        total_w   += w

    final = weighted / total_w if total_w > 0 else 0.0
    final = round(float(np.clip(final, 0.0, 1.0)), 4)

    dominant = max(module_scores, key=lambda k: module_scores[k])

    # Confidence: how many modules agree?
    above_threshold = sum(1 for s in module_scores.values() if s > 0.50)
    if above_threshold >= 3:
        confidence = "high"
    elif above_threshold >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "final_score"        : final,
        "weighted_breakdown" : breakdown,
        "dominant_module"    : dominant,
        "confidence"         : confidence,
        "modules_above_50pct": above_threshold,
    }
