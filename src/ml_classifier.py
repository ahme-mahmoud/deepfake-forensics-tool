"""
ml_classifier.py
================
ML Classifier Layer  —  SVM + Random Forest per forensic module

Each module (splicing, ai_gen, deepfake) gets its own pair of models:
  - RandomForestClassifier (handles non-linear relationships, gives importances)
  - SVC with probability=True (strong margin-based classifier)

Final module score = mean of [RF_proba, SVM_proba]

Trained models are saved to  models/<module>_rf.pkl  and  models/<module>_svm.pkl
If no trained model exists the classifier returns a signal-processing fallback.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger("ml_classifier")

_ROOT       = Path(__file__).parent.parent
MODELS_DIR  = _ROOT / "models"

# Module names used as file prefixes
MODULES = ("splicing", "ai_gen", "deepfake")


# ═══════════════════════════════════════════════════════════════════════════════
# Model bundle: scaler + RF + SVM stored together
# ═══════════════════════════════════════════════════════════════════════════════

class ForensicClassifier:
    """
    Holds a trained scaler + Random Forest + SVM for one forensic module.

    Usage:
        clf = ForensicClassifier.load("splicing")
        score = clf.predict_proba(feature_vector)   # returns float [0,1]
    """

    def __init__(self, module: str):
        self.module  = module
        self.scaler  : Optional[StandardScaler]  = None
        self.rf      : Optional[RandomForestClassifier] = None
        self.svm     : Optional[SVC]             = None
        self.trained : bool = False
        self.feature_importances_: Optional[np.ndarray] = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _paths(self) -> Tuple[Path, Path, Path]:
        d = MODELS_DIR
        return (d / f"{self.module}_scaler.pkl",
                d / f"{self.module}_rf.pkl",
                d / f"{self.module}_svm.pkl")

    def save(self) -> None:
        """Persist scaler + RF + SVM to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        sp, rp, svp = self._paths()
        pickle.dump(self.scaler, open(sp,  "wb"))
        pickle.dump(self.rf,     open(rp,  "wb"))
        pickle.dump(self.svm,    open(svp, "wb"))
        logger.info("Saved %s models → %s", self.module, MODELS_DIR)

    @classmethod
    def load(cls, module: str) -> "ForensicClassifier":
        """Load a previously trained classifier.  Returns untrained instance if not found."""
        obj = cls(module)
        sp  = MODELS_DIR / f"{module}_scaler.pkl"
        rp  = MODELS_DIR / f"{module}_rf.pkl"
        svp = MODELS_DIR / f"{module}_svm.pkl"

        if sp.exists() and rp.exists() and svp.exists():
            try:
                obj.scaler  = pickle.load(open(sp,  "rb"))
                obj.rf      = pickle.load(open(rp,  "rb"))
                obj.svm     = pickle.load(open(svp, "rb"))
                obj.trained = True
                if obj.rf is not None and hasattr(obj.rf, "feature_importances_"):
                    obj.feature_importances_ = obj.rf.feature_importances_
                logger.info("Loaded %s models from disk", module)
            except Exception as e:
                logger.warning("Could not load %s models: %s", module, e)
        else:
            logger.warning("No saved models for '%s' → ML scoring disabled for this module", module)
        return obj

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForensicClassifier":
        """
        Train both RF and SVM on feature matrix X with binary labels y.
        y: 0 = authentic, 1 = manipulated/fake
        """
        logger.info("Training %s — samples=%d  features=%d", self.module, len(y), X.shape[1])

        # Scale features (critical for SVM)
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)

        # ── Random Forest ─────────────────────────────────────────────────────
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.rf.fit(X_s, y)
        self.feature_importances_ = self.rf.feature_importances_
        logger.info("RF trained  — OOB-like train accuracy: %.3f",
                    self.rf.score(X_s, y))

        # ── SVM with probability calibration ─────────────────────────────────
        base_svm = SVC(kernel="rbf", C=10.0, gamma="scale",
                       class_weight="balanced", probability=True,
                       random_state=42)
        self.svm = CalibratedClassifierCV(base_svm, cv=3)
        self.svm.fit(X_s, y)
        logger.info("SVM trained — train accuracy: %.3f",
                    self.svm.score(X_s, y))

        self.trained = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, x: np.ndarray) -> float:
        """
        Return manipulation probability in [0, 1].
        If not trained → returns None (caller falls back to signal-processing).

        Aggregation: mean(RF_proba, SVM_proba)
        """
        if not self.trained or self.scaler is None:
            return None  # type: ignore[return-value]

        x_2d = x.reshape(1, -1)
        x_s  = self.scaler.transform(x_2d)

        rf_prob  = float(self.rf.predict_proba(x_s)[0][1])
        svm_prob = float(self.svm.predict_proba(x_s)[0][1])
        score    = (rf_prob + svm_prob) / 2.0

        logger.debug("%s ML score: RF=%.3f SVM=%.3f → %.3f",
                     self.module, rf_prob, svm_prob, score)
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def predict_detail(self, x: np.ndarray) -> Dict:
        """Return detailed prediction including individual model scores."""
        if not self.trained:
            return {"trained": False, "score": None}

        x_s      = self.scaler.transform(x.reshape(1, -1))
        rf_prob  = float(self.rf.predict_proba(x_s)[0][1])
        svm_prob = float(self.svm.predict_proba(x_s)[0][1])
        combined = (rf_prob + svm_prob) / 2.0

        return {
            "trained"   : True,
            "rf_score"  : round(rf_prob,  4),
            "svm_score" : round(svm_prob, 4),
            "score"     : round(float(np.clip(combined, 0, 1)), 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier Registry  (singleton pattern)
# ═══════════════════════════════════════════════════════════════════════════════

_registry: Dict[str, ForensicClassifier] = {}

def get_classifier(module: str) -> ForensicClassifier:
    """Load (or return cached) classifier for a given module."""
    global _registry
    if module not in _registry:
        _registry[module] = ForensicClassifier.load(module)
    return _registry[module]


def reload_all() -> None:
    """Force reload all classifiers (call after training)."""
    global _registry
    _registry = {}
    for m in MODULES:
        _registry[m] = ForensicClassifier.load(m)
    logger.info("All classifiers reloaded")
