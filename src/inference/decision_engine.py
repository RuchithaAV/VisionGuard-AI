"""
VisionGuard AI - Calibrated 3-State Industrial Decision Engine (PASS / REVIEW / FAIL)

Calibrates decision boundaries strictly from held-out normal validation scores
(zero test leakage) and translates raw PatchCore anomaly distances + heatmaps
into factory routing decisions, normalized risk scores, and defect area metrics.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class DecisionResult:
    """Structured inspection outcome for a single industrial part."""
    decision: str                # "PASS", "REVIEW", or "FAIL"
    raw_score: float             # Raw PatchCore max patch distance
    normalized_score: float      # Calibrated risk score in [0.0, 1.0]
    z_score: float               # Standard deviations above normal validation mean
    defect_area_pct: float       # Percentage of image pixels exceeding pass threshold
    pass_threshold: float        # Upper boundary for PASS
    fail_threshold: float        # Upper boundary for REVIEW (above = FAIL)
    category: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionEngine:
    """
    Leak-free statistical threshold calibrator and 3-state quality classifier.

    Decision logic:
        - raw_score <= pass_threshold                  -> "PASS"
        - pass_threshold < raw_score <= fail_threshold -> "REVIEW"
        - raw_score > fail_threshold                   -> "FAIL"
    """

    def __init__(
        self,
        pass_std_multiplier: float = 2.0,
        fail_std_multiplier: float = 3.5,
        category: str = "unknown",
        val_mean: float | None = None,
        val_std: float | None = None,
    ):
        if pass_std_multiplier <= 0 or fail_std_multiplier <= pass_std_multiplier:
            raise ValueError(
                "Require 0 < pass_std_multiplier < fail_std_multiplier."
            )
        self.pass_std_multiplier = float(pass_std_multiplier)
        self.fail_std_multiplier = float(fail_std_multiplier)
        self.category = category
        self.val_mean: float | None = float(val_mean) if val_mean is not None else None
        self.val_std: float | None = float(val_std) if val_std is not None else None

    @property
    def is_calibrated(self) -> bool:
        return self.val_mean is not None and self.val_std is not None

    @property
    def pass_threshold(self) -> float:
        if not self.is_calibrated:
            raise ValueError("DecisionEngine is not calibrated. Call fit() first.")
        return self.val_mean + self.pass_std_multiplier * self.val_std

    @property
    def fail_threshold(self) -> float:
        if not self.is_calibrated:
            raise ValueError("DecisionEngine is not calibrated. Call fit() first.")
        return self.val_mean + self.fail_std_multiplier * self.val_std

    def fit(self, val_normal_scores: np.ndarray) -> "DecisionEngine":
        """
        Calibrates thresholds using ONLY held-out normal validation scores.

        Args:
            val_normal_scores: 1D array of anomaly scores from defect-free validation parts.
        """
        scores = np.asarray(val_normal_scores, dtype=np.float64).ravel()
        if scores.size < 2:
            raise ValueError("Need at least 2 validation normal scores to calibrate thresholds.")

        self.val_mean = float(np.mean(scores))
        # Guard against zero variance on synthetic/constant inputs
        self.val_std = float(max(np.std(scores, ddof=1), 1e-6))
        return self

    def _normalize_score(self, raw_score: float) -> float:
        """
        Maps raw anomaly distance onto [0.0, 1.0]:
            val_mean       -> 0.25
            pass_threshold -> 0.50
            fail_threshold -> 0.80
        """
        t_pass = self.pass_threshold
        t_fail = self.fail_threshold

        if raw_score <= self.val_mean:
            span = max(t_pass - self.val_mean, 1e-6)
            norm = 0.25 - 0.25 * ((self.val_mean - raw_score) / span)
        elif raw_score <= t_pass:
            norm = 0.25 + 0.25 * ((raw_score - self.val_mean) / max(t_pass - self.val_mean, 1e-6))
        elif raw_score <= t_fail:
            norm = 0.50 + 0.30 * ((raw_score - t_pass) / max(t_fail - t_pass, 1e-6))
        else:
            span = max(t_fail - t_pass, 1e-6)
            norm = 0.80 + 0.20 * min(1.0, (raw_score - t_fail) / span)

        return float(np.clip(norm, 0.0, 1.0))

    def evaluate(
        self,
        raw_score: float,
        heatmap: np.ndarray | None = None,
    ) -> DecisionResult:
        """
        Classifies a single image into PASS, REVIEW, or FAIL and computes diagnostics.

        Args:
            raw_score: Scalar image anomaly score from PatchCoreMemoryBank.
            heatmap: Optional 2D (H, W) spatial anomaly map for defect footprint estimation.
        """
        t_pass = self.pass_threshold
        t_fail = self.fail_threshold
        score = float(raw_score)

        if score <= t_pass:
            decision = "PASS"
        elif score <= t_fail:
            decision = "REVIEW"
        else:
            decision = "FAIL"

        z_score = (score - self.val_mean) / self.val_std
        norm_score = self._normalize_score(score)

        defect_area_pct = 0.0
        if heatmap is not None:
            hmap = np.asarray(heatmap, dtype=np.float64)
            loc_thresh = max(t_fail, float(np.max(hmap)) * 0.65) if score > t_pass else t_pass
            defect_area_pct = float(np.mean(hmap > loc_thresh) * 100.0)

        return DecisionResult(
            decision=decision,
            raw_score=round(score, 4),
            normalized_score=round(norm_score, 4),
            z_score=round(float(z_score), 4),
            defect_area_pct=round(defect_area_pct, 4),
            pass_threshold=round(t_pass, 4),
            fail_threshold=round(t_fail, 4),
            category=self.category,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes calibration parameters for storage inside metadata.json."""
        return {
            "category": self.category,
            "pass_std_multiplier": self.pass_std_multiplier,
            "fail_std_multiplier": self.fail_std_multiplier,
            "val_mean": self.val_mean,
            "val_std": self.val_std,
            "pass_threshold": self.pass_threshold if self.is_calibrated else None,
            "fail_threshold": self.fail_threshold if self.is_calibrated else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionEngine":
        """Reconstructs a calibrated DecisionEngine from metadata.json."""
        return cls(
            pass_std_multiplier=data.get("pass_std_multiplier", 2.0),
            fail_std_multiplier=data.get("fail_std_multiplier", 3.5),
            category=data.get("category", "unknown"),
            val_mean=data.get("val_mean"),
            val_std=data.get("val_std"),
        )
