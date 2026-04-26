"""
utils_sampling.py — Auto-scaling sampling utilities for large datasets.

Provides intelligent sample size calculation based on dataset size,
and confidence interval estimation for risk scores.

FIX (stress test):
  - compute_auto_sample_size: when total_rows < MIN_SAMPLE_SIZE, return total_rows
    (use 100% of data, not an inflated MIN_SAMPLE_SIZE you don't have).
    Old: max(MIN_SAMPLE_SIZE, ...) would return 1000 for a 100-row dataset.
    New: if total_rows <= MIN_SAMPLE_SIZE, return total_rows immediately.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from .config import (
    MIN_SAMPLE_SIZE,
    MAX_SAMPLE_SIZE,
    SAMPLE_SCALE_FACTORS,
    CONFIDENCE_BOOTSTRAP_ITERATIONS,
    CONFIDENCE_LEVEL,
)


@dataclass
class ConfidenceInterval:
    """Represents a confidence interval for a risk score."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    standard_error: float

    def to_dict(self) -> dict:
        return {
            "point_estimate": round(self.point_estimate, 4),
            "lower_bound":    round(self.lower_bound, 4),
            "upper_bound":    round(self.upper_bound, 4),
            "confidence_level": self.confidence_level,
            "standard_error": round(self.standard_error, 4),
        }


def compute_auto_sample_size(total_rows: int, requested_size: Optional[int] = None) -> int:
    """Compute optimal sample size based on dataset size.

    Rules:
      - total_rows == 0              → MIN_SAMPLE_SIZE (safe sentinel)
      - total_rows <= MIN_SAMPLE_SIZE → total_rows     (use all, don't inflate)
      - requested_size given         → clamp to [total_rows min, MAX_SAMPLE_SIZE]
      - auto-scale                   → percentage from SAMPLE_SCALE_FACTORS,
                                       clamped to [MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE]
    """
    if total_rows == 0:
        return MIN_SAMPLE_SIZE

    # FIX: tiny datasets → use everything, don't inflate
    if total_rows <= MIN_SAMPLE_SIZE:
        return total_rows

    # User-requested size: respect it, clamp to valid range
    if requested_size is not None:
        return max(MIN_SAMPLE_SIZE, min(requested_size, MAX_SAMPLE_SIZE, total_rows))

    # Auto-scale based on dataset size bracket
    percentage = 1.0
    for (lo, hi), pct in SAMPLE_SCALE_FACTORS.items():
        if lo <= total_rows < hi:
            percentage = pct
            break

    computed = int(total_rows * percentage)
    return max(MIN_SAMPLE_SIZE, min(computed, MAX_SAMPLE_SIZE, total_rows))


def compute_bootstrap_confidence_interval(
    scores: np.ndarray,
    confidence_level: float = CONFIDENCE_LEVEL,
    n_iterations: int = CONFIDENCE_BOOTSTRAP_ITERATIONS,
    rng: Optional[np.random.Generator] = None,
) -> ConfidenceInterval:
    """Bootstrap confidence interval for a set of scores."""
    if rng is None:
        rng = np.random.default_rng()

    scores = np.asarray(scores)
    if len(scores) == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence_level, 0.0)
    if len(scores) == 1:
        return ConfidenceInterval(scores[0], scores[0], scores[0], confidence_level, 0.0)

    point_estimate = float(np.mean(scores))

    bootstrap_means = np.empty(n_iterations)
    for i in range(n_iterations):
        resampled = rng.choice(scores, size=len(scores), replace=True)
        bootstrap_means[i] = np.mean(resampled)

    standard_error = float(np.std(bootstrap_means, ddof=1))

    alpha = 1 - confidence_level
    lower_bound = float(np.percentile(bootstrap_means, (alpha / 2) * 100))
    upper_bound = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))

    return ConfidenceInterval(
        point_estimate=point_estimate,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=confidence_level,
        standard_error=standard_error,
    )


def compute_risk_confidence_interval(
    base_score: float,
    sample_size: int,
    variance_estimate: float = 100.0,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> ConfidenceInterval:
    """Normal-approximation CI for a single risk score given sample size."""
    if sample_size <= 1:
        return ConfidenceInterval(base_score, base_score, base_score, confidence_level, 0.0)

    standard_error = float(np.sqrt(variance_estimate / sample_size))

    from scipy import stats
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    margin = z * standard_error

    return ConfidenceInterval(
        point_estimate=base_score,
        lower_bound=max(0.0, base_score - margin),
        upper_bound=min(100.0, base_score + margin),
        confidence_level=confidence_level,
        standard_error=standard_error,
    )


def add_confidence_intervals_to_risk_scores(
    risk_scores: dict,
    sample_sizes: dict,
) -> dict:
    """Wrap each risk score with its confidence interval."""
    result = {}
    for name, score in risk_scores.items():
        if score is None:
            result[name] = {"value": None, "confidence_interval": None}
            continue
        ci = compute_risk_confidence_interval(
            base_score=float(score),
            sample_size=sample_sizes.get(name, 1000),
        )
        result[name] = {
            "value":               round(float(score), 2),
            "confidence_interval": ci.to_dict(),
        }
    return result
