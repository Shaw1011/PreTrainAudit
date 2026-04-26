"""
test_utils_sampling.py — Tests for sampling utilities.
"""

import pytest
import numpy as np

from modules.utils_sampling import (
    compute_auto_sample_size,
    compute_bootstrap_confidence_interval,
    compute_risk_confidence_interval,
    add_confidence_intervals_to_risk_scores,
    ConfidenceInterval,
)


class TestAutoSampleSize:
    """Tests for auto-scaling sample size calculation."""

    def test_tiny_dataset_full_sample(self) -> None:
        """Tiny datasets should sample 100%."""
        size = compute_auto_sample_size(5_000)
        assert size == 5_000  # All rows

    def test_small_dataset_half_sample(self) -> None:
        """Small datasets should sample 50%."""
        size = compute_auto_sample_size(50_000)
        assert size == 25_000  # 50%

    def test_medium_dataset_twenty_percent(self) -> None:
        """Medium datasets should sample 20%."""
        size = compute_auto_sample_size(500_000)
        assert size == 100_000  # 20%

    def test_large_dataset_five_percent(self) -> None:
        """Large datasets should sample 5%."""
        size = compute_auto_sample_size(5_000_000)
        assert size == 250_000  # 5%

    def test_huge_dataset_one_percent(self) -> None:
        """Huge datasets should sample 1%."""
        size = compute_auto_sample_size(100_000_000)
        assert size == 1_000_000  # 1% capped at max

    def test_respects_user_requested_size(self) -> None:
        """User-requested size should be respected within bounds."""
        # User wants 50k from 1M dataset
        size = compute_auto_sample_size(1_000_000, requested_size=50_000)
        assert size == 50_000

    def test_enforces_minimum(self) -> None:
        """Very small requested size should hit minimum."""
        size = compute_auto_sample_size(1_000_000, requested_size=100)
        assert size >= 1_000  # MIN_SAMPLE_SIZE

    def test_zero_rows_returns_minimum(self) -> None:
        """Empty dataset should return minimum sample size."""
        size = compute_auto_sample_size(0)
        assert size == 1_000


class TestConfidenceInterval:
    """Tests for confidence interval computation."""

    def test_confidence_interval_basic(self) -> None:
        """Test basic confidence interval computation."""
        scores = np.array([10, 20, 30, 40, 50])
        ci = compute_bootstrap_confidence_interval(scores, n_iterations=50)
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.point_estimate == 30.0  # Mean
        assert ci.lower_bound <= ci.point_estimate
        assert ci.upper_bound >= ci.point_estimate
        assert ci.confidence_level == 0.95

    def test_confidence_interval_single_value(self) -> None:
        """Single value should have zero-width interval."""
        scores = np.array([50.0])
        ci = compute_bootstrap_confidence_interval(scores)
        
        assert ci.lower_bound == 50.0
        assert ci.upper_bound == 50.0
        assert ci.standard_error == 0.0

    def test_confidence_interval_empty(self) -> None:
        """Empty array should return zeros."""
        scores = np.array([])
        ci = compute_bootstrap_confidence_interval(scores)
        
        assert ci.point_estimate == 0.0
        assert ci.lower_bound == 0.0
        assert ci.upper_bound == 0.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        ci = ConfidenceInterval(
            point_estimate=50.0,
            lower_bound=45.0,
            upper_bound=55.0,
            confidence_level=0.95,
            standard_error=2.5,
        )
        d = ci.to_dict()
        
        assert d["point_estimate"] == 50.0
        assert d["lower_bound"] == 45.0
        assert d["upper_bound"] == 55.0


class TestRiskConfidenceInterval:
    """Tests for risk score confidence interval computation."""

    def test_risk_ci_basic(self) -> None:
        """Test risk CI with normal approximation."""
        ci = compute_risk_confidence_interval(
            base_score=50.0,
            sample_size=10_000,
        )
        
        assert ci.point_estimate == 50.0
        assert ci.lower_bound < 50.0
        assert ci.upper_bound > 50.0
        assert ci.lower_bound >= 0.0
        assert ci.upper_bound <= 100.0

    def test_risk_ci_small_sample(self) -> None:
        """Small sample should have wider interval."""
        ci_small = compute_risk_confidence_interval(50.0, sample_size=100)
        ci_large = compute_risk_confidence_interval(50.0, sample_size=10_000)
        
        # Smaller sample = larger standard error = wider interval
        assert ci_small.standard_error > ci_large.standard_error

    def test_risk_ci_single_sample(self) -> None:
        """Single sample should have exact bounds."""
        ci = compute_risk_confidence_interval(50.0, sample_size=1)
        
        assert ci.lower_bound == 50.0
        assert ci.upper_bound == 50.0


class TestAddConfidenceIntervals:
    """Tests for adding CIs to risk score dicts."""

    def test_adds_intervals_to_all_scores(self) -> None:
        """Should add CI to each score."""
        risk_scores = {
            "adversarial": 30.0,
            "memorization": 45.0,
            "fairness": 20.0,
        }
        sample_sizes = {
            "adversarial": 5_000,
            "memorization": 10_000,
            "fairness": 5_000,
        }
        
        result = add_confidence_intervals_to_risk_scores(risk_scores, sample_sizes)
        
        assert "adversarial" in result
        assert "value" in result["adversarial"]
        assert "confidence_interval" in result["adversarial"]

    def test_handles_none_scores(self) -> None:
        """Should handle None scores gracefully."""
        risk_scores = {"adversarial": None, "memorization": 50.0}
        sample_sizes = {"adversarial": 1000, "memorization": 1000}
        
        result = add_confidence_intervals_to_risk_scores(risk_scores, sample_sizes)
        
        assert result["adversarial"]["value"] is None
        assert result["adversarial"]["confidence_interval"] is None

    def test_uses_default_sample_size(self) -> None:
        """Should use default sample size if not provided."""
        result = add_confidence_intervals_to_risk_scores(
            {"test": 50.0},
            {},  # No sample sizes
        )
        
        # Should still produce a CI
        assert result["test"]["confidence_interval"] is not None
