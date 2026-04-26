"""
test_quality.py — Tests for the quality scoring module.
"""

import pytest

from modules.quality import compute_quality_score, _grade
from modules.utils import clamp


class TestQualityScoreTabular:
    """Tests for tabular data quality scoring."""

    def test_quality_basic_csv(self, sample_csv: str) -> None:
        """Test quality score for basic CSV dataset."""
        from modules.profiler import profile_dataset
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_csv)
        profile = profile_dataset(sample_csv)
        result = compute_quality_score(profile)

        assert "total_score" in result
        assert 0 <= result["total_score"] <= 100
        assert "grade" in result
        assert result["grade"] in ["A", "B", "C", "D", "F"]
        assert "dimensions" in result
        assert "issues" in result

    def test_quality_imbalanced_data(self, sample_csv_imbalanced: str) -> None:
        """Test that severe imbalance affects quality score."""
        from modules.profiler import profile_dataset

        profile = profile_dataset(sample_csv_imbalanced)
        result = compute_quality_score(profile)

        # Should detect imbalance in issues
        assert any("imbalance" in issue.lower() or "dominant" in issue.lower() 
                   for issue in result["issues"])

    def test_quality_missing_values(self, sample_csv_missing: str) -> None:
        """Test that missing values affect quality score."""
        from modules.profiler import profile_dataset

        profile = profile_dataset(sample_csv_missing)
        result = compute_quality_score(profile)

        # Should have completeness issues
        assert any("missing" in issue.lower() for issue in result["issues"])

    def test_quality_duplicates(self, sample_csv_duplicates: str) -> None:
        """Test that duplicates affect quality score."""
        from modules.profiler import profile_dataset

        profile = profile_dataset(sample_csv_duplicates)
        result = compute_quality_score(profile)

        # Should detect duplicates
        assert any("duplicate" in issue.lower() for issue in result["issues"])


class TestQualityScoreText:
    """Tests for text data quality scoring."""

    def test_quality_text_basic(self, sample_text: str) -> None:
        """Test quality score for text dataset."""
        from modules.profiler import profile_dataset

        profile = profile_dataset(sample_text, sample_size=1000)
        result = compute_quality_score(profile)

        assert "total_score" in result
        assert result["total_score"] >= 0

    def test_quality_text_small(self, sample_text_small: str) -> None:
        """Test quality score for very small text dataset."""
        from modules.profiler import profile_dataset

        profile = profile_dataset(sample_text_small, sample_size=1000)
        result = compute_quality_score(profile)

        # Small datasets should have structural issues
        assert result["total_score"] < 50


class TestQualityHelpers:
    """Tests for helper functions."""

    def test_clamp_within_bounds(self) -> None:
        """Test clamp keeps values in bounds."""
        assert clamp(50) == 50
        assert clamp(-10) == 0
        assert clamp(150) == 100
        assert clamp(0) == 0
        assert clamp(100) == 100

    def test_grade_boundaries(self) -> None:
        """Test grade assignment at boundaries."""
        assert _grade(85) == "A"
        assert _grade(84.99) == "B"
        assert _grade(70) == "B"
        assert _grade(69.99) == "C"
        assert _grade(55) == "C"
        assert _grade(54.99) == "D"
        assert _grade(40) == "D"
        assert _grade(39.99) == "F"
        assert _grade(0) == "F"
