"""
test_compatibility.py — Tests for the compatibility evaluation module.
"""

import pytest

from modules.compatibility import (
    evaluate_compatibility,
    _recommend_best_model,
    SIZE_REQUIREMENTS,
    VALID_DATA_TYPES,
    ARCH_TASK_FIT,
)
from modules.utils import clamp


class TestCompatibilityEvaluation:
    """Tests for compatibility evaluation."""

    def test_compatibility_tabular_data(self, sample_csv: str, default_ai_config: dict) -> None:
        """Test compatibility for tabular data with default config."""
        from modules.profiler import profile_dataset
        from modules.quality import compute_quality_score

        profile = profile_dataset(sample_csv)
        quality = compute_quality_score(profile)
        result = evaluate_compatibility(profile, quality, default_ai_config)

        assert "compatibility_score" in result
        assert 0 <= result["compatibility_score"] <= 100
        assert "suitability" in result
        assert result["suitability"] in ["SUITABLE", "PARTIALLY_SUITABLE", "NOT_SUITABLE"]
        assert "issues" in result
        assert "recommendations" in result

    def test_compatibility_text_data(self, sample_text: str, default_ai_config: dict) -> None:
        """Test compatibility for text data with NLP config."""
        from modules.profiler import profile_dataset
        from modules.quality import compute_quality_score

        profile = profile_dataset(sample_text, sample_size=1000)
        quality = compute_quality_score(profile)
        result = evaluate_compatibility(profile, quality, default_ai_config)

        assert "compatibility_score" in result

    def test_compatibility_type_mismatch(self, sample_csv: str, vision_ai_config: dict) -> None:
        """Test that type mismatch reduces score."""
        from modules.profiler import profile_dataset
        from modules.quality import compute_quality_score

        profile = profile_dataset(sample_csv)
        quality = compute_quality_score(profile)
        result = evaluate_compatibility(profile, quality, vision_ai_config)

        # Vision model with tabular data should have issues
        assert any("MISMATCH" in issue.upper() for issue in result["issues"])
        assert result["compatibility_score"] < 50

    def test_compatibility_architecture_mismatch(self, sample_csv: str) -> None:
        """Test architecture vs task mismatch."""
        from modules.profiler import profile_dataset
        from modules.quality import compute_quality_score

        # RNN with Detection task (suboptimal)
        config = {
            "ai_type": "NLP",
            "model_size": "Medium",
            "architecture": "RNN",
            "domain": "General",
            "task_type": "Detection",
        }

        profile = profile_dataset(sample_csv)
        quality = compute_quality_score(profile)
        result = evaluate_compatibility(profile, quality, config)

        # Should note architecture mismatch
        assert any("architecture" in issue.lower() for issue in result["issues"])


class TestModelRecommendation:
    """Tests for best model recommendation."""

    def test_recommend_text_model(self) -> None:
        """Test recommendation for text data."""
        result = _recommend_best_model("text", 70.0, 100_000_000, "tokens")
        assert result["ai_type"] == "NLP"
        assert result["architecture"] == "Transformer"

    def test_recommend_tabular_model(self) -> None:
        """Test recommendation for tabular data."""
        result = _recommend_best_model("tabular", 70.0, 100_000, "rows")
        assert result["ai_type"] == "Recommendation"

    def test_recommend_image_model(self) -> None:
        """Test recommendation for image data."""
        result = _recommend_best_model("image", 70.0, 50_000, "images")
        assert result["ai_type"] == "Computer Vision"
        assert result["architecture"] == "CNN"


class TestCompatibilityHelpers:
    """Tests for helper functions."""

    def test_clamp_bounds(self) -> None:
        """Test clamp function bounds."""
        assert clamp(50) == 50
        assert clamp(-10) == 0
        assert clamp(150) == 100


class TestConfigurationConstants:
    """Tests for configuration constants."""

    def test_size_requirements_defined(self) -> None:
        """Verify size requirements are defined."""
        assert "Small" in SIZE_REQUIREMENTS
        assert "Medium" in SIZE_REQUIREMENTS
        assert "Large" in SIZE_REQUIREMENTS

    def test_valid_data_types_defined(self) -> None:
        """Verify valid data types per AI type."""
        assert "NLP" in VALID_DATA_TYPES
        assert "text" in VALID_DATA_TYPES["NLP"]

    def test_arch_task_fit_defined(self) -> None:
        """Verify architecture-task fit mappings."""
        assert "Transformer" in ARCH_TASK_FIT
        assert "CNN" in ARCH_TASK_FIT
