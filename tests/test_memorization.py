"""
test_memorization.py — Tests for the memorization risk module.
"""

import pytest

from modules.memorization import compute_memorization_risk, RISK_THRESHOLDS


class TestMemorizationRisk:
    """Tests for memorization risk computation."""

    def test_memorization_text_data(self, sample_text: str) -> None:
        """Test memorization risk for text data."""
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_text)
        result = compute_memorization_risk(summary, param_count_millions=125.0)

        assert "risk_level" in result
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL", "MINIMAL", "UNKNOWN"]
        assert "token_count_estimate" in result
        assert "token_to_param_ratio" in result
        assert "memorization_probability" in result
        assert "privacy_leakage_score" in result

    def test_memorization_tabular_data(self, sample_csv: str) -> None:
        """Test memorization risk for tabular data."""
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_csv)
        result = compute_memorization_risk(summary, param_count_millions=10.0)

        assert "risk_level" in result
        assert result["token_count_estimate"] == 100 * 5  # rows * cols

    def test_memorization_large_model_high_risk(self, sample_text_small: str) -> None:
        """Test that large model with small data = high risk."""
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_text_small)
        result = compute_memorization_risk(summary, param_count_millions=7000.0)

        # 7B params with tiny dataset = critical/high risk
        assert result["risk_level"] in ["CRITICAL", "HIGH"]

    def test_memorization_small_model_lower_risk(self, sample_csv: str) -> None:
        """Test that smaller model reduces risk level."""
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_csv)

        # Small model (10M params)
        result_small = compute_memorization_risk(summary, param_count_millions=10.0)

        # Large model (7B params)
        result_large = compute_memorization_risk(summary, param_count_millions=7000.0)

        # Larger model should have lower token-to-param ratio
        assert result_small["token_to_param_ratio"] > result_large["token_to_param_ratio"]

    def test_memorization_gdpr_flag(self, sample_text: str) -> None:
        """Test GDPR flag is set for high risk scenarios."""
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_text)
        result = compute_memorization_risk(summary, param_count_millions=7000.0)

        # Check GDPR flag exists
        assert "gdpr_compliance_flag" in result
        assert isinstance(result["gdpr_compliance_flag"], bool)

    def test_memorization_issues_and_recommendations(self, sample_text: str) -> None:
        """Test that issues and recommendations are populated."""
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_text)
        result = compute_memorization_risk(summary, param_count_millions=7000.0)

        assert "issues" in result
        assert "recommendations" in result
        assert isinstance(result["issues"], list)
        assert isinstance(result["recommendations"], list)


class TestRiskThresholds:
    """Tests for risk threshold definitions."""

    def test_thresholds_ordered(self) -> None:
        """Verify thresholds are in ascending order."""
        thresholds = [t[0] for t in RISK_THRESHOLDS]
        assert thresholds == sorted(thresholds)

    def test_thresholds_coverage(self) -> None:
        """Verify thresholds cover all risk levels."""
        levels = [t[1] for t in RISK_THRESHOLDS]
        assert "CRITICAL" in levels
        assert "HIGH" in levels
        assert "MEDIUM" in levels
        assert "LOW" in levels
