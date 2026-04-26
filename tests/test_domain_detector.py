"""
test_domain_detector.py — Tests for the domain detection module.
"""

import pytest

from modules.domain_detector import (
    detect_domain,
    check_domain_mismatch,
    DOMAIN_SIGNATURES,
    DOMAIN_ALIASES,
)


class TestDomainDetection:
    """Tests for automatic domain detection."""

    def test_detect_domain_healthcare(self, sample_healthcare_csv: str) -> None:
        """Test healthcare domain detection."""
        from modules.profiler import profile_dataset
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_healthcare_csv)
        profile = profile_dataset(sample_healthcare_csv)
        result = detect_domain(profile, summary)

        assert "detected_domain" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
        # Healthcare keywords should boost healthcare domain
        assert result["detected_domain"] == "Healthcare"

    def test_detect_domain_finance(self, sample_finance_csv: str) -> None:
        """Test finance domain detection."""
        from modules.profiler import profile_dataset
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_finance_csv)
        profile = profile_dataset(sample_finance_csv)
        result = detect_domain(profile, summary)

        # Finance keywords should boost finance domain
        assert result["detected_domain"] == "Finance"

    def test_detect_domain_general(self, sample_csv: str) -> None:
        """Test detection on general/unspecific data."""
        from modules.profiler import profile_dataset
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_csv)
        profile = profile_dataset(sample_csv)
        result = detect_domain(profile, summary)

        assert "detected_domain" in result
        assert "ranked" in result
        assert len(result["ranked"]) > 0


class TestDomainMismatch:
    """Tests for domain mismatch detection."""

    def test_no_mismatch_same_domain(self, sample_healthcare_csv: str) -> None:
        """Test no mismatch when domains match."""
        from modules.profiler import profile_dataset
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_healthcare_csv)
        profile = profile_dataset(sample_healthcare_csv)
        detection = detect_domain(profile, summary)

        result = check_domain_mismatch(detection, "Healthcare")

        assert result["mismatch"] is False

    def test_mismatch_different_domain(self, sample_healthcare_csv: str) -> None:
        """Test mismatch when domains differ."""
        from modules.profiler import profile_dataset
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_healthcare_csv)
        profile = profile_dataset(sample_healthcare_csv)
        detection = detect_domain(profile, summary)

        result = check_domain_mismatch(detection, "Finance")

        # Should detect mismatch (healthcare vs finance)
        assert result["mismatch"] is True
        assert "severity" in result
        assert result["severity"] in ["LOW", "MEDIUM", "HIGH"]

    def test_no_mismatch_low_confidence(self, sample_csv: str) -> None:
        """Test no mismatch when detection confidence is low."""
        from modules.profiler import profile_dataset
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_csv)
        profile = profile_dataset(sample_csv)
        detection = detect_domain(profile, summary)

        # Low confidence detection shouldn't trigger mismatch
        if detection["confidence"] < 0.20:
            result = check_domain_mismatch(detection, "Healthcare")
            assert result["mismatch"] is False
            assert "confidence is low" in result["message"].lower()


class TestDomainSignatures:
    """Tests for domain signature definitions."""

    def test_all_domains_have_signatures(self) -> None:
        """Verify all domains have signature definitions."""
        assert "Healthcare" in DOMAIN_SIGNATURES
        assert "Finance" in DOMAIN_SIGNATURES
        assert "Social Media" in DOMAIN_SIGNATURES
        assert "Computer Vision" in DOMAIN_SIGNATURES
        assert "E-Commerce" in DOMAIN_SIGNATURES
        assert "General" in DOMAIN_SIGNATURES

    def test_domain_aliases_complete(self) -> None:
        """Verify all domains have aliases."""
        for domain in DOMAIN_SIGNATURES:
            assert domain in DOMAIN_ALIASES
