"""
test_contamination.py — Tests for the contamination detection module.
"""

import pytest

from modules.contamination import (
    detect_benchmark_contamination,
    _ngram_fingerprint_check,
    _RISK,
    KNOWN_BENCHMARKS,
)


class TestContaminationDetection:
    """Tests for benchmark contamination detection."""

    def test_contamination_text_basic(self, sample_text: str) -> None:
        """Test contamination detection on normal text."""
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_text)
        result = detect_benchmark_contamination(sample_text, summary)

        assert "contamination_score" in result
        assert "risk_level" in result
        assert "suspicious_benchmarks" in result
        assert "recommendations" in result
        assert 0 <= result["contamination_score"] <= 100

    def test_contamination_benchmark_like_text(self, sample_text_benchmark_like: str) -> None:
        """Test detection on benchmark-like text data."""
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_text_benchmark_like)
        result = detect_benchmark_contamination(sample_text_benchmark_like, summary)

        # Should detect n-gram overlaps
        assert result["ngram_overlap_signals"] > 0

    def test_contamination_tabular(self, sample_csv: str) -> None:
        """Test contamination detection on tabular data."""
        from modules.ingestion import ingest_dataset

        summary = ingest_dataset(sample_csv)
        result = detect_benchmark_contamination(sample_csv, summary)

        assert "contamination_score" in result
        assert "risk_level" in result


class TestNgramFingerprint:
    """Tests for n-gram fingerprint checking."""

    def test_ngram_check_no_matches(self, sample_text: str) -> None:
        """Test n-gram check on clean text."""
        hits = _ngram_fingerprint_check(sample_text)
        assert hits >= 0

    def test_ngram_check_with_matches(self, sample_text_benchmark_like: str) -> None:
        """Test n-gram check on benchmark-like text."""
        hits = _ngram_fingerprint_check(sample_text_benchmark_like)
        # Should detect benchmark n-grams
        assert hits > 0


class TestRiskLevels:
    """Tests for risk level assignment."""

    def test_risk_level_low(self) -> None:
        """Test LOW risk level."""
        assert _RISK.classify(0) == "LOW"
        assert _RISK.classify(10) == "LOW"
        assert _RISK.classify(14.99) == "LOW"

    def test_risk_level_medium(self) -> None:
        """Test MEDIUM risk level."""
        assert _RISK.classify(15) == "MEDIUM"
        assert _RISK.classify(25) == "MEDIUM"
        assert _RISK.classify(39.99) == "MEDIUM"

    def test_risk_level_high(self) -> None:
        """Test HIGH risk level."""
        assert _RISK.classify(40) == "HIGH"
        assert _RISK.classify(55) == "HIGH"
        assert _RISK.classify(69.99) == "HIGH"

    def test_risk_level_critical(self) -> None:
        """Test CRITICAL risk level."""
        assert _RISK.classify(70) == "CRITICAL"
        assert _RISK.classify(100) == "CRITICAL"


class TestKnownBenchmarks:
    """Tests for known benchmark definitions."""

    def test_benchmarks_defined(self) -> None:
        """Verify expected benchmarks are defined."""
        assert "MMLU" in KNOWN_BENCHMARKS
        assert "GSM8K" in KNOWN_BENCHMARKS
        assert "HumanEval" in KNOWN_BENCHMARKS

    def test_benchmarks_have_required_fields(self) -> None:
        """Verify each benchmark has required fields."""
        for name, meta in KNOWN_BENCHMARKS.items():
            assert "avg_length_range" in meta
            assert "line_count_range" in meta
            assert isinstance(meta["avg_length_range"], tuple)
            assert isinstance(meta["line_count_range"], tuple)
