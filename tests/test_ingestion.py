"""
test_ingestion.py — Tests for the ingestion module.
"""

import pytest

from modules.ingestion import ingest_dataset, SUPPORTED_TABULAR, SUPPORTED_TEXT


class TestIngestTabular:
    """Tests for tabular data ingestion."""

    def test_ingest_csv_basic(self, sample_csv: str) -> None:
        """Test basic CSV ingestion returns correct structure."""
        result = ingest_dataset(sample_csv)

        assert result["data_type"] == "tabular"
        assert result["format"] == ".csv"
        assert result["row_count"] == 100
        assert result["column_count"] == 5
        assert "columns" in result
        assert len(result["sample_rows"]) == 5

    def test_ingest_csv_empty(self, empty_csv: str) -> None:
        """Test ingestion of empty CSV (headers only)."""
        result = ingest_dataset(empty_csv)

        assert result["data_type"] == "tabular"
        assert result["row_count"] == 0
        assert "warning" in result

    def test_ingest_parquet(self, sample_parquet: str) -> None:
        """Test Parquet ingestion."""
        result = ingest_dataset(sample_parquet)

        assert result["data_type"] == "tabular"
        assert result["format"] == ".parquet"
        assert result["row_count"] == 1000

    def test_unsupported_format(self, tmp_path) -> None:
        """Test that unsupported formats raise ValueError."""
        unsupported_file = tmp_path / "data.xyz"
        unsupported_file.write_text("some data")

        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_dataset(str(unsupported_file))


class TestIngestText:
    """Tests for text data ingestion."""

    def test_ingest_text_basic(self, sample_text: str) -> None:
        """Test basic text file ingestion."""
        result = ingest_dataset(sample_text)

        assert result["data_type"] == "text"
        assert result["format"] == ".txt"
        assert result["line_count"] == 100
        assert "char_count" in result
        assert "estimated_tokens" in result
        assert len(result["sample_lines"]) == 5

    def test_ingest_text_small(self, sample_text_small: str) -> None:
        """Test small text file ingestion."""
        result = ingest_dataset(sample_text_small)

        assert result["data_type"] == "text"
        assert result["line_count"] == 3


class TestSupportedFormats:
    """Tests for supported format detection."""

    def test_supported_tabular_formats(self) -> None:
        """Verify expected tabular formats."""
        assert ".csv" in SUPPORTED_TABULAR
        assert ".parquet" in SUPPORTED_TABULAR
        assert ".xlsx" in SUPPORTED_TABULAR

    def test_supported_text_formats(self) -> None:
        """Verify expected text formats."""
        assert ".txt" in SUPPORTED_TEXT
        assert ".jsonl" in SUPPORTED_TEXT
        assert ".json" in SUPPORTED_TEXT
