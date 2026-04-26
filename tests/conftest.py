"""
conftest.py — Shared pytest fixtures for PreTrainAudit tests.
"""

import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest


# ── Sample Data Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_csv() -> Generator[str, None, None]:
    """Create a temporary CSV file with sample tabular data."""
    df = pd.DataFrame(
        {
            "id": range(1, 101),
            "name": [f"user_{i}" for i in range(1, 101)],
            "age": [20 + (i % 50) for i in range(1, 101)],
            "income": [30_000 + (i * 500) for i in range(1, 101)],
            "category": ["A", "B", "C", "A", "B"] * 20,
        }
    )
    # Use UTF-8 with explicit newline for Windows compatibility
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False, encoding="utf-8")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def sample_csv_imbalanced() -> Generator[str, None, None]:
    """Create CSV with severe class imbalance for fairness testing."""
    df = pd.DataFrame(
        {
            "id": range(1, 1001),
            "label": ["positive"] * 900 + ["negative"] * 100,  # 90% positive
            "feature_a": range(1, 1001),
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False, encoding="utf-8")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def sample_csv_missing() -> Generator[str, None, None]:
    """Create CSV with missing values for quality testing."""
    df = pd.DataFrame(
        {
            "id": range(1, 51),
            "col_a": [1, 2, None, 4, 5] * 10,
            "col_b": [None] * 25 + list(range(1, 26)),  # 50% missing
            "col_c": range(1, 51),
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False, encoding="utf-8")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def sample_csv_duplicates() -> Generator[str, None, None]:
    """Create CSV with duplicate rows."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 1, 2, 4, 5, 5, 5],  # duplicates: 1, 2, 5
            "value": [10, 20, 30, 10, 20, 40, 50, 50, 50],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False, encoding="utf-8")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def sample_text() -> Generator[str, None, None]:
    """Create a temporary text file with sample text data."""
    lines = [
        "This is a sample line of text for testing purposes.",
        "Machine learning models require high quality datasets.",
        "Data quality is essential for model performance.",
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is a fascinating field.",
    ] * 20  # 100 lines
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("\n".join(lines))
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def sample_text_small() -> Generator[str, None, None]:
    """Create a very small text file for edge case testing."""
    lines = ["Short line.", "Another line.", "Third line."]
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("\n".join(lines))
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def sample_text_benchmark_like() -> Generator[str, None, None]:
    """Create text that looks like benchmark data for contamination testing."""
    lines = [
        "The following question asks about machine learning concepts.",
        "Choose the correct answer from the options below please.",
        "Which of the following is true about neural networks?",
        "According to the passage above, what is the capital of France?",
        "Read the passage and answer the question that follows.",
    ] * 50
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("\n".join(lines))
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def empty_csv() -> Generator[str, None, None]:
    """Create an empty CSV file (headers only)."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8", newline="") as f:
        f.write("id,name,value\n")  # Headers only
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def sample_parquet() -> Generator[str, None, None]:
    """Create a temporary Parquet file."""
    df = pd.DataFrame(
        {
            "id": range(1, 1001),
            "feature_a": [i * 0.5 for i in range(1, 1001)],
            "feature_b": [i % 10 for i in range(1, 1001)],
            "label": ["A", "B"] * 500,
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f, index=False)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


# ── Healthcare Domain Fixture ────────────────────────────────────────────────


@pytest.fixture
def sample_healthcare_csv() -> Generator[str, None, None]:
    """Create CSV with healthcare domain columns."""
    df = pd.DataFrame(
        {
            "patient_id": range(1, 101),
            "age": [30 + (i % 40) for i in range(1, 101)],
            "diagnosis": ["diabetes", "hypertension", "asthma", "healthy"] * 25,
            "blood_pressure_systolic": [120 + (i % 20) for i in range(1, 101)],
            "medication": ["metformin", "lisinopril", "albuterol", "none"] * 25,
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False, encoding="utf-8")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


# ── Finance Domain Fixture ────────────────────────────────────────────────────


@pytest.fixture
def sample_finance_csv() -> Generator[str, None, None]:
    """Create CSV with finance domain columns."""
    df = pd.DataFrame(
        {
            "transaction_id": range(1, 101),
            "amount": [100.0 + i * 10 for i in range(1, 101)],
            "merchant": [f"merchant_{i % 10}" for i in range(1, 101)],
            "is_fraud": [0, 0, 0, 0, 1] * 20,  # 20% fraud
            "card_type": ["visa", "mastercard", "amex"] * 33 + ["visa"],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False, encoding="utf-8")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


# ── AI Config Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def default_ai_config() -> dict:
    """Default AI configuration for testing."""
    return {
        "ai_type": "NLP",
        "model_size": "Medium",
        "architecture": "Transformer",
        "domain": "General",
        "task_type": "Classification",
        "param_count_millions": 125.0,
    }


@pytest.fixture
def large_ai_config() -> dict:
    """Large model AI configuration."""
    return {
        "ai_type": "NLP",
        "model_size": "Large",
        "architecture": "Transformer",
        "domain": "General",
        "task_type": "Generation",
        "param_count_millions": 7000.0,
    }


@pytest.fixture
def vision_ai_config() -> dict:
    """Computer vision AI configuration."""
    return {
        "ai_type": "Computer Vision",
        "model_size": "Medium",
        "architecture": "CNN",
        "domain": "General",
        "task_type": "Classification",
        "param_count_millions": 50.0,
    }
