"""
contamination.py — Benchmark contamination detector.
Checks dataset overlap with known public test set fingerprints.
Detects train/test leakage that would inflate benchmark scores.
"""

import duckdb
from pathlib import Path

from .utils import RiskScale

_RISK = RiskScale(LOW=15, MEDIUM=40, HIGH=70, default="CRITICAL")


# Known benchmark fingerprints (line count, avg_length, key n-grams)
# In production: replace with actual hashed n-gram indices of benchmark test sets
KNOWN_BENCHMARKS = {
    "MMLU": {"avg_length_range": (80, 200), "line_count_range": (14000, 16000)},
    "HellaSwag": {"avg_length_range": (100, 250), "line_count_range": (9000, 11000)},
    "TruthfulQA": {"avg_length_range": (50, 150), "line_count_range": (700, 900)},
    "GSM8K": {"avg_length_range": (100, 300), "line_count_range": (1300, 1400)},
    "HumanEval": {"avg_length_range": (200, 600), "line_count_range": (150, 175)},
    "SQuAD": {"avg_length_range": (100, 400), "line_count_range": (10000, 12000)},
    "GLUE_SST2": {"avg_length_range": (50, 150), "line_count_range": (67000, 70000)},
    "ImageNet_test": {"avg_length_range": (0, 0), "line_count_range": (0, 0)},  # tabular/image
}


def detect_benchmark_contamination(path: str, summary: dict) -> dict:
    data_type = summary.get("data_type", "tabular")

    if data_type == "text":
        return _check_text_contamination(path, summary)
    elif data_type == "tabular":
        return _check_tabular_contamination(path, summary)
    else:
        return _default("Contamination check not available for this data type")


# ─── Text ─────────────────────────────────────────────────────────────────────

def _check_text_contamination(path: str, summary: dict) -> dict:
    line_count = summary.get("line_count", 0)
    avg_length = summary.get("avg_line_length", 0)

    suspicious = []
    flags = []

    for benchmark, meta in KNOWN_BENCHMARKS.items():
        lc_min, lc_max = meta["line_count_range"]
        al_min, al_max = meta["avg_length_range"]
        if lc_min == 0:
            continue

        lc_overlap = lc_min <= line_count <= lc_max
        al_overlap = al_min <= avg_length <= al_max

        if lc_overlap and al_overlap:
            suspicious.append(benchmark)
            flags.append(
                f"Dataset statistics match {benchmark} test set "
                f"(lines: {line_count} in [{lc_min},{lc_max}], "
                f"avg_len: {avg_length} in [{al_min},{al_max}])"
            )

    # N-gram fingerprint check (lightweight — sample first 1000 lines)
    ngram_hits = _ngram_fingerprint_check(path)

    contamination_score = min(100, len(suspicious) * 30 + ngram_hits * 10)
    risk_level = _RISK.classify(contamination_score)

    recommendations = []
    if suspicious:
        recommendations.append(f"Manually verify dataset does not contain {', '.join(suspicious)} test split")
        recommendations.append("Use dataset-decontamination pipelines (e.g., exact/near-dedup against benchmark sets)")
    if ngram_hits > 0:
        recommendations.append("N-gram overlap detected — run full decontamination before benchmark evaluation")

    return {
        "contamination_score": contamination_score,
        "risk_level": risk_level,
        "suspicious_benchmarks": suspicious,
        "ngram_overlap_signals": ngram_hits,
        "flags": flags,
        "recommendations": recommendations,
        "note": "Full decontamination requires benchmark test set indices (not bundled for legal reasons)",
    }


def _ngram_fingerprint_check(path: str, n: int = 5, sample_lines: int = 1000) -> int:
    """
    Lightweight n-gram check using known high-frequency benchmark phrases.
    Returns count of hits.
    """
    # Known distinctive n-grams from benchmark test sets.
    # Longer and more specific than v1.0 to reduce false positives:
    #   - Removed generic phrases like "return result", "given the context"
    #     which appear in any programming/instructional corpus.
    #   - Added multi-word phrases that are strongly benchmark-specific.
    BENCHMARK_NGRAMS = [
        "the following question asks about",
        "choose the correct answer from the options",
        "which of the following is true about",
        "according to the passage above",
        "what is the capital of",
        "def solution(input_str):",
        "complete the following sentence by choosing",
        "read the passage and answer the question",
        "the answer to the above question is",
    ]

    hits = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break
                lower = line.lower()
                for ngram in BENCHMARK_NGRAMS:
                    if ngram in lower:
                        hits += 1
                        break
    except (OSError, IOError, UnicodeDecodeError) as e:
        # File access issues - return 0 hits (no contamination detected)
        pass

    return hits


# ─── Tabular ─────────────────────────────────────────────────────────────────

def _check_tabular_contamination(path: str, summary: dict) -> dict:
    row_count = summary.get("row_count", 0)
    col_count = summary.get("column_count", 0)

    # Tabular benchmark fingerprint heuristics
    flags = []
    suspicious = []

    # GLUE/SuperGLUE-style tabular datasets
    if 800 <= row_count <= 1100 and col_count in (2, 3):
        suspicious.append("TruthfulQA-style tabular")
        flags.append(f"Row/column count matches TruthfulQA-style benchmark ({row_count} rows, {col_count} cols)")

    contamination_score = min(100, len(suspicious) * 25)
    risk_level = _RISK.classify(contamination_score)

    recommendations = []
    if suspicious:
        recommendations.append("Cross-reference dataset provenance against known benchmarks")

    return {
        "contamination_score": contamination_score,
        "risk_level": risk_level,
        "suspicious_benchmarks": suspicious,
        "flags": flags,
        "recommendations": recommendations,
        "note": "Tabular contamination detection relies on statistical fingerprints only",
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _default(note: str) -> dict:
    return {
        "contamination_score": 0,
        "risk_level": "UNKNOWN",
        "flags": [],
        "recommendations": [],
        "note": note,
    }
