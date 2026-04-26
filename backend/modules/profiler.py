"""
profiler.py — Chunked data profiling engine.

FIX HISTORY:
  - Removed all_varchar=true for CSV profiling. Ingestion uses all_varchar
    for safe string previews, but the profiler MUST let DuckDB auto-detect
    types so numeric columns are properly identified for statistical analysis.
  - Fixed double-close of DuckDB connection (except+finally both closing).
  - Replaced local _safe_path with shared safe_sql_path from utils.
  - TABLESAMPLE percentage clamped to minimum 0.01% to prevent 0% edge case.
  - All-null columns skipped in distribution analysis.
  - Empty dataframe guard after sampling.
  - Correlation matrix only computed with >=2 cols with non-zero variance.
  - std/mean default to 0.0 instead of NaN propagation.
  - fillna(0) on correlation matrix before tolist() to prevent JSON NaN.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

from .utils import safe_sql_path


def profile_dataset(path: str, sample_size: int = 100_000) -> dict:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return _profile_tabular_duckdb(path, "csv", sample_size)
    elif ext == ".parquet":
        return _profile_tabular_duckdb(path, "parquet", sample_size)
    elif ext in (".txt", ".jsonl", ".json"):
        return _profile_text(path, sample_size)
    elif ext in (".xlsx", ".xls"):
        return _profile_excel(path, sample_size)
    else:
        raise ValueError(f"Unsupported format for profiling: {ext}")


# ─── Tabular ──────────────────────────────────────────────────────────────────

def _profile_tabular_duckdb(path: str, fmt: str, sample_size: int) -> dict:
    safe = safe_sql_path(path)
    con  = duckdb.connect()

    # CSV: auto-detect types (NOT all_varchar) so numeric columns are identified.
    # Ingestion uses all_varchar for safe string previews; the profiler needs real types.
    reader = (f"read_csv_auto('{safe}', header=true, ignore_errors=true)"
              if fmt == "csv" else f"read_parquet('{safe}')")

    try:
        total_rows = con.execute(f"SELECT COUNT(*) FROM {reader}").fetchone()[0]

        if total_rows == 0:
            return {
                "data_type": "tabular", "total_rows": 0, "sampled_rows": 0,
                "column_count": 0, "numeric_columns": [], "categorical_columns": [],
                "missing_values": {}, "missing_pct": {}, "duplicate_count": 0,
                "duplicate_pct": 0.0, "distributions": {}, "correlation": None,
                "class_balance": {}, "descriptive_stats": {},
                "warning": "Dataset is empty — no rows to profile",
            }

        # Clamp sample fraction: min 0.01%, max 100%
        sample_frac_pct = max(0.01, min(100.0, (sample_size / total_rows) * 100))

        if sample_frac_pct >= 100.0:
            sample_q = f"SELECT * FROM {reader}"
        else:
            sample_q = f"SELECT * FROM {reader} USING SAMPLE {round(sample_frac_pct, 4)} PERCENT (bernoulli)"

        df: pd.DataFrame = con.execute(sample_q).df()
    except Exception as e:
        raise ValueError(f"Profiling failed: {e}") from e
    finally:
        # Single close point — no double-close regardless of exception path
        con.close()

    if df.empty:
        return {
            "data_type": "tabular", "total_rows": total_rows, "sampled_rows": 0,
            "warning": "Sampled dataframe is empty — try increasing sample_size",
        }

    # Skip all-null columns from numeric analysis
    numeric_cols     = [c for c in df.select_dtypes(include=[np.number]).columns
                        if df[c].notna().sum() > 0]
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing     = df.isnull().sum().to_dict()
    missing_pct = {k: round(v / max(len(df), 1) * 100, 2) for k, v in missing.items()}
    dup_count   = int(df.duplicated().sum())
    dup_pct     = round(dup_count / max(len(df), 1) * 100, 2)

    # Distributions — skip zero-range and tiny-sample columns
    distributions = {}
    for col in numeric_cols[:10]:
        hist_data = df[col].dropna()
        if len(hist_data) < 2:
            continue
        std_val  = float(hist_data.std())  if not np.isnan(hist_data.std())  else 0.0
        mean_val = float(hist_data.mean()) if not np.isnan(hist_data.mean()) else 0.0
        counts, bins = np.histogram(hist_data, bins=30)
        distributions[col] = {
            "type":   "histogram",
            "bins":   bins.tolist(),
            "counts": counts.tolist(),
            "mean":   mean_val,
            "std":    std_val,
            "min":    float(hist_data.min()),
            "max":    float(hist_data.max()),
            "median": float(hist_data.median()),
        }

    # Correlation — >=2 non-zero-variance columns only
    correlation = None
    valid_num = [c for c in numeric_cols if df[c].std() > 0]
    if len(valid_num) >= 2:
        try:
            corr = df[valid_num].corr().round(3).fillna(0)
            correlation = {
                "columns": valid_num,
                "matrix":  corr.values.tolist(),
            }
        except Exception:
            pass

    # Class balance
    balance = {}
    for col in categorical_cols[:5]:
        vc = df[col].dropna().value_counts(normalize=True).head(20)
        if not vc.empty:
            balance[col] = {
                "labels":      vc.index.astype(str).tolist(),
                "proportions": vc.values.tolist(),
            }

    # Descriptive stats
    try:
        desc = df.describe(include="all").fillna("").astype(str).to_dict()
    except Exception:
        desc = {}

    return {
        "data_type":           "tabular",
        "total_rows":          total_rows,
        "sampled_rows":        len(df),
        "column_count":        len(df.columns),
        "numeric_columns":     numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values":      missing,
        "missing_pct":         missing_pct,
        "duplicate_count":     dup_count,
        "duplicate_pct":       dup_pct,
        "distributions":       distributions,
        "correlation":         correlation,
        "class_balance":       balance,
        "descriptive_stats":   desc,
    }


# ─── Text ──────────────────────────────────────────────────────────────────────

def _profile_text(path: str, sample_size: int) -> dict:
    lines        = []
    line_lengths = []
    word_counts  = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            lines.append(stripped)
            line_lengths.append(len(stripped))
            word_counts.append(len(stripped.split()))
            if i >= sample_size:
                break

    if not lines:
        return {
            "data_type": "text", "sampled_lines": 0, "unique_lines": 0,
            "duplicate_lines": 0, "duplicate_pct": 0, "avg_line_length": 0,
            "avg_word_count": 0, "max_line_length": 0, "min_line_length": 0,
            "vocab_size_estimate": 0, "type_token_ratio": 0,
            "length_distribution": {},
            "warning": "No non-empty lines found",
        }

    total_lines  = len(lines)
    unique_lines = len(set(lines))
    dup_count    = total_lines - unique_lines
    dup_pct      = round(dup_count / max(total_lines, 1) * 100, 2)

    len_arr  = np.array(line_lengths)
    word_arr = np.array(word_counts)
    counts, bins = np.histogram(len_arr, bins=30)

    all_words  = " ".join(lines[:10_000]).lower().split()
    vocab_size = len(set(all_words))
    ttr        = round(vocab_size / max(len(all_words), 1), 4)

    return {
        "data_type":           "text",
        "sampled_lines":       total_lines,
        "unique_lines":        unique_lines,
        "duplicate_lines":     dup_count,
        "duplicate_pct":       dup_pct,
        "avg_line_length":     round(float(len_arr.mean()), 1),
        "avg_word_count":      round(float(word_arr.mean()), 1),
        "max_line_length":     int(len_arr.max()),
        "min_line_length":     int(len_arr.min()),
        "vocab_size_estimate": vocab_size,
        "type_token_ratio":    ttr,
        "length_distribution": {"bins": bins.tolist(), "counts": counts.tolist()},
    }


# ─── Excel ────────────────────────────────────────────────────────────────────

def _profile_excel(path: str, sample_size: int) -> dict:
    df  = pd.read_excel(path, nrows=sample_size)
    tmp = path + "_tmp_profile.csv"
    try:
        df.to_csv(tmp, index=False)
        return _profile_tabular_duckdb(tmp, "csv", sample_size)
    finally:
        import os
        if os.path.exists(tmp):
            os.remove(tmp)
