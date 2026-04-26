"""
ingestion.py — Dataset ingestion and type detection.

FIXES:
  - _safe_path uses str.replace only (no Path.resolve) — Path.resolve on Windows
    paths run from Linux gives wrong results; simple replace is cross-platform safe
    since all upload paths are already absolute
  - Empty file detection before DuckDB queries
  - try/except around all DuckDB queries with clear error messages
  - fillna("") before to_dict prevents NaN serialization errors in JSON
"""

import os
import duckdb
import pandas as pd
from pathlib import Path


SUPPORTED_TABULAR = {".csv", ".parquet", ".arrow", ".xlsx", ".xls"}
SUPPORTED_TEXT    = {".txt", ".jsonl", ".json"}


def _safe_path(path: str) -> str:
    """
    Convert Windows backslashes to forward slashes for DuckDB SQL strings.
    Uses simple replace — NOT Path.resolve() — because the path is already
    absolute (from FastAPI upload handler) and resolve() behaves differently
    per OS when given a path string from a different OS.
    
    SECURITY: Escapes single quotes to prevent SQL injection via file paths.
    """
    # Convert backslashes to forward slashes
    safe = path.replace("\\", "/")
    # Escape single quotes to prevent SQL injection
    safe = safe.replace("'", "''")
    return safe


def ingest_dataset(path: str) -> dict:
    ext             = Path(path).suffix.lower()
    file_size_bytes = os.path.getsize(path)
    file_size_mb    = round(file_size_bytes / (1024 ** 2), 2)

    if file_size_bytes == 0:
        raise ValueError("Uploaded file is empty (0 bytes)")

    if ext in SUPPORTED_TABULAR:
        return _ingest_tabular(path, ext, file_size_mb)
    elif ext in SUPPORTED_TEXT:
        return _ingest_text(path, ext, file_size_mb)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {sorted(SUPPORTED_TABULAR | SUPPORTED_TEXT)}"
        )


def get_dataset_summary(path: str) -> dict:
    return ingest_dataset(path)


# ─── Tabular ──────────────────────────────────────────────────────────────────

def _ingest_tabular(path: str, ext: str, size_mb: float) -> dict:
    if ext in (".xlsx", ".xls"):
        return _ingest_excel(path, size_mb)

    safe = _safe_path(path)
    con  = duckdb.connect()

    try:
        if ext == ".csv":
            # Explicit CSV options for cross-platform compatibility
            reader = f"read_csv_auto('{safe}', header=true, ignore_errors=true, all_varchar=true)"
        elif ext == ".parquet":
            reader = f"read_parquet('{safe}')"
        else:
            reader = f"'{safe}'"

        row_count = con.execute(f"SELECT COUNT(*) FROM {reader}").fetchone()[0]

        if row_count == 0:
            col_info = con.execute(f"DESCRIBE SELECT * FROM {reader}").df()
            columns  = col_info["column_name"].tolist() if "column_name" in col_info.columns else []
            return {
                "data_type": "tabular", "format": ext, "file_size_mb": size_mb,
                "row_count": 0, "column_count": len(columns), "columns": columns,
                "dtypes": {}, "sample_rows": [], "engine": "duckdb",
                "warning": "Dataset has no data rows (headers only)",
            }

        sample   = con.execute(f"SELECT * FROM {reader} LIMIT 5").df()
        col_info = con.execute(f"DESCRIBE SELECT * FROM {reader}").df()

        columns = col_info["column_name"].tolist() if "column_name" in col_info.columns else sample.columns.tolist()
        dtypes  = (dict(zip(col_info["column_name"], col_info["column_type"]))
                   if "column_type" in col_info.columns else {})

        return {
            "data_type":    "tabular",
            "format":        ext,
            "file_size_mb":  size_mb,
            "row_count":     row_count,
            "column_count":  len(columns),
            "columns":       columns,
            "dtypes":        dtypes,
            "sample_rows":   sample.head(5).fillna("").astype(str).to_dict(orient="records"),
            "engine":        "duckdb",
        }
    except Exception as e:
        raise ValueError(f"Failed to read {ext} file: {e}") from e
    finally:
        con.close()


def _ingest_excel(path: str, size_mb: float) -> dict:
    try:
        full_df = pd.read_excel(path)
        return {
            "data_type":    "tabular",
            "format":        ".xlsx",
            "file_size_mb":  size_mb,
            "row_count":     len(full_df),
            "column_count":  len(full_df.columns),
            "columns":       list(full_df.columns),
            "dtypes":        {col: str(dt) for col, dt in full_df.dtypes.items()},
            "sample_rows":   full_df.head(5).fillna("").astype(str).to_dict(orient="records"),
            "engine":        "pandas",
        }
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}") from e


# ─── Text ──────────────────────────────────────────────────────────────────────

def _ingest_text(path: str, ext: str, size_mb: float) -> dict:
    line_count   = 0
    char_count   = 0
    sample_lines = []

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_count += 1
                char_count += len(line)
                if line_count <= 5:
                    sample_lines.append(line.strip())
    except Exception as e:
        raise ValueError(f"Failed to read text file: {e}") from e

    if line_count == 0:
        raise ValueError("Text file is empty — no lines found")

    return {
        "data_type":       "text",
        "format":           ext,
        "file_size_mb":     size_mb,
        "line_count":       line_count,
        "char_count":       char_count,
        "avg_line_length":  round(char_count / line_count, 1),
        "estimated_tokens": int(char_count / 4),
        "sample_lines":     sample_lines,
        "engine":           "streaming",
    }
