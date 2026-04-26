"""
multi_domain_analyzer.py — Mixed dataset segmentation and per-group analysis.
"""

from __future__ import annotations
import math
from typing import Optional

import duckdb
import pandas as pd
import numpy as np

from .domain_detector import DOMAIN_SIGNATURES, detect_domain
from .quality import compute_quality_score
from .utils import clamp, safe_sql_path


def _assign_columns_to_domains(columns: list[str]) -> dict[str, list[str]]:
    import re
    assignments: dict[str, list[str]] = {d: [] for d in DOMAIN_SIGNATURES if d != "General"}
    assignments["General"] = []

    for col in columns:
        col_lower = col.lower()
        col_tokens = set(re.split(r"[_\-\s\.]+", col_lower))
        best_domain = "General"
        best_score = 0.0
        for domain, sigs in DOMAIN_SIGNATURES.items():
            if domain == "General":
                continue
            for col_kws, _, weight in sigs:
                exact   = len(col_tokens & col_kws)
                partial = sum(1 for kw in col_kws if kw in col_lower)
                score   = weight * (exact * 2.0 + partial * 0.8)
                if score > best_score:
                    best_score, best_domain = score, domain
        assignments[best_domain].append(col)

    return {d: cols for d, cols in assignments.items() if cols}


def _profile_column_group(df: pd.DataFrame, cols: list[str], domain: str, total_rows: int) -> dict:
    sub = df[cols].copy()
    numeric_cols     = [c for c in sub.select_dtypes(include=[np.number]).columns if sub[c].notna().sum() > 0]
    categorical_cols = sub.select_dtypes(exclude=[np.number]).columns.tolist()

    missing_counts = sub.isnull().sum().to_dict()
    missing_pct    = {k: round(v / max(len(sub), 1) * 100, 6) for k, v in missing_counts.items()}
    avg_missing    = round(sum(missing_pct.values()) / max(len(missing_pct), 1), 6)
    dup_count      = int(sub.duplicated().sum())
    dup_pct        = round(dup_count / max(len(sub), 1) * 100, 6)
    total_cells    = len(sub) * len(cols)
    null_cells     = sub.isnull().sum().sum()
    coverage       = round((total_cells - null_cells) / max(total_cells, 1), 6)

    distributions: dict = {}
    for col in numeric_cols[:8]:
        series = sub[col].dropna()
        if len(series) < 2:
            continue
        counts, bins = np.histogram(series, bins=30)
        distributions[col] = {
            "mean":       round(float(series.mean()), 6),
            "std":        round(float(series.std()), 6),
            "min":        round(float(series.min()), 6),
            "max":        round(float(series.max()), 6),
            "median":     round(float(series.median()), 6),
            "skew":       round(float(series.skew()), 6),
            "null_count": int(sub[col].isnull().sum()),
            "bins":       bins.tolist(),
            "counts":     counts.tolist(),
        }

    class_balance: dict = {}
    for col in categorical_cols[:5]:
        vc = sub[col].dropna().value_counts(normalize=True).head(20)
        if not vc.empty:
            class_balance[col] = {
                "labels":      vc.index.astype(str).tolist(),
                "proportions": [round(v, 6) for v in vc.values.tolist()],
                "n_classes":   int(vc.shape[0]),
                "entropy":     round(float(-sum(p * math.log2(p) for p in vc.values if p > 0)), 6),
                "gini":        round(float(1 - sum(p**2 for p in vc.values)), 6),
            }

    return {
        "data_type":           "tabular",
        "domain":              domain,
        "columns":             cols,
        "column_count":        len(cols),
        "total_rows":          total_rows,
        "sampled_rows":        len(sub),
        "numeric_columns":     numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values":      missing_counts,
        "missing_pct":         missing_pct,
        "avg_missing_pct":     avg_missing,
        "duplicate_count":     dup_count,
        "duplicate_pct":       dup_pct,
        "data_coverage":       coverage,
        "distributions":       distributions,
        "class_balance":       class_balance,
    }


def _compute_group_sufficiency(total_rows: int, n_columns: int, domain: str) -> dict:
    THRESHOLDS = {
        "Healthcare": 50_000, "Finance": 100_000, "Social Media": 500_000,
        "Computer Vision": 10_000, "Music/Audio": 10_000, "E-Commerce": 50_000,
        "NLP/Text": 100_000, "General": 10_000,
    }
    threshold = THRESHOLDS.get(domain, 10_000)
    ratio     = round(total_rows / max(threshold, 1), 6)
    pct       = round(ratio * 100, 4)

    if ratio >= 2.0:   verdict, note = "ABUNDANT",               f"{total_rows:,} rows — well above {threshold:,}"
    elif ratio >= 1.0: verdict, note = "SUFFICIENT",             f"{total_rows:,} rows meets {threshold:,} recommended"
    elif ratio >= 0.5: verdict, note = "MARGINAL",               f"{total_rows:,} is {pct:.2f}% of {threshold:,} recommended"
    elif ratio >= 0.1: verdict, note = "INSUFFICIENT",           f"Only {pct:.2f}% of {threshold:,} recommended"
    else:              verdict, note = "CRITICALLY_INSUFFICIENT", f"Far below minimum — {total_rows:,} vs {threshold:,} needed"

    return {
        "verdict":              verdict,
        "total_rows":           total_rows,
        "recommended_rows":     threshold,
        "ratio":                ratio,
        "pct_of_recommended":   pct,
        "note":                 note,
        "column_count":         n_columns,
        "effective_data_points": total_rows * n_columns,
    }


def analyze_multi_domain(path: str, summary: dict, sample_size: int = 200_000) -> dict:
    from pathlib import Path
    ext       = Path(path).suffix.lower()
    safe_p    = safe_sql_path(path)
    con       = duckdb.connect()
    total_rows = summary.get("row_count", 0) or 0

    sample_frac_pct = max(0.01, min(100.0, (sample_size / max(total_rows, 1)) * 100))

    # Explicit CSV options for cross-platform compatibility
    if ext == ".csv":
        reader = f"read_csv_auto('{safe_p}', header=true, ignore_errors=true, all_varchar=true)"
    elif ext == ".parquet":
        reader = f"read_parquet('{safe_p}')"
    else:
        con.close()
        return {"error": f"Multi-domain analysis supports CSV and Parquet only. Got: {ext}"}

    q_str = (f"SELECT * FROM {reader}" if sample_frac_pct >= 100.0
             else f"SELECT * FROM {reader} USING SAMPLE {round(sample_frac_pct, 4)} PERCENT (bernoulli)")

    try:
        df: pd.DataFrame = con.execute(q_str).df()
    except Exception as e:
        con.close()
        return {"error": f"Failed to load dataset: {e}"}
    finally:
        con.close()

    if df.empty:
        return {"error": "Dataset is empty after sampling"}

    all_columns  = df.columns.tolist()
    assignments  = _assign_columns_to_domains(all_columns)
    group_reports: dict[str, dict] = {}

    for domain, cols in assignments.items():
        if not cols:
            continue
        group_profile     = _profile_column_group(df, cols, domain, total_rows)
        group_quality     = compute_quality_score(group_profile)
        group_sufficiency = _compute_group_sufficiency(total_rows, len(cols), domain)

        # Boost precision
        group_quality["total_score"] = round(group_quality["total_score"], 6)
        group_quality["dimensions"]  = {k: round(v, 6) for k, v in group_quality.get("dimensions", {}).items()}

        null_map = {
            col: {
                "null_count": int(df[col].isnull().sum()),
                "null_pct":   round(df[col].isnull().mean() * 100, 6),
                "unique":     int(df[col].nunique()),
                "dtype":      str(df[col].dtype),
                "fill_rate":  round(df[col].notna().mean(), 6),
            }
            for col in cols
        }

        num_cols = [c for c in cols if c in df.select_dtypes(include=[np.number]).columns and df[c].std() > 0]
        correlation = None
        if len(num_cols) >= 2:
            try:
                corr = df[num_cols].corr().round(6).fillna(0)
                correlation = {"columns": num_cols, "matrix": corr.values.tolist()}
            except Exception:
                pass

        group_reports[domain] = {
            "domain":       domain,
            "columns":      cols,
            "column_count": len(cols),
            "column_share": round(len(cols) / max(len(all_columns), 1), 6),
            "profile":      group_profile,
            "quality":      group_quality,
            "sufficiency":  group_sufficiency,
            "null_map":     null_map,
            "correlation":  correlation,
        }

    def _rank_score(gr: dict) -> float:
        q = gr["quality"].get("total_score", 0) / 100.0
        s = min(gr["sufficiency"].get("ratio", 0), 2.0) / 2.0
        return round(q * 0.6 + s * 0.4, 6)

    ranked = sorted(group_reports.items(), key=lambda x: _rank_score(x[1]), reverse=True)

    weighted_quality = round(sum(
        gr["quality"]["total_score"] * (gr["column_count"] / max(len(all_columns), 1))
        for gr in group_reports.values()
    ), 6)

    dominant = ranked[0][0] if ranked else "General"
    global_null_pct = round(df.isnull().mean().mean() * 100, 6)
    domain_col_ratio = {d: round(len(c)/max(len(all_columns),1),6) for d,c in assignments.items()}

    return {
        "multi_domain":                     True,
        "total_columns":                    len(all_columns),
        "total_rows":                       total_rows,
        "sampled_rows":                     len(df),
        "n_domain_groups":                  len(group_reports),
        "global_null_pct":                  global_null_pct,
        "dominant_domain":                  dominant,
        "dominant_domain_column_share":     round(group_reports.get(dominant, {}).get("column_share", 0) * 100, 4),
        "weighted_quality_score":           weighted_quality,
        "domain_column_distribution":       domain_col_ratio,
        "group_reports":                    group_reports,
        "ranked_groups": [
            {
                "rank":        i + 1,
                "domain":      d,
                "rank_score":  round(_rank_score(gr), 6),
                "quality":     gr["quality"]["total_score"],
                "sufficiency": gr["sufficiency"]["verdict"],
                "columns":     gr["columns"],
            }
            for i, (d, gr) in enumerate(ranked)
        ],
        "recommendations": _recommendations(group_reports, weighted_quality),
    }


def _recommendations(group_reports: dict, weighted_quality: float) -> list[str]:
    recs = []
    for domain, gr in group_reports.items():
        q, s = gr["quality"]["total_score"], gr["sufficiency"]
        if s["verdict"] in ("CRITICALLY_INSUFFICIENT", "INSUFFICIENT"):
            recs.append(f"[{domain}] Only {s['total_rows']:,} rows ({s['pct_of_recommended']:.2f}% of {s['recommended_rows']:,} recommended). Collect more data.")
        if q < 55:
            recs.append(f"[{domain}] Quality {q:.4f}/100 is below threshold. Avg missing: {gr['profile'].get('avg_missing_pct',0):.4f}%, dupes: {gr['profile'].get('duplicate_pct',0):.4f}%.")
        high_null = [c for c, info in gr.get("null_map", {}).items() if info["null_pct"] > 20.0]
        if high_null:
            recs.append(f"[{domain}] High nullity (>20%): {high_null[:4]}. Consider imputation or removal.")
    if weighted_quality < 60:
        recs.append(f"Overall weighted quality {weighted_quality:.4f}/100 is low. Address per-domain issues before combined training.")
    if not recs:
        recs.append("All domain groups meet minimum quality and sufficiency thresholds.")
    return recs
