"""
quality.py — Data Quality Score computation (precision edition).

All intermediate values computed at full float64 precision.
All output scores rounded to 6 decimal places.
Final total_score rounded to 4 dp for readability.

Dimensions (tabular):
  completeness   0.20 — missing value rate
  duplication    0.15 — exact duplicate row rate
  noise          0.15 — coefficient-of-variation proxy
  balance        0.20 — class imbalance (exponential penalty)
  structural     0.30 — row-count adequacy (hard caps for tiny data)

Dimensions (text):
  completeness         0.20
  duplication          0.25
  vocabulary_diversity 0.25
  noise                0.10
  structural           0.20
"""

from __future__ import annotations
import math
import numpy as np

from .utils import clamp


def compute_quality_score(profile: dict) -> dict:
    data_type = profile.get("data_type", "tabular")
    if data_type == "tabular":
        return _score_tabular(profile)
    elif data_type == "text":
        return _score_text(profile)
    else:
        return {
            "total_score": 50.0000,
            "grade":       "C",
            "dimensions":  {},
            "issues":      ["Unknown data type — defaulting to 50/100"],
            "recommendations": [],
        }


# ── Tabular ────────────────────────────────────────────────────────────────────

def _score_tabular(profile: dict) -> dict:
    issues: list[str] = []
    recommendations: list[str] = []

    # ── 1. Completeness ───────────────────────────────────────────────────────
    missing_pct: dict = profile.get("missing_pct", {})
    if missing_pct:
        per_col = list(missing_pct.values())
        avg_missing = sum(per_col) / len(per_col)
        max_missing = max(per_col)
        # Penalty: linear on avg, extra hit for any column >50% missing
        completeness = clamp(100.0 - avg_missing * 1.8 - (max_missing > 50) * 8.0)
    else:
        avg_missing = 0.0
        completeness = 100.0

    if avg_missing > 5.0:
        issues.append(
            f"Missing values: avg {avg_missing:.4f}% across {len(missing_pct)} columns. "
            f"Worst column: {max(missing_pct, key=missing_pct.get)} "
            f"({max(missing_pct.values()):.4f}% null)"
        )
        recommendations.append("Impute or drop columns with >20% missing values")

    # ── 2. Duplication ────────────────────────────────────────────────────────
    dup_pct: float = float(profile.get("duplicate_pct", 0) or 0)
    # Exponential penalty above 2% — duplicates compound memorization risk
    if dup_pct <= 2.0:
        duplication_score = clamp(100.0 - dup_pct * 2.0)
    else:
        duplication_score = clamp(100.0 - 4.0 - (dup_pct - 2.0) ** 1.4 * 4.0)

    if dup_pct > 2.0:
        issues.append(f"Duplicate rows: {dup_pct:.4f}% ({profile.get('duplicate_count', '?')} rows)")
        recommendations.append("Deduplicate before training — duplicates amplify memorization risk")

    # ── 3. Noise (composite: CV + outlier fraction) ────────────────────────────
    #   CV alone is misleading for zero-mean or heavy-tailed data.
    #   Adding outlier fraction gives a more robust signal.
    distributions: dict = profile.get("distributions", {})
    noise_penalties: list[float] = []
    for col, stats in distributions.items():
        rng      = stats.get("max", 0) - stats.get("min", 0)
        mean_val = stats.get("mean", 0) or 0
        std_val  = stats.get("std", 0) or 0
        if rng <= 0:
            continue  # constant column — no noise

        # CV component (0→2, capped)
        if abs(mean_val) > 1e-12:
            cv = std_val / abs(mean_val)
        elif std_val > 0:
            cv = 1.5  # nonzero spread around zero center
        else:
            cv = 0.0

        # Outlier fraction component — proportion of data beyond 3σ
        bins   = stats.get("bins", [])
        counts = stats.get("counts", [])
        if bins and counts:
            total_pts = sum(counts)
            lo_3s = mean_val - 3 * (std_val + 1e-9)
            hi_3s = mean_val + 3 * (std_val + 1e-9)
            outlier_pts = sum(
                c for c, b_lo, b_hi in zip(counts, bins[:-1], bins[1:])
                if b_hi < lo_3s or b_lo > hi_3s
            )
            outlier_frac = outlier_pts / max(total_pts, 1)
        else:
            outlier_frac = 0.0

        # Composite: CV for general spread, outlier_frac for tail heaviness
        composite = min(cv, 2.0) * 0.7 + outlier_frac * 10.0 * 0.3
        noise_penalties.append(min(composite, 1.5))

    if noise_penalties:
        avg_noise = sum(noise_penalties) / len(noise_penalties)
        noise_score = clamp(100.0 - avg_noise * 22.0)
    else:
        avg_noise = 0.0
        noise_score = 100.0

    if avg_noise > 0.7:
        issues.append(
            f"High noise signal: mean CV = {avg_noise:.4f} across numeric columns"
        )
        recommendations.append(
            "Apply outlier detection (IQR or z-score) and consider feature normalization"
        )

    # ── 4. Class balance (exponential penalty) ────────────────────────────────
    balance: dict = profile.get("class_balance", {})
    balance_metrics: list[dict] = []

    for col, data in balance.items():
        proportions: list[float] = data.get("proportions", [])
        if not proportions:
            continue
        max_prop = max(proportions)
        gini = 1.0 - sum(p ** 2 for p in proportions)
        entropy = -sum(p * math.log2(p) for p in proportions if p > 0)
        # Maximum entropy for n classes
        n = len(proportions)
        max_entropy = math.log2(n) if n > 1 else 1.0
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 1.0
        balance_metrics.append({
            "col": col, "max_prop": max_prop,
            "gini": gini, "entropy_ratio": entropy_ratio,
        })

    if balance_metrics:
        avg_max_prop = sum(m["max_prop"] for m in balance_metrics) / len(balance_metrics)
        avg_entropy  = sum(m["entropy_ratio"] for m in balance_metrics) / len(balance_metrics)
        excess = max(0.0, avg_max_prop - 0.5)
        # Primary: exponential penalty on dominance; Secondary: entropy bonus
        balance_score = clamp(
            100.0 - (excess ** 1.5) * 380.0 + avg_entropy * 5.0
        )
        for m in balance_metrics:
            if m["max_prop"] > 0.80:
                issues.append(
                    f"Severe imbalance in '{m['col']}': "
                    f"dominant class = {m['max_prop']*100:.2f}%, "
                    f"Gini = {m['gini']:.4f}, entropy ratio = {m['entropy_ratio']:.4f}"
                )
                recommendations.append(
                    f"Column '{m['col']}': use SMOTE / class-weighted loss / stratified sampling"
                )
            elif m["max_prop"] > 0.65:
                issues.append(
                    f"Moderate imbalance in '{m['col']}': "
                    f"dominant class = {m['max_prop']*100:.2f}%"
                )
    else:
        balance_score = 100.0  # No categorical columns — not penalized

    # ── 5. Structural integrity (rows × cols aware) ─────────────────────────────
    #   data_points = rows × cols gives a fairer picture than rows alone.
    #   A 100k×2 dataset is NOT equivalent to 100k×500.
    row_count: int = int(profile.get("total_rows", 0) or 0)
    col_count: int = int(profile.get("column_count", 1) or 1)
    data_points = row_count * col_count

    # Column richness bonus: log-scaled so diminishing returns past ~50 cols
    col_bonus = min(10.0, math.log2(max(col_count, 1)) * 2.5)

    if row_count < 10:
        structural_score = 0.0
        issues.append(
            f"CRITICALLY SMALL: {row_count} rows × {col_count} cols = "
            f"{data_points:,} data points — unusable for training"
        )
        recommendations.append("Minimum 1,000+ rows required for any meaningful model training")
    elif row_count < 100:
        structural_score = 10.0
        issues.append(
            f"Very small: {row_count} rows ({data_points:,} data points) — "
            f"only toy experiments possible"
        )
        recommendations.append("Collect at least 1k rows before attempting training")
    elif row_count < 1_000:
        structural_score = clamp(10.0 + (row_count - 100) / 900.0 * 35.0 + col_bonus * 0.3)
        issues.append(f"Small dataset: {row_count:,} rows × {col_count} cols — suitable for small models only")
        recommendations.append("Target 10k+ rows for reliable generalization")
    elif row_count < 10_000:
        structural_score = clamp(45.0 + (row_count - 1_000) / 9_000.0 * 30.0 + col_bonus * 0.5)
        issues.append(
            f"Moderate dataset: {row_count:,} rows × {col_count} cols — "
            f"acceptable for Small/Medium models"
        )
    elif row_count < 100_000:
        structural_score = clamp(75.0 + (row_count - 10_000) / 90_000.0 * 20.0 + col_bonus * 0.3)
    else:
        structural_score = clamp(100.0 + col_bonus * 0.2)

    # ── Weighted total ─────────────────────────────────────────────────────────
    total = (
        completeness     * 0.20 +
        duplication_score * 0.15 +
        noise_score      * 0.15 +
        balance_score    * 0.20 +
        structural_score * 0.30
    )
    total = clamp(total)

    # Hard caps for tiny data
    if row_count < 10:
        total = min(total, 20.0)
    elif row_count < 100:
        total = min(total, 45.0)
    elif row_count < 1_000:
        total = min(total, 65.0)

    if not issues:
        issues.append("No major quality issues detected")

    return {
        "total_score": round(total, 4),
        "grade":       _grade(total),
        "dimensions": {
            "completeness":         round(completeness, 6),
            "duplication":          round(duplication_score, 6),
            "noise":                round(noise_score, 6),
            "balance":              round(balance_score, 6),
            "structural_integrity": round(structural_score, 6),
        },
        "raw_inputs": {
            "avg_missing_pct":  round(avg_missing, 6),
            "duplicate_pct":    round(dup_pct, 6),
            "avg_noise_cv":     round(avg_noise, 6),
            "row_count":        row_count,
            "column_count":     col_count,
            "data_points":      data_points,
        },
        "issues":          issues,
        "recommendations": recommendations,
    }


# ── Text ────────────────────────────────────────────────────────────────────────

def _score_text(profile: dict) -> dict:
    issues: list[str] = []
    recommendations: list[str] = []

    completeness = 95.0

    dup_pct: float = float(profile.get("duplicate_pct", 0) or 0)
    duplication_score = clamp(100.0 - dup_pct * 2.2)
    if dup_pct > 10.0:
        issues.append(f"Duplicate lines: {dup_pct:.4f}%")
        recommendations.append("Deduplicate corpus using MinHash near-dedup")

    ttr: float = float(profile.get("type_token_ratio", 0.5) or 0.01)
    # Scale: TTR=0.05 → 10, TTR=0.5 → 100
    diversity_score = clamp(math.log1p(ttr * 20) / math.log1p(10) * 100)
    if ttr < 0.15:
        issues.append(
            f"Low vocabulary diversity: TTR = {ttr:.6f} — corpus may be highly repetitive"
        )
        recommendations.append("Add diverse sources; current corpus lacks lexical variety")

    avg_len: float = float(profile.get("avg_line_length", 0) or 0)
    max_len: float = float(profile.get("max_line_length", 0) or 0)
    if max_len > avg_len * 12.0 and avg_len > 0:
        noise_score = 60.0
        issues.append(
            f"Extreme length outliers: max={max_len:.0f} chars vs avg={avg_len:.2f} chars "
            f"(ratio={max_len/avg_len:.2f}x)"
        )
        recommendations.append("Filter lines outside ±3σ of mean length")
    else:
        noise_score = 82.0

    sampled: int = int(profile.get("sampled_lines", 0) or 0)
    if sampled < 1_000:
        structural_score = 0.0
        issues.append(f"Critically small corpus: {sampled:,} lines")
        recommendations.append("Language models require millions of lines at minimum")
    elif sampled < 10_000:
        structural_score = clamp(10.0 + (sampled - 1_000) / 9_000.0 * 35.0)
        issues.append(f"Small corpus: {sampled:,} lines — only fine-tuning on very small models feasible")
    elif sampled < 100_000:
        structural_score = clamp(45.0 + (sampled - 10_000) / 90_000.0 * 30.0)
    elif sampled < 1_000_000:
        structural_score = clamp(75.0 + (sampled - 100_000) / 900_000.0 * 20.0)
    else:
        structural_score = 100.0

    total = clamp(
        completeness     * 0.20 +
        duplication_score * 0.25 +
        diversity_score  * 0.25 +
        noise_score      * 0.10 +
        structural_score * 0.20
    )

    if sampled < 1_000:
        total = min(total, 20.0)
    elif sampled < 10_000:
        total = min(total, 50.0)

    if not issues:
        issues.append("No major quality issues detected")

    return {
        "total_score": round(total, 4),
        "grade":       _grade(total),
        "dimensions": {
            "completeness":         round(completeness, 6),
            "duplication":          round(duplication_score, 6),
            "vocabulary_diversity": round(diversity_score, 6),
            "noise":                round(noise_score, 6),
            "structural_integrity": round(structural_score, 6),
        },
        "raw_inputs": {
            "duplicate_pct":     round(dup_pct, 6),
            "type_token_ratio":  round(ttr, 6),
            "avg_line_length":   round(avg_len, 4),
            "sampled_lines":     sampled,
        },
        "issues":          issues,
        "recommendations": recommendations,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _grade(score: float) -> str:
    if score >= 85.0: return "A"
    if score >= 70.0: return "B"
    if score >= 55.0: return "C"
    if score >= 40.0: return "D"
    return "F"
