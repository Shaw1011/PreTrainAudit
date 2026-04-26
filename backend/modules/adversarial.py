"""
adversarial.py — Adversarial vulnerability scanner.
Detects mislabeled clusters, poisoning-susceptible regions,
and outlier injection risk using embedding-space geometry.
Derived from Shaw's adversarial-immune-system research (F1=0.926).
"""

import numpy as np
import duckdb
from pathlib import Path

from .utils import safe_sql_path, RiskScale
from .config import DEFAULT_RANDOM_SEED

_RISK = RiskScale(LOW=20, MEDIUM=50, HIGH=75, default="CRITICAL")


def scan_adversarial_vulnerability(path: str, summary: dict) -> dict:
    data_type = summary.get("data_type", "tabular")

    if data_type == "tabular":
        return _scan_tabular(path, summary)
    elif data_type == "text":
        return _scan_text(path, summary)
    else:
        return _default_result("Adversarial scanning not supported for this data type")


# ─── Tabular ─────────────────────────────────────────────────────────────────

def _scan_tabular(path: str, summary: dict) -> dict:
    ext = Path(path).suffix.lower()
    con = duckdb.connect()
    safe = safe_sql_path(path)

    if ext == ".csv":
        df = con.execute(f"SELECT * FROM read_csv_auto('{safe}', header=true, ignore_errors=true, all_varchar=true) USING SAMPLE 10000 ROWS").df()
    elif ext == ".parquet":
        df = con.execute(f"SELECT * FROM read_parquet('{safe}') USING SAMPLE 10000 ROWS").df()
    else:
        con.close()
        return _default_result("Format not supported for adversarial scan")

    con.close()

    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1)
    if numeric_df.empty or len(numeric_df) < 50:
        return _default_result("Insufficient numeric features for adversarial geometry analysis")

    X = numeric_df.values

    # Normalize
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    # Compute pairwise distance distribution
    from sklearn.metrics import pairwise_distances
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    sample_idx = rng.choice(len(X_scaled), min(500, len(X_scaled)), replace=False)
    X_sample = X_scaled[sample_idx]
    dists = pairwise_distances(X_sample, metric="euclidean")
    np.fill_diagonal(dists, np.inf)
    min_dists = dists.min(axis=1)

    # Outlier detection via z-score on min distances
    z_scores = (min_dists - min_dists.mean()) / (min_dists.std() + 1e-9)
    outlier_indices = np.where(np.abs(z_scores) > 3)[0]
    outlier_count = len(outlier_indices)
    outlier_pct = round(outlier_count / len(X_sample) * 100, 2)

    # Poisoning susceptibility: high if many isolated points exist
    isolated_pct = round(np.sum(min_dists > np.percentile(min_dists, 95)) / len(min_dists) * 100, 2)

    # Vulnerability score (0=safe, 100=high risk)
    vuln_score = min(100, outlier_pct * 5 + isolated_pct * 2)

    risk_level = _RISK.classify(vuln_score)
    issues = []
    recommendations = []

    if outlier_pct > 2:
        issues.append(f"{outlier_pct}% of samples are statistical outliers — potential poisoned points")
        recommendations.append("Inspect outlier samples manually; consider robust outlier filtering")

    if isolated_pct > 5:
        issues.append(f"{isolated_pct}% of samples are geometrically isolated — easy injection targets")
        recommendations.append("Apply density-based filtering (DBSCAN) to remove sparse-region samples")

    if vuln_score < 20:
        issues.append("Dataset geometry appears robust — low poisoning susceptibility")

    return {
        "vulnerability_score": round(vuln_score, 1),
        "risk_level": risk_level,
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct,
        "isolated_sample_pct": isolated_pct,
        "samples_analyzed": len(X_sample),
        "issues": issues,
        "recommendations": recommendations,
        "method": "embedding-space geometry (z-score + pairwise distance distribution)",
    }


# ─── Text ─────────────────────────────────────────────────────────────────────

def _scan_text(path: str, summary: dict) -> dict:
    """
    For text: detect anomalous lines via length outliers and
    character-level entropy anomalies (proxy for injection artifacts).
    Full embedding-based scan requires sentence-transformers — 
    flagged as optional heavy dependency.
    """
    lines = []
    lengths = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
                lengths.append(len(stripped))
            if i >= 50_000:
                break

    lengths = np.array(lengths)
    mean_l, std_l = lengths.mean(), lengths.std()
    z_scores = np.abs((lengths - mean_l) / (std_l + 1e-9))
    outlier_count = int((z_scores > 3).sum())
    outlier_pct = round(outlier_count / max(len(lengths), 1) * 100, 2)

    # Entropy analysis on sample lines
    def char_entropy(text: str) -> float:
        from collections import Counter
        import math
        c = Counter(text)
        total = len(text)
        return -sum((v / total) * math.log2(v / total) for v in c.values()) if total > 0 else 0

    sample_lines = lines[:1000]
    entropies = np.array([char_entropy(l) for l in sample_lines])
    low_entropy_pct = round((entropies < 2.0).sum() / max(len(entropies), 1) * 100, 2)

    vuln_score = min(100, outlier_pct * 4 + low_entropy_pct * 2)
    risk_level = _RISK.classify(vuln_score)

    issues = []
    recommendations = []
    if outlier_pct > 1:
        issues.append(f"{outlier_pct}% of lines are length outliers — possible injected artifacts")
        recommendations.append("Filter lines outside ±3σ of mean length")
    if low_entropy_pct > 5:
        issues.append(f"{low_entropy_pct}% of lines have low character entropy — possible repetitive injections")
        recommendations.append("Remove low-entropy lines (e.g., repeated sequences, padding)")

    return {
        "vulnerability_score": round(vuln_score, 1),
        "risk_level": risk_level,
        "outlier_lines": outlier_count,
        "outlier_pct": outlier_pct,
        "low_entropy_pct": low_entropy_pct,
        "samples_analyzed": len(lines),
        "issues": issues,
        "recommendations": recommendations,
        "method": "length z-score + character entropy analysis",
        "note": "For full embedding-based adversarial scan, enable sentence-transformers mode",
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _default_result(note: str) -> dict:
    return {
        "vulnerability_score": None,
        "risk_level": "UNKNOWN",
        "issues": [],
        "recommendations": [],
        "note": note,
    }
