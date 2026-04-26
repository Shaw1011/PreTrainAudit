"""
fairness.py — Fairness drift risk predictor.
Analyzes demographic distribution proxies and class imbalance
to estimate the probability of fairness degradation in production.
Derived from Shaw's fairness-drift research (ADWIN/EWMA/Page-Hinkley).
"""

import numpy as np

from .utils import RiskScale

_RISK = RiskScale(LOW=20, MEDIUM=45, HIGH=70, default="CRITICAL")


def predict_fairness_drift(profile: dict) -> dict:
    data_type = profile.get("data_type", "tabular")

    if data_type == "tabular":
        return _analyze_tabular_fairness(profile)
    elif data_type == "text":
        return _analyze_text_fairness(profile)
    else:
        return {"risk_level": "UNKNOWN", "note": "Unsupported data type for fairness analysis"}


# ─── Tabular ─────────────────────────────────────────────────────────────────

def _analyze_tabular_fairness(profile: dict) -> dict:
    balance = profile.get("class_balance", {})
    issues = []
    recommendations = []
    drift_signals = []

    # Detect highly imbalanced categorical columns (proxy for protected attribute skew)
    for col, data in balance.items():
        proportions = data.get("proportions", [])
        if not proportions:
            continue

        max_prop = max(proportions)
        min_prop = min(proportions)
        gini = _gini_impurity(proportions)

        # High concentration = potential representation bias
        if max_prop > 0.70:
            issues.append(
                f"Column '{col}': dominant class holds {round(max_prop*100,1)}% — "
                f"underrepresented groups may suffer degraded model performance"
            )
            drift_signals.append(1 - gini)
            recommendations.append(
                f"Balance '{col}' via stratified sampling or group-weighted loss"
            )
        elif max_prop > 0.50:
            issues.append(f"Column '{col}': moderate imbalance ({round(max_prop*100,1)}% dominant)")
            drift_signals.append((1 - gini) * 0.5)

        # Very few classes = low group diversity
        if len(proportions) < 3:
            issues.append(f"Column '{col}': only {len(proportions)} distinct values — limited group coverage")

    # Order-independent drift score: peak signal + breadth penalty.
    #   Peak: the single worst imbalance dominates (this IS the fairness risk).
    #   Breadth: multiple imbalanced columns compound production risk.
    #   Previous EWMA was order-dependent (dict iteration order = arbitrary weight).
    if drift_signals:
        peak_signal   = max(drift_signals)
        breadth_bonus = min(0.2, len(drift_signals) * 0.05)
        drift_risk_score = round(min((peak_signal + breadth_bonus) * 100, 100.0), 1)
    else:
        drift_risk_score = 10.0

    # Distribution shift vulnerability (based on number of skewed columns)
    skewed_cols = sum(1 for col, d in balance.items() if d.get("proportions") and max(d["proportions"]) > 0.6)
    total_cat_cols = max(len(balance), 1)
    shift_vulnerability = round(skewed_cols / total_cat_cols * 100, 1)

    risk_level = _RISK.classify(drift_risk_score)

    if not issues:
        issues.append("No severe fairness imbalances detected in categorical columns")

    return {
        "risk_level": risk_level,
        "drift_risk_score": drift_risk_score,
        "distribution_shift_vulnerability": shift_vulnerability,
        "skewed_columns": skewed_cols,
        "total_categorical_columns": total_cat_cols,
        "issues": issues,
        "recommendations": recommendations,
        "method": "Peak Gini impurity + breadth penalty across categorical columns",
        "note": "For time-series fairness drift (ADWIN/Page-Hinkley), provide timestamped data",
    }


# ─── Text ─────────────────────────────────────────────────────────────────────

def _analyze_text_fairness(profile: dict) -> dict:
    """
    Text fairness analysis uses vocabulary diversity as a proxy.
    Low TTR + high duplication = corpus likely dominated by one source/perspective.
    """
    ttr = profile.get("type_token_ratio", 0.5)
    dup_pct = profile.get("duplicate_pct", 0)

    issues = []
    recommendations = []

    # Low diversity = likely single-source corpus = representation bias risk
    diversity_score = ttr * 100
    drift_risk = max(0, 100 - diversity_score * 1.5 + dup_pct)

    if ttr < 0.15:
        issues.append("Very low lexical diversity — corpus likely dominated by single source or style")
        recommendations.append("Diversify corpus with multiple sources, styles, and demographic perspectives")

    if dup_pct > 15:
        issues.append(f"{dup_pct}% duplicate lines — repeated content amplifies existing biases")
        recommendations.append("Deduplicate before training to prevent bias amplification")

    risk_level = _RISK.classify(min(drift_risk, 100))

    return {
        "risk_level": risk_level,
        "drift_risk_score": round(min(drift_risk, 100), 1),
        "lexical_diversity": round(ttr, 4),
        "issues": issues,
        "recommendations": recommendations,
        "method": "Lexical diversity (TTR) + duplication rate as representation bias proxy",
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gini_impurity(proportions: list) -> float:
    """Gini impurity: 0 = perfectly imbalanced, 1 = perfectly balanced."""
    return 1 - sum(p ** 2 for p in proportions)
