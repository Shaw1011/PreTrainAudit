"""
memorization.py — Memorization risk and privacy leakage estimator.
Uses dataset size vs model parameter count ratio to estimate
the probability that a model will memorize training samples verbatim.
Based on empirical findings from Carlini et al. (2021, 2022).
"""

import math


# Empirical thresholds from memorization research
# ratio = tokens / parameters
# < 1     → severe memorization risk
# 1-10    → high risk
# 10-100  → moderate risk  
# > 100   → low risk

RISK_THRESHOLDS = [
    (1,    "CRITICAL", "Model parameters exceed token count — near-certain verbatim memorization"),
    (10,   "HIGH",     "Low token-to-parameter ratio — high memorization probability"),
    (40,   "MEDIUM",   "Moderate memorization risk — model may memorize rare or repeated samples"),
    (100,  "LOW",      "Acceptable ratio — standard regularization should suffice"),
    (float("inf"), "MINIMAL", "High token-to-parameter ratio — low memorization risk"),
]


def compute_memorization_risk(summary: dict, param_count_millions: float) -> dict:
    data_type = summary.get("data_type", "tabular")
    param_count = param_count_millions * 1_000_000

    if data_type == "text":
        # Estimate tokens from character count
        char_count = summary.get("char_count", 0)
        token_count = int(char_count / 4)  # ~4 chars/token heuristic
        data_points = summary.get("line_count", 0)
    elif data_type == "tabular":
        row_count = summary.get("row_count", 0)
        col_count = summary.get("column_count", 1)
        # Treat each cell as ~1 token equivalent
        token_count = row_count * col_count
        data_points = row_count
    else:
        return {"risk_level": "UNKNOWN", "note": "Unsupported data type"}

    # Duplication amplifies memorization — repeated samples are memorised first.
    # If profiling data is available via summary, factor it in.
    dup_pct = float(summary.get("duplicate_pct", 0) or 0)
    dup_amplifier = 1.0 + (dup_pct / 100.0) * 2.0  # 10% dupes → 1.2× effective ratio reduction

    ratio = token_count / max(param_count, 1)
    # Effective ratio accounts for duplication: more dupes → lower effective diversity
    effective_ratio = ratio / max(dup_amplifier, 1.0)

    risk_level = "UNKNOWN"
    description = ""
    for threshold, level, desc in RISK_THRESHOLDS:
        if effective_ratio < threshold:
            risk_level = level
            description = desc
            break

    # Memorization probability estimate (sigmoid-based heuristic)
    # P(mem) ≈ sigmoid(-0.5 * log(ratio))
    try:
        log_ratio = math.log(max(effective_ratio, 1e-6))
        mem_probability = 1 / (1 + math.exp(0.8 * log_ratio))
    except (ValueError, OverflowError, ZeroDivisionError):
        mem_probability = 0.5

    # Privacy leakage score (0–100, higher = more risk)
    leakage_score = round(mem_probability * 100, 1)

    issues = []
    recommendations = []

    if risk_level in ("CRITICAL", "HIGH"):
        issues.append(f"Effective token-to-parameter ratio = {round(effective_ratio, 2)} "
                      f"(raw: {round(ratio, 2)}, dup amplifier: {round(dup_amplifier, 2)}×) "
                      f"— model will likely memorize training data")
        recommendations.append("Increase dataset size significantly before training")
        recommendations.append("Apply differential privacy (DP-SGD) during training")
        recommendations.append("Use deduplication — memorization risk compounds on repeated samples")

    if risk_level == "MEDIUM":
        issues.append(f"Effective token-to-parameter ratio = {round(effective_ratio, 2)} — moderate memorization risk")
        recommendations.append("Apply gradient clipping and early stopping")
        recommendations.append("Consider training with a smaller model size")

    if data_type == "text" and token_count < 1_000_000:
        issues.append(f"Corpus is small ({token_count:,} estimated tokens) — memorization is amplified")

    if dup_pct > 5.0:
        issues.append(f"Duplication rate {dup_pct:.1f}% amplifies memorization risk "
                      f"(effective ratio reduced from {round(ratio, 2)} to {round(effective_ratio, 2)})")
        recommendations.append("Deduplicate dataset before training — repeated samples are memorised first")

    gdpr_flag = risk_level in ("CRITICAL", "HIGH") and data_type in ("text", "tabular")

    return {
        "risk_level": risk_level,
        "description": description,
        "token_count_estimate": token_count,
        "data_points": data_points,
        "param_count_millions": param_count_millions,
        "token_to_param_ratio": round(ratio, 4),
        "effective_ratio": round(effective_ratio, 4),
        "duplication_amplifier": round(dup_amplifier, 4),
        "memorization_probability": round(mem_probability, 4),
        "privacy_leakage_score": leakage_score,
        "gdpr_compliance_flag": gdpr_flag,
        "issues": issues,
        "recommendations": recommendations,
        "reference": "Carlini et al. (2021) — Extracting Training Data from Large Language Models",
    }
