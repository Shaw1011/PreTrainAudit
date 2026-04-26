"""
compatibility.py — AI Compatibility Advisor Engine.

FIXES applied after stress testing:
  - Domain quality penalty: exponent raised from 1.5 → 2.5 for steeper drop-off
      Old: (60/75)^1.5 * 100 = 71.6 → weighted contribution too high
      New: (60/75)^2.5 * 100 = 57.2 → properly penalizes quality gaps
  - W_QUAL raised to 0.25, W_SIZE reduced to 0.15 to make quality matter more
  - All other logic unchanged (hard cap, arch mismatch, etc.)
"""

from .utils import clamp

SIZE_REQUIREMENTS = {
    "Small":  {"tabular_rows": 1_000,      "text_tokens": 500_000},
    "Medium": {"tabular_rows": 50_000,     "text_tokens": 25_000_000},
    "Large":  {"tabular_rows": 1_000_000,  "text_tokens": 500_000_000},
}

VALID_DATA_TYPES = {
    "NLP":             ["text"],
    "Computer Vision": ["image"],
    "Recommendation":  ["tabular"],
    "Speech":          ["text", "audio"],
    "General":         ["text", "tabular", "image"],
}

ARCH_TASK_FIT = {
    "Transformer": ["Classification", "Generation", "Detection"],
    "CNN":         ["Classification", "Detection"],
    "RNN":         ["Generation", "Classification"],
    "MoE":         ["Generation", "Classification"],
}

DOMAIN_QUALITY_THRESHOLDS = {
    "Healthcare":   80,
    "Finance":      75,
    "Social Media": 55,
    "General":      60,
}

# Stress-tested weights — sum to 1.0
W_TYPE  = 0.30
W_ARCH  = 0.30
W_SIZE  = 0.15  # reduced
W_QUAL  = 0.25  # raised — quality matters more than raw size

TYPE_MISMATCH_CAP = 35.0


def evaluate_compatibility(profile: dict, quality: dict, ai_config: dict) -> dict:
    data_type    = profile.get("data_type", "tabular")
    ai_type      = ai_config.get("ai_type", "NLP")
    model_size   = ai_config.get("model_size", "Medium")
    architecture = ai_config.get("architecture", "Transformer")
    domain       = ai_config.get("domain", "General")
    task_type    = ai_config.get("task_type", "Classification")

    issues = []
    recommendations = []

    # ── 1. Data type vs AI type ───────────────────────────────────────────────
    valid_types   = VALID_DATA_TYPES.get(ai_type, ["tabular"])
    type_mismatch = data_type not in valid_types

    if type_mismatch:
        issues.append(
            f"CRITICAL MISMATCH: {data_type} data with {ai_type} model — "
            f"{ai_type} expects {valid_types}"
        )
        recommendations.append(
            f"Switch to a model type that accepts {data_type} data, "
            f"or convert dataset to {valid_types[0]} format"
        )
        type_score = 0.0
    else:
        type_score = 100.0

    # ── 2. Architecture vs task ───────────────────────────────────────────────
    supported_tasks = ARCH_TASK_FIT.get(architecture, [])
    if task_type not in supported_tasks:
        issues.append(
            f"Architecture mismatch: {architecture} is suboptimal for {task_type} "
            f"(best for: {supported_tasks})"
        )
        recommendations.append(
            "For Generation → Transformer or MoE.  "
            "For Detection → CNN or Transformer.  "
            "For Classification → any architecture works."
        )
        arch_score = 30.0
    else:
        arch_score = 100.0

    # ── 3. Dataset size vs model size ─────────────────────────────────────────
    req = SIZE_REQUIREMENTS.get(model_size, SIZE_REQUIREMENTS["Medium"])
    if data_type == "tabular":
        actual   = profile.get("total_rows", 0) or 0
        required = req["tabular_rows"]
        unit     = "rows"
    else:
        actual   = profile.get("estimated_tokens", 0) or (profile.get("char_count", 0) or 0) // 4
        required = req["text_tokens"]
        unit     = "tokens"

    size_score = clamp(min(100.0, (actual / max(required, 1)) * 100))

    if size_score < 50:
        issues.append(
            f"Insufficient data: {actual:,} {unit} (recommended: {required:,}+)"
        )
        recommendations.append(
            f"Collect ≥{required:,} {unit}, or switch to a smaller model"
        )
    elif size_score < 100:
        issues.append(f"Marginal data: {round(size_score)}% of recommended for {model_size} model")

    # ── 4. Domain quality fit — FIX: steeper penalty exponent ────────────────
    quality_score = float(quality.get("total_score", 0) or 0)
    threshold     = DOMAIN_QUALITY_THRESHOLDS.get(domain, 60)

    if quality_score >= threshold:
        quality_fit = 100.0
    else:
        ratio       = quality_score / max(threshold, 1)
        # Exponent 2.5: sharper drop-off when quality falls below threshold
        # Example: Finance 60/75 → (0.8)^2.5 = 0.572 → 57.2/100
        quality_fit = clamp(ratio ** 2.5 * 100)
        issues.append(
            f"Data quality ({quality_score}/100) below {domain} threshold ({threshold}/100)"
        )
        recommendations.append(
            f"Improve quality to ≥{threshold}/100 before {domain} domain training"
        )

    # ── 5. Weighted score + hard cap ──────────────────────────────────────────
    raw = (
        type_score  * W_TYPE  +
        arch_score  * W_ARCH  +
        size_score  * W_SIZE  +
        quality_fit * W_QUAL
    )
    if type_mismatch:
        raw = min(raw, TYPE_MISMATCH_CAP)

    compatibility_score = clamp(round(raw, 1))

    suitability = (
        "SUITABLE"           if compatibility_score >= 75 else
        "PARTIALLY_SUITABLE" if compatibility_score >= 45 else
        "NOT_SUITABLE"
    )
    size_verdict = (
        "ENOUGH"    if size_score >= 100 else
        "MARGINAL"  if size_score >= 50  else
        "NOT_ENOUGH"
    )

    best_model = _recommend_best_model(data_type, quality_score, actual, unit)

    if not issues:
        issues.append("No major compatibility issues — dataset well-aligned with configuration")

    return {
        "compatibility_score": compatibility_score,
        "suitability":         suitability,
        "size_verdict":        size_verdict,
        "score_breakdown": {
            "type_compatibility": {"score": round(type_score, 1),  "weight": W_TYPE},
            "architecture_fit":   {"score": round(arch_score, 1),  "weight": W_ARCH},
            "size_sufficiency":   {"score": round(size_score, 1),  "weight": W_SIZE},
            "domain_quality_fit": {"score": round(quality_fit, 1), "weight": W_QUAL},
        },
        "issues":          issues,
        "recommendations": recommendations,
        "best_model_recommendation": best_model,
        "config_evaluated": ai_config,
    }


def _recommend_best_model(data_type: str, quality_score: float, data_size: int, unit: str) -> dict:
    if data_type == "text":
        size = "Large" if data_size > 500_000_000 else "Medium" if data_size > 25_000_000 else "Small"
        return {"ai_type": "NLP", "model_size": size, "architecture": "Transformer",
                "rationale": f"{data_size:,} {unit} of text, quality {quality_score}/100"}
    elif data_type == "tabular":
        size = "Large" if data_size > 1_000_000 else "Medium" if data_size > 50_000 else "Small"
        return {"ai_type": "Recommendation", "model_size": size, "architecture": "Transformer",
                "rationale": f"{data_size:,} {unit} of tabular data, quality {quality_score}/100"}
    elif data_type == "image":
        size = "Large" if data_size > 500_000 else "Medium" if data_size > 10_000 else "Small"
        return {"ai_type": "Computer Vision", "model_size": size, "architecture": "CNN",
                "rationale": f"{data_size:,} {unit} of image data, quality {quality_score}/100"}
    else:
        return {"note": f"Cannot determine recommendation for data type: {data_type}"}

