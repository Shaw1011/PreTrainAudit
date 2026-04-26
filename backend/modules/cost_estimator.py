"""
cost_estimator.py — Training cost estimator.

REDESIGNED:
  - GPU time now derived from total_flops / GPU_compute_throughput.
    Previously: time = tokens / TPS — which ignored model parameter count
    entirely, making a 7B model show the same training time as a 10M model.
  - Constants imported from config.py (single source of truth).
  - GPU utilization factor (MFU) applied for realistic estimates.
  - Small model param counts are realistic (10M, not 125M).
"""

from .config import GPU_BENCHMARKS, MODEL_SIZE_PARAMS, GPU_UTILIZATION

FLOPS_PER_TOKEN_PER_PARAM = 6   # 6x for forward + backward pass


def estimate_training_cost(summary: dict, ai_config: dict) -> dict:
    data_type  = summary.get("data_type", "tabular")
    ai_type    = ai_config.get("ai_type", "NLP")
    model_size = ai_config.get("model_size", "Medium")

    # Token/sample count
    if data_type == "text":
        token_count = int(summary.get("estimated_tokens", 0) or
                          (summary.get("char_count", 0) or 0) // 4)
    elif data_type == "tabular":
        row_count = summary.get("row_count", 0) or 0
        col_count = summary.get("column_count", 1) or 1
        token_count = row_count * col_count
    else:
        token_count = 0

    # Param count — fall back to Medium if unknown size
    size_map   = MODEL_SIZE_PARAMS.get(model_size, MODEL_SIZE_PARAMS["Medium"])
    param_m    = size_map.get(ai_type, size_map.get("NLP", 125))
    param_count = param_m * 1_000_000

    # Chinchilla: optimal tokens ≈ 20x parameters
    chinchilla_optimal = param_count * 20
    chinchilla_ratio = round(token_count / max(chinchilla_optimal, 1), 4)

    # FLOPs — the real compute cost of training
    total_flops   = FLOPS_PER_TOKEN_PER_PARAM * token_count * param_count
    total_pflops  = round(total_flops / 1e15, 4)

    # GPU estimates — derived from actual compute, not just data throughput.
    #   time = total_flops / (GPU_peak_flops × utilization)
    #   This correctly scales with model size: a 7B model is ~700× slower
    #   than a 10M model on the same hardware.
    gpu_estimates = {}
    for gpu_name, specs in GPU_BENCHMARKS.items():
        if total_flops == 0:
            gpu_estimates[gpu_name] = {"hours": 0, "cost_usd": 0}
            continue
        # Effective throughput = peak TFLOPS × 1e12 (to FLOPS) × utilization
        effective_flops_per_sec = specs["tflops_fp16"] * 1e12 * GPU_UTILIZATION
        hours = total_flops / (effective_flops_per_sec * 3600)
        cost  = hours * specs["cost_per_hour"]
        gpu_estimates[gpu_name] = {
            "hours":    round(hours, 2),
            "cost_usd": round(cost, 2),
        }

    # Sufficiency verdict
    if token_count == 0:
        sufficiency = "CRITICALLY_INSUFFICIENT"
        note = "No tokens detected — dataset may be empty or unreadable"
    elif chinchilla_ratio >= 1.0:
        sufficiency = "SUFFICIENT"
        note = "Dataset meets Chinchilla optimal token count for this model size"
    elif chinchilla_ratio >= 0.5:
        sufficiency = "MARGINAL"
        note = f"Dataset is {round(chinchilla_ratio*100)}% of Chinchilla optimal — acceptable for fine-tuning"
    elif chinchilla_ratio >= 0.1:
        sufficiency = "INSUFFICIENT"
        note = f"Only {round(chinchilla_ratio*100)}% of optimal — underfitting expected"
    else:
        sufficiency = "CRITICALLY_INSUFFICIENT"
        note = "Dataset far too small — reduce model size or collect significantly more data"

    recommendations = []
    if chinchilla_ratio < 0.5 and token_count > 0:
        recommendations.append(
            f"For a {model_size} {ai_type} model, target ~{int(chinchilla_optimal):,} tokens "
            f"(currently {token_count:,})"
        )
        recommendations.append("Consider a smaller model size to better match your data budget")
    if total_pflops > 100:
        recommendations.append(
            "Computationally intensive — use gradient checkpointing and mixed precision (fp16/bf16)"
        )

    return {
        "token_count":              token_count,
        "param_count_millions":     param_m,
        "total_flops_petaflops":    total_pflops,
        "chinchilla_optimal_tokens": int(chinchilla_optimal),
        "chinchilla_ratio":         chinchilla_ratio,
        "data_sufficiency":         sufficiency,
        "sufficiency_note":         note,
        "gpu_estimates":            gpu_estimates,
        "gpu_utilization_assumed":  GPU_UTILIZATION,
        "recommendations":          recommendations,
        "reference": "Hoffmann et al. (2022) — Chinchilla Scaling Laws",
    }
