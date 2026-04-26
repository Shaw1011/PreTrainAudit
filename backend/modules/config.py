"""
config.py — Centralized configuration for PreTrainAudit.

All configurable constants in one place for easy tuning and maintenance.
"""

from typing import Final

# ── Upload Settings ───────────────────────────────────────────────────────────

MAX_UPLOAD_BYTES: Final[int] = 500 * 1024 * 1024  # 500 MB
CHUNK_SIZE_BYTES: Final[int] = 1024 * 1024  # 1 MB chunks for streaming

# ── Session Settings ───────────────────────────────────────────────────────────

SESSION_TTL_SECONDS: Final[int] = 3600  # 1 hour default TTL
CLEANUP_INTERVAL_SECONDS: Final[int] = 300  # Run cleanup every 5 minutes
MAX_SESSIONS: Final[int] = 100  # Maximum concurrent sessions

# ── Sampling Settings ───────────────────────────────────────────────────────────

DEFAULT_SAMPLE_SIZE: Final[int] = 100_000
MAX_SAMPLE_SIZE: Final[int] = 1_000_000
MIN_SAMPLE_SIZE: Final[int] = 1_000

# Auto-scaling sample sizes based on dataset size
SAMPLE_SCALE_FACTORS = {
    # (min_rows, max_rows): sample_percentage
    (0, 10_000): 1.0,        # Sample 100% for tiny datasets
    (10_000, 100_000): 0.5,  # Sample 50% for small datasets
    (100_000, 1_000_000): 0.2,  # Sample 20% for medium datasets
    (1_000_000, 10_000_000): 0.05,  # Sample 5% for large datasets
    (10_000_000, float("inf")): 0.01,  # Sample 1% for huge datasets
}

# ── Timeout Settings ───────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_SECONDS: Final[int] = 300  # 5 minutes max per operation
PROFILE_TIMEOUT_SECONDS: Final[int] = 600  # 10 minutes for profiling
RISK_ANALYSIS_TIMEOUT_SECONDS: Final[int] = 300  # 5 minutes for risk analysis

# ── Risk Score Confidence Intervals ─────────────────────────────────────────────

# Bootstrap iterations for confidence interval estimation
CONFIDENCE_BOOTSTRAP_ITERATIONS: Final[int] = 100
CONFIDENCE_LEVEL: Final[float] = 0.95  # 95% confidence interval

# ── DuckDB Settings ───────────────────────────────────────────────────────────

DUCKDB_MEMORY_LIMIT: Final[str] = "4GB"  # Limit DuckDB memory usage
DUCKDB_THREADS: Final[int] = 4  # Number of threads for parallel processing

# ── Model Size Parameters (millions) ───────────────────────────────────────────

MODEL_SIZE_PARAMS = {
    "Small": {
        "NLP": 10,              # ~10M (tiny BERT, DistilBERT-tiny)
        "Computer Vision": 5,   # ~5M (MobileNet-tiny)
        "Recommendation": 2,
        "Speech": 10,
    },
    "Medium": {
        "NLP": 125,             # ~125M (BERT-base, GPT-2)
        "Computer Vision": 50,  # ~50M (ResNet-50)
        "Recommendation": 20,
        "Speech": 74,           # ~74M (Whisper-small)
    },
    "Large": {
        "NLP": 7_000,           # ~7B (LLaMA-7B)
        "Computer Vision": 300, # ~300M (ViT-L)
        "Recommendation": 100,
        "Speech": 1_500,        # ~1.5B (Whisper-large)
    },
}

# ── GPU Benchmarks (TFLOPS-based for compute-accurate cost estimation) ────────
# tflops_fp16 = peak FP16 TFLOPS from vendor specs

GPU_BENCHMARKS = {
    "A100": {"tflops_fp16": 312,  "cost_per_hour": 3.0},
    "H100": {"tflops_fp16": 990,  "cost_per_hour": 8.0},
    "V100": {"tflops_fp16": 125,  "cost_per_hour": 2.5},
    "A10G": {"tflops_fp16": 71,   "cost_per_hour": 1.5},
    "T4":   {"tflops_fp16": 65,   "cost_per_hour": 0.5},
}

# Model FLOPS Utilization — real-world training typically achieves 30–50%
GPU_UTILIZATION: Final[float] = 0.4

# ── Domain Quality Thresholds ─────────────────────────────────────────────────

DOMAIN_QUALITY_THRESHOLDS = {
    "Healthcare": 80,
    "Finance": 75,
    "Social Media": 55,
    "General": 60,
}

# ── Valid Enum Values ───────────────────────────────────────────────────────────

VALID_AI_TYPES = {"NLP", "Computer Vision", "Recommendation", "Speech", "General"}
VALID_MODEL_SIZES = {"Small", "Medium", "Large"}
VALID_ARCHITECTURES = {"Transformer", "CNN", "RNN", "MoE"}
VALID_DOMAINS = {
    "Healthcare", "Finance", "Social Media", "General",
    "Computer Vision", "Music/Audio", "E-Commerce", "NLP/Text",
}
VALID_TASK_TYPES = {"Classification", "Generation", "Detection", "Recommendation"}
VALID_EXTENSIONS = {".csv", ".parquet", ".arrow", ".xlsx", ".xls", ".txt", ".jsonl", ".json"}

# ── Reproducibility ────────────────────────────────────────────────────────────

TOOL_VERSION: Final[str] = "1.1.0"
DEFAULT_RANDOM_SEED: Final[int] = 42
