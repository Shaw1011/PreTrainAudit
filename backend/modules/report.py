"""
report.py — Full structured audit report generator.

FIXES:
  - None-safe score extraction: all risk scores default to 0 if None/missing
  - avg_risk calculation uses or-0 pattern to avoid NaN contamination
  - generate_report handles empty module dicts gracefully
  - verdict logic handles None compat_score
"""

from datetime import datetime, timezone
import random
import os

from .config import TOOL_VERSION, DEFAULT_RANDOM_SEED, CONFIDENCE_LEVEL
from .utils import RiskScale
from .utils_sampling import compute_risk_confidence_interval

_RISK = RiskScale(LOW=20, MEDIUM=45, HIGH=70, default="CRITICAL")


def _safe(d: dict, *keys, default=0):
    """Traverse nested dict safely, returning default on any miss or None."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d if d is not None else default


def generate_report(
    summary:       dict,
    profile:       dict,
    quality:       dict,
    adversarial:   dict,
    memorization:  dict,
    fairness:      dict,
    contamination: dict,
    cost:          dict,
    compatibility: dict,
    ai_config:     dict,
    filename:      str,
    random_seed:   int = None,
    sample_sizes:  dict = None,
) -> dict:
    """Generate a comprehensive audit report.
    
    Args:
        random_seed: Seed used for random operations (for reproducibility)
        sample_sizes: Dict of {risk_type: sample_size} for confidence intervals
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Use provided seed or generate one for reproducibility
    if random_seed is None:
        random_seed = random.randint(0, 2**31 - 1)

    # None-safe risk scores
    risk_scores = {
        "adversarial":    _safe(adversarial,   "vulnerability_score",    default=0),
        "memorization":   _safe(memorization,  "privacy_leakage_score",  default=0),
        "fairness":       _safe(fairness,       "drift_risk_score",       default=0),
        "contamination":  _safe(contamination,  "contamination_score",    default=0),
    }
    # All values coerced to float with fallback to 0
    risk_scores = {k: float(v or 0) for k, v in risk_scores.items()}
    avg_risk     = sum(risk_scores.values()) / max(len(risk_scores), 1)
    overall_risk = _RISK.classify(avg_risk)
    
    # Compute confidence intervals for risk scores
    sample_sizes = sample_sizes or {}
    risk_scores_with_ci = {}
    for name, score in risk_scores.items():
        sample_size = sample_sizes.get(name, 1000)
        ci = compute_risk_confidence_interval(score, sample_size)
        risk_scores_with_ci[name] = {
            "value": round(score, 2),
            "confidence_interval": ci.to_dict(),
        }

    # Collect all issues and recommendations
    all_issues = (
        _safe_list(quality,       "issues")       +
        _safe_list(adversarial,   "issues")       +
        _safe_list(memorization,  "issues")       +
        _safe_list(fairness,      "issues")       +
        _safe_list(contamination, "flags")        +
        _safe_list(compatibility, "issues")
    )
    all_recs = (
        _safe_list(quality,       "recommendations") +
        _safe_list(adversarial,   "recommendations") +
        _safe_list(memorization,  "recommendations") +
        _safe_list(fairness,      "recommendations") +
        _safe_list(contamination, "recommendations") +
        _safe_list(compatibility, "recommendations") +
        _safe_list(cost,          "recommendations")
    )

    # Deduplicate preserving order
    seen, deduped_recs = set(), []
    for r in all_recs:
        if r not in seen:
            seen.add(r)
            deduped_recs.append(r)

    # Split critical vs warning
    critical_keywords = {"CRITICAL", "MISMATCH", "SEVERE", "INSUFFICIENT", "GDPR"}
    critical_issues = [i for i in all_issues if any(kw in i.upper() for kw in critical_keywords)]
    other_issues    = [i for i in all_issues if i not in critical_issues]

    compat_score  = float(_safe(compatibility, "compatibility_score", default=0))
    quality_score = float(_safe(quality,       "total_score",         default=0))
    quality_grade = _safe(quality, "grade", default="?")

    return {
        "report_meta": {
            "generated_at": timestamp,
            "filename":     filename,
            "tool":         f"PreTrainAudit v{TOOL_VERSION}",
            "author":       "github.com/Shaw1011",
            "random_seed":  random_seed,
            "confidence_level": CONFIDENCE_LEVEL,
            "environment": {
                "python_version": _get_python_version(),
                "platform": _get_platform(),
            },
        },
        "executive_summary": {
            "compatibility_score":  compat_score,
            "suitability":          _safe(compatibility, "suitability", default="UNKNOWN"),
            "data_quality_score":   quality_score,
            "quality_grade":        quality_grade,
            "overall_risk_level":   overall_risk,
            "size_verdict":         _safe(compatibility, "size_verdict", default="UNKNOWN"),
            "total_issues_found":   len(all_issues),
            "critical_issues":      len(critical_issues),
        },
        "dataset_summary":   summary,
        "data_quality":      quality,
        "risk_audit": {
            "adversarial_vulnerability": adversarial,
            "memorization_risk":         memorization,
            "fairness_drift":            fairness,
            "benchmark_contamination":   contamination,
            "risk_scores":               risk_scores_with_ci,
            "overall_risk_level":        overall_risk,
        },
        "ai_compatibility":      compatibility,
        "training_cost_estimate": cost,
        "prioritized_issues": {
            "critical": critical_issues,
            "warnings": other_issues,
        },
        "recommendations":          deduped_recs,
        "best_model_recommendation": _safe(compatibility, "best_model_recommendation", default={}),
        "ai_config_evaluated":       ai_config,
        "verdict": _final_verdict(compat_score, quality_score, overall_risk, len(critical_issues)),
    }


def _safe_list(d: dict, key: str) -> list:
    val = d.get(key, []) if isinstance(d, dict) else []
    return val if isinstance(val, list) else []


def _final_verdict(compat: float, quality: float, risk: str, critical: int) -> dict:
    if critical > 0:
        return {
            "status":  "NOT_READY",
            "message": f"{critical} critical issue(s) must be resolved before training. "
                       "Proceeding risks failed runs or unsafe model behavior.",
            "color":   "red",
        }
    if compat >= 75 and quality >= 70 and risk in ("LOW", "MEDIUM"):
        return {
            "status":  "READY",
            "message": "Dataset is ready for training with the selected configuration.",
            "color":   "green",
        }
    if compat >= 50 or quality >= 55:
        return {
            "status":  "CONDITIONALLY_READY",
            "message": "Dataset is usable but has known weaknesses. Address recommendations before training.",
            "color":   "yellow",
        }
    return {
        "status":  "NOT_READY",
        "message": "Dataset requires significant improvement before training can proceed effectively.",
        "color":   "red",
    }





def _get_python_version() -> str:
    """Get Python version string."""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_platform() -> str:
    """Get platform information."""
    import platform
    return f"{platform.system()} {platform.release()}"
