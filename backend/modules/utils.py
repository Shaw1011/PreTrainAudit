"""
utils.py — Shared utilities for PreTrainAudit modules.

Centralises helper functions used across multiple modules to eliminate
code duplication and ensure consistent behaviour.

Consolidates:
  - _clamp()      (was defined 3× in quality, compatibility, utils)
  - _safe_path()  (was defined 4× in ingestion, profiler, adversarial, multi_domain)
  - _risk_level() (was defined 4× with different thresholds — now RiskScale)
"""

from __future__ import annotations

from dataclasses import dataclass


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a numeric value to [lo, hi] range."""
    return max(lo, min(hi, float(v)))


# Backward-compat alias (multi_domain_analyzer imports _clamp)
_clamp = clamp


def safe_sql_path(path: str) -> str:
    """Sanitise a filesystem path for use in DuckDB SQL queries.

    - Converts Windows backslashes to forward slashes
    - Escapes single quotes to prevent SQL injection

    The path must already be absolute (FastAPI upload handler guarantees this).
    We intentionally avoid Path.resolve() because it behaves differently
    per OS when given a path string from a different OS.
    """
    return path.replace("\\", "/").replace("'", "''")


# ── Configurable risk classification ──────────────────────────────────────────


@dataclass(frozen=True)
class _Tier:
    """Internal: one level in a RiskScale."""
    ceiling: float
    label: str


class RiskScale:
    """Configurable risk classifier — maps a numeric score to a severity label.

    Each module creates its own RiskScale with domain-appropriate thresholds,
    eliminating the need for duplicate ``_risk_level()`` functions.

    Parameters are keyword-only: ``LABEL=ceiling_value``.
    A score below the ceiling maps to that label.
    Scores above all ceilings get the ``default`` label.

    Example
    -------
    >>> scale = RiskScale(LOW=20, MEDIUM=50, HIGH=75, default="CRITICAL")
    >>> scale.classify(15)
    'LOW'
    >>> scale.classify(60)
    'HIGH'
    >>> scale.classify(90)
    'CRITICAL'
    """

    def __init__(self, default: str = "CRITICAL", **thresholds: float):
        self._tiers = sorted(
            [_Tier(ceiling=v, label=k) for k, v in thresholds.items()],
            key=lambda t: t.ceiling,
        )
        self._default = default

    def classify(self, score: float) -> str:
        """Return the risk label for the given numeric score."""
        for tier in self._tiers:
            if score < tier.ceiling:
                return tier.label
        return self._default

    def __repr__(self) -> str:
        tiers = ", ".join(f"{t.label}<{t.ceiling}" for t in self._tiers)
        return f"RiskScale({tiers}, default={self._default})"
