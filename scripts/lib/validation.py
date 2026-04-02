"""Validation utilities for the profile HMM error model.

Compares trained model predictions against skiver summary CSV files
to verify internal consistency.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from .encoding import (
    BASE_TO_IDX,
    ERROR_TYPE_SUB_A,
    NUM_BASES,
)
from .profile_hmm import marginal_error_rate, stationary_distribution

logger = logging.getLogger(__name__)

IDX_TO_BASE = {v: k for k, v in BASE_TO_IDX.items()}

# ─── Summary CSV loaders ─────────────────────────────────────────────────────


def load_summary_error_rate(prefix: str | Path) -> dict[str, float]:
    """Load summary_error_rate.csv and return key metrics.

    Returns:
        Dict with keys like 'per_base_error_rate', 'effective_error_rate'.
    """
    path = Path(f"{prefix}.summary_error_rate.csv")
    if not path.exists():
        logger.warning("File not found: %s", path)
        return {}

    result = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if "per_base_error_rate" in row:
                result["per_base_error_rate"] = float(row["per_base_error_rate"])
            if "effective_error_rate" in row:
                result["effective_error_rate"] = float(row["effective_error_rate"])
    return result


def load_summary_phred(prefix: str | Path) -> list[dict[str, float]]:
    """Load summary_phred.csv.

    Returns:
        List of dicts with 'phred', 'error_rate', 'count' keys.
    """
    path = Path(f"{prefix}.summary_phred.csv")
    if not path.exists():
        logger.warning("File not found: %s", path)
        return []

    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "phred": float(row.get("phred", row.get("phred_score", 0))),
                "error_rate": float(row.get("error_rate", 0)),
                "count": float(row.get("count", row.get("total", 0))),
            })
    return rows


def load_summary_error_spectrum(
    prefix: str | Path,
) -> list[dict[str, str | float]]:
    """Load summary_error_spectrum.csv.

    Returns:
        List of dicts with 'operation', 'prev_base', 'next_base', 'total',
        'forward' keys.
    """
    path = Path(f"{prefix}.summary_error_spectrum.csv")
    if not path.exists():
        logger.warning("File not found: %s", path)
        return []

    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "operation": row["operation"],
                "prev_base": row.get("prev_base", ""),
                "next_base": row.get("next_base", ""),
                "total": float(row.get("total", 0)),
                "forward": float(row.get("forward", 0)),
            })
    return rows


# ─── Comparison functions ─────────────────────────────────────────────────────


def compare_error_rate(
    params: dict[str, torch.Tensor],
    prefix: str | Path,
) -> dict[str, float]:
    """Compare model marginal error rate to summary_error_rate.csv.

    Args:
        params: Trained model parameters.
        prefix: Skiver dump output prefix.

    Returns:
        Dict with 'model_error_rate', 'summary_error_rate', 'ratio'.
    """
    summary = load_summary_error_rate(prefix)
    if not summary:
        return {}

    pi = stationary_distribution(params)
    err_rates = marginal_error_rate(params)
    model_rate = float((pi * err_rates).sum().item())
    summary_rate = summary.get(
        "per_base_error_rate",
        summary.get("effective_error_rate", 0.0),
    )

    ratio = model_rate / summary_rate if summary_rate > 0 else float("inf")
    logger.info(
        "Error rate — model: %.6f, summary: %.6f, ratio: %.3f",
        model_rate, summary_rate, ratio,
    )
    return {
        "model_error_rate": model_rate,
        "summary_error_rate": summary_rate,
        "ratio": ratio,
    }


def model_substitution_spectrum(
    params: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Extract substitution spectrum from model parameters.

    Marginalises over states, positions, and prev_base context to produce
    relative rates for each substitution type.

    Args:
        params: Trained model parameters.

    Returns:
        Dict mapping operation string (e.g. 'A>C') to relative rate.
    """
    pi = stationary_distribution(params)
    logits = params["error_type_logits"]  # [S, 16, T, 10]
    probs = F.softmax(logits, dim=-1)

    spectrum = {}
    for tb_idx in range(NUM_BASES):
        for ob_idx in range(NUM_BASES):
            if tb_idx == ob_idx:
                continue
            tb = IDX_TO_BASE[tb_idx]
            ob = IDX_TO_BASE[ob_idx]
            # Substitution error_type index = 1 + obs_base_idx.
            et_idx = ERROR_TYPE_SUB_A + ob_idx
            # Weight by contexts where true_base = tb_idx.
            relevant_contexts = [
                pb_idx * NUM_BASES + tb_idx for pb_idx in range(NUM_BASES)
            ]
            rate = 0.0
            for ci in relevant_contexts:
                # probs weighted by state: sum over states.
                state_avg = (probs[:, ci, :, et_idx] * pi.unsqueeze(-1)).sum(0)
                rate += state_avg.mean().item()
            spectrum[f"{tb}>{ob}"] = rate / NUM_BASES

    return spectrum


def compare_substitution_spectrum(
    params: dict[str, torch.Tensor],
    prefix: str | Path,
) -> dict[str, dict[str, float]]:
    """Compare model substitution spectrum to summary_error_spectrum.csv.

    Args:
        params: Trained model parameters.
        prefix: Skiver dump output prefix.

    Returns:
        Dict mapping operation to {'model': rate, 'summary': count}.
    """
    summary_rows = load_summary_error_spectrum(prefix)
    if not summary_rows:
        return {}

    model_spectrum = model_substitution_spectrum(params)

    # Aggregate summary by operation (sum over contexts).
    summary_by_op: dict[str, float] = {}
    for row in summary_rows:
        op = str(row["operation"])
        if ">" in op and "-" not in op:
            summary_by_op[op] = summary_by_op.get(op, 0) + float(row["total"])

    result = {}
    for op in sorted(set(model_spectrum.keys()) | set(summary_by_op.keys())):
        result[op] = {
            "model": model_spectrum.get(op, 0.0),
            "summary": summary_by_op.get(op, 0.0),
        }
        logger.info(
            "  %s — model: %.6f, summary count: %.0f",
            op, result[op]["model"], result[op]["summary"],
        )

    return result
