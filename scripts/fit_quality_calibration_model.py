#!/usr/bin/env python3
"""Fit genome-blender Q-to-error calibration models from Skiver dump TSVs."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from lib.encoding import ERROR_TYPE_MATCH, NUM_ERROR_TYPES, encode_error_type

logger = logging.getLogger(__name__)

PHRED_MIN: Final[int] = 0
PHRED_MAX: Final[int] = 60
N_PHRED: Final[int] = PHRED_MAX - PHRED_MIN + 1
EPS: Final[float] = 1e-9


@dataclass(frozen=True)
class CalibrationCounts:
    """Deduplicated calibration counts by integer Phred score."""

    train_total: np.ndarray
    train_error: np.ndarray
    validation_total: np.ndarray
    validation_error: np.ndarray
    raw_rows: int
    unique_rows: int
    duplicate_rows: int
    conflicting_duplicates: int
    skipped_rows: int


@dataclass(frozen=True)
class FitSummary:
    """Fitted calibration parameters and evaluation metrics."""

    model: str
    params: dict[str, float]
    train_nll: float
    validation_nll: float
    train_observations: int
    validation_observations: int
    num_parameters: int

    @property
    def validation_aic(self) -> float:
        """Return an AIC-style validation criterion."""
        return 2.0 * self.num_parameters + 2.0 * self.validation_nll


def _parse_bool(value: str) -> bool:
    """Parse Skiver boolean text."""
    return value.lower() == "true"


def _normalise_q(phred: int) -> int:
    """Clip an integer Phred score to the supported range."""
    return min(max(phred, PHRED_MIN), PHRED_MAX)


def _nll(total: torch.Tensor, error: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """Return binomial negative log likelihood up to a constant."""
    prob = prob.clamp(EPS, 1.0 - EPS)
    return -((error * torch.log(prob)) + ((total - error) * torch.log1p(-prob))).sum()


def _log_linear_probs(
    q_values: torch.Tensor,
    raw: torch.Tensor,
    *,
    fit_bounds: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return log-linear probabilities and decoded parameters."""
    intercept = raw[0]
    slope = raw[1]
    if fit_bounds:
        floor = torch.sigmoid(raw[2]) * 0.05
        ceiling = floor + (0.5 - floor) * torch.sigmoid(raw[3])
    else:
        floor = torch.tensor(1e-7, dtype=q_values.dtype)
        ceiling = torch.tensor(0.5, dtype=q_values.dtype)
    prob = torch.pow(10.0, intercept + slope * q_values).clamp(floor, ceiling)
    params = {
        "qcal_intercept": float(intercept.detach().item()),
        "qcal_slope": float(slope.detach().item()),
        "qcal_floor": float(floor.detach().item()),
        "qcal_ceiling": float(ceiling.detach().item()),
    }
    return prob, params


def _sigmoid_probs(
    q_values: torch.Tensor,
    raw: torch.Tensor,
    *,
    fit_bounds: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return sigmoid probabilities and decoded parameters."""
    steepness = F.softplus(raw[0]) + 1e-6
    midpoint = raw[1]
    if fit_bounds:
        floor = torch.sigmoid(raw[2]) * 0.05
        ceiling = floor + (0.5 - floor) * torch.sigmoid(raw[3])
    else:
        floor = torch.tensor(1e-6, dtype=q_values.dtype)
        ceiling = torch.tensor(0.5, dtype=q_values.dtype)
    prob = floor + (ceiling - floor) * torch.sigmoid(-steepness * (q_values - midpoint))
    params = {
        "qcal_steepness": float(steepness.detach().item()),
        "qcal_midpoint": float(midpoint.detach().item()),
        "qcal_floor": float(floor.detach().item()),
        "qcal_ceiling": float(ceiling.detach().item()),
    }
    return prob, params


def _initial_log_linear(counts: CalibrationCounts) -> torch.Tensor:
    """Return a robust initial point for the log-linear model."""
    total = counts.train_total.astype(np.float64)
    error = counts.train_error.astype(np.float64)
    mask = total > 0
    q = np.arange(PHRED_MIN, PHRED_MAX + 1, dtype=np.float64)[mask]
    empirical = (error[mask] + 0.5) / (total[mask] + 1.0)
    y = np.log10(np.clip(empirical, 1e-7, 0.5))
    if q.size >= 2:
        slope, intercept = np.polyfit(q, y, 1)
    else:
        intercept, slope = -0.3, -0.08
    return torch.tensor([intercept, slope, -12.0, 0.0], dtype=torch.float64)


def _initial_sigmoid(counts: CalibrationCounts) -> torch.Tensor:
    """Return a stable initial point for the sigmoid model."""
    del counts
    return torch.tensor([math.log(math.exp(0.25) - 1.0), 20.0, -12.0, 0.0], dtype=torch.float64)


def _fit_model(
    counts: CalibrationCounts,
    *,
    model: str,
    fit_bounds: bool,
    steps: int,
    lr: float,
) -> FitSummary:
    """Fit one calibration model to aggregated Q counts."""
    q_values = torch.arange(PHRED_MIN, PHRED_MAX + 1, dtype=torch.float64)
    train_total = torch.tensor(counts.train_total, dtype=torch.float64)
    train_error = torch.tensor(counts.train_error, dtype=torch.float64)
    val_total = torch.tensor(counts.validation_total, dtype=torch.float64)
    val_error = torch.tensor(counts.validation_error, dtype=torch.float64)

    if model == "log-linear":
        raw = _initial_log_linear(counts).requires_grad_(True)
        prob_fn = _log_linear_probs
    elif model == "sigmoid":
        raw = _initial_sigmoid(counts).requires_grad_(True)
        prob_fn = _sigmoid_probs
    else:
        raise ValueError(f"Unknown calibration model: {model}")

    optimiser = torch.optim.Adam([raw], lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        probs, _ = prob_fn(q_values, raw, fit_bounds=fit_bounds)
        loss = _nll(train_total, train_error, probs)
        loss.backward()
        optimiser.step()

    optimiser_lbfgs = torch.optim.LBFGS(
        [raw],
        lr=0.25,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimiser_lbfgs.zero_grad()
        probs_inner, _ = prob_fn(q_values, raw, fit_bounds=fit_bounds)
        loss_inner = _nll(train_total, train_error, probs_inner)
        loss_inner.backward()
        return loss_inner

    optimiser_lbfgs.step(closure)
    probs, params = prob_fn(q_values, raw, fit_bounds=fit_bounds)
    train_nll = float(_nll(train_total, train_error, probs).detach().item())
    validation_nll = float(_nll(val_total, val_error, probs).detach().item())
    num_parameters = 4 if fit_bounds else 2
    return FitSummary(
        model=model,
        params=params,
        train_nll=train_nll,
        validation_nll=validation_nll,
        train_observations=int(counts.train_total.sum()),
        validation_observations=int(counts.validation_total.sum()),
        num_parameters=num_parameters,
    )


def _iter_tsv_paths(prefixes: Sequence[Path], tsvs: Sequence[Path]) -> list[Path]:
    """Resolve input prefixes and explicit TSV paths."""
    paths = [Path(f"{prefix}.base_observations.tsv") for prefix in prefixes]
    paths.extend(tsvs)
    if not paths:
        raise ValueError("At least one --input-prefix or --input-tsv is required")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input TSV(s): {missing}")
    return paths


def collect_counts(
    paths: Sequence[Path],
    *,
    validation_fraction: float,
    seed: int,
    passes_filter_only: bool,
    max_rows: int | None,
) -> CalibrationCounts:
    """Collect de-duplicated binary error counts by Phred score.

    Args:
        paths: Input Skiver base-observation TSV paths.
        validation_fraction: Fraction of unique rows assigned to validation.
        seed: Random seed for deterministic train/validation assignment.
        passes_filter_only: Whether to discard rows failing Skiver key filters.
        max_rows: Optional cap on unique training rows for smoke tests.

    Returns:
        Aggregated counts and de-duplication metadata.
    """
    rng = random.Random(seed)
    train_total = np.zeros(N_PHRED, dtype=np.int64)
    train_error = np.zeros(N_PHRED, dtype=np.int64)
    validation_total = np.zeros(N_PHRED, dtype=np.int64)
    validation_error = np.zeros(N_PHRED, dtype=np.int64)
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    raw_rows = 0
    unique_rows = 0
    duplicate_rows = 0
    conflicting_duplicates = 0
    skipped_rows = 0

    for path in paths:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {
                "read_pos",
                "true_base",
                "obs_base",
                "edit_op",
                "phred",
                "passes_filter",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
            id_field = (
                "read_id"
                if "read_id" in (reader.fieldnames or [])
                else "obs_id"
                if "obs_id" in (reader.fieldnames or [])
                else None
            )
            if id_field is None:
                raise ValueError(f"{path} must contain either read_id or obs_id")

            for row in reader:
                raw_rows += 1
                if max_rows is not None and unique_rows >= max_rows:
                    continue
                if passes_filter_only and not _parse_bool(row["passes_filter"]):
                    skipped_rows += 1
                    continue
                try:
                    phred = int(row["phred"])
                    read_id = int(row[id_field])
                    read_pos = int(row["read_pos"])
                except ValueError:
                    skipped_rows += 1
                    continue
                if phred < 0:
                    skipped_rows += 1
                    continue

                error_type = encode_error_type(
                    row["true_base"],
                    row["obs_base"],
                    row["edit_op"],
                )
                event = (_normalise_q(phred), int(error_type != ERROR_TYPE_MATCH))
                key = (read_id, read_pos)
                previous = seen.get(key)
                if previous is not None:
                    duplicate_rows += 1
                    if previous != event:
                        conflicting_duplicates += 1
                    continue
                seen[key] = event

                q, is_error = event
                if rng.random() < validation_fraction:
                    validation_total[q] += 1
                    validation_error[q] += is_error
                else:
                    train_total[q] += 1
                    train_error[q] += is_error
                unique_rows += 1

    return CalibrationCounts(
        train_total=train_total,
        train_error=train_error,
        validation_total=validation_total,
        validation_error=validation_error,
        raw_rows=raw_rows,
        unique_rows=unique_rows,
        duplicate_rows=duplicate_rows,
        conflicting_duplicates=conflicting_duplicates,
        skipped_rows=skipped_rows,
    )


def _curve_for_model(fit: FitSummary) -> list[float]:
    """Return fitted probabilities for all integer Q values."""
    q = np.arange(PHRED_MIN, PHRED_MAX + 1, dtype=np.float64)
    if fit.model == "log-linear":
        p = np.power(10.0, fit.params["qcal_intercept"] + fit.params["qcal_slope"] * q)
        p = np.clip(p, fit.params["qcal_floor"], fit.params["qcal_ceiling"])
    elif fit.model == "sigmoid":
        floor = fit.params["qcal_floor"]
        ceiling = fit.params["qcal_ceiling"]
        steepness = fit.params["qcal_steepness"]
        midpoint = fit.params["qcal_midpoint"]
        p = floor + (ceiling - floor) / (1.0 + np.exp(steepness * (q - midpoint)))
    else:
        raise ValueError(f"Unknown calibration model: {fit.model}")
    return p.astype(float).tolist()


def _counts_table(counts: CalibrationCounts) -> list[dict[str, int | float]]:
    """Return per-Q empirical count summaries."""
    rows = []
    total = counts.train_total + counts.validation_total
    error = counts.train_error + counts.validation_error
    for q in range(N_PHRED):
        n = int(total[q])
        e = int(error[q])
        rows.append(
            {
                "q": q + PHRED_MIN,
                "n": n,
                "errors": e,
                "empirical_error_rate": float(e / n) if n else 0.0,
                "train_n": int(counts.train_total[q]),
                "train_errors": int(counts.train_error[q]),
                "validation_n": int(counts.validation_total[q]),
                "validation_errors": int(counts.validation_error[q]),
            }
        )
    return rows


def _fit_summaries_to_json(fits: Sequence[FitSummary]) -> list[dict[str, object]]:
    """Return JSON-serialisable fit summaries."""
    return [
        {
            "model": fit.model,
            "params": fit.params,
            "train_nll": fit.train_nll,
            "validation_nll": fit.validation_nll,
            "validation_aic": fit.validation_aic,
            "train_observations": fit.train_observations,
            "validation_observations": fit.validation_observations,
            "num_parameters": fit.num_parameters,
            "fitted_error_rate_by_q": _curve_for_model(fit),
        }
        for fit in fits
    ]


def fit_quality_calibration(
    paths: Sequence[Path],
    *,
    models: Sequence[str],
    validation_fraction: float,
    seed: int,
    passes_filter_only: bool,
    max_rows: int | None,
    fit_bounds: bool,
    steps: int,
    lr: float,
) -> dict[str, object]:
    """Fit candidate quality calibration models and return an artifact dict."""
    counts = collect_counts(
        paths,
        validation_fraction=validation_fraction,
        seed=seed,
        passes_filter_only=passes_filter_only,
        max_rows=max_rows,
    )
    fits = [
        _fit_model(
            counts,
            model=model,
            fit_bounds=fit_bounds,
            steps=steps,
            lr=lr,
        )
        for model in models
    ]
    selected = min(fits, key=lambda fit: fit.validation_aic)
    qcal_params = {
        "quality_calibration_model": selected.model,
        **selected.params,
    }
    return {
        "artifact_type": "skiver_genome_blender_quality_calibration",
        "selected_model": selected.model,
        "selected_params": selected.params,
        "genome_blender_config": qcal_params,
        "phred_min": PHRED_MIN,
        "phred_max": PHRED_MAX,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "passes_filter_only": passes_filter_only,
        "fit_bounds": fit_bounds,
        "input_paths": [str(path) for path in paths],
        "deduplication": {
            "raw_rows": counts.raw_rows,
            "unique_rows": counts.unique_rows,
            "duplicate_rows": counts.duplicate_rows,
            "conflicting_duplicates": counts.conflicting_duplicates,
            "skipped_rows": counts.skipped_rows,
        },
        "counts_by_q": _counts_table(counts),
        "candidate_fits": _fit_summaries_to_json(fits),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit genome-blender Q-to-error calibration models from Skiver dumps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-prefix", type=Path, action="append", default=[])
    parser.add_argument("--input-tsv", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("log-linear", "sigmoid", "both"),
        default="log-linear",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--passes-filter-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--fixed-bounds",
        action="store_true",
        help="Keep genome-blender default floor/ceiling values instead of fitting them.",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the calibration model fitter."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")

    paths = _iter_tsv_paths(args.input_prefix, args.input_tsv)
    models = ("log-linear", "sigmoid") if args.model == "both" else (args.model,)
    artifact = fit_quality_calibration(
        paths,
        models=models,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        passes_filter_only=args.passes_filter_only,
        max_rows=args.max_rows,
        fit_bounds=not args.fixed_bounds,
        steps=args.steps,
        lr=args.lr,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")

    logger.info(
        "Wrote %s calibration to %s",
        artifact["selected_model"],
        args.output_json,
    )
    logger.info("Genome-blender config: %s", artifact["genome_blender_config"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
