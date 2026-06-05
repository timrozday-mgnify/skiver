#!/usr/bin/env python3
"""Diagnose observed, truth, and fitted marginal error-rate discrepancies."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from benchmark_simulated_context_model import (  # noqa: E402
    ERROR_TYPE_NAMES,
    NUM_ERROR_TYPES,
    _error_rate,
    _normalise,
    _observed_counts,
    _observed_records_from_dump,
)
from simulate_errors import load_model, probabilities_for_context  # noqa: E402
from lib.context_h5_cache import (  # noqa: E402
    MISSING_CONTEXT_BASE,
    _rolling_context_codes,
    _row_cache_context_events,
)


def _load_json(path: Path) -> dict[str, object]:
    """Load a JSON object."""
    with open(path) as handle:
        return json.load(handle)


def _load_artifact(path: Path) -> dict[str, object]:
    """Load a torch artifact."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _weibull_from_records(
    model_path: Path,
    records: Sequence[tuple[int, int, int, int]],
    v: int,
    *,
    use_vi: bool = False,
) -> dict[str, object]:
    """Compute model-predicted marginal Weibull parameters from sparse records.

    Uses a mixture-of-geometrics approximation: each unique context c contributes
    a geometric first-error-time distribution with rate (1 − p_match_c), weighted
    by its total count in ``records``.  The marginal survival function
    S(t) = Σ_c w_c · p_match_c^t is then linearised to fit Weibull λ and β.

    Args:
        model_path: Path to a trained ``.pt`` artifact.
        records: Sparse (context_index, true_base, target, count) tuples.
        v: Value-window length; survival is evaluated at t = 1 … v.
        use_vi: Use variational-inference posterior-mean parameters.

    Returns:
        Dict with keys ``lambda``, ``beta``, ``window_averaged_rate``, and
        ``survival`` (list length v).
    """
    model = load_model(model_path, use_vi=use_vi)
    context_total: dict[int, float] = defaultdict(float)
    context_match: dict[int, float] = {}
    for context_index, _true_base, _target, count in records:
        context_total[context_index] += count
        if context_index not in context_match:
            probs = probabilities_for_context(model, context_index, None)
            context_match[context_index] = float(probs[0])

    if not context_total:
        nan = float("nan")
        return {"lambda": nan, "beta": nan, "window_averaged_rate": nan, "survival": []}

    total = sum(context_total.values())
    t = np.arange(1, v + 1, dtype=float)
    S = np.zeros(v, dtype=float)
    for ctx, w in context_total.items():
        p = context_match[ctx]
        S += (w / total) * np.power(p, t)

    # Fit Weibull via log(-log(S(t))) = log λ + β·log(t)
    valid = (S > 1e-15) & (S < 1.0 - 1e-15)
    if valid.sum() < 2:
        nan = float("nan")
        return {"lambda": nan, "beta": nan, "window_averaged_rate": nan, "survival": S.tolist()}
    y = np.log(-np.log(S[valid]))
    x = np.log(t[valid])
    coeffs = np.polyfit(x, y, 1)
    beta = float(max(coeffs[0], 1e-6))
    lam = float(np.exp(coeffs[1]))
    rate = (1.0 - math.exp(-lam * v**beta)) / v
    return {
        "lambda": lam,
        "beta": beta,
        "window_averaged_rate": rate,
        "survival": S.tolist(),
    }


def _weibull_from_csv(csv_path: Path, v: int) -> dict[str, object] | None:
    """Return observed Weibull parameters from a skiver analyze summary CSV.

    Args:
        csv_path: Path to ``*.summary_error_rate.csv`` produced by ``skiver analyze``.
        v: Value-window length for computing the window-averaged rate.

    Returns:
        Dict with ``lambda_mean``, ``beta_mean``, ``window_averaged_rate`` and
        per-key lists, or None if the file is missing / unparseable.
    """
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("lambda") and row.get("beta"):
                rows.append(row)
    if not rows:
        return None
    lambdas = [float(r["lambda"]) for r in rows]
    betas = [float(r["beta"]) for r in rows]
    rates = [
        (1.0 - math.exp(-lam * v**beta)) / v
        for lam, beta in zip(lambdas, betas)
    ]
    return {
        "lambda_mean": float(np.mean(lambdas)),
        "beta_mean": float(np.mean(betas)),
        "window_averaged_rate": float(np.mean(rates)),
        "lambda_values": lambdas,
        "beta_values": betas,
    }


def _record_counts(records: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
    """Return observed counts by error type from sparse context records."""
    return _observed_counts(records)


def _expected_counts(
    model_path: Path,
    records: Sequence[tuple[int, int, int, int]],
    *,
    use_vi: bool = False,
    mask_impossible_substitutions: bool,
) -> np.ndarray:
    """Return model-expected counts on the record exposure."""
    model = load_model(model_path, use_vi=use_vi)
    opportunities: dict[tuple[int, int | None], int] = defaultdict(int)
    for context_index, true_base, _, count in records:
        key_true_base = true_base if mask_impossible_substitutions and true_base >= 0 else None
        opportunities[(context_index, key_true_base)] += count

    expected = np.zeros(NUM_ERROR_TYPES, dtype=np.float64)
    for (context_index, true_base), count in opportunities.items():
        expected += count * probabilities_for_context(model, context_index, true_base)
    return expected


def _counts_summary(counts: np.ndarray) -> dict[str, object]:
    """Return count, probability, and error-rate summary."""
    return {
        "total": float(counts.sum()),
        "counts": counts.astype(float).tolist(),
        "probs": _normalise(counts).astype(float).tolist(),
        "error_rate": _error_rate(counts),
    }


def _model_exposure_summary(
    *,
    label: str,
    model_path: Path,
    records: Sequence[tuple[int, int, int, int]],
    include_vi: bool,
) -> list[dict[str, object]]:
    """Return masked/unmasked model marginal rates for one exposure."""
    rows = []
    for inference, use_vi in [("maximum_likelihood", False), ("variational_inference", True)]:
        if use_vi and not include_vi:
            continue
        for masking_label, mask in [
            ("training_unmasked", False),
            ("generation_masked", True),
        ]:
            counts = _expected_counts(
                model_path,
                records,
                use_vi=use_vi,
                mask_impossible_substitutions=mask,
            )
            rows.append(
                {
                    "label": label,
                    "model": str(model_path),
                    "inference": inference,
                    "probability_mode": masking_label,
                    **_counts_summary(counts),
                }
            )
    return rows


def _records_from_h5_cache(path: Path, context_length: int) -> list[tuple[int, int, int, int]]:
    """Return sparse context/true-base/error records from a context row cache."""
    import h5py  # type: ignore[import-not-found]

    with h5py.File(path, "r") as handle:
        rows_group = handle["rows"]
        obs_starts = rows_group["obs_start"][:].astype(bool, copy=False)
        prev_bases = rows_group["prev_base"][:]
        true_bases = rows_group["true_base"][:]
        targets = rows_group["target"][:]
        event_bases, row_event_ends, available_history = _row_cache_context_events(
            obs_starts,
            prev_bases,
            true_bases,
        )

    valid_rows = available_history >= context_length
    if not np.any(valid_rows):
        return []
    context_codes_by_end = _rolling_context_codes(event_bases, context_length)
    context_indices = context_codes_by_end[row_event_ends[valid_rows]]
    valid_true_bases = true_bases[valid_rows].astype(np.int64, copy=False)
    valid_true_bases = np.where(
        valid_true_bases == MISSING_CONTEXT_BASE,
        -1,
        valid_true_bases,
    )
    valid_targets = targets[valid_rows].astype(np.int64, copy=False)
    combined = (
        context_indices.astype(np.int64, copy=False) * (5 * NUM_ERROR_TYPES)
        + (valid_true_bases + 1) * NUM_ERROR_TYPES
        + valid_targets
    )
    counts = np.bincount(combined)
    records = []
    nonzero = np.flatnonzero(counts)
    for idx in nonzero:
        count = int(counts[idx])
        target = int(idx % NUM_ERROR_TYPES)
        tmp = idx // NUM_ERROR_TYPES
        true_base = int(tmp % 5) - 1
        context_index = int(tmp // 5)
        records.append((context_index, true_base, target, count))
    return records


def _simulated_split_summary(
    *,
    split: str,
    synthetic_results: Path,
    source_model: Path,
    retrained_model: Path,
    v: int,
) -> dict[str, object]:
    """Return simulated split diagnostic summaries."""
    records = _observed_records_from_dump(
        synthetic_results / f"dump_{split}.base_observations.tsv",
        context_length=int(_load_artifact(retrained_model)["context_length"]),
    )
    observed_counts = _record_counts(records)
    match_metrics = _load_json(synthetic_results / "skiver_bam_error_match_metrics.json")
    rate_metrics = _load_json(synthetic_results / "rate_recovery_metrics.json")
    split_match = next(row for row in match_metrics["splits"] if row["split"] == split)
    split_rate = next(row for row in rate_metrics["splits"] if row["split"] == split)

    denominator_rows = []
    bam_truth = split_rate.get("physical_bam_truth", {})
    if bam_truth.get("available", False):
        denominator_rows.append({
            "quantity": "full_bam_truth",
            "denominator": sum(bam_truth["counts"]),
            "errors": bam_truth["counts"][1:],
            "error_count": sum(bam_truth["counts"][1:]),
            "error_rate": bam_truth["error_rate"],
        })
    covered = split_match.get("skiver_covered_coordinate_confusion")
    if covered is not None:
        denominator_rows.extend([
            {
                "quantity": "skiver_covered_bam_truth",
                "denominator": covered["universe"],
                "error_count": covered["true_positive"] + covered["false_negative"],
                "error_rate": (
                    (covered["true_positive"] + covered["false_negative"]) / covered["universe"]
                ),
            },
            {
                "quantity": "skiver_covered_skiver_calls",
                "denominator": covered["universe"],
                "error_count": covered["true_positive"] + covered["false_positive"],
                "error_rate": (
                    (covered["true_positive"] + covered["false_positive"]) / covered["universe"]
                ),
            },
        ])
    denominator_rows.append({
        "quantity": "skiver_window_rows",
        "denominator": float(observed_counts.sum()),
        "error_count": float(observed_counts[1:].sum()),
        "error_rate": _error_rate(observed_counts),
    })

    # Observed Weibull from skiver analyze on the train data (only meaningful for train split).
    weibull_csv = synthetic_results / f"dump_{split}_weibull.summary_error_rate.csv"
    observed_weibull = _weibull_from_csv(weibull_csv, v)

    # Also surface the stored weibull_result from rate_recovery_metrics (train only).
    if split == "train" and rate_metrics.get("observed_weibull") is not None:
        observed_weibull = rate_metrics["observed_weibull"]

    # Compute marginal Weibull predicted by source and retrained models.
    retrained_artifact = _load_artifact(retrained_model)
    retrained_weibull_mle = retrained_artifact.get("marginal_weibull", {}).get("maximum_likelihood")
    retrained_weibull_vi = retrained_artifact.get("marginal_weibull", {}).get("variational_inference")

    return {
        "split": split,
        "denominators": denominator_rows,
        "observed_window": _counts_summary(observed_counts),
        "observed_weibull": observed_weibull,
        "source_model_weibull": _weibull_from_records(source_model, records, v, use_vi=False),
        "retrained_model_weibull_mle": (
            retrained_weibull_mle
            if retrained_weibull_mle is not None
            else _weibull_from_records(retrained_model, records, v, use_vi=False)
        ),
        "retrained_model_weibull_vi": (
            retrained_weibull_vi
            if retrained_weibull_vi is not None
            else _weibull_from_records(retrained_model, records, v, use_vi=True)
            if retrained_artifact.get("variational_inference") is not None
            else None
        ),
        "source_model_expected": _model_exposure_summary(
            label=f"synthetic_{split}_source",
            model_path=source_model,
            records=records,
            include_vi=False,
        ),
        "retrained_model_expected": _model_exposure_summary(
            label=f"synthetic_{split}_retrained",
            model_path=retrained_model,
            records=records,
            include_vi=True,
        ),
    }


def _real_split_summary(
    *,
    split: str,
    cache_path: Path,
    real_model: Path,
    existing_model: Path | None,
    v: int,
) -> dict[str, object]:
    """Return real-metagenome observed-vs-fitted rate summaries."""
    context_length = int(_load_artifact(real_model)["context_length"])
    records = _records_from_h5_cache(cache_path, context_length)
    observed_counts = _record_counts(records)
    model_rows = _model_exposure_summary(
        label=f"real_{split}_rerun",
        model_path=real_model,
        records=records,
        include_vi=True,
    )
    weibull_rows = [
        {
            "label": f"real_{split}_rerun",
            "inference": "maximum_likelihood",
            **_weibull_from_records(real_model, records, v, use_vi=False),
        },
        {
            "label": f"real_{split}_rerun",
            "inference": "variational_inference",
            **_weibull_from_records(real_model, records, v, use_vi=True),
        },
    ]
    if existing_model is not None and existing_model.exists():
        model_rows.extend(
            _model_exposure_summary(
                label=f"real_{split}_existing_source",
                model_path=existing_model,
                records=records,
                include_vi=True,
            )
        )
        weibull_rows.extend([
            {
                "label": f"real_{split}_existing_source",
                "inference": "maximum_likelihood",
                **_weibull_from_records(existing_model, records, v, use_vi=False),
            },
        ])
    return {
        "split": split,
        "cache": str(cache_path),
        "observed": _counts_summary(observed_counts),
        "model_expected": model_rows,
        "model_weibull": weibull_rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Summarise marginal-rate diagnostics for synthetic and real data.",
    )
    parser.add_argument("--synthetic-results", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--retrained-model", type=Path, required=True)
    parser.add_argument("--real-model", type=Path, required=True)
    parser.add_argument("--existing-real-model", type=Path, default=None)
    parser.add_argument("--real-cache-dir", type=Path, default=Path("context_error_cache"))
    parser.add_argument("--platform", default="hq-illumina")
    parser.add_argument(
        "--v",
        type=int,
        default=None,
        help=(
            "Value-window length used to compute marginal Weibull parameters. "
            "Defaults to the value stored in rate_recovery_metrics.json, or 13."
        ),
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run diagnostics."""
    args = parse_args(argv)
    synthetic_results = args.synthetic_results.resolve()
    real_cache_dir = args.real_cache_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Determine v: from CLI flag, then from rate_recovery_metrics, then default 13.
    v: int = args.v or 13
    rate_metrics_path = synthetic_results / "rate_recovery_metrics.json"
    if rate_metrics_path.exists():
        rate_metrics = _load_json(rate_metrics_path)
        v = int(args.v or rate_metrics.get("v", 13))  # type: ignore[arg-type]

    result = {
        "schema_version": 1,
        "description": (
            "Marginal-rate diagnostics comparing Skiver observed rates, BAM truth "
            "rates, and fitted model expected rates under masked and unmasked "
            "probability modes."
        ),
        "error_type_names": list(ERROR_TYPE_NAMES),
        "v": v,
        "synthetic": [
            _simulated_split_summary(
                split=split,
                synthetic_results=synthetic_results,
                source_model=args.source_model.resolve(),
                retrained_model=args.retrained_model.resolve(),
                v=v,
            )
            for split in ("train", "test")
        ],
        "real_metagenome": [
            _real_split_summary(
                split=split,
                cache_path=real_cache_dir / f"{args.platform}_{split}_filtered_rows.h5",
                real_model=args.real_model.resolve(),
                existing_model=(
                    args.existing_real_model.resolve()
                    if args.existing_real_model is not None
                    else None
                ),
                v=v,
            )
            for split in ("train", "test")
        ],
    }
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
