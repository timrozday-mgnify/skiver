#!/usr/bin/env python3
"""Train non-HMM Pyro context error models and compare them with AIC."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from lib.context_error_models import (
    BayesianFitResult,
    ContextCounts,
    ContextLengthScreenCounts,
    FitResult,
    aggregate_context_length_screen_counts,
    calibrate_to_rate,
    compute_marginal_error_rate,
    fit_bayesian_and_test,
    fit_and_test,
    subsample_context_counts,
)
from lib.context_h5_cache import (
    cache_path,
    load_counts_from_row_cache_h5,
    require_h5py,
    save_row_cache_h5,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path("../skiver_run")
DEFAULT_PLATFORMS = ("hq-illumina", "lq-illumina", "ont", "pacbio")
DEFAULT_MODEL_CONFIG = Path("model_config.json")
PROGRESS_INTERVAL = 50_000

MODEL_TYPES = ("combinatorial_context", "additive_context")


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a single model to train.

    Attributes:
        id: Unique identifier used in artifact filenames and CSV rows.
        type: Parameterisation — ``"combinatorial_context"`` (full joint) or
            ``"additive_context"`` (factored over positions).
        context_length: Number of preceding consensus bases to condition on.
        phred_lags: Number of previous Phred quality scores to use as covariates.
        position_features: Number of read-position features to use as covariates
            (0–2). Feature order: log1p(dist_to_end), log1p(read_pos).
    """

    id: str
    type: str
    context_length: int
    phred_lags: int = 0
    position_features: int = 0

    def __post_init__(self) -> None:
        if self.type not in MODEL_TYPES:
            raise ValueError(
                f"Unknown model type {self.type!r}. Must be one of {MODEL_TYPES}."
            )
        if self.context_length < 1:
            raise ValueError("context_length must be at least 1")
        if self.phred_lags < 0:
            raise ValueError("phred_lags must be non-negative")
        if not (0 <= self.position_features <= 2):
            raise ValueError("position_features must be 0, 1, or 2")

    @property
    def additive_context(self) -> bool:
        """Return True when this spec uses additive context parameterisation."""
        return self.type == "additive_context"


def load_model_config(path: Path) -> list[ModelSpec]:
    """Load and validate a JSON model config file.

    Args:
        path: Path to a JSON file with a ``"models"`` list.

    Returns:
        Ordered list of validated model specs.

    Raises:
        ValueError: If the config is malformed or contains duplicate IDs.
    """
    with open(path) as handle:
        raw = json.load(handle)

    if "models" not in raw:
        raise ValueError(f"{path}: JSON must have a top-level 'models' key")

    specs = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw["models"]):
        for field in ("id", "type", "context_length"):
            if field not in entry:
                raise ValueError(f"{path}: model[{i}] is missing field {field!r}")
        spec = ModelSpec(
            id=str(entry["id"]),
            type=str(entry["type"]),
            context_length=int(entry["context_length"]),
            phred_lags=int(entry.get("phred_lags", 0)),
            position_features=int(entry.get("position_features", 0)),
        )
        if spec.id in seen_ids:
            raise ValueError(f"{path}: duplicate model id {spec.id!r}")
        seen_ids.add(spec.id)
        specs.append(spec)

    if not specs:
        raise ValueError(f"{path}: 'models' list is empty")
    return specs


def _prefix_from_base_path(path: Path) -> Path:
    """Return the skiver prefix for a base observations file path."""
    suffix = ".base_observations.tsv"
    text = str(path)
    if not text.endswith(suffix):
        raise ValueError(f"Unexpected base observations path: {path}")
    return Path(text[: -len(suffix)])


def discover_prefixes(data_root: Path, platform: str, split: str | None) -> list[Path]:
    """Return sorted skiver dump prefixes for a platform/split.

    When ``split`` is ``None`` (--no-split mode), globs directly in
    ``data_root/platform/`` without a train/test subdirectory.
    """
    search_dir = data_root / platform if split is None else data_root / platform / split
    paths = sorted(search_dir.glob("*.base_observations.tsv"))
    return [_prefix_from_base_path(path) for path in paths]


def _prefix_summary(prefixes: Sequence[Path]) -> str:
    """Return a compact human-readable prefix summary."""
    if not prefixes:
        return "none"
    return ", ".join(prefix.name for prefix in prefixes)


def _counts_for_spec(
    spec: ModelSpec,
    screen_counts: ContextLengthScreenCounts,
    *,
    max_contexts: int | None = None,
    phred_lags_override: int = 0,
    position_features_override: int = 0,
) -> ContextCounts:
    """Return context counts with additive_context and phred/position lags set per the model spec.

    For additive_context models, optionally cap to the top-``max_contexts``
    most-observed context rows to bound training time. Phred and position sums are
    truncated to the effective lag/feature count (max of spec and override), or
    stripped when 0.
    """
    base = screen_counts.by_length[spec.context_length]
    if base.additive_context != spec.additive_context:
        base = dataclasses.replace(base, additive_context=spec.additive_context)
    effective_lags = max(spec.phred_lags, phred_lags_override)
    phred_sums = base.phred_context_sums
    if phred_sums is not None:
        if effective_lags == 0:
            phred_sums = None
        elif phred_sums.shape[-1] > effective_lags:
            phred_sums = phred_sums[..., :effective_lags]
    if phred_sums is not base.phred_context_sums:
        base = dataclasses.replace(base, phred_context_sums=phred_sums)
    effective_pos = max(spec.position_features, position_features_override)
    position_sums = base.position_context_sums
    if position_sums is not None:
        if effective_pos == 0:
            position_sums = None
        elif position_sums.shape[-1] > effective_pos:
            position_sums = position_sums[..., :effective_pos]
    if position_sums is not base.position_context_sums:
        base = dataclasses.replace(base, position_context_sums=position_sums)
    if spec.additive_context and max_contexts is not None:
        base = subsample_context_counts(base, max_contexts)
    return base


def _save_artifact(
    path: Path,
    *,
    spec: ModelSpec,
    platform: str,
    train_prefixes: Sequence[Path],
    test_prefixes: Sequence[Path],
    data_root: Path,
    mle_fit: FitResult,
    vi_fit: BayesianFitResult,
    train_total: int,
    test_total: int,
    low_count_contexts: int,
    calibration: dict[str, float] | None = None,
) -> None:
    """Save a fitted model artifact."""
    mle_run_transform_values = _run_transform_values(mle_fit.params)
    vi_run_transform_values = _run_transform_values(vi_fit.params_mean)

    artifact = {
        "model_id": spec.id,
        "model_type": spec.type,
        "platform": platform,
        "data_root": str(data_root),
        "train_prefixes": [str(p) for p in train_prefixes],
        "test_prefixes": [str(p) for p in test_prefixes],
        "n_train": train_total,
        "n_test": test_total,
        "low_count_contexts": low_count_contexts,
        "context_length": spec.context_length,
        "parameterization": spec.type,
        "target": "error_type",
        "notes": (
            "Conditional categorical error model. Sequence context is treated as "
            "given; this is not a full generative model of read sequence."
        ),
        "maximum_likelihood": {
            "params": mle_fit.params,
            "losses": torch.tensor(mle_fit.losses),
            "train_log_likelihood": mle_fit.train_log_likelihood,
            "test_log_likelihood": mle_fit.test_log_likelihood,
            "num_parameters": mle_fit.num_parameters,
            "aic": mle_fit.aic,
            "run_transform_values": mle_run_transform_values,
        },
        "variational_inference": {
            "params_mean": vi_fit.params_mean,
            "params_stdev": vi_fit.params_stdev,
            "inference_params": vi_fit.inference_params,
            "losses": torch.tensor(vi_fit.losses),
            "train_log_likelihood": vi_fit.train_log_likelihood,
            "test_log_likelihood": vi_fit.test_log_likelihood,
            "train_elbo": vi_fit.train_elbo,
            "test_elbo": vi_fit.test_elbo,
            "prior_scale": vi_fit.prior_scale,
            "run_transform_values_mean": vi_run_transform_values,
        },
        "calibration": calibration,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    logger.info("Saved %s", path)


def _run_transform_values(params: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Return learned monotone repeat-count transform values when present."""
    if "run_step_unconstrained" not in params:
        return None
    run_steps = F.softplus(params["run_step_unconstrained"])
    return torch.cat(
        [torch.zeros(1, dtype=run_steps.dtype), torch.cumsum(run_steps, dim=0)]
    )


def _write_comparison(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write AIC comparison rows to CSV."""
    fieldnames = [
        "model_id",
        "model_type",
        "platform",
        "context_length",
        "inference",
        "n_train",
        "n_test",
        "train_log_likelihood",
        "test_log_likelihood",
        "train_elbo",
        "test_elbo",
        "num_parameters",
        "aic",
        "low_count_contexts",
        "prior_scale",
        "weibull_target_rate",
        "uncalibrated_rate",
        "calibration_offset",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train non-HMM Pyro context error models and compare AIC.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Root containing platform train/test folders (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--platform",
        action="append",
        choices=DEFAULT_PLATFORMS,
        help="Platform to train. Repeat to train multiple. Default: all platforms.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("context_error_models"),
        help="Directory for .pt artifacts and AIC CSV.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("context_error_cache"),
        help="Directory containing HDF5 preparsed row caches.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore HDF5 caches and parse TSV files directly.",
    )
    parser.add_argument(
        "--write-cache",
        action="store_true",
        help="Write an HDF5 row cache when TSV fallback parsing is needed.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help=f"JSON file listing models to train (default: {DEFAULT_MODEL_CONFIG}).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of SVI steps per model (default: 1000).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate (default: 0.05).",
    )
    parser.add_argument(
        "--vi-steps",
        type=int,
        default=None,
        help="Number of variational inference steps. Default: same as --steps.",
    )
    parser.add_argument(
        "--vi-lr",
        type=float,
        default=0.01,
        help="Variational inference learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--prior-scale",
        type=float,
        default=2.0,
        help="Normal prior scale for Bayesian parameters (default: 2.0).",
    )
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=10.0,
        help="Gradient clip norm (default: 10).",
    )
    parser.add_argument(
        "--pseudo-count",
        type=float,
        default=0.5,
        help="Positive value used only to initialise logits (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Include observations from keys that failed the outlier filter.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help=(
            "Disable train/test split: look for TSV files directly in "
            "<data-root>/<platform>/ instead of the train/ and test/ subdirectories. "
            "Training data is used for both fitting and evaluation."
        ),
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=None,
        metavar="N",
        help=(
            "For additive_context models: cap the training count tensor to the N "
            "most-observed contexts before fitting. Has no effect on "
            "combinatorial_context models. Recommended: 4096–16384 for "
            "context_length >= 8 (default: no cap)."
        ),
    )
    parser.add_argument(
        "--weibull-rate",
        type=float,
        default=None,
        metavar="RATE",
        help=(
            "Target per-base error rate from skiver Weibull fit "
            "(``per_base_error_rate`` column of ``{prefix}.summary_error_rate.csv``). "
            "When provided, a calibration offset is computed post-training and stored "
            "in the artifact.  Not provided by default (no calibration)."
        ),
    )
    parser.add_argument(
        "--phred-lags",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of previous Phred quality scores to use as context covariates "
            "(overrides per-model phred_lags from model config; 0 = disabled). "
            "Applied as a global override: all model specs get at least this many lags."
        ),
    )
    parser.add_argument(
        "--position-features",
        type=int,
        default=0,
        metavar="N",
        choices=(0, 1, 2),
        help=(
            "Number of read-position features to use as context covariates (0–2). "
            "Feature order: log1p(dist_to_end), log1p(read_pos). "
            "Overrides per-model position_features from model config; 0 = disabled."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def _aggregate_with_progress(
    *,
    platform: str,
    split: str,
    prefixes: Sequence[Path],
    include_outliers: bool,
    context_lengths: Sequence[int],
    num_phred_lags: int = 0,
    num_position_features: int = 0,
) -> ContextLengthScreenCounts:
    """Aggregate all context-length counts while rendering a Rich progress bar."""
    state = {"scanned": 0, "accepted": 0, "skipped": 0}
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn(
            "scanned={task.fields[scanned]} "
            "accepted={task.fields[accepted]} "
            "skipped={task.fields[skipped]}"
        ),
        TimeElapsedColumn(),
    )

    with progress:
        task_id = progress.add_task(
            f"{platform} aggregate {split}",
            total=None,
            scanned="0",
            accepted="0",
            skipped="0",
        )

        def update_progress(
            path: Path,
            scanned_delta: int,
            accepted_delta: int,
            skipped_delta: int,
        ) -> None:
            del path
            state["scanned"] += scanned_delta
            state["accepted"] += accepted_delta
            state["skipped"] += skipped_delta
            progress.update(
                task_id,
                advance=scanned_delta,
                scanned=f"{state['scanned']:,}",
                accepted=f"{state['accepted']:,}",
                skipped=f"{state['skipped']:,}",
            )

        return aggregate_context_length_screen_counts(
            prefixes,
            context_lengths=context_lengths,
            passes_filter_only=not include_outliers,
            progress_callback=update_progress,
            progress_interval=PROGRESS_INTERVAL,
            num_phred_lags=num_phred_lags,
            num_position_features=num_position_features,
        )


def _row_cache_with_progress(
    *,
    platform: str,
    split: str,
    prefixes: Sequence[Path],
    include_outliers: bool,
    cache_dir: Path,
) -> Path:
    """Write an HDF5 row cache while rendering a Rich progress bar."""
    h5_path = cache_path(cache_dir, platform, split, include_outliers)
    state = {"scanned": 0, "accepted": 0, "skipped": 0}
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn(
            "scanned={task.fields[scanned]} "
            "accepted={task.fields[accepted]} "
            "skipped={task.fields[skipped]}"
        ),
        TimeElapsedColumn(),
    )

    with progress:
        task_id = progress.add_task(
            f"{platform} cache {split}",
            total=None,
            scanned="0",
            accepted="0",
            skipped="0",
        )

        def update_progress(
            path: Path,
            scanned_delta: int,
            accepted_delta: int,
            skipped_delta: int,
        ) -> None:
            del path
            state["scanned"] += scanned_delta
            state["accepted"] += accepted_delta
            state["skipped"] += skipped_delta
            progress.update(
                task_id,
                advance=scanned_delta,
                scanned=f"{state['scanned']:,}",
                accepted=f"{state['accepted']:,}",
                skipped=f"{state['skipped']:,}",
            )

        save_row_cache_h5(
            h5_path,
            platform=platform,
            split=split,
            prefixes=prefixes,
            include_outliers=include_outliers,
            progress_callback=update_progress,
            progress_interval=PROGRESS_INTERVAL,
        )
    return h5_path


def _aggregate_row_cache_with_progress(
    *,
    platform: str,
    split: str,
    h5_path: Path,
    prefixes: Sequence[Path],
    include_outliers: bool,
    context_lengths: Sequence[int],
) -> ContextLengthScreenCounts | None:
    """Aggregate requested context lengths from an HDF5 row cache."""
    state = {"scanned": 0}
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("cached_rows={task.fields[scanned]}"),
        TimeElapsedColumn(),
    )

    with progress:
        task_id = progress.add_task(
            f"{platform} load cache {split}",
            total=None,
            scanned="0",
        )

        def update_progress(
            path: Path,
            scanned_delta: int,
            accepted_delta: int,
            skipped_delta: int,
        ) -> None:
            del path, accepted_delta, skipped_delta
            state["scanned"] += scanned_delta
            progress.update(
                task_id,
                advance=scanned_delta,
                scanned=f"{state['scanned']:,}",
            )

        return load_counts_from_row_cache_h5(
            h5_path,
            prefixes=prefixes,
            context_lengths=context_lengths,
            include_outliers=include_outliers,
            progress_callback=update_progress,
        )


def _load_counts(
    *,
    platform: str,
    split: str,
    prefixes: Sequence[Path],
    include_outliers: bool,
    context_lengths: Sequence[int],
    num_phred_lags: int = 0,
    num_position_features: int = 0,
    args: argparse.Namespace,
) -> ContextLengthScreenCounts:
    """Load counts from HDF5 cache when possible, otherwise parse TSVs."""
    # H5 cache does not store Phred or position scores; bypass when either is active.
    use_cache = not args.no_cache and num_phred_lags == 0 and num_position_features == 0
    if use_cache:
        h5_path = cache_path(args.cache_dir, platform, split, include_outliers)
        cached_counts = _aggregate_row_cache_with_progress(
            platform=platform,
            split=split,
            h5_path=h5_path,
            prefixes=prefixes,
            include_outliers=include_outliers,
            context_lengths=context_lengths,
        )
        if cached_counts is not None:
            return cached_counts
        if args.write_cache:
            logger.info("%s/%s HDF5 row cache unavailable; writing it", platform, split)
            _row_cache_with_progress(
                platform=platform,
                split=split,
                prefixes=prefixes,
                include_outliers=include_outliers,
                cache_dir=args.cache_dir,
            )
            cached_counts = _aggregate_row_cache_with_progress(
                platform=platform,
                split=split,
                h5_path=h5_path,
                prefixes=prefixes,
                include_outliers=include_outliers,
                context_lengths=context_lengths,
            )
            if cached_counts is not None:
                return cached_counts
        logger.info("%s/%s HDF5 row cache unavailable; parsing TSV files", platform, split)
    elif num_phred_lags > 0 or num_position_features > 0:
        logger.info(
            "%s/%s phred_lags=%d position_features=%d: bypassing H5 cache, parsing TSV files",
            platform, split, num_phred_lags, num_position_features,
        )
    else:
        logger.info("%s/%s cache disabled; parsing TSV files", platform, split)

    counts = _aggregate_with_progress(
        platform=platform,
        split=split,
        prefixes=prefixes,
        include_outliers=include_outliers,
        context_lengths=context_lengths,
        num_phred_lags=num_phred_lags,
        num_position_features=num_position_features,
    )
    if args.write_cache and num_phred_lags == 0:
        require_h5py()
        _row_cache_with_progress(
            platform=platform,
            split=split,
            prefixes=prefixes,
            include_outliers=include_outliers,
            cache_dir=args.cache_dir,
        )
    return counts


def _fit_with_progress(
    *,
    platform: str,
    model_id: str,
    train_counts: ContextCounts,
    test_counts: ContextCounts,
    args: argparse.Namespace,
) -> FitResult:
    """Fit a model while rendering a Rich progress bar."""
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("loss={task.fields[loss]}"),
        TimeElapsedColumn(),
    )
    with progress:
        task_id = progress.add_task(
            f"{platform}/{model_id} train",
            total=args.steps,
            loss="n/a",
        )

        def update_progress(step: int, loss: float) -> None:
            del step
            progress.update(task_id, advance=1, loss=f"{loss:.4f}")

        return fit_and_test(
            train_counts,
            test_counts,
            lr=args.lr,
            num_steps=args.steps,
            clip_norm=args.clip_norm,
            pseudo_count=args.pseudo_count,
            seed=args.seed,
            progress_callback=update_progress,
        )


def _fit_vi_with_progress(
    *,
    platform: str,
    model_id: str,
    train_counts: ContextCounts,
    test_counts: ContextCounts,
    args: argparse.Namespace,
) -> BayesianFitResult:
    """Fit a variational posterior while rendering a Rich progress bar."""
    vi_steps = args.vi_steps if args.vi_steps is not None else args.steps
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("loss={task.fields[loss]}"),
        TimeElapsedColumn(),
    )
    with progress:
        task_id = progress.add_task(
            f"{platform}/{model_id} VI",
            total=vi_steps,
            loss="n/a",
        )

        def update_progress(step: int, loss: float) -> None:
            del step
            progress.update(task_id, advance=1, loss=f"{loss:.4f}")

        return fit_bayesian_and_test(
            train_counts,
            test_counts,
            lr=args.vi_lr,
            num_steps=vi_steps,
            clip_norm=args.clip_norm,
            pseudo_count=args.pseudo_count,
            prior_scale=args.prior_scale,
            seed=args.seed,
            progress_callback=update_progress,
        )


def _run_model(
    *,
    platform: str,
    spec: ModelSpec,
    train_counts: ContextCounts,
    test_counts: ContextCounts,
    args: argparse.Namespace,
) -> tuple[FitResult, BayesianFitResult, int, int, int]:
    """Train a model from precomputed counts and return fit metadata."""
    logger.info("Starting %s/%s fit from cached counts", platform, spec.id)
    logger.info(
        "%s/%s train counts: accepted=%d skipped=%d low_count_contexts=%d",
        platform,
        spec.id,
        train_counts.total_observations,
        train_counts.skipped_rows,
        train_counts.low_count_contexts,
    )
    logger.info(
        "%s/%s test counts: accepted=%d skipped=%d low_count_contexts=%d",
        platform,
        spec.id,
        test_counts.total_observations,
        test_counts.skipped_rows,
        test_counts.low_count_contexts,
    )
    if train_counts.total_observations == 0 or test_counts.total_observations == 0:
        raise ValueError(f"No usable observations for {platform}/{spec.id}")

    logger.info(
        "%s/%s MLE fitting: context_length=%d additive=%s steps=%d lr=%s clip_norm=%s "
        "pseudo_count=%s",
        platform,
        spec.id,
        spec.context_length,
        spec.additive_context,
        args.steps,
        args.lr,
        args.clip_norm,
        args.pseudo_count,
    )
    fit = _fit_with_progress(
        platform=platform,
        model_id=spec.id,
        train_counts=train_counts,
        test_counts=test_counts,
        args=args,
    )
    logger.info(
        "%s/%s MLE complete: train_log_likelihood=%.4f test_log_likelihood=%.4f",
        platform,
        spec.id,
        fit.train_log_likelihood,
        fit.test_log_likelihood,
    )
    vi_steps = args.vi_steps if args.vi_steps is not None else args.steps
    logger.info(
        "%s/%s VI fitting: steps=%d lr=%s prior_scale=%s",
        platform,
        spec.id,
        vi_steps,
        args.vi_lr,
        args.prior_scale,
    )
    vi_fit = _fit_vi_with_progress(
        platform=platform,
        model_id=spec.id,
        train_counts=train_counts,
        test_counts=test_counts,
        args=args,
    )
    logger.info(
        "%s/%s VI complete: train_log_likelihood=%.4f test_log_likelihood=%.4f",
        platform,
        spec.id,
        vi_fit.train_log_likelihood,
        vi_fit.test_log_likelihood,
    )
    return (
        fit,
        vi_fit,
        train_counts.total_observations,
        test_counts.total_observations,
        train_counts.low_count_contexts,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run model training and AIC comparison."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger.info("Data root: %s", args.data_root)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Cache directory: %s", args.cache_dir)
    logger.info("Use cache: %s", not args.no_cache)
    logger.info("Write cache on TSV fallback: %s", args.write_cache)
    logger.info("Include outliers: %s", args.include_outliers)
    logger.info("Model config: %s", args.model_config)

    model_specs = load_model_config(args.model_config)
    logger.info(
        "Loaded %d model specs: %s",
        len(model_specs),
        ", ".join(s.id for s in model_specs),
    )

    context_lengths = sorted({spec.context_length for spec in model_specs})
    logger.info("Context lengths required: %s", ", ".join(str(i) for i in context_lengths))
    max_phred_lags = max(max(spec.phred_lags for spec in model_specs), args.phred_lags)
    if max_phred_lags > 0:
        logger.info("Max phred_lags across all specs: %d", max_phred_lags)
    max_position_features = max(max(spec.position_features for spec in model_specs), args.position_features)
    if max_position_features > 0:
        logger.info("Max position_features across all specs: %d", max_position_features)

    platforms = tuple(args.platform) if args.platform else DEFAULT_PLATFORMS
    logger.info("Platforms: %s", ", ".join(platforms))

    rows: list[dict[str, object]] = []

    for platform in platforms:
        if args.no_split:
            train_prefixes = discover_prefixes(args.data_root, platform, None)
            test_prefixes = train_prefixes
            logger.info(
                "%s discovered prefixes (no-split): %d",
                platform,
                len(train_prefixes),
            )
        else:
            train_prefixes = discover_prefixes(args.data_root, platform, "train")
            test_prefixes = discover_prefixes(args.data_root, platform, "test")
            logger.info(
                "%s discovered prefixes: train=%d test=%d",
                platform,
                len(train_prefixes),
                len(test_prefixes),
            )
        if not train_prefixes or not test_prefixes:
            logger.warning("Skipping %s: missing train or test prefixes", platform)
            continue
        logger.debug("Train prefixes: %s", _prefix_summary(train_prefixes))
        if not args.no_split:
            logger.debug("Test prefixes: %s", _prefix_summary(test_prefixes))

        logger.info(
            "%s loading data for context lengths %s",
            platform,
            ", ".join(str(i) for i in context_lengths),
        )
        train_platform_counts = _load_counts(
            platform=platform,
            split="train",
            prefixes=train_prefixes,
            include_outliers=args.include_outliers,
            context_lengths=context_lengths,
            num_phred_lags=max_phred_lags,
            num_position_features=max_position_features,
            args=args,
        )
        logger.info(
            "%s train loaded: accepted=%d skipped=%d",
            platform,
            train_platform_counts.total_observations,
            train_platform_counts.skipped_rows,
        )
        if args.no_split:
            test_platform_counts = train_platform_counts
        else:
            test_platform_counts = _load_counts(
                platform=platform,
                split="test",
                prefixes=test_prefixes,
                include_outliers=args.include_outliers,
                context_lengths=context_lengths,
                num_phred_lags=max_phred_lags,
                num_position_features=max_position_features,
                args=args,
            )
        logger.info(
            "%s test loaded: accepted=%d skipped=%d",
            platform,
            test_platform_counts.total_observations,
            test_platform_counts.skipped_rows,
        )

        for spec in model_specs:
            train_counts = _counts_for_spec(spec, train_platform_counts, max_contexts=args.max_contexts, phred_lags_override=args.phred_lags, position_features_override=args.position_features)
            test_counts = _counts_for_spec(spec, test_platform_counts, phred_lags_override=args.phred_lags, position_features_override=args.position_features)

            mle_fit, vi_fit, n_train, n_test, low_count_contexts = _run_model(
                platform=platform,
                spec=spec,
                train_counts=train_counts,
                test_counts=test_counts,
                args=args,
            )

            calibration: dict[str, float] | None = None
            if args.weibull_rate is not None:
                raw_mle = compute_marginal_error_rate(
                    train_counts.counts,
                    mle_fit.params,
                    run_values=train_counts.run_values,
                    additive_context=train_counts.additive_context,
                    context_indices=train_counts.context_indices,
                    phred_context_sums=train_counts.phred_context_sums,
                    position_context_sums=train_counts.position_context_sums,
                )
                delta_mle = calibrate_to_rate(
                    train_counts.counts,
                    mle_fit.params,
                    args.weibull_rate,
                    run_values=train_counts.run_values,
                    additive_context=train_counts.additive_context,
                    context_indices=train_counts.context_indices,
                    phred_context_sums=train_counts.phred_context_sums,
                    position_context_sums=train_counts.position_context_sums,
                )
                raw_vi = compute_marginal_error_rate(
                    train_counts.counts,
                    vi_fit.params_mean,
                    run_values=train_counts.run_values,
                    additive_context=train_counts.additive_context,
                    context_indices=train_counts.context_indices,
                    phred_context_sums=train_counts.phred_context_sums,
                    position_context_sums=train_counts.position_context_sums,
                )
                delta_vi = calibrate_to_rate(
                    train_counts.counts,
                    vi_fit.params_mean,
                    args.weibull_rate,
                    run_values=train_counts.run_values,
                    additive_context=train_counts.additive_context,
                    context_indices=train_counts.context_indices,
                    phred_context_sums=train_counts.phred_context_sums,
                    position_context_sums=train_counts.position_context_sums,
                )
                calibration = {
                    "weibull_target_rate": args.weibull_rate,
                    "uncalibrated_mle_rate": raw_mle,
                    "calibration_offset_mle": delta_mle,
                    "uncalibrated_vi_rate": raw_vi,
                    "calibration_offset_vi": delta_vi,
                }
                logger.info(
                    "%s/%s MLE calibration: rate %.6f → %.6f (δ=%.4f)",
                    platform,
                    spec.id,
                    raw_mle,
                    args.weibull_rate,
                    delta_mle,
                )
                logger.info(
                    "%s/%s VI  calibration: rate %.6f → %.6f (δ=%.4f)",
                    platform,
                    spec.id,
                    raw_vi,
                    args.weibull_rate,
                    delta_vi,
                )

            model_path = args.output_dir / f"{spec.id}_{platform}.pt"
            _save_artifact(
                model_path,
                spec=spec,
                platform=platform,
                train_prefixes=train_prefixes,
                test_prefixes=test_prefixes,
                data_root=args.data_root,
                mle_fit=mle_fit,
                vi_fit=vi_fit,
                train_total=n_train,
                test_total=n_test,
                low_count_contexts=low_count_contexts,
                calibration=calibration,
            )
            rows.append(
                {
                    "model_id": spec.id,
                    "model_type": spec.type,
                    "platform": platform,
                    "context_length": spec.context_length,
                    "inference": "maximum_likelihood",
                    "n_train": n_train,
                    "n_test": n_test,
                    "train_log_likelihood": mle_fit.train_log_likelihood,
                    "test_log_likelihood": mle_fit.test_log_likelihood,
                    "train_elbo": "",
                    "test_elbo": "",
                    "num_parameters": mle_fit.num_parameters,
                    "aic": mle_fit.aic,
                    "low_count_contexts": low_count_contexts,
                    "prior_scale": "",
                    "weibull_target_rate": args.weibull_rate if args.weibull_rate is not None else "",
                    "uncalibrated_rate": calibration["uncalibrated_mle_rate"] if calibration else "",
                    "calibration_offset": calibration["calibration_offset_mle"] if calibration else "",
                }
            )
            rows.append(
                {
                    "model_id": spec.id,
                    "model_type": spec.type,
                    "platform": platform,
                    "context_length": spec.context_length,
                    "inference": "variational_inference",
                    "n_train": n_train,
                    "n_test": n_test,
                    "train_log_likelihood": vi_fit.train_log_likelihood,
                    "test_log_likelihood": vi_fit.test_log_likelihood,
                    "train_elbo": vi_fit.train_elbo,
                    "test_elbo": vi_fit.test_elbo,
                    "num_parameters": "",
                    "aic": "",
                    "low_count_contexts": low_count_contexts,
                    "prior_scale": vi_fit.prior_scale,
                    "weibull_target_rate": args.weibull_rate if args.weibull_rate is not None else "",
                    "uncalibrated_rate": calibration["uncalibrated_vi_rate"] if calibration else "",
                    "calibration_offset": calibration["calibration_offset_vi"] if calibration else "",
                }
            )
            logger.info(
                "%s/%s MLE test_log_likelihood=%.4f k=%d AIC=%.4f",
                platform,
                spec.id,
                mle_fit.test_log_likelihood,
                mle_fit.num_parameters,
                mle_fit.aic,
            )
            logger.info(
                "%s/%s VI test_log_likelihood=%.4f test_elbo=%.4f",
                platform,
                spec.id,
                vi_fit.test_log_likelihood,
                vi_fit.test_elbo,
            )

    if not rows:
        logger.error("No models were trained.")
        return 1

    _write_comparison(args.output_dir / "context_model_aic.csv", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
