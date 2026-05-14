#!/usr/bin/env python3
"""Train non-HMM Pyro context error models and compare them with AIC."""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Sequence
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
    PreviousBasesErrorModel,
    aggregate_context_length_screen_counts,
    fit_bayesian_and_test,
    fit_and_test,
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
DEFAULT_CONTEXT_MIN = 1
DEFAULT_CONTEXT_MAX = 10
PROGRESS_INTERVAL = 50_000


def _prefix_from_base_path(path: Path) -> Path:
    """Return the skiver prefix for a base observations file path."""
    suffix = ".base_observations.tsv"
    text = str(path)
    if not text.endswith(suffix):
        raise ValueError(f"Unexpected base observations path: {path}")
    return Path(text[: -len(suffix)])


def discover_prefixes(data_root: Path, platform: str, split: str) -> list[Path]:
    """Return sorted skiver dump prefixes for a platform/split."""
    split_dir = data_root / platform / split
    paths = sorted(split_dir.glob("*.base_observations.tsv"))
    return [_prefix_from_base_path(path) for path in paths]


def _prefix_summary(prefixes: Sequence[Path]) -> str:
    """Return a compact human-readable prefix summary."""
    if not prefixes:
        return "none"
    return ", ".join(prefix.name for prefix in prefixes)


def _save_artifact(
    path: Path,
    *,
    platform: str,
    model_name: str,
    mle_fit: FitResult,
    vi_fit: BayesianFitResult,
    train_total: int,
    test_total: int,
    low_count_contexts: int,
    context_length: int,
) -> None:
    """Save a fitted model artifact."""
    mle_run_transform_values = _run_transform_values(mle_fit.params)
    vi_run_transform_values = _run_transform_values(vi_fit.params_mean)

    artifact = {
        "platform": platform,
        "model": model_name,
        "n_train": train_total,
        "n_test": test_total,
        "low_count_contexts": low_count_contexts,
        "context_length": context_length,
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
        "platform",
        "model",
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
        "--context-min",
        type=int,
        default=DEFAULT_CONTEXT_MIN,
        help=f"Minimum previous-base context length (default: {DEFAULT_CONTEXT_MIN}).",
    )
    parser.add_argument(
        "--context-max",
        type=int,
        default=DEFAULT_CONTEXT_MAX,
        help=f"Maximum previous-base context length (default: {DEFAULT_CONTEXT_MAX}).",
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
    args: argparse.Namespace,
) -> ContextLengthScreenCounts:
    """Load counts from HDF5 cache when possible, otherwise parse TSVs."""
    h5_path = cache_path(args.cache_dir, platform, split, include_outliers)
    if not args.no_cache:
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
    else:
        logger.info("%s/%s cache disabled; parsing TSV files", platform, split)

    counts = _aggregate_with_progress(
        platform=platform,
        split=split,
        prefixes=prefixes,
        include_outliers=include_outliers,
        context_lengths=context_lengths,
    )
    if args.write_cache:
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
    model_name: str,
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
            f"{platform}/{model_name} train",
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
    model_name: str,
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
            f"{platform}/{model_name} VI",
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
    model: PreviousBasesErrorModel,
    train_counts: ContextCounts,
    test_counts: ContextCounts,
    args: argparse.Namespace,
) -> tuple[FitResult, BayesianFitResult, int, int, int]:
    """Train a model from precomputed counts and return fit metadata."""
    logger.info(
        "Starting %s/%s fit from cached counts",
        platform,
        model.name,
    )
    logger.info(
        "%s/%s train counts: accepted=%d skipped=%d low_count_contexts=%d",
        platform,
        model.name,
        train_counts.total_observations,
        train_counts.skipped_rows,
        train_counts.low_count_contexts,
    )
    logger.info(
        "%s/%s test counts: accepted=%d skipped=%d low_count_contexts=%d",
        platform,
        model.name,
        test_counts.total_observations,
        test_counts.skipped_rows,
        test_counts.low_count_contexts,
    )
    if train_counts.total_observations == 0 or test_counts.total_observations == 0:
        raise ValueError(f"No usable observations for {platform}/{model.name}")

    logger.info(
        "%s/%s MLE fitting: context_length=%d steps=%d lr=%s clip_norm=%s "
        "pseudo_count=%s",
        platform,
        model.name,
        model.context_length,
        args.steps,
        args.lr,
        args.clip_norm,
        args.pseudo_count,
    )
    fit = _fit_with_progress(
        platform=platform,
        model_name=model.name,
        train_counts=train_counts,
        test_counts=test_counts,
        args=args,
    )
    logger.info(
        "%s/%s MLE complete: train_log_likelihood=%.4f test_log_likelihood=%.4f",
        platform,
        model.name,
        fit.train_log_likelihood,
        fit.test_log_likelihood,
    )
    vi_steps = args.vi_steps if args.vi_steps is not None else args.steps
    logger.info(
        "%s/%s VI fitting: steps=%d lr=%s prior_scale=%s",
        platform,
        model.name,
        vi_steps,
        args.vi_lr,
        args.prior_scale,
    )
    vi_fit = _fit_vi_with_progress(
        platform=platform,
        model_name=model.name,
        train_counts=train_counts,
        test_counts=test_counts,
        args=args,
    )
    logger.info(
        "%s/%s VI complete: train_log_likelihood=%.4f test_log_likelihood=%.4f",
        platform,
        model.name,
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
    platforms = tuple(args.platform) if args.platform else DEFAULT_PLATFORMS
    logger.info("Platforms: %s", ", ".join(platforms))
    if args.context_min < 1:
        raise ValueError("--context-min must be at least 1")
    if args.context_max < args.context_min:
        raise ValueError("--context-max must be greater than or equal to --context-min")
    context_lengths = tuple(range(args.context_min, args.context_max + 1))
    logger.info("Context lengths: %s", ", ".join(str(i) for i in context_lengths))
    rows: list[dict[str, object]] = []

    for platform in platforms:
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
        logger.debug("Test prefixes: %s", _prefix_summary(test_prefixes))

        logger.info(
            "%s loading train/test data once for context lengths %s",
            platform,
            ", ".join(str(i) for i in context_lengths),
        )
        train_platform_counts = _load_counts(
            platform=platform,
            split="train",
            prefixes=train_prefixes,
            include_outliers=args.include_outliers,
            context_lengths=context_lengths,
            args=args,
        )
        logger.info(
            "%s train loaded: accepted=%d skipped=%d",
            platform,
            train_platform_counts.total_observations,
            train_platform_counts.skipped_rows,
        )
        test_platform_counts = _load_counts(
            platform=platform,
            split="test",
            prefixes=test_prefixes,
            include_outliers=args.include_outliers,
            context_lengths=context_lengths,
            args=args,
        )
        logger.info(
            "%s test loaded: accepted=%d skipped=%d",
            platform,
            test_platform_counts.total_observations,
            test_platform_counts.skipped_rows,
        )

        models = tuple(PreviousBasesErrorModel(length) for length in context_lengths)
        for model in models:
            train_counts = train_platform_counts.by_length[model.context_length]
            test_counts = test_platform_counts.by_length[model.context_length]

            mle_fit, vi_fit, n_train, n_test, low_count_contexts = _run_model(
                platform=platform,
                model=model,
                train_counts=train_counts,
                test_counts=test_counts,
                args=args,
            )
            model_path = args.output_dir / f"context_{model.name}_{platform}.pt"
            _save_artifact(
                model_path,
                platform=platform,
                model_name=model.name,
                mle_fit=mle_fit,
                vi_fit=vi_fit,
                train_total=n_train,
                test_total=n_test,
                low_count_contexts=low_count_contexts,
                context_length=model.context_length,
            )
            rows.append(
                {
                    "platform": platform,
                    "model": model.name,
                    "context_length": model.context_length,
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
                }
            )
            rows.append(
                {
                    "platform": platform,
                    "model": model.name,
                    "context_length": model.context_length,
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
                }
            )
            logger.info(
                "%s/%s MLE test_log_likelihood=%.4f k=%d AIC=%.4f",
                platform,
                model.name,
                mle_fit.test_log_likelihood,
                mle_fit.num_parameters,
                mle_fit.aic,
            )
            logger.info(
                "%s/%s VI test_log_likelihood=%.4f test_elbo=%.4f",
                platform,
                model.name,
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
