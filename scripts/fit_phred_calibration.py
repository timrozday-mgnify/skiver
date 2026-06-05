#!/usr/bin/env python3
"""Fit P(Q | error_type) calibration tables from skiver dump --base TSV outputs.

For each platform, accumulates per-Phred-score counts broken down by error type
and writes a JSON calibration file.  The calibration can then be passed to
simulate_errors.py to sample realistic quality scores conditioned on the actual
sampled error type, rather than the context probability.

Usage::

    python scripts/fit_phred_calibration.py \\
        --data-root ../skiver_run \\
        --platform hq-illumina \\
        --cache-dir context_error_cache

    python scripts/fit_phred_calibration.py \\
        --data-root ../skiver_run \\
        --cache-dir context_error_cache   # all platforms, train split only
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from lib.encoding import (
    NUM_ERROR_TYPES,
    encode_error_type,
)
from preparse_context_error_data import DEFAULT_PLATFORMS, discover_prefixes

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_PHRED_MIN: Final[int] = 0
_PHRED_MAX: Final[int] = 60
_N_PHRED: Final[int] = _PHRED_MAX - _PHRED_MIN + 1  # 61 values

_ERROR_TYPE_NAMES: Final[tuple[str, ...]] = (
    "match",
    "sub_to_A",
    "sub_to_C",
    "sub_to_G",
    "sub_to_T",
    "ins_A",
    "ins_C",
    "ins_G",
    "ins_T",
    "deletion",
)


# ── Core accumulation ─────────────────────────────────────────────────────────


def _accumulate_tsv(
    tsv_path: Path,
    counts: np.ndarray,
    *,
    include_outliers: bool,
) -> int:
    """Accumulate Phred counts from a single base_observations TSV.

    Args:
        tsv_path: Path to a ``*.base_observations.tsv`` file.
        counts: Int64 array of shape [10, 61] to update in-place.
        include_outliers: If False, skip rows where ``passes_filter`` is false.

    Returns:
        Number of rows processed (after filtering).
    """
    n_processed = 0
    logger.info(
        "Accumulating Phred counts from %s (%.0f MB)…",
        tsv_path.name,
        tsv_path.stat().st_size / 1e6,
    )
    _PROGRESS = 500_000
    n_rows = 0
    with open(tsv_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            n_rows += 1
            if n_rows % _PROGRESS == 0:
                logger.info("  phred calibration: %d rows read", n_rows)
            if not include_outliers and row.get("passes_filter", "true") == "false":
                continue

            phred_raw = row.get("phred", "-1")
            try:
                phred = int(phred_raw)
            except ValueError:
                continue

            if phred < 0:
                # Deletion or FASTA (no quality score) — skip for Q accumulation.
                continue

            true_base = row.get("true_base", "N")
            obs_base = row.get("obs_base", "N")
            edit_op = row.get("edit_op", "NA")

            error_type = encode_error_type(true_base, obs_base, edit_op)
            q = min(max(phred, _PHRED_MIN), _PHRED_MAX)
            counts[error_type, q] += 1
            n_processed += 1

    logger.info("  phred calibration: %d rows read, %d accepted", n_rows, n_processed)
    return n_processed


def fit_calibration(
    data_root: Path,
    platform: str,
    split: str,
    *,
    include_outliers: bool = False,
) -> dict:
    """Fit P(Q | error_type) from all TSV files for a platform/split.

    Args:
        data_root: Root directory containing platform sub-directories.
        platform: Platform name (e.g. ``"hq-illumina"``).
        split: Data split (``"train"`` or ``"test"``).
        include_outliers: If True, include observations from keys that failed
            the outlier filter.

    Returns:
        Calibration dict with keys: platform, split, error_type_names,
        phred_min, phred_max, counts, probs, n_observations.
    """
    prefixes = discover_prefixes(data_root, platform, split)
    if not prefixes:
        logger.warning("No TSV prefixes found for %s/%s — skipping", platform, split)
        return {}

    counts = np.zeros((NUM_ERROR_TYPES, _N_PHRED), dtype=np.int64)
    total_rows = 0

    for prefix in prefixes:
        tsv_path = Path(str(prefix) + ".base_observations.tsv")
        if not tsv_path.exists():
            logger.warning("Missing TSV: %s", tsv_path)
            continue
        n = _accumulate_tsv(tsv_path, counts, include_outliers=include_outliers)
        total_rows += n
        logger.debug("  %s: %d rows", tsv_path.name, n)

    logger.info(
        "%s/%s: %d total rows across %d file(s)",
        platform, split, total_rows, len(prefixes),
    )

    # Laplace smoothing: avoids zero probabilities for unobserved (error, Q) pairs.
    # Deletion row (index 9) has no phred observations; probs[9] will be uniform-ish
    # after smoothing — the simulator checks for this and emits no quality byte.
    probs = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + _N_PHRED)

    return {
        "platform": platform,
        "split": split,
        "include_outliers": include_outliers,
        "error_type_names": list(_ERROR_TYPE_NAMES),
        "phred_min": _PHRED_MIN,
        "phred_max": _PHRED_MAX,
        "counts": counts.tolist(),
        "probs": probs.astype(float).tolist(),
        "n_observations": int(total_rows),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────


def fit_calibration_from_tsv(
    tsv_path: Path,
    *,
    platform: str = "unknown",
    include_outliers: bool = False,
) -> dict:
    """Fit P(Q | error_type) from a single base_observations TSV file.

    Args:
        tsv_path: Path to a ``*.base_observations.tsv`` file.
        platform: Platform label to embed in the output (informational).
        include_outliers: If False, skip rows where ``passes_filter`` is false.

    Returns:
        Calibration dict in the same format as :func:`fit_calibration`.
    """
    counts = np.zeros((NUM_ERROR_TYPES, _N_PHRED), dtype=np.int64)
    n = _accumulate_tsv(tsv_path, counts, include_outliers=include_outliers)
    logger.info("fit_calibration_from_tsv: %d rows from %s", n, tsv_path)
    probs = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + _N_PHRED)
    return {
        "platform": platform,
        "split": "train",
        "include_outliers": include_outliers,
        "error_type_names": list(_ERROR_TYPE_NAMES),
        "phred_min": _PHRED_MIN,
        "phred_max": _PHRED_MAX,
        "counts": counts.tolist(),
        "probs": probs.astype(float).tolist(),
        "n_observations": int(n),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fit P(Q | error_type) Phred calibration tables from "
            "skiver dump --base TSV files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base-tsv",
        type=Path,
        metavar="FILE",
        help=(
            "Fit directly from a single base_observations TSV file. "
            "When set, --data-root and --platform are ignored and "
            "--output is required."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Output JSON path. Required when --base-tsv is given.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("../skiver_run"),
        metavar="DIR",
        help="Root directory containing platform sub-directories.",
    )
    p.add_argument(
        "--platform",
        action="append",
        choices=DEFAULT_PLATFORMS,
        metavar="PLATFORM",
        help=(
            "Platform to fit. Repeat to process multiple. "
            f"Default: all ({', '.join(DEFAULT_PLATFORMS)})."
        ),
    )
    p.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="train",
        help="Which data split to use for fitting.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("context_error_cache"),
        metavar="DIR",
        help="Directory in which calibration JSON files are written.",
    )
    p.add_argument(
        "--include-outliers",
        action="store_true",
        help="Include observations from keys that failed the outlier filter.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run Phred calibration fitting.

    Args:
        argv: Command-line arguments; defaults to sys.argv.

    Returns:
        Exit code (0 on success, 1 if nothing was written).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.base_tsv is not None:
        if args.output is None:
            logger.error("--output is required when --base-tsv is given.")
            return 1
        cal = fit_calibration_from_tsv(
            args.base_tsv,
            include_outliers=args.include_outliers,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(cal, fh)
        logger.info("Wrote %s  (n=%d)", args.output, cal["n_observations"])
        return 0

    platforms = tuple(args.platform) if args.platform else DEFAULT_PLATFORMS
    splits = ("train", "test") if args.split == "both" else (args.split,)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    wrote_any = False

    for platform in platforms:
        for split in splits:
            logger.info("Fitting calibration for %s / %s …", platform, split)
            cal = fit_calibration(
                args.data_root,
                platform,
                split,
                include_outliers=args.include_outliers,
            )
            if not cal:
                continue

            outlier_tag = "_with_outliers" if args.include_outliers else ""
            out_path = args.cache_dir / f"{platform}_{split}{outlier_tag}_phred_calibration.json"
            with open(out_path, "w") as fh:
                json.dump(cal, fh)
            logger.info("Wrote %s  (n=%d)", out_path, cal["n_observations"])
            wrote_any = True

    if not wrote_any:
        logger.error("No calibration files were written.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
