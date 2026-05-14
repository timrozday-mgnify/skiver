#!/usr/bin/env python3
"""Preparse skiver base-observation TSVs into row-level HDF5 caches."""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from lib.context_h5_cache import cache_path, require_h5py, save_row_cache_h5

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path("../skiver_run")
DEFAULT_PLATFORMS = ("hq-illumina", "lq-illumina", "ont", "pacbio")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preparse skiver base-observation TSVs into row-level HDF5 caches.",
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
        help="Platform to preparse. Repeat to process multiple. Default: all platforms.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("context_error_cache"),
        help="Directory for HDF5 row caches.",
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


def _preparse_with_progress(
    *,
    platform: str,
    split: str,
    prefixes: Sequence[Path],
    include_outliers: bool,
    cache_dir: Path,
) -> Path:
    """Preparse TSV rows into an HDF5 cache with a Rich progress bar."""
    output_path = cache_path(cache_dir, platform, split, include_outliers)
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
            f"{platform} preparse {split}",
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
            output_path,
            platform=platform,
            split=split,
            prefixes=prefixes,
            include_outliers=include_outliers,
            progress_callback=update_progress,
            progress_interval=PROGRESS_INTERVAL,
        )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run TSV preparsing and HDF5 row-cache writing."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    require_h5py()

    platforms = tuple(args.platform) if args.platform else DEFAULT_PLATFORMS
    logger.info("Data root: %s", args.data_root)
    logger.info("Cache directory: %s", args.cache_dir)
    logger.info("Platforms: %s", ", ".join(platforms))
    logger.info("Include outliers: %s", args.include_outliers)

    wrote_any = False
    for platform in platforms:
        for split in ("train", "test"):
            prefixes = discover_prefixes(args.data_root, platform, split)
            if not prefixes:
                logger.warning("Skipping %s/%s: no base observation TSVs", platform, split)
                continue

            logger.info(
                "%s/%s preparsing %d prefix(es)",
                platform,
                split,
                len(prefixes),
            )
            output_path = _preparse_with_progress(
                platform=platform,
                split=split,
                prefixes=prefixes,
                include_outliers=args.include_outliers,
                cache_dir=args.cache_dir,
            )
            logger.info("Wrote %s", output_path)
            wrote_any = True

    if not wrote_any:
        logger.error("No HDF5 caches were written.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
