"""HDF5 row-cache helpers for context error model training."""
from __future__ import annotations

import csv
import importlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import torch

from .context_error_models import (
    BASE_TO_IDX,
    NUM_CONTEXT_BASES,
    ContextCounts,
    ContextLengthScreenCounts,
    PreviousBasesErrorModel,
    ProgressCallback,
    _normalise_base,
    _parse_bool,
)
from .encoding import NUM_ERROR_TYPES, encode_error_type

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final[int] = 2
CACHE_KIND: Final[str] = "context_row_cache"
ROW_CHUNK_SIZE: Final[int] = 1_000_000


def cache_path(
    cache_dir: Path,
    platform: str,
    split: str,
    include_outliers: bool,
) -> Path:
    """Return the canonical row-cache path for a platform/split."""
    suffix = "all" if include_outliers else "filtered"
    return cache_dir / f"{platform}_{split}_{suffix}_rows.h5"


def require_h5py() -> object:
    """Import h5py or raise a message that names the missing dependency."""
    try:
        return importlib.import_module("h5py")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "h5py is required for context row HDF5 caches. Activate the skiver "
            "Python environment or install h5py."
        ) from error


def source_metadata(prefixes: Sequence[Path]) -> list[dict[str, int | str]]:
    """Return source TSV metadata used to detect stale caches."""
    metadata = []
    for prefix in prefixes:
        path = Path(f"{prefix}.base_observations.tsv")
        stat = path.stat()
        metadata.append(
            {
                "prefix": str(prefix.resolve()),
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return metadata


def save_row_cache_h5(
    path: Path,
    *,
    platform: str,
    split: str,
    prefixes: Sequence[Path],
    include_outliers: bool,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 10_000,
) -> None:
    """Parse base-observation TSVs and save accepted rows to an HDF5 cache."""
    h5py = require_h5py()
    path.parent.mkdir(parents=True, exist_ok=True)
    source_files = source_metadata(prefixes)
    total_observations = 0
    skipped_rows = 0
    scanned_rows = 0

    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["kind"] = CACHE_KIND
        handle.attrs["platform"] = platform
        handle.attrs["split"] = split
        handle.attrs["include_outliers"] = include_outliers
        handle.attrs["source_files_json"] = json.dumps(source_files)
        rows_group = handle.create_group("rows")
        datasets = {
            "obs_start": _create_row_dataset(rows_group, "obs_start"),
            "prev_base": _create_row_dataset(rows_group, "prev_base"),
            "true_base": _create_row_dataset(rows_group, "true_base"),
            "target": _create_row_dataset(rows_group, "target"),
        }

        for prefix in prefixes:
            tsv_path = Path(f"{prefix}.base_observations.tsv")
            file_total, file_skipped, file_scanned = _append_tsv_rows(
                tsv_path,
                datasets,
                passes_filter_only=not include_outliers,
                progress_callback=progress_callback,
                progress_interval=progress_interval,
            )
            total_observations += file_total
            skipped_rows += file_skipped
            scanned_rows += file_scanned
            logger.info("Preparsed %d accepted rows from %s", file_total, tsv_path)

        handle.attrs["total_observations"] = total_observations
        handle.attrs["skipped_rows"] = skipped_rows
        handle.attrs["scanned_rows"] = scanned_rows
    logger.info("Saved HDF5 row cache: %s", path)


def load_counts_from_row_cache_h5(
    path: Path,
    *,
    prefixes: Sequence[Path],
    context_lengths: Sequence[int],
    include_outliers: bool,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = ROW_CHUNK_SIZE,
) -> ContextLengthScreenCounts | None:
    """Load a valid row cache and aggregate requested context-length counts."""
    if not path.exists():
        return None

    h5py = require_h5py()
    with h5py.File(path, "r") as handle:
        invalid_reason = _cache_invalid_reason(
            handle,
            prefixes=prefixes,
            include_outliers=include_outliers,
        )
        if invalid_reason is not None:
            logger.warning("Ignoring HDF5 row cache %s: %s", path, invalid_reason)
            return None

        counts = _aggregate_row_cache(
            handle,
            path=path,
            context_lengths=context_lengths,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
    logger.info("Loaded and aggregated HDF5 row cache: %s", path)
    return counts


def _create_row_dataset(group: object, name: str) -> object:
    """Create an appendable uint8 row dataset."""
    return group.create_dataset(
        name,
        shape=(0,),
        maxshape=(None,),
        chunks=(ROW_CHUNK_SIZE,),
        dtype="u1",
        compression="lzf",
        shuffle=True,
    )


def _append_batch(datasets: dict[str, object], batch: dict[str, list[int]]) -> None:
    """Append one in-memory row batch to HDF5 datasets."""
    if not batch["target"]:
        return
    current_size = int(datasets["target"].shape[0])
    batch_size = len(batch["target"])
    new_size = current_size + batch_size
    for name, dataset in datasets.items():
        dataset.resize((new_size,))
        dataset[current_size:new_size] = np.asarray(batch[name], dtype=np.uint8)
        batch[name].clear()


def _append_tsv_rows(
    path: Path,
    datasets: dict[str, object],
    *,
    passes_filter_only: bool,
    progress_callback: ProgressCallback | None,
    progress_interval: int,
) -> tuple[int, int, int]:
    """Append accepted rows from one TSV file to HDF5 datasets."""
    current_obs_id: int | None = None
    total_observations = 0
    skipped_rows = 0
    scanned_rows = 0
    scanned_since_callback = 0
    accepted_since_callback = 0
    skipped_since_callback = 0
    batch = {"obs_start": [], "prev_base": [], "true_base": [], "target": []}

    def flush_progress() -> None:
        nonlocal scanned_since_callback, accepted_since_callback, skipped_since_callback
        if progress_callback is None or scanned_since_callback == 0:
            return
        progress_callback(
            path,
            scanned_since_callback,
            accepted_since_callback,
            skipped_since_callback,
        )
        scanned_since_callback = 0
        accepted_since_callback = 0
        skipped_since_callback = 0

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "obs_id",
            "true_base",
            "obs_base",
            "prev_base",
            "edit_op",
            "passes_filter",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            scanned_rows += 1
            scanned_since_callback += 1
            if passes_filter_only and not _parse_bool(row["passes_filter"]):
                skipped_rows += 1
                skipped_since_callback += 1
                if scanned_since_callback >= progress_interval:
                    flush_progress()
                continue

            obs_id = int(row["obs_id"])
            obs_start = int(obs_id != current_obs_id)
            current_obs_id = obs_id
            batch["obs_start"].append(obs_start)
            batch["prev_base"].append(BASE_TO_IDX[_normalise_base(row["prev_base"])])
            batch["true_base"].append(BASE_TO_IDX[_normalise_base(row["true_base"])])
            batch["target"].append(
                encode_error_type(row["true_base"], row["obs_base"], row["edit_op"])
            )

            total_observations += 1
            accepted_since_callback += 1

            if len(batch["target"]) >= ROW_CHUNK_SIZE:
                _append_batch(datasets, batch)
            if scanned_since_callback >= progress_interval:
                flush_progress()

    _append_batch(datasets, batch)
    flush_progress()
    return total_observations, skipped_rows, scanned_rows


def _aggregate_row_cache(
    handle: object,
    *,
    path: Path,
    context_lengths: Sequence[int],
    progress_callback: ProgressCallback | None,
    progress_interval: int,
) -> ContextLengthScreenCounts:
    """Aggregate previous-base context counts from encoded row datasets."""
    if not context_lengths:
        raise ValueError("context_lengths must not be empty")
    models = [PreviousBasesErrorModel(length) for length in context_lengths]
    count_tensors = {
        model.context_length: torch.zeros(
            *model.context_shape,
            NUM_ERROR_TYPES,
            dtype=torch.float32,
        )
        for model in models
    }

    rows_group = handle["rows"]
    total_observations = int(handle.attrs["total_observations"])
    skipped_rows = int(handle.attrs["skipped_rows"])
    row_count = int(rows_group["target"].shape[0])
    history: list[int] = []
    scanned_since_callback = 0

    for start in range(0, row_count, progress_interval):
        stop = min(start + progress_interval, row_count)
        obs_starts = rows_group["obs_start"][start:stop]
        prev_bases = rows_group["prev_base"][start:stop]
        true_bases = rows_group["true_base"][start:stop]
        targets = rows_group["target"][start:stop]

        for obs_start, prev_base, true_base, target in zip(
            obs_starts,
            prev_bases,
            true_bases,
            targets,
            strict=True,
        ):
            if obs_start:
                history = [int(prev_base)]
            for model in models:
                context_idx = _context_index_from_encoded_history(
                    history,
                    model.context_length,
                )
                count_tensors[model.context_length][context_idx, int(target)] += 1
            if int(true_base) != BASE_TO_IDX["N"]:
                history.append(int(true_base))

        scanned_since_callback += stop - start
        if progress_callback is not None:
            progress_callback(path, scanned_since_callback, scanned_since_callback, 0)
            scanned_since_callback = 0

    by_length = {}
    for model in models:
        counts = count_tensors[model.context_length]
        context_totals = counts.sum(dim=-1)
        by_length[model.context_length] = ContextCounts(
            counts=counts,
            run_values=None,
            total_observations=total_observations,
            skipped_rows=skipped_rows,
            low_count_contexts=int((context_totals < 10).sum().item()),
            context_shape=model.context_shape,
            scalar_run=False,
        )

    return ContextLengthScreenCounts(
        by_length=by_length,
        total_observations=total_observations,
        skipped_rows=skipped_rows,
    )


def _context_index_from_encoded_history(history: Sequence[int], length: int) -> int:
    """Return a flat previous-base context index from encoded base history."""
    missing = max(0, length - len(history))
    context = [BASE_TO_IDX["N"]] * missing
    context.extend(history[-length:])
    flat_index = 0
    for base_idx in context:
        flat_index = flat_index * NUM_CONTEXT_BASES + int(base_idx)
    return flat_index


def _cache_invalid_reason(
    handle: object,
    *,
    prefixes: Sequence[Path],
    include_outliers: bool,
) -> str | None:
    """Return a reason if an open HDF5 handle is incompatible with this request."""
    if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
        return "schema version mismatch"
    if handle.attrs.get("kind") != CACHE_KIND:
        return "cache kind mismatch"
    if bool(handle.attrs.get("include_outliers", False)) != include_outliers:
        return "include_outliers mismatch"
    if "rows" not in handle:
        return "missing rows group"

    rows_group = handle["rows"]
    for name in ("obs_start", "prev_base", "true_base", "target"):
        if name not in rows_group:
            return f"missing rows/{name} dataset"
    row_count = rows_group["target"].shape[0]
    for name in ("obs_start", "prev_base", "true_base"):
        if rows_group[name].shape[0] != row_count:
            return "row dataset length mismatch"

    cached_sources = json.loads(handle.attrs.get("source_files_json", "[]"))
    try:
        current_sources = source_metadata(prefixes)
    except FileNotFoundError as error:
        return f"source TSV is missing: {error.filename}"
    if cached_sources != current_sources:
        return "source TSV metadata changed"

    return None
