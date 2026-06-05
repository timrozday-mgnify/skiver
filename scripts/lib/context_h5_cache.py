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
    MISSING_CONTEXT_BASE,
    NUM_CONTEXT_BASES,
    NUM_TRUE_BASE_BINS,
    UNKNOWN_TRUE_BASE_BIN,
    ContextCounts,
    ContextLengthScreenCounts,
    PreviousBasesErrorModel,
    ProgressCallback,
    _normalise_context_base,
    _parse_bool,
)
from .encoding import NUM_ERROR_TYPES, encode_error_type

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final[int] = 4
CACHE_KIND: Final[str] = "context_row_cache"
ROW_CHUNK_SIZE: Final[int] = 1_000_000
AGGREGATE_CHUNK_ROWS: Final[int] = 10_000_000
_OBS_START_SCAN_BATCH: Final[int] = 4096
_OBS_START_SCAN_CAP: Final[int] = 1_000_000


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
            prev_base = _normalise_context_base(row["prev_base"])
            if prev_base is None:
                skipped_rows += 1
                skipped_since_callback += 1
                if scanned_since_callback >= progress_interval:
                    flush_progress()
                continue
            current_obs_id = obs_id
            true_base = _normalise_context_base(row["true_base"])
            batch["obs_start"].append(obs_start)
            batch["prev_base"].append(BASE_TO_IDX[prev_base])
            batch["true_base"].append(
                BASE_TO_IDX[true_base]
                if true_base is not None
                else MISSING_CONTEXT_BASE
            )
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
    """Aggregate previous-base context counts from encoded row datasets.

    Streams the HDF5 datasets in observation-aligned chunks of roughly
    ``AGGREGATE_CHUNK_ROWS`` rows, accumulating per-context-length counts into
    pre-allocated flat int64 buffers. Result is bit-identical to a single-shot
    aggregation because rolling context only depends on the last
    ``context_length`` events within the current observation, and chunks always
    end strictly before the next observation start.
    """
    del progress_interval  # chunked loop drives progress directly
    if not context_lengths:
        raise ValueError("context_lengths must not be empty")
    models = [PreviousBasesErrorModel(length) for length in context_lengths]
    rows_group = handle["rows"]
    cache_skipped_rows = int(handle.attrs["skipped_rows"])
    row_count = int(rows_group["target"].shape[0])

    obs_start_ds = rows_group["obs_start"]
    prev_base_ds = rows_group["prev_base"]
    true_base_ds = rows_group["true_base"]
    target_ds = rows_group["target"]

    flat_counts: dict[int, np.ndarray] = {
        model.context_length: np.zeros(
            NUM_CONTEXT_BASES**model.context_length
            * NUM_TRUE_BASE_BINS
            * NUM_ERROR_TYPES,
            dtype=np.int64,
        )
        for model in models
    }

    rows_seen = 0
    start = 0
    while start < row_count:
        end = _find_next_obs_start(
            obs_start_ds,
            row_count,
            start + AGGREGATE_CHUNK_ROWS,
        )
        obs_starts = obs_start_ds[start:end][:].astype(bool, copy=False)
        prev_bases = prev_base_ds[start:end][:]
        true_bases = true_base_ds[start:end][:]
        targets = target_ds[start:end][:]

        event_bases, row_event_ends, available_history = _row_cache_context_events(
            obs_starts,
            prev_bases,
            true_bases,
        )
        for model in models:
            _accumulate_context_length_flat(
                flat_counts[model.context_length],
                event_bases,
                row_event_ends,
                available_history,
                true_bases,
                targets,
                model.context_length,
            )

        del obs_starts, prev_bases, true_bases, targets
        del event_bases, row_event_ends, available_history

        chunk_rows = end - start
        rows_seen += chunk_rows
        if progress_callback is not None:
            progress_callback(path, chunk_rows, chunk_rows, 0)
        start = end

    by_length = {}
    for model in models:
        cl = model.context_length
        shape = (
            NUM_CONTEXT_BASES**cl,
            NUM_TRUE_BASE_BINS,
            NUM_ERROR_TYPES,
        )
        counts_np = flat_counts[cl].reshape(shape)
        total_observations = int(counts_np.sum())
        context_totals = counts_np.reshape(shape[0], -1).sum(axis=-1)
        counts_tensor = torch.from_numpy(
            counts_np.astype(np.float32, copy=False)
        )
        by_length[cl] = ContextCounts(
            counts=counts_tensor,
            run_values=None,
            total_observations=total_observations,
            skipped_rows=cache_skipped_rows + row_count - total_observations,
            low_count_contexts=int((context_totals < 10).sum()),
            context_shape=model.context_shape,
            scalar_run=False,
        )

    return ContextLengthScreenCounts(
        by_length=by_length,
        total_observations=row_count,
        skipped_rows=cache_skipped_rows,
    )


def _find_next_obs_start(obs_start_ds: object, row_count: int, target: int) -> int:
    """Return the first row index >= ``target`` where ``obs_start`` is True.

    Probes the HDF5 dataset forward in small batches. If the target is at or
    past the dataset end, returns ``row_count``. Falls back to ``row_count``
    after scanning ``_OBS_START_SCAN_CAP`` rows without finding a True (covers
    pathological inputs with no further observation starts).
    """
    if target >= row_count:
        return row_count
    pos = int(target)
    scanned = 0
    while pos < row_count and scanned < _OBS_START_SCAN_CAP:
        end = min(pos + _OBS_START_SCAN_BATCH, row_count)
        batch = obs_start_ds[pos:end][:].astype(bool, copy=False)
        hits = np.flatnonzero(batch)
        if hits.size > 0:
            return pos + int(hits[0])
        scanned += end - pos
        pos = end
    return row_count


def _accumulate_context_length_flat(
    flat_counts: np.ndarray,
    event_bases: np.ndarray,
    row_event_ends: np.ndarray,
    available_history: np.ndarray,
    true_bases: np.ndarray,
    targets: np.ndarray,
    context_length: int,
) -> None:
    """Add a chunk's context-by-error counts into ``flat_counts`` in place.

    ``flat_counts`` is a 1-D ``int64`` buffer with shape
    ``(NUM_CONTEXT_BASES**context_length * NUM_TRUE_BASE_BINS * NUM_ERROR_TYPES,)``.
    """
    if targets.size == 0:
        return
    valid_rows = available_history >= context_length
    if not np.any(valid_rows):
        return
    context_codes_by_end = _rolling_context_codes(event_bases, context_length)
    context_indices = context_codes_by_end[row_event_ends[valid_rows]]
    true_base_bins = true_bases[valid_rows].astype(np.int64, copy=False)
    true_base_bins = np.where(
        true_base_bins == MISSING_CONTEXT_BASE,
        UNKNOWN_TRUE_BASE_BIN,
        true_base_bins,
    )
    combined_indices = (
        context_indices.astype(np.int64, copy=False)
        * NUM_TRUE_BASE_BINS
        * NUM_ERROR_TYPES
        + true_base_bins * NUM_ERROR_TYPES
        + targets[valid_rows].astype(np.int64, copy=False)
    )
    chunk_counts = np.bincount(combined_indices, minlength=flat_counts.size)
    if chunk_counts.size == flat_counts.size:
        flat_counts += chunk_counts
    else:
        flat_counts[: chunk_counts.size] += chunk_counts


def _row_cache_context_events(
    obs_starts: np.ndarray,
    prev_bases: np.ndarray,
    true_bases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return context event stream and per-row history metadata.

    Each observation contributes its row-level ``prev_base`` before the first
    row. Rows with A/C/G/T true bases contribute that base after the row. Gap
    rows remain valid targets but do not extend history.
    """
    row_count = int(true_bases.shape[0])
    if row_count == 0:
        empty = np.array([], dtype=np.int64)
        return empty.astype(np.uint8), empty, empty
    if not bool(obs_starts[0]):
        raise ValueError("HDF5 row cache must mark the first row as an observation start")

    true_is_context = true_bases != MISSING_CONTEXT_BASE
    obs_events_before_or_at_row = np.cumsum(obs_starts, dtype=np.int64)
    true_events_before_or_at_row = np.cumsum(true_is_context, dtype=np.int64)
    true_events_before_row = true_events_before_or_at_row.copy()
    true_events_before_row -= true_is_context.astype(np.int64)
    row_event_ends = obs_events_before_or_at_row + true_events_before_row

    event_count = int(obs_events_before_or_at_row[-1] + true_events_before_or_at_row[-1])
    event_bases = np.empty(event_count, dtype=np.uint8)
    event_bases[row_event_ends[obs_starts] - 1] = prev_bases[obs_starts]
    true_event_positions = obs_events_before_or_at_row + true_events_before_or_at_row
    event_bases[true_event_positions[true_is_context] - 1] = true_bases[
        true_is_context
    ]

    obs_ids = obs_events_before_or_at_row - 1
    obs_start_rows = np.flatnonzero(obs_starts)
    obs_event_starts = row_event_ends[obs_start_rows] - 1
    available_history = row_event_ends - obs_event_starts[obs_ids]
    return event_bases, row_event_ends, available_history


def _aggregate_context_length_from_events(
    event_bases: np.ndarray,
    row_event_ends: np.ndarray,
    available_history: np.ndarray,
    true_bases: np.ndarray,
    targets: np.ndarray,
    context_length: int,
) -> torch.Tensor:
    """Return dense context-by-error counts for one previous-base length."""
    context_count = NUM_CONTEXT_BASES**context_length
    counts_shape = (context_count, NUM_TRUE_BASE_BINS, NUM_ERROR_TYPES)
    if targets.size == 0:
        return torch.zeros(*counts_shape, dtype=torch.float32)

    valid_rows = available_history >= context_length
    if not np.any(valid_rows):
        return torch.zeros(*counts_shape, dtype=torch.float32)

    context_codes_by_end = _rolling_context_codes(event_bases, context_length)
    context_indices = context_codes_by_end[row_event_ends[valid_rows]]
    true_base_bins = true_bases[valid_rows].astype(np.int64, copy=False)
    true_base_bins = np.where(
        true_base_bins == MISSING_CONTEXT_BASE,
        UNKNOWN_TRUE_BASE_BIN,
        true_base_bins,
    )
    combined_indices = (
        context_indices.astype(np.int64, copy=False)
        * NUM_TRUE_BASE_BINS
        * NUM_ERROR_TYPES
        + true_base_bins * NUM_ERROR_TYPES
        + targets[valid_rows].astype(np.int64, copy=False)
    )
    counts = np.bincount(
        combined_indices,
        minlength=context_count * NUM_TRUE_BASE_BINS * NUM_ERROR_TYPES,
    ).reshape(counts_shape)
    return torch.from_numpy(counts.astype(np.float32, copy=False))


def _rolling_context_codes(event_bases: np.ndarray, context_length: int) -> np.ndarray:
    """Return base-4 context codes keyed by exclusive event end position."""
    event_count = int(event_bases.shape[0])
    codes_by_end = np.full(event_count + 1, -1, dtype=np.int64)
    if event_count < context_length:
        return codes_by_end

    code_count = event_count - context_length + 1
    codes = np.zeros(code_count, dtype=np.int64)
    for offset in range(context_length):
        codes *= NUM_CONTEXT_BASES
        codes += event_bases[offset : offset + code_count].astype(
            np.int64,
            copy=False,
        )
    codes_by_end[context_length:] = codes
    return codes_by_end


def _context_index_from_encoded_history(history: Sequence[int], length: int) -> int:
    """Return a flat previous-base context index from encoded base history."""
    if len(history) < length:
        raise ValueError("history is shorter than length")
    context = history[-length:]
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
