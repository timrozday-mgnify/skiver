"""Context-dependent, non-HMM sequencing error models.

These models treat each aligned base observation from ``skiver dump --base`` as
conditionally independent given sequence-context covariates.  They are fitted by
maximum likelihood with Pyro parameters and a categorical log-likelihood.
"""
from __future__ import annotations

import concurrent.futures
import csv
import logging
import math
import os
import sys
from collections import deque
from collections.abc import Callable, Iterable, Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.initialization import init_to_value

from .encoding import NUM_ERROR_TYPES, ERROR_TYPE_SUB_A, encode_error_type

logger = logging.getLogger(__name__)

BASES: Final[tuple[str, ...]] = ("A", "C", "G", "T")
BASE_TO_IDX: Final[dict[str, int]] = {base: idx for idx, base in enumerate(BASES)}
NUM_CONTEXT_BASES: Final[int] = len(BASES)
MISSING_CONTEXT_BASE: Final[int] = 255
UNKNOWN_TRUE_BASE_BIN: Final[int] = NUM_CONTEXT_BASES
NUM_TRUE_BASE_BINS: Final[int] = NUM_CONTEXT_BASES + 1
DEFAULT_MAX_RUN: Final[int] = 8
ProgressCallback = Callable[[Path, int, int, int], object]
TrainProgressCallback = Callable[[int, float], object]


@dataclass(frozen=True)
class ContextCounts:
    """Aggregated target counts for a conditional error model."""

    counts: torch.Tensor
    run_values: torch.Tensor | None
    total_observations: int
    skipped_rows: int
    low_count_contexts: int
    context_shape: tuple[int, ...]
    scalar_run: bool = False
    additive_context: bool = False
    context_indices: torch.Tensor | None = None
    phred_context_sums: torch.Tensor | None = None
    position_context_sums: torch.Tensor | None = None
    strand_context_sums: torch.Tensor | None = None
    fragment_count_per_context: torch.Tensor | None = None


@dataclass(frozen=True)
class FitResult:
    """Fitted model parameters and evaluation metrics."""

    params: dict[str, torch.Tensor]
    losses: list[float]
    train_log_likelihood: float
    test_log_likelihood: float
    num_parameters: int
    aic: float


@dataclass(frozen=True)
class BayesianFitResult:
    """Variational Bayesian fit summaries and metrics."""

    params_mean: dict[str, torch.Tensor]
    params_stdev: dict[str, torch.Tensor]
    inference_params: dict[str, torch.Tensor]
    losses: list[float]
    train_log_likelihood: float
    test_log_likelihood: float
    train_elbo: float
    test_elbo: float
    prior_scale: float


@dataclass(frozen=True)
class PlatformCounts:
    """Reusable aggregated counts for all context models."""

    prev2: ContextCounts
    prev2_hpoly: ContextCounts
    total_observations: int
    skipped_rows: int


@dataclass(frozen=True)
class ContextLengthScreenCounts:
    """Reusable aggregated counts for previous-base context length screening."""

    by_length: dict[int, ContextCounts]
    total_observations: int
    skipped_rows: int


class PreviousBasesErrorModel:
    """Predict error type from a configurable number of previous bases."""

    scalar_run = False

    def __init__(self, context_length: int) -> None:
        if context_length < 1:
            raise ValueError("context_length must be at least 1")
        self.context_length = context_length
        self.name = f"prev{context_length}"
        self.context_shape = (NUM_CONTEXT_BASES**context_length,)

    def context_index_from_history(self, history: Sequence[str]) -> tuple[int]:
        """Return a flat context index from the last context_length bases."""
        if len(history) < self.context_length:
            raise ValueError("history is shorter than context_length")
        context = history[-self.context_length:]
        flat_index = 0
        for base in context:
            flat_index = flat_index * NUM_CONTEXT_BASES + _base_index(base)
        return (flat_index,)


class Prev2ErrorModel:
    """Predict error type from the previous two consensus bases."""

    name = "prev2"
    context_shape = (NUM_CONTEXT_BASES, NUM_CONTEXT_BASES)
    scalar_run = False

    @classmethod
    def context_index(
        cls,
        prev2_base: str,
        prev1_base: str,
        run_base: str,
        run_length: int,
    ) -> tuple[int, int]:
        """Return the context index for a row."""
        del run_base, run_length
        return (_base_index(prev2_base), _base_index(prev1_base))


class Prev2HomopolymerErrorModel:
    """Predict error type from previous two bases and homopolymer run context."""

    name = "prev2_hpoly"
    scalar_run = True

    def __init__(self, max_run: int = DEFAULT_MAX_RUN) -> None:
        if max_run < 1:
            raise ValueError("max_run must be at least 1")
        self.max_run = max_run
        self.context_shape = (
            NUM_CONTEXT_BASES,
            NUM_CONTEXT_BASES,
            NUM_CONTEXT_BASES,
        )

    def context_index(
        self,
        prev2_base: str,
        prev1_base: str,
        run_base: str,
        run_length: int,
    ) -> tuple[int, int, int]:
        """Return the context index for a row."""
        del run_length
        return (
            _base_index(prev2_base),
            _base_index(prev1_base),
            _base_index(run_base),
        )

    def run_value(self, run_length: int) -> float:
        """Return the clipped integer repeat-count bin."""
        return float(min(max(run_length, 0), self.max_run))


def _base_index(base: str) -> int:
    """Return a stable A/C/G/T base index."""
    return BASE_TO_IDX[base]


def _normalise_context_base(base: str) -> str | None:
    """Return A/C/G/T for context reconstruction, or None if unusable."""
    return base if base in BASE_TO_IDX else None


def _true_base_bin(base: str) -> int:
    """Return A/C/G/T true-base bin, or unknown for gap/ambiguous rows."""
    true_base = _normalise_context_base(base)
    return BASE_TO_IDX[true_base] if true_base is not None else UNKNOWN_TRUE_BASE_BIN


def _is_true_base_conditioned(counts: torch.Tensor, logits: torch.Tensor) -> bool:
    """Return whether counts include a true-base bin before error type."""
    return (
        counts.dim() == logits.dim() + 1
        and counts.shape[-2] == NUM_TRUE_BASE_BINS
        and tuple(counts.shape[:-2]) == tuple(logits.shape[:-1])
    )


def _collapse_true_base_counts(counts: torch.Tensor) -> torch.Tensor:
    """Sum over the true-base conditioning axis when present."""
    if counts.dim() >= 3 and counts.shape[-2] == NUM_TRUE_BASE_BINS:
        return counts.sum(dim=-2)
    return counts


def _context_totals(counts: torch.Tensor) -> torch.Tensor:
    """Return one total per model context row, including all auxiliary axes."""
    return counts.reshape(counts.shape[0], -1).sum(dim=-1)


def _model_counts_for_initial_logits(
    counts: torch.Tensor,
    run_values: torch.Tensor | None,
) -> torch.Tensor:
    """Collapse count axes that are conditioned but not directly parameterised."""
    collapsed = _collapse_true_base_counts(counts)
    if run_values is not None:
        collapsed = collapsed.sum(dim=-2)
    return collapsed


def _context_shape_for_counts(
    counts: torch.Tensor,
    run_values: torch.Tensor | None,
) -> tuple[int, ...]:
    """Return the model context shape from a count tensor."""
    if counts.dim() >= 3 and counts.shape[-2] == NUM_TRUE_BASE_BINS:
        end = -3 if run_values is not None else -2
    else:
        end = -2 if run_values is not None else -1
    return tuple(counts.shape[:end])


def _masked_log_probs_for_counts(
    logits: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Return log probabilities with impossible self-substitutions masked.

    Generation masks substitution-to-self probabilities using the emitted true
    base. Training uses the same constraint by keeping a true-base bin in the
    aggregated counts. Unknown/gap true-base rows, mainly insertions, are left
    unmasked.
    """
    if not _is_true_base_conditioned(counts, logits):
        return F.log_softmax(logits, dim=-1)

    expanded_logits = logits.unsqueeze(-2).expand(*counts.shape[:-1], NUM_ERROR_TYPES)
    mask = torch.zeros(
        NUM_TRUE_BASE_BINS,
        NUM_ERROR_TYPES,
        dtype=torch.bool,
        device=expanded_logits.device,
    )
    for true_base_idx in range(NUM_CONTEXT_BASES):
        mask[true_base_idx, ERROR_TYPE_SUB_A + true_base_idx] = True
    mask_shape = (1,) * (expanded_logits.dim() - 2) + mask.shape
    masked_logits = expanded_logits.masked_fill(mask.reshape(mask_shape), -torch.inf)
    return F.log_softmax(masked_logits, dim=-1)


def _count_weighted_log_prob_sum(
    counts: torch.Tensor,
    log_probs: torch.Tensor,
) -> torch.Tensor:
    """Return sum(counts * log_probs) without 0 * -inf NaNs."""
    return torch.where(counts > 0, counts * log_probs, torch.zeros_like(log_probs)).sum()


def _parse_bool(value: str) -> bool:
    """Parse skiver TSV boolean text."""
    return value.lower() == "true"


def _update_run(history: list[str]) -> tuple[str, int]:
    """Return the immediate homopolymer run ending before the current base."""
    if not history:
        raise ValueError("history is empty")

    run_base = history[-1]
    run_length = 0
    for base in reversed(history):
        if base != run_base:
            break
        run_length += 1
    return run_base, run_length


def _raw_key_contexts_for_base_path(
    base_path: Path,
    context_lengths: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (obs_ids, context_codes, max_context) from raw dump as compact numpy arrays.

    obs_ids and context_codes are parallel int64 arrays sorted by obs_id.
    Each context_code encodes max_context bases as a base-4 integer (MSB = leftmost base).
    Returns empty arrays when the raw file is absent or context is not needed.
    """
    max_context = max(context_lengths, default=0)
    raw_path = base_path.with_name(
        base_path.name.replace(".base_observations.tsv", ".raw_observations.tsv")
    )
    _empty: tuple[np.ndarray, np.ndarray, int] = (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        max_context,
    )
    if max_context <= 0 or not raw_path.exists():
        return _empty

    obs_id_list: list[int] = []
    code_list: list[int] = []
    logger.info(
        "Reading raw context history from %s (%.0f MB)…",
        raw_path.name,
        raw_path.stat().st_size / 1e6,
    )
    _RAW_PROGRESS = 500_000
    n_raw_rows = 0
    csv.field_size_limit(sys.maxsize)
    with open(raw_path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "obs_id" not in (reader.fieldnames or []) or "key_str" not in (
            reader.fieldnames or []
        ):
            return _empty
        for row in reader:
            n_raw_rows += 1
            if n_raw_rows % _RAW_PROGRESS == 0:
                logger.info("  raw context: %d rows read", n_raw_rows)
            key = row["key_str"].upper()
            if len(key) < max_context or any(base not in BASE_TO_IDX for base in key):
                continue
            code = 0
            for base in key[-max_context:]:
                code = code * 4 + BASE_TO_IDX[base]
            obs_id_list.append(int(row["obs_id"]))
            code_list.append(code)
    logger.info(
        "  raw context: %d rows total, %d retained", n_raw_rows, len(obs_id_list)
    )

    if not obs_id_list:
        return _empty

    obs_ids_arr = np.array(obs_id_list, dtype=np.int64)
    codes_arr = np.array(code_list, dtype=np.int64)
    order = np.argsort(obs_ids_arr, kind="stable")
    return obs_ids_arr[order], codes_arr[order], max_context


def _lookup_raw_context_history(
    obs_ids: np.ndarray,
    codes: np.ndarray,
    max_context: int,
    obs_id: int,
) -> list[str]:
    """Return decoded context history for obs_id, or [] if not found."""
    if obs_ids.size == 0:
        return []
    idx = int(np.searchsorted(obs_ids, obs_id))
    if idx < obs_ids.size and obs_ids[idx] == obs_id:
        code = int(codes[idx])
        return [BASES[(code >> (2 * (max_context - 1 - i))) & 3] for i in range(max_context)]
    return []


def aggregate_counts(
    prefixes: Iterable[str | Path],
    model: type[Prev2ErrorModel] | Prev2HomopolymerErrorModel,
    *,
    passes_filter_only: bool = True,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 10_000,
) -> ContextCounts:
    """Aggregate error-type counts from base observation TSV files.

    Args:
        prefixes: Skiver dump prefixes. Each prefix must have a matching
            ``.base_observations.tsv`` file.
        model: Model class/instance that maps reconstructed context to indices.
        passes_filter_only: If true, ignore rows from outlier keys.
        progress_callback: Optional callback receiving path, scanned rows,
            accepted rows, and skipped rows since the previous callback.
        progress_interval: Row interval between progress callback invocations.

    Returns:
        Aggregated context-by-error-type counts.
    """
    if model.scalar_run:
        counts = torch.zeros(
            *model.context_shape,
            model.max_run + 1,
            NUM_TRUE_BASE_BINS,
            NUM_ERROR_TYPES,
            dtype=torch.float32,
        )
        run_values = torch.tensor(
            [model.run_value(length) for length in range(model.max_run + 1)],
            dtype=torch.float32,
        )
    else:
        counts = torch.zeros(
            *model.context_shape,
            NUM_TRUE_BASE_BINS,
            NUM_ERROR_TYPES,
            dtype=torch.float32,
        )
        run_values = None
    total_observations = 0
    skipped_rows = 0

    for prefix in prefixes:
        path = Path(f"{prefix}.base_observations.tsv")
        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            continue
        file_total, file_skipped = _aggregate_file(
            path,
            counts,
            model,
            passes_filter_only=passes_filter_only,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
        total_observations += file_total
        skipped_rows += file_skipped
        logger.info("Aggregated %d rows from %s", file_total, path)

    context_totals = _context_totals(counts)
    low_count_contexts = int((context_totals < 10).sum().item())
    return ContextCounts(
        counts=counts,
        run_values=run_values,
        total_observations=total_observations,
        skipped_rows=skipped_rows,
        low_count_contexts=low_count_contexts,
        context_shape=model.context_shape,
        scalar_run=model.scalar_run,
    )


def aggregate_platform_counts(
    prefixes: Iterable[str | Path],
    *,
    max_run: int = DEFAULT_MAX_RUN,
    passes_filter_only: bool = True,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 10_000,
) -> PlatformCounts:
    """Aggregate all model count tensors in one pass over base observation TSVs.

    Args:
        prefixes: Skiver dump prefixes. Each prefix must have a matching
            ``.base_observations.tsv`` file.
        max_run: Maximum clipped homopolymer run bin.
        passes_filter_only: If true, ignore rows from outlier keys.
        progress_callback: Optional callback receiving path, scanned rows,
            accepted rows, and skipped rows since the previous callback.
        progress_interval: Row interval between progress callback invocations.

    Returns:
        Counts for both context models from the same accepted rows.
    """
    hpoly_model = Prev2HomopolymerErrorModel(max_run=max_run)
    prev2_counts = torch.zeros(
        *Prev2ErrorModel.context_shape,
        NUM_TRUE_BASE_BINS,
        NUM_ERROR_TYPES,
        dtype=torch.float32,
    )
    hpoly_counts = torch.zeros(
        *hpoly_model.context_shape,
        max_run + 1,
        NUM_TRUE_BASE_BINS,
        NUM_ERROR_TYPES,
        dtype=torch.float32,
    )

    total_observations = 0
    skipped_rows = 0
    for prefix in prefixes:
        path = Path(f"{prefix}.base_observations.tsv")
        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            continue
        file_total, file_skipped = _aggregate_platform_file(
            path,
            prev2_counts,
            hpoly_counts,
            hpoly_model,
            passes_filter_only=passes_filter_only,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
        total_observations += file_total
        skipped_rows += file_skipped
        logger.info("Aggregated %d rows from %s", file_total, path)

    run_values = torch.tensor(
        [hpoly_model.run_value(length) for length in range(max_run + 1)],
        dtype=torch.float32,
    )
    prev2_context_totals = _context_totals(prev2_counts)
    hpoly_context_totals = _context_totals(hpoly_counts)
    return PlatformCounts(
        prev2=ContextCounts(
            counts=prev2_counts,
            run_values=None,
            total_observations=total_observations,
            skipped_rows=skipped_rows,
            low_count_contexts=int((prev2_context_totals < 10).sum().item()),
            context_shape=Prev2ErrorModel.context_shape,
            scalar_run=False,
        ),
        prev2_hpoly=ContextCounts(
            counts=hpoly_counts,
            run_values=run_values,
            total_observations=total_observations,
            skipped_rows=skipped_rows,
            low_count_contexts=int((hpoly_context_totals < 10).sum().item()),
            context_shape=hpoly_model.context_shape,
            scalar_run=True,
        ),
        total_observations=total_observations,
        skipped_rows=skipped_rows,
    )


def _aggregate_one_prefix_screen(
    prefix: str,
    context_lengths: Sequence[int],
    passes_filter_only: bool,
    num_phred_lags: int = 0,
    num_position_features: int = 0,
    use_strand: bool = False,
    use_fragment_overdispersion: bool = False,
) -> tuple[dict[int, "torch.Tensor"], dict[int, int], dict[int, int], int, int, dict[int, "torch.Tensor"], dict[int, "torch.Tensor"], dict[int, "torch.Tensor"], "dict[int, dict[int, set[int]]] | None"] | None:
    """Aggregate a single prefix into its own count tensors (process-pool worker).

    Returns ``None`` if the file is missing. The per-file count tensors are summed
    by the caller — addition is associative, so the parallel result matches the
    serial one. Runs without a progress callback (not picklable across processes).
    """
    path = Path(f"{prefix}.base_observations.tsv")
    if not path.exists():
        logger.warning("Skipping missing file: %s", path)
        return None
    models = [PreviousBasesErrorModel(length) for length in context_lengths]
    count_tensors = {
        model.context_length: torch.zeros(
            *model.context_shape, NUM_TRUE_BASE_BINS, NUM_ERROR_TYPES, dtype=torch.float32
        )
        for model in models
    }
    phred_sums = {
        model.context_length: torch.zeros(
            *model.context_shape, num_phred_lags, dtype=torch.float32
        )
        for model in models
    } if num_phred_lags > 0 else {}
    position_sums = {
        model.context_length: torch.zeros(
            *model.context_shape, num_position_features, dtype=torch.float32
        )
        for model in models
    } if num_position_features > 0 else {}
    strand_sums = {
        model.context_length: torch.zeros(
            *model.context_shape, 1, dtype=torch.float32
        )
        for model in models
    } if use_strand else {}
    from collections import defaultdict
    fragment_id_sets: dict[int, dict[int, set[int]]] | None = (
        {model.context_length: defaultdict(set) for model in models}
        if use_fragment_overdispersion else None
    )
    file_totals, file_skipped_by_length, file_total, file_skipped = (
        _aggregate_context_length_screen_file(
            path,
            models,
            count_tensors,
            phred_context_sums_by_length=phred_sums,
            num_phred_lags=num_phred_lags,
            position_context_sums_by_length=position_sums,
            num_position_features=num_position_features,
            strand_context_sums_by_length=strand_sums,
            use_strand=use_strand,
            fragment_id_sets_by_length=fragment_id_sets,
            passes_filter_only=passes_filter_only,
            progress_callback=None,
            progress_interval=10_000,
        )
    )
    return count_tensors, file_totals, file_skipped_by_length, file_total, file_skipped, phred_sums, position_sums, strand_sums, fragment_id_sets


def aggregate_context_length_screen_counts(
    prefixes: Iterable[str | Path],
    *,
    context_lengths: Sequence[int],
    passes_filter_only: bool = True,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 10_000,
    max_workers: int | None = None,
    num_phred_lags: int = 0,
    num_position_features: int = 0,
    use_strand: bool = False,
    use_fragment_overdispersion: bool = False,
) -> ContextLengthScreenCounts:
    """Aggregate previous-base context models in one pass over TSV files.

    Args:
        prefixes: Skiver dump prefixes. Each prefix must have a matching
            ``.base_observations.tsv`` file.
        context_lengths: Previous-base context lengths to aggregate.
        passes_filter_only: If true, ignore rows from outlier keys.
        progress_callback: Optional callback receiving path, scanned rows,
            accepted rows, and skipped rows since the previous callback. Only used
            on the serial path (single file, or ``max_workers == 1``).
        progress_interval: Row interval between progress callback invocations.
        max_workers: Parallelise per-file aggregation across this many processes
            (CSV parsing is CPU-bound Python; processes bypass the GIL). Defaults to
            ``min(#files, os.cpu_count())``. ``1`` forces the serial path. The
            per-file count tensors are summed, which is associative — the result is
            identical to serial aggregation.
        num_phred_lags: Number of previous Phred quality scores to accumulate as
            context covariates. 0 disables Phred context (default).
        num_position_features: Number of read-position features to accumulate as
            context covariates. 0 disables position context (default). Features in
            order: log1p(dist_to_end), log1p(read_pos).
        use_strand: If True, accumulate the forward-strand fraction per context as
            a scalar covariate (from the ``is_forward`` column).
        use_fragment_overdispersion: If True, track unique fragment IDs per context
            to populate ``fragment_count_per_context`` in each ``ContextCounts``.
            Requires a ``fragment_id`` column in the TSV files.

    Returns:
        Counts keyed by context length.
    """
    if not context_lengths:
        raise ValueError("context_lengths must not be empty")

    prefix_list = [str(p) for p in prefixes]
    models = [PreviousBasesErrorModel(length) for length in context_lengths]
    count_tensors = {
        model.context_length: torch.zeros(
            *model.context_shape,
            NUM_TRUE_BASE_BINS,
            NUM_ERROR_TYPES,
            dtype=torch.float32,
        )
        for model in models
    }
    phred_sum_tensors: dict[int, torch.Tensor] = {
        model.context_length: torch.zeros(*model.context_shape, num_phred_lags, dtype=torch.float32)
        for model in models
    } if num_phred_lags > 0 else {}
    position_sum_tensors: dict[int, torch.Tensor] = {
        model.context_length: torch.zeros(*model.context_shape, num_position_features, dtype=torch.float32)
        for model in models
    } if num_position_features > 0 else {}
    strand_sum_tensors: dict[int, torch.Tensor] = {
        model.context_length: torch.zeros(*model.context_shape, 1, dtype=torch.float32)
        for model in models
    } if use_strand else {}
    from collections import defaultdict
    fragment_id_sets_global: dict[int, dict[int, set[int]]] | None = (
        {model.context_length: defaultdict(set) for model in models}
        if use_fragment_overdispersion else None
    )
    totals_by_length = {model.context_length: 0 for model in models}
    skipped_by_length = {model.context_length: 0 for model in models}

    total_observations = 0
    skipped_rows = 0

    if max_workers is None:
        max_workers = min(len(prefix_list), os.cpu_count() or 1)
    parallel = max_workers > 1 and len(prefix_list) > 1

    def accumulate(result) -> None:
        nonlocal total_observations, skipped_rows
        if result is None:
            return
        file_tensors, file_totals, file_skipped_by_length, file_total, file_skipped, file_phred, file_position, file_strand, file_frag_sets = result
        for length, tensor in file_tensors.items():
            count_tensors[length] += tensor
        for length, psums in file_phred.items():
            phred_sum_tensors[length] += psums
        for length, possums in file_position.items():
            position_sum_tensors[length] += possums
        for length, ssums in file_strand.items():
            strand_sum_tensors[length] += ssums
        if file_frag_sets is not None and fragment_id_sets_global is not None:
            for length, ctx_sets in file_frag_sets.items():
                for ctx_idx, fids in ctx_sets.items():
                    fragment_id_sets_global[length][ctx_idx].update(fids)
        for length, total in file_totals.items():
            totals_by_length[length] += total
        for length, skipped in file_skipped_by_length.items():
            skipped_by_length[length] += skipped
        total_observations += file_total
        skipped_rows += file_skipped

    if parallel:
        logger.info("Aggregating %d files across %d processes…", len(prefix_list), max_workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _aggregate_one_prefix_screen,
                    prefix,
                    list(context_lengths),
                    passes_filter_only,
                    num_phred_lags,
                    num_position_features,
                    use_strand,
                    use_fragment_overdispersion,
                ): prefix
                for prefix in prefix_list
            }
            for fut in concurrent.futures.as_completed(futures):
                accumulate(fut.result())
                logger.info("Aggregated %s", futures[fut])
    else:
        # Serial path: accumulate directly into the shared tensors (keeps the
        # progress callback, which can't cross a process boundary).
        for prefix in prefix_list:
            path = Path(f"{prefix}.base_observations.tsv")
            if not path.exists():
                logger.warning("Skipping missing file: %s", path)
                continue
            file_totals, file_skipped_by_length, file_total, file_skipped = (
                _aggregate_context_length_screen_file(
                    path,
                    models,
                    count_tensors,
                    phred_context_sums_by_length=phred_sum_tensors,
                    num_phred_lags=num_phred_lags,
                    position_context_sums_by_length=position_sum_tensors,
                    num_position_features=num_position_features,
                    strand_context_sums_by_length=strand_sum_tensors,
                    use_strand=use_strand,
                    fragment_id_sets_by_length=fragment_id_sets_global,
                    passes_filter_only=passes_filter_only,
                    progress_callback=progress_callback,
                    progress_interval=progress_interval,
                )
            )
            for length, total in file_totals.items():
                totals_by_length[length] += total
            for length, skipped in file_skipped_by_length.items():
                skipped_by_length[length] += skipped
            total_observations += file_total
            skipped_rows += file_skipped
            logger.info("Aggregated %d usable rows from %s", sum(file_totals.values()), path)

    by_length = {}
    for model in models:
        counts = count_tensors[model.context_length]
        context_totals = _context_totals(counts)
        phred_sums = phred_sum_tensors.get(model.context_length)
        position_sums = position_sum_tensors.get(model.context_length)
        strand_sums = strand_sum_tensors.get(model.context_length)
        fragment_counts: torch.Tensor | None = None
        if fragment_id_sets_global is not None:
            fid_sets = fragment_id_sets_global[model.context_length]
            n_ctx = math.prod(model.context_shape)
            fragment_counts = torch.tensor(
                [len(fid_sets[i]) for i in range(n_ctx)],
                dtype=torch.float32,
            )
        by_length[model.context_length] = ContextCounts(
            counts=counts,
            run_values=None,
            total_observations=totals_by_length[model.context_length],
            skipped_rows=skipped_by_length[model.context_length],
            low_count_contexts=int((context_totals < 10).sum().item()),
            context_shape=model.context_shape,
            scalar_run=False,
            phred_context_sums=phred_sums,
            position_context_sums=position_sums,
            strand_context_sums=strand_sums,
            fragment_count_per_context=fragment_counts,
        )

    return ContextLengthScreenCounts(
        by_length=by_length,
        total_observations=total_observations,
        skipped_rows=skipped_rows,
    )


def _aggregate_context_length_screen_file(
    path: Path,
    models: Sequence[PreviousBasesErrorModel],
    count_tensors: dict[int, torch.Tensor],
    *,
    phred_context_sums_by_length: dict[int, torch.Tensor],
    num_phred_lags: int = 0,
    position_context_sums_by_length: dict[int, torch.Tensor],
    num_position_features: int = 0,
    strand_context_sums_by_length: dict[int, torch.Tensor],
    use_strand: bool = False,
    fragment_id_sets_by_length: "dict[int, dict[int, set[int]]] | None" = None,
    passes_filter_only: bool,
    progress_callback: ProgressCallback | None,
    progress_interval: int,
) -> tuple[dict[int, int], dict[int, int], int, int]:
    """Aggregate one TSV file into all previous-base context length tensors."""
    raw_obs_ids, raw_codes, raw_max_context = _raw_key_contexts_for_base_path(
        path,
        [model.context_length for model in models],
    )
    current_obs_id: int | None = None
    history: list[str] = []
    phred_history: deque[float] = deque([0.0] * num_phred_lags, maxlen=num_phred_lags) if num_phred_lags > 0 else deque()
    totals_by_length = {model.context_length: 0 for model in models}
    skipped_by_length = {model.context_length: 0 for model in models}
    total_observations = 0
    skipped_rows = 0
    scanned_since_callback = 0
    accepted_since_callback = 0
    skipped_since_callback = 0
    total_scanned_so_far = 0

    def flush_progress() -> None:
        nonlocal scanned_since_callback, accepted_since_callback, skipped_since_callback, total_scanned_so_far
        if scanned_since_callback == 0:
            return
        total_scanned_so_far += scanned_since_callback
        logger.info("  %s: %d rows scanned…", path.name, total_scanned_so_far)
        if progress_callback is not None:
            progress_callback(
                path,
                scanned_since_callback,
                accepted_since_callback,
                skipped_since_callback,
            )
        scanned_since_callback = 0
        accepted_since_callback = 0
        skipped_since_callback = 0

    logger.info("Scanning %s (%.0f MB)…", path.name, path.stat().st_size / 1e6)
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
        if num_phred_lags > 0:
            required.add("phred")
        if num_position_features > 0:
            required.update(["dist_to_end", "read_pos"])
        if use_strand:
            required.add("is_forward")
        if fragment_id_sets_by_length is not None:
            required.add("fragment_id")
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            scanned_since_callback += 1
            if passes_filter_only and not _parse_bool(row["passes_filter"]):
                skipped_rows += 1
                skipped_since_callback += 1
                if scanned_since_callback >= progress_interval:
                    flush_progress()
                for model in models:
                    skipped_by_length[model.context_length] += 1
                continue

            row_prev_base = _normalise_context_base(row["prev_base"])
            obs_id = int(row["obs_id"])
            if obs_id != current_obs_id:
                current_obs_id = obs_id
                history = _lookup_raw_context_history(raw_obs_ids, raw_codes, raw_max_context, obs_id)
                if not history:
                    history = [row_prev_base] if row_prev_base is not None else []
                if num_phred_lags > 0:
                    phred_history = deque([0.0] * num_phred_lags, maxlen=num_phred_lags)

            target = encode_error_type(
                row["true_base"],
                row["obs_base"],
                row["edit_op"],
            )
            true_base_bin = _true_base_bin(row["true_base"])
            pos_feats: list[float] = []
            if num_position_features > 0:
                dist_to_end = max(0.0, float(row["dist_to_end"]))
                read_pos = max(0.0, float(row["read_pos"]))
                all_pos = [math.log1p(dist_to_end), math.log1p(read_pos)]
                pos_feats = all_pos[:num_position_features]
            strand_val = 1.0 if (use_strand and _parse_bool(row["is_forward"])) else 0.0
            for model in models:
                if len(history) < model.context_length:
                    skipped_by_length[model.context_length] += 1
                    continue
                context_idx = model.context_index_from_history(history)
                count_tensors[model.context_length][
                    (*context_idx, true_base_bin, target)
                ] += 1
                totals_by_length[model.context_length] += 1
                if num_phred_lags > 0:
                    flat_ctx = context_idx[0]
                    phred_list = list(phred_history)
                    for k in range(num_phred_lags):
                        lag_idx = len(phred_list) - 1 - k
                        lag_val = phred_list[lag_idx] if lag_idx >= 0 else 0.0
                        phred_context_sums_by_length[model.context_length][flat_ctx, k] += lag_val
                if num_position_features > 0:
                    flat_ctx = context_idx[0]
                    for f, pval in enumerate(pos_feats):
                        position_context_sums_by_length[model.context_length][flat_ctx, f] += pval
                if use_strand:
                    flat_ctx = context_idx[0]
                    strand_context_sums_by_length[model.context_length][flat_ctx, 0] += strand_val
                if fragment_id_sets_by_length is not None:
                    flat_ctx = context_idx[0]
                    frag_id = int(row["fragment_id"])
                    fragment_id_sets_by_length[model.context_length][flat_ctx].add(frag_id)

            total_observations += 1
            accepted_since_callback += 1

            true_base = _normalise_context_base(row["true_base"])
            if true_base is not None:
                history.append(true_base)
                if num_phred_lags > 0:
                    phred_val = max(0.0, float(row["phred"]))
                    phred_history.append(phred_val)

            if scanned_since_callback >= progress_interval:
                flush_progress()

    flush_progress()
    return totals_by_length, skipped_by_length, total_observations, skipped_rows


def _aggregate_platform_file(
    path: Path,
    prev2_counts: torch.Tensor,
    hpoly_counts: torch.Tensor,
    hpoly_model: Prev2HomopolymerErrorModel,
    *,
    passes_filter_only: bool,
    progress_callback: ProgressCallback | None,
    progress_interval: int,
) -> tuple[int, int]:
    """Aggregate a single TSV file into all model count tensors."""
    current_obs_id: int | None = None
    history: list[str] = []
    total_observations = 0
    skipped_rows = 0
    scanned_since_callback = 0
    accepted_since_callback = 0
    skipped_since_callback = 0

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
            scanned_since_callback += 1
            if passes_filter_only and not _parse_bool(row["passes_filter"]):
                skipped_rows += 1
                skipped_since_callback += 1
                if scanned_since_callback >= progress_interval:
                    flush_progress()
                continue

            row_prev_base = _normalise_context_base(row["prev_base"])
            if row_prev_base is None:
                skipped_rows += 1
                skipped_since_callback += 1
                if scanned_since_callback >= progress_interval:
                    flush_progress()
                continue
            obs_id = int(row["obs_id"])
            if obs_id != current_obs_id:
                current_obs_id = obs_id
                history = [row_prev_base]

            if len(history) < 2:
                skipped_rows += 1
                skipped_since_callback += 1
                true_base = _normalise_context_base(row["true_base"])
                if true_base is not None:
                    history.append(true_base)
                if scanned_since_callback >= progress_interval:
                    flush_progress()
                continue

            prev1_base = history[-1]
            prev2_base = history[-2]
            run_base, run_length = _update_run(history)
            target = encode_error_type(
                row["true_base"],
                row["obs_base"],
                row["edit_op"],
            )
            true_base_bin = _true_base_bin(row["true_base"])

            prev2_idx = Prev2ErrorModel.context_index(
                prev2_base,
                prev1_base,
                run_base,
                run_length,
            )
            hpoly_idx = hpoly_model.context_index(
                prev2_base,
                prev1_base,
                run_base,
                run_length,
            )
            run_bin = min(max(run_length, 0), hpoly_model.max_run)
            prev2_counts[(*prev2_idx, true_base_bin, target)] += 1
            hpoly_counts[(*hpoly_idx, run_bin, true_base_bin, target)] += 1

            total_observations += 1
            accepted_since_callback += 1

            true_base = _normalise_context_base(row["true_base"])
            if true_base is not None:
                history.append(true_base)

            if scanned_since_callback >= progress_interval:
                flush_progress()

    flush_progress()
    return total_observations, skipped_rows


def _aggregate_file(
    path: Path,
    counts: torch.Tensor,
    model: type[Prev2ErrorModel] | Prev2HomopolymerErrorModel,
    *,
    passes_filter_only: bool,
    progress_callback: ProgressCallback | None,
    progress_interval: int,
) -> tuple[int, int]:
    """Aggregate a single TSV file into the provided counts tensor."""
    current_obs_id: int | None = None
    history: list[str] = []
    total_observations = 0
    skipped_rows = 0
    scanned_since_callback = 0
    accepted_since_callback = 0
    skipped_since_callback = 0

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
            scanned_since_callback += 1
            if passes_filter_only and not _parse_bool(row["passes_filter"]):
                skipped_rows += 1
                skipped_since_callback += 1
                if scanned_since_callback >= progress_interval:
                    flush_progress()
                continue

            row_prev_base = _normalise_context_base(row["prev_base"])
            if row_prev_base is None:
                skipped_rows += 1
                skipped_since_callback += 1
                if scanned_since_callback >= progress_interval:
                    flush_progress()
                continue
            obs_id = int(row["obs_id"])
            if obs_id != current_obs_id:
                current_obs_id = obs_id
                history = [row_prev_base]

            if len(history) < 2:
                skipped_rows += 1
                skipped_since_callback += 1
                true_base = _normalise_context_base(row["true_base"])
                if true_base is not None:
                    history.append(true_base)
                if scanned_since_callback >= progress_interval:
                    flush_progress()
                continue

            prev1_base = history[-1]
            prev2_base = history[-2]
            run_base, run_length = _update_run(history)

            target = encode_error_type(
                row["true_base"],
                row["obs_base"],
                row["edit_op"],
            )
            true_base_bin = _true_base_bin(row["true_base"])
            context_idx = model.context_index(
                prev2_base,
                prev1_base,
                run_base,
                run_length,
            )
            if model.scalar_run:
                run_bin = min(max(run_length, 0), model.max_run)
                counts[(*context_idx, run_bin, true_base_bin, target)] += 1
            else:
                counts[(*context_idx, true_base_bin, target)] += 1
            total_observations += 1
            accepted_since_callback += 1

            true_base = _normalise_context_base(row["true_base"])
            if true_base is not None:
                history.append(true_base)

            if scanned_since_callback >= progress_interval:
                flush_progress()

    flush_progress()
    return total_observations, skipped_rows


def context_error_model(
    counts: torch.Tensor,
    init_logits: torch.Tensor,
    run_values: torch.Tensor | None,
    additive_context: bool = False,
    context_indices: torch.Tensor | None = None,
    phred_context_sums: torch.Tensor | None = None,
    position_context_sums: torch.Tensor | None = None,
    strand_context_sums: torch.Tensor | None = None,
    fragment_count_per_context: torch.Tensor | None = None,
) -> None:
    """Pyro model for aggregated conditional categorical observations."""
    if additive_context:
        intercept_logits = pyro.param("intercept_logits", init_logits[0])
        base_logits = pyro.param("base_logits", init_logits[1])
        logits = _compose_additive_context_logits(
            intercept_logits,
            base_logits,
            counts.shape[0],
            context_indices=context_indices,
        )
    else:
        logits = pyro.param("logits", init_logits)
    if phred_context_sums is not None or position_context_sums is not None or strand_context_sums is not None:
        context_totals = counts.reshape(counts.shape[0], -1).sum(dim=-1, keepdim=True).clamp(min=1)
    if phred_context_sums is not None:
        num_lags = phred_context_sums.shape[-1]
        mean_phred = phred_context_sums / context_totals
        phred_weights = pyro.param(
            "phred_weights", torch.zeros(num_lags, NUM_ERROR_TYPES, dtype=logits.dtype)
        )
        logits = logits + mean_phred @ phred_weights
    if position_context_sums is not None:
        num_pos = position_context_sums.shape[-1]
        mean_pos = position_context_sums / context_totals
        position_weights = pyro.param(
            "position_weights", torch.zeros(num_pos, NUM_ERROR_TYPES, dtype=logits.dtype)
        )
        logits = logits + mean_pos @ position_weights
    if strand_context_sums is not None:
        mean_strand = strand_context_sums / context_totals
        strand_weights = pyro.param(
            "strand_weights", torch.zeros(1, NUM_ERROR_TYPES, dtype=logits.dtype)
        )
        logits = logits + mean_strand @ strand_weights
    if run_values is not None:
        run_slopes = pyro.param("run_slopes", torch.zeros_like(init_logits))
        run_step_unconstrained = pyro.param(
            "run_step_unconstrained",
            torch.zeros(run_values.numel() - 1, dtype=init_logits.dtype),
        )
        run_steps = F.softplus(run_step_unconstrained)
        learned_run_values = torch.cat(
            [torch.zeros(1, dtype=init_logits.dtype), torch.cumsum(run_steps, dim=0)]
        )
        run_shape = (1,) * (logits.dim() - 1) + (run_values.numel(), 1)
        run_x = learned_run_values.reshape(run_shape)
        logits = logits.unsqueeze(-2) + run_slopes.unsqueeze(-2) * run_x
    if fragment_count_per_context is not None:
        log_phi = pyro.param("log_phi_unconstrained", torch.tensor(3.0, dtype=logits.dtype))
        phi = F.softplus(log_phi)
        counts_flat = counts.reshape(counts.shape[0], -1)
        logits_flat = logits.reshape(counts.shape[0], -1)
        probs_flat = torch.softmax(logits_flat, dim=-1)
        frag = fragment_count_per_context.to(dtype=logits.dtype)
        concentration = probs_flat * (phi * frag.unsqueeze(-1))
        n_c = counts_flat.sum(dim=-1)
        mask = n_c > 0
        dm = torch.distributions.DirichletMultinomial(
            total_count=n_c[mask], concentration=concentration[mask]
        )
        log_lik = dm.log_prob(counts_flat[mask]).sum()
        pyro.factor("error_type_log_likelihood", log_lik)
    else:
        log_probs = _masked_log_probs_for_counts(logits, counts)
        pyro.factor("error_type_log_likelihood", _count_weighted_log_prob_sum(counts, log_probs))


def _compose_logits(
    logits: torch.Tensor,
    run_values: torch.Tensor | None,
    run_slopes: torch.Tensor | None = None,
    run_step_unconstrained: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return final logits after optional scalar homopolymer effect."""
    if run_values is None:
        return logits
    if run_slopes is None or run_step_unconstrained is None:
        raise ValueError("run_slopes and run_step_unconstrained are required")

    run_steps = F.softplus(run_step_unconstrained)
    learned_run_values = torch.cat(
        [torch.zeros(1, dtype=logits.dtype), torch.cumsum(run_steps, dim=0)]
    )
    run_shape = (1,) * (logits.dim() - 1) + (run_values.numel(), 1)
    run_x = learned_run_values.reshape(run_shape)
    return logits.unsqueeze(-2) + run_slopes.unsqueeze(-2) * run_x


def _context_length_from_shape(context_shape: Sequence[int]) -> int:
    """Return previous-base context length represented by a flat base-5 shape."""
    if len(context_shape) != 1:
        raise ValueError("additive context models require a flat context shape")
    context_count = int(context_shape[0])
    context_length = 0
    value = 1
    while value < context_count:
        value *= NUM_CONTEXT_BASES
        context_length += 1
    if value != context_count or context_length < 1:
        raise ValueError(f"Invalid additive context shape: {context_shape}")
    return context_length


def _context_base_indices_for_position(
    num_contexts: int,
    context_length: int,
    position: int,
    *,
    device: torch.device,
    context_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return base indices for one position in each flat context index."""
    divisor = NUM_CONTEXT_BASES ** (context_length - position - 1)
    idx = (
        context_indices.to(device)
        if context_indices is not None
        else torch.arange(num_contexts, device=device)
    )
    return (idx // divisor) % NUM_CONTEXT_BASES


def _compose_additive_context_logits(
    intercept_logits: torch.Tensor,
    base_logits: torch.Tensor,
    num_contexts: int,
    *,
    context_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose full context logits from additive position/base contributions."""
    context_length = int(base_logits.shape[0])
    base_logits = base_logits - base_logits.mean(dim=1, keepdim=True)
    logits = intercept_logits.expand(num_contexts, -1).clone()
    for position in range(context_length):
        base_indices = _context_base_indices_for_position(
            num_contexts,
            context_length,
            position,
            device=base_logits.device,
            context_indices=context_indices,
        )
        position_logits = base_logits[position].index_select(0, base_indices)
        logits = logits + position_logits
    return logits


def _initialise_additive_context_logits(
    counts: torch.Tensor,
    *,
    context_length: int,
    context_indices: torch.Tensor | None = None,
    pseudo_count: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return empirical initial additive intercept/base logits."""
    if pseudo_count <= 0:
        raise ValueError("pseudo_count must be positive")
    counts = _collapse_true_base_counts(counts)
    flat_counts = counts.reshape(-1, NUM_ERROR_TYPES)
    intercept_logits = torch.log(flat_counts.sum(dim=0) + pseudo_count)
    base_logits = torch.zeros(
        context_length,
        NUM_CONTEXT_BASES,
        NUM_ERROR_TYPES,
        dtype=counts.dtype,
        device=counts.device,
    )

    num_contexts = flat_counts.shape[0]
    for position in range(context_length):
        base_indices = _context_base_indices_for_position(
            num_contexts,
            context_length,
            position,
            device=counts.device,
            context_indices=context_indices,
        )
        position_counts = torch.zeros(
            NUM_CONTEXT_BASES,
            NUM_ERROR_TYPES,
            dtype=counts.dtype,
            device=counts.device,
        )
        position_counts.index_add_(0, base_indices, flat_counts)
        base_logits[position] = torch.log(position_counts + pseudo_count)
    base_logits = base_logits - base_logits.mean(dim=1, keepdim=True)
    return intercept_logits, base_logits


def bayesian_context_error_model(
    counts: torch.Tensor,
    init_logits: torch.Tensor,
    run_values: torch.Tensor | None,
    prior_scale: float,
    additive_context: bool = False,
    context_indices: torch.Tensor | None = None,
    phred_context_sums: torch.Tensor | None = None,
    position_context_sums: torch.Tensor | None = None,
    strand_context_sums: torch.Tensor | None = None,
    fragment_count_per_context: torch.Tensor | None = None,
) -> None:
    """Bayesian context error model with Normal priors over logit parameters."""
    if additive_context:
        intercept_logits = pyro.sample(
            "intercept_logits",
            dist.Normal(torch.zeros_like(init_logits[0]), prior_scale).to_event(1),
        )
        base_logits = pyro.sample(
            "base_logits",
            dist.Normal(torch.zeros_like(init_logits[1]), prior_scale).to_event(3),
        )
        logits = _compose_additive_context_logits(
            intercept_logits,
            base_logits,
            counts.shape[0],
            context_indices=context_indices,
        )
    else:
        logits = pyro.sample(
            "logits",
            dist.Normal(torch.zeros_like(init_logits), prior_scale).to_event(
                init_logits.dim()
            ),
        )
    if phred_context_sums is not None or position_context_sums is not None or strand_context_sums is not None:
        context_totals = counts.reshape(counts.shape[0], -1).sum(dim=-1, keepdim=True).clamp(min=1)
    if phred_context_sums is not None:
        num_lags = phred_context_sums.shape[-1]
        mean_phred = phred_context_sums / context_totals
        phred_weights = pyro.sample(
            "phred_weights",
            dist.Normal(
                torch.zeros(num_lags, NUM_ERROR_TYPES, dtype=logits.dtype),
                prior_scale,
            ).to_event(2),
        )
        logits = logits + mean_phred @ phred_weights
    if position_context_sums is not None:
        num_pos = position_context_sums.shape[-1]
        mean_pos = position_context_sums / context_totals
        position_weights = pyro.sample(
            "position_weights",
            dist.Normal(
                torch.zeros(num_pos, NUM_ERROR_TYPES, dtype=logits.dtype),
                prior_scale,
            ).to_event(2),
        )
        logits = logits + mean_pos @ position_weights
    if strand_context_sums is not None:
        mean_strand = strand_context_sums / context_totals
        strand_weights = pyro.sample(
            "strand_weights",
            dist.Normal(
                torch.zeros(1, NUM_ERROR_TYPES, dtype=logits.dtype),
                prior_scale,
            ).to_event(2),
        )
        logits = logits + mean_strand @ strand_weights
    run_slopes = None
    run_step_unconstrained = None
    if run_values is not None:
        run_slopes = pyro.sample(
            "run_slopes",
            dist.Normal(torch.zeros_like(init_logits), prior_scale).to_event(
                init_logits.dim()
            ),
        )
        run_step_unconstrained = pyro.sample(
            "run_step_unconstrained",
            dist.Normal(
                torch.zeros(run_values.numel() - 1, dtype=init_logits.dtype),
                prior_scale,
            ).to_event(1),
        )

    final_logits = _compose_logits(
        logits,
        run_values,
        run_slopes,
        run_step_unconstrained,
    )
    if fragment_count_per_context is not None:
        log_phi = pyro.sample(
            "log_phi_unconstrained",
            dist.Normal(torch.tensor(3.0, dtype=final_logits.dtype), prior_scale),
        )
        phi = F.softplus(log_phi)
        counts_flat = counts.reshape(counts.shape[0], -1)
        logits_flat = final_logits.reshape(counts.shape[0], -1)
        probs_flat = torch.softmax(logits_flat, dim=-1)
        frag = fragment_count_per_context.to(dtype=final_logits.dtype)
        concentration = probs_flat * (phi * frag.unsqueeze(-1))
        n_c = counts_flat.sum(dim=-1)
        mask = n_c > 0
        dm = torch.distributions.DirichletMultinomial(
            total_count=n_c[mask], concentration=concentration[mask]
        )
        log_lik = dm.log_prob(counts_flat[mask]).sum()
        pyro.factor("error_type_log_likelihood", log_lik)
    else:
        log_probs = _masked_log_probs_for_counts(final_logits, counts)
        pyro.factor("error_type_log_likelihood", _count_weighted_log_prob_sum(counts, log_probs))


def empty_guide(
    counts: torch.Tensor,
    init_logits: torch.Tensor,
    run_values: torch.Tensor | None,
    additive_context: bool = False,
    context_indices: torch.Tensor | None = None,
    phred_context_sums: torch.Tensor | None = None,
    position_context_sums: torch.Tensor | None = None,
    strand_context_sums: torch.Tensor | None = None,
    fragment_count_per_context: torch.Tensor | None = None,
) -> None:
    """Empty guide for maximum-likelihood optimisation."""
    del counts, init_logits, run_values, additive_context, context_indices, phred_context_sums, position_context_sums, strand_context_sums, fragment_count_per_context


def initialise_logits(counts: torch.Tensor, *, pseudo_count: float = 0.5) -> torch.Tensor:
    """Return stable empirical logits for optimisation initialisation."""
    if pseudo_count <= 0:
        raise ValueError("pseudo_count must be positive")
    return torch.log(_collapse_true_base_counts(counts) + pseudo_count)


def train_counts(
    counts: torch.Tensor,
    *,
    run_values: torch.Tensor | None = None,
    additive_context: bool = False,
    context_length: int | None = None,
    context_indices: torch.Tensor | None = None,
    phred_context_sums: torch.Tensor | None = None,
    position_context_sums: torch.Tensor | None = None,
    strand_context_sums: torch.Tensor | None = None,
    fragment_count_per_context: torch.Tensor | None = None,
    lr: float = 0.05,
    num_steps: int = 1000,
    clip_norm: float = 10.0,
    pseudo_count: float = 0.5,
    seed: int = 42,
    log_every: int = 100,
    progress_callback: TrainProgressCallback | None = None,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Fit conditional categorical parameters by maximum likelihood.

    Args:
        counts: Context-by-target count tensor.
        run_values: Optional run-length values for homopolymer models.
        additive_context: Whether to use the additive context parameterisation.
        context_length: Context length for additive models; derived from shape
            when None and counts has shape [4^N, E].
        context_indices: Original 4^N context indices for rows in counts after
            subsampling; None means counts rows are sequential (0, 1, …).
        lr: Adam learning rate.
        num_steps: Number of optimisation steps.
        clip_norm: Gradient clipping norm.
        pseudo_count: Positive value used only for logit initialisation.
        seed: Random seed for reproducibility.
        log_every: Log progress every this many steps.
        progress_callback: Optional callback receiving step index and loss.

    Returns:
        Parameter tensors and per-step losses.
    """
    if additive_context and run_values is not None:
        raise ValueError("additive_context is not supported with run_values")
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    if additive_context:
        if context_length is None:
            context_length = _context_length_from_shape(
                _context_shape_for_counts(counts, run_values)
            )
        init_logits = _initialise_additive_context_logits(
            counts,
            context_length=context_length,
            context_indices=context_indices,
            pseudo_count=pseudo_count,
        )
    else:
        init_counts = _model_counts_for_initial_logits(counts, run_values)
        init_logits = initialise_logits(init_counts, pseudo_count=pseudo_count)
    optimiser = pyro.optim.ClippedAdam({"lr": lr, "clip_norm": clip_norm})
    svi = SVI(context_error_model, empty_guide, optimiser, loss=Trace_ELBO())

    losses = []
    for step in range(num_steps):
        loss = float(svi.step(counts, init_logits, run_values, additive_context, context_indices, phred_context_sums, position_context_sums, strand_context_sums, fragment_count_per_context))
        losses.append(loss)
        if progress_callback is not None:
            progress_callback(step, loss)
        if step % log_every == 0 or step == num_steps - 1:
            logger.info("Step %d/%d loss %.4f", step, num_steps, loss)

    params = {
        name: value.detach().clone()
        for name, value in pyro.get_param_store().items()
    }
    return params, losses


def _posterior_summary_from_param_store() -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Extract AutoNormal posterior means, stdevs, and raw guide parameters."""
    params_mean = {}
    params_stdev = {}
    inference_params = {}

    for name, value in pyro.get_param_store().items():
        detached = value.detach().clone()
        inference_params[name] = detached
        if name.startswith("AutoNormal.locs."):
            params_mean[name.removeprefix("AutoNormal.locs.")] = detached
        elif name.startswith("AutoNormal.scales."):
            params_stdev[name.removeprefix("AutoNormal.scales.")] = detached

    return params_mean, params_stdev, inference_params


def train_bayesian_counts(
    counts: torch.Tensor,
    *,
    run_values: torch.Tensor | None = None,
    additive_context: bool = False,
    context_length: int | None = None,
    context_indices: torch.Tensor | None = None,
    phred_context_sums: torch.Tensor | None = None,
    position_context_sums: torch.Tensor | None = None,
    strand_context_sums: torch.Tensor | None = None,
    fragment_count_per_context: torch.Tensor | None = None,
    lr: float = 0.01,
    num_steps: int = 1000,
    clip_norm: float = 10.0,
    pseudo_count: float = 0.5,
    prior_scale: float = 2.0,
    seed: int = 42,
    log_every: int = 100,
    progress_callback: TrainProgressCallback | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], list[float]]:
    """Fit a mean-field variational posterior over context model parameters."""
    if prior_scale <= 0:
        raise ValueError("prior_scale must be positive")
    if additive_context and run_values is not None:
        raise ValueError("additive_context is not supported with run_values")

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    if additive_context:
        if context_length is None:
            context_length = _context_length_from_shape(
                _context_shape_for_counts(counts, run_values)
            )
        init_logits = _initialise_additive_context_logits(
            counts,
            context_length=context_length,
            context_indices=context_indices,
            pseudo_count=pseudo_count,
        )
        init_values = {
            "intercept_logits": init_logits[0],
            "base_logits": init_logits[1],
        }
    else:
        init_counts = _model_counts_for_initial_logits(counts, run_values)
        init_logits = initialise_logits(init_counts, pseudo_count=pseudo_count)
        init_values = {"logits": init_logits}
    if run_values is not None:
        init_values["run_slopes"] = torch.zeros_like(init_logits)
        init_values["run_step_unconstrained"] = torch.zeros(
            run_values.numel() - 1,
            dtype=init_logits.dtype,
        )
    _logit_dtype = init_logits[0].dtype if isinstance(init_logits, tuple) else init_logits.dtype
    if phred_context_sums is not None:
        num_lags = phred_context_sums.shape[-1]
        init_values["phred_weights"] = torch.zeros(num_lags, NUM_ERROR_TYPES, dtype=_logit_dtype)
    if position_context_sums is not None:
        num_pos = position_context_sums.shape[-1]
        init_values["position_weights"] = torch.zeros(num_pos, NUM_ERROR_TYPES, dtype=_logit_dtype)
    if strand_context_sums is not None:
        init_values["strand_weights"] = torch.zeros(1, NUM_ERROR_TYPES, dtype=_logit_dtype)
    if fragment_count_per_context is not None:
        init_values["log_phi_unconstrained"] = torch.tensor(3.0, dtype=_logit_dtype)
    guide = AutoNormal(
        bayesian_context_error_model,
        init_loc_fn=init_to_value(values=init_values),
    )
    optimiser = pyro.optim.ClippedAdam({"lr": lr, "clip_norm": clip_norm})
    svi = SVI(
        bayesian_context_error_model,
        guide,
        optimiser,
        loss=Trace_ELBO(),
    )

    losses = []
    for step in range(num_steps):
        loss = float(
            svi.step(counts, init_logits, run_values, prior_scale, additive_context, context_indices, phred_context_sums, position_context_sums, strand_context_sums, fragment_count_per_context)
        )
        losses.append(loss)
        if progress_callback is not None:
            progress_callback(step, loss)
        if step % log_every == 0 or step == num_steps - 1:
            logger.info("VI step %d/%d loss %.4f", step, num_steps, loss)

    params_mean, params_stdev, inference_params = _posterior_summary_from_param_store()
    return params_mean, params_stdev, inference_params, losses


def log_likelihood(
    counts: torch.Tensor,
    params: dict[str, torch.Tensor],
    run_values: torch.Tensor | None = None,
    additive_context: bool = False,
    context_indices: torch.Tensor | None = None,
    phred_context_sums: torch.Tensor | None = None,
    position_context_sums: torch.Tensor | None = None,
    strand_context_sums: torch.Tensor | None = None,
    fragment_count_per_context: torch.Tensor | None = None,
) -> float:
    """Return the conditional categorical log likelihood for counts."""
    logits = _get_full_logits(params, counts, run_values, additive_context, context_indices, phred_context_sums, position_context_sums, strand_context_sums)
    if fragment_count_per_context is not None and "log_phi_unconstrained" in params:
        phi = F.softplus(params["log_phi_unconstrained"])
        counts_flat = counts.reshape(counts.shape[0], -1)
        logits_flat = logits.reshape(counts.shape[0], -1)
        probs_flat = torch.softmax(logits_flat, dim=-1)
        frag = fragment_count_per_context.to(dtype=logits.dtype)
        concentration = probs_flat * (phi * frag.unsqueeze(-1))
        n_c = counts_flat.sum(dim=-1)
        mask = n_c > 0
        dm = torch.distributions.DirichletMultinomial(
            total_count=n_c[mask], concentration=concentration[mask]
        )
        return float(dm.log_prob(counts_flat[mask]).sum().item())
    log_probs = _masked_log_probs_for_counts(logits, counts)
    return float(_count_weighted_log_prob_sum(counts, log_probs).item())


def elbo_loss(
    counts: torch.Tensor,
    params: dict[str, torch.Tensor],
    *,
    run_values: torch.Tensor | None = None,
    additive_context: bool = False,
    prior_scale: float = 2.0,
    context_indices: torch.Tensor | None = None,
    phred_context_sums: torch.Tensor | None = None,
    position_context_sums: torch.Tensor | None = None,
    strand_context_sums: torch.Tensor | None = None,
    fragment_count_per_context: torch.Tensor | None = None,
) -> float:
    """Return joint negative log posterior at a parameter point."""
    if prior_scale <= 0:
        raise ValueError("prior_scale must be positive")

    loss = -log_likelihood(counts, params, run_values, additive_context, context_indices, phred_context_sums, position_context_sums, strand_context_sums, fragment_count_per_context)
    for value in params.values():
        loss -= float(
            dist.Normal(torch.zeros_like(value), prior_scale)
            .log_prob(value)
            .sum()
            .item()
        )
    return loss


def _get_full_logits(
    params: dict[str, torch.Tensor],
    counts: torch.Tensor,
    run_values: torch.Tensor | None,
    additive_context: bool,
    context_indices: torch.Tensor | None,
    phred_context_sums: torch.Tensor | None = None,
    position_context_sums: torch.Tensor | None = None,
    strand_context_sums: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose full logit tensor from fitted parameters."""
    if additive_context:
        logits = _compose_additive_context_logits(
            params["intercept_logits"],
            params["base_logits"],
            counts.shape[0],
            context_indices=context_indices,
        )
    else:
        logits = params["logits"]
    context_totals = counts.reshape(counts.shape[0], -1).sum(dim=-1, keepdim=True).clamp(min=1)
    if phred_context_sums is not None and "phred_weights" in params:
        mean_phred = phred_context_sums / context_totals
        logits = logits + mean_phred @ params["phred_weights"]
    if position_context_sums is not None and "position_weights" in params:
        mean_pos = position_context_sums / context_totals
        logits = logits + mean_pos @ params["position_weights"]
    if strand_context_sums is not None and "strand_weights" in params:
        mean_strand = strand_context_sums / context_totals
        logits = logits + mean_strand @ params["strand_weights"]
    return _compose_logits(
        logits,
        run_values,
        params.get("run_slopes"),
        params.get("run_step_unconstrained"),
    )


def _weighted_error_rate_with_offset(
    logits: torch.Tensor,
    counts: torch.Tensor,
    offset: float,
) -> float:
    """Return empirical-weighted error rate with scalar offset on non-match logits."""
    calibrated = logits.clone()
    calibrated[..., 1:] += offset
    log_probs = _masked_log_probs_for_counts(calibrated, counts)
    p_match = torch.exp(log_probs[..., 0])
    cell_weights = counts.sum(dim=-1)
    total = float(cell_weights.sum())
    if total == 0.0:
        return 0.0
    return float(((1.0 - p_match) * cell_weights).sum() / total)


def compute_marginal_error_rate(
    counts: torch.Tensor,
    params: dict[str, torch.Tensor],
    run_values: torch.Tensor | None = None,
    additive_context: bool = False,
    context_indices: torch.Tensor | None = None,
    phred_context_sums: torch.Tensor | None = None,
    position_context_sums: torch.Tensor | None = None,
    strand_context_sums: torch.Tensor | None = None,
) -> float:
    """Return model marginal error rate weighted by the empirical training distribution.

    Args:
        counts: Context-by-target count tensor from aggregation.
        params: Fitted parameter dict (MLE or VI posterior mean).
        run_values: Optional run-length values for homopolymer models.
        additive_context: Whether params use the additive parameterisation.
        context_indices: Subsampled context indices when counts rows are a subset.
        phred_context_sums: Optional Phred sums tensor; shape [num_contexts, num_lags].
        position_context_sums: Optional position sums tensor; shape [num_contexts, num_pos].
        strand_context_sums: Optional strand sums tensor; shape [num_contexts, 1].

    Returns:
        Weighted-average probability of any error across all training contexts.
    """
    logits = _get_full_logits(params, counts, run_values, additive_context, context_indices, phred_context_sums, position_context_sums, strand_context_sums)
    return _weighted_error_rate_with_offset(logits, counts, 0.0)


def calibrate_to_rate(
    counts: torch.Tensor,
    params: dict[str, torch.Tensor],
    target_rate: float,
    run_values: torch.Tensor | None = None,
    additive_context: bool = False,
    context_indices: torch.Tensor | None = None,
    phred_context_sums: torch.Tensor | None = None,
    position_context_sums: torch.Tensor | None = None,
    strand_context_sums: torch.Tensor | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Find calibration offset δ to add to non-match logits to match target_rate.

    Adds a single global scalar to all non-match logit columns so that the
    empirical-weighted marginal error rate equals ``target_rate``.  The
    context-conditional structure and error-type ratios are preserved.

    Args:
        counts: Context-by-target count tensor from aggregation.
        params: Fitted parameter dict (MLE or VI posterior mean).
        target_rate: Desired marginal error rate (e.g. Weibull estimate).
        run_values: Optional run-length values for homopolymer models.
        additive_context: Whether params use the additive parameterisation.
        context_indices: Subsampled context indices when counts rows are a subset.
        tol: Convergence tolerance on error rate.
        max_iter: Maximum bisection iterations.

    Returns:
        Scalar δ such that adding it to all non-match logits gives the target rate.

    Raises:
        ValueError: If target_rate is outside the achievable range.
    """
    logits = _get_full_logits(params, counts, run_values, additive_context, context_indices, phred_context_sums, position_context_sums, strand_context_sums)

    lo, hi = -50.0, 50.0
    rate_lo = _weighted_error_rate_with_offset(logits, counts, lo)
    rate_hi = _weighted_error_rate_with_offset(logits, counts, hi)

    if target_rate < rate_lo or target_rate > rate_hi:
        raise ValueError(
            f"target_rate {target_rate:.6g} is outside the achievable range "
            f"[{rate_lo:.6g}, {rate_hi:.6g}]"
        )

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        rate_mid = _weighted_error_rate_with_offset(logits, counts, mid)
        if abs(rate_mid - target_rate) < tol:
            return mid
        if rate_mid < target_rate:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def num_free_parameters(
    context_shape: Sequence[int],
    *,
    scalar_run: bool = False,
    additive_context: bool = False,
    max_run: int = DEFAULT_MAX_RUN,
    num_phred_lags: int = 0,
    num_position_features: int = 0,
    use_strand: bool = False,
    use_fragment_overdispersion: bool = False,
) -> int:
    """Return free categorical parameters after row-wise softmax invariance."""
    if additive_context:
        context_length = _context_length_from_shape(context_shape)
        base = (1 + (NUM_CONTEXT_BASES - 1) * context_length) * (NUM_ERROR_TYPES - 1)
    else:
        multiplier = 2 if scalar_run else 1
        transform_params = max_run if scalar_run else 0
        base = math.prod(context_shape) * multiplier * (NUM_ERROR_TYPES - 1) + transform_params
    return base + (num_phred_lags + num_position_features + int(use_strand)) * (NUM_ERROR_TYPES - 1) + int(use_fragment_overdispersion)


def aic(log_lik: float, num_parameters: int) -> float:
    """Return Akaike information criterion."""
    return 2.0 * num_parameters - 2.0 * log_lik


# ── Marginal Weibull ──────────────────────────────────────────────────────────

def _fit_weibull_to_survival(
    S: "np.ndarray",
    v: int,
) -> dict[str, float]:
    """Fit Weibull parameters to a survival function evaluated at t = 1 … v.

    Uses the Weibull linearisation: log(−log(S(t))) = log λ + β·log(t).
    Points where S is indistinguishable from 0 or 1 are excluded.

    Args:
        S: Survival probabilities at t = 1, 2, …, v (length-v array).
        v: Value-window length (used to compute the window-averaged rate).

    Returns:
        Dict with keys ``lambda``, ``beta``, and ``window_averaged_rate``.
        Values are ``nan`` when the fit cannot be performed.
    """
    t = np.arange(1, v + 1, dtype=float)
    valid = (S > 1e-15) & (S < 1.0 - 1e-15)
    if valid.sum() < 2:
        return {"lambda": float("nan"), "beta": float("nan"), "window_averaged_rate": float("nan")}
    y = np.log(-np.log(S[valid]))
    x = np.log(t[valid])
    coeffs = np.polyfit(x, y, 1)
    beta = float(max(coeffs[0], 1e-6))
    lam = float(np.exp(coeffs[1]))
    rate = (1.0 - math.exp(-lam * v**beta)) / v
    return {"lambda": lam, "beta": beta, "window_averaged_rate": rate}


def compute_marginal_weibull(
    counts: torch.Tensor,
    params: dict[str, torch.Tensor],
    v: int,
    run_values: torch.Tensor | None = None,
    additive_context: bool = False,
    context_indices: torch.Tensor | None = None,
    calibration_offset: float = 0.0,
    phred_context_sums: torch.Tensor | None = None,
    position_context_sums: torch.Tensor | None = None,
    strand_context_sums: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute marginal Weibull parameters predicted by the model.

    Uses a mixture-of-geometrics approximation: each context c contributes a
    geometric first-error-time distribution with rate (1 − p_match_c), weighted
    by the empirical context frequency in ``counts``.  The marginal survival
    function S(t) = Σ_c w_c · p_match_c^t is then fitted with a Weibull.

    The approximation ignores within-observation position-to-position context
    changes (the true context shifts as bases are revealed), but it provides a
    principled summary of the hazard-rate profile implied by the model and is
    directly comparable across source / retrained / real models on the same
    training distribution.

    Args:
        counts: Context-by-target count tensor from aggregation.
        params: Fitted parameter dict (MLE or VI posterior mean).
        v: Value-window length (max t in the survival function).
        run_values: Optional run-length values for homopolymer models.
        additive_context: Whether params use the additive parameterisation.
        context_indices: Subsampled context indices when counts rows are a subset.
        calibration_offset: Scalar offset applied to non-match logits (baked-in
            calibration delta).

    Returns:
        Dict with keys ``lambda``, ``beta``, ``window_averaged_rate``, and
        ``survival`` (list of S(1), …, S(v)).
    """
    logits = _get_full_logits(params, counts, run_values, additive_context, context_indices, phred_context_sums, position_context_sums, strand_context_sums)
    calibrated = logits.clone()
    calibrated[..., 1:] += calibration_offset

    log_probs = _masked_log_probs_for_counts(calibrated, counts)
    # p_match and cell_weights may be multi-dimensional when run_values are present
    # (e.g. shape [n_contexts, n_run_values]).  Flatten to 1-D for the mixture.
    p_match = torch.exp(log_probs[..., 0]).reshape(-1)
    cell_weights = counts.sum(dim=-1).float().reshape(-1)

    total = cell_weights.sum()
    if total == 0.0:
        nan = float("nan")
        return {"lambda": nan, "beta": nan, "window_averaged_rate": nan, "survival": []}

    w = cell_weights / total
    t_vals = torch.arange(1, v + 1, dtype=torch.float64)

    # S(t) = Σ_c w_c · p_match_c^t
    log_pm = torch.log(p_match.double().clamp(min=1e-15))  # [n_flat]
    log_surv = log_pm.unsqueeze(1) * t_vals.unsqueeze(0)   # [n_flat, v]
    S = (w.double().unsqueeze(1) * torch.exp(log_surv)).sum(dim=0).numpy()  # [v]

    result = _fit_weibull_to_survival(S, v)
    result["survival"] = S.tolist()
    return result


def subsample_context_counts(
    counts_obj: ContextCounts,
    max_contexts: int,
) -> ContextCounts:
    """Return a ContextCounts capped to at most `max_contexts` rows.

    Rows are ranked by descending total observation count so the most
    informative contexts are always retained. If the tensor already has
    at most `max_contexts` rows, it is returned unchanged.

    Args:
        counts_obj: Source counts to subsample.
        max_contexts: Maximum number of context rows to keep.

    Returns:
        A new ``ContextCounts`` with trimmed ``counts`` tensor and updated
        ``total_observations`` and ``low_count_contexts`` fields.
    """
    if counts_obj.counts.shape[0] <= max_contexts:
        return counts_obj
    row_totals = _context_totals(counts_obj.counts)
    _, keep_idx = torch.topk(row_totals, k=max_contexts, largest=True, sorted=False)
    trimmed = counts_obj.counts[keep_idx]
    new_total = int(trimmed.sum().item())
    context_totals = _context_totals(trimmed)
    new_low = int((context_totals < 10).sum().item())
    trimmed_phred = counts_obj.phred_context_sums[keep_idx] if counts_obj.phred_context_sums is not None else None
    trimmed_position = counts_obj.position_context_sums[keep_idx] if counts_obj.position_context_sums is not None else None
    trimmed_strand = counts_obj.strand_context_sums[keep_idx] if counts_obj.strand_context_sums is not None else None
    trimmed_frag = counts_obj.fragment_count_per_context[keep_idx] if counts_obj.fragment_count_per_context is not None else None
    logger.debug(
        "Subsampled context counts from %d to %d rows (%d → %d observations)",
        counts_obj.counts.shape[0],
        max_contexts,
        counts_obj.total_observations,
        new_total,
    )
    return dataclasses.replace(
        counts_obj,
        counts=trimmed,
        total_observations=new_total,
        low_count_contexts=new_low,
        context_indices=keep_idx,
        phred_context_sums=trimmed_phred,
        position_context_sums=trimmed_position,
        strand_context_sums=trimmed_strand,
        fragment_count_per_context=trimmed_frag,
    )


def fit_and_test(
    train_context_counts: ContextCounts,
    test_context_counts: ContextCounts,
    *,
    lr: float = 0.05,
    num_steps: int = 1000,
    clip_norm: float = 10.0,
    pseudo_count: float = 0.5,
    seed: int = 42,
    progress_callback: TrainProgressCallback | None = None,
) -> FitResult:
    """Train on aggregated counts and evaluate AIC on test counts."""
    _cl = (
        _context_length_from_shape(train_context_counts.context_shape)
        if train_context_counts.additive_context
        else None
    )
    params, losses = train_counts(
        train_context_counts.counts,
        run_values=train_context_counts.run_values,
        additive_context=train_context_counts.additive_context,
        context_length=_cl,
        context_indices=train_context_counts.context_indices,
        phred_context_sums=train_context_counts.phred_context_sums,
        position_context_sums=train_context_counts.position_context_sums,
        strand_context_sums=train_context_counts.strand_context_sums,
        fragment_count_per_context=train_context_counts.fragment_count_per_context,
        lr=lr,
        num_steps=num_steps,
        clip_norm=clip_norm,
        pseudo_count=pseudo_count,
        seed=seed,
        progress_callback=progress_callback,
    )
    train_ll = log_likelihood(
        train_context_counts.counts,
        params,
        train_context_counts.run_values,
        train_context_counts.additive_context,
        context_indices=train_context_counts.context_indices,
        phred_context_sums=train_context_counts.phred_context_sums,
        position_context_sums=train_context_counts.position_context_sums,
        strand_context_sums=train_context_counts.strand_context_sums,
        fragment_count_per_context=train_context_counts.fragment_count_per_context,
    )
    test_ll = log_likelihood(
        test_context_counts.counts,
        params,
        test_context_counts.run_values,
        test_context_counts.additive_context,
        context_indices=test_context_counts.context_indices,
        phred_context_sums=test_context_counts.phred_context_sums,
        position_context_sums=test_context_counts.position_context_sums,
        strand_context_sums=test_context_counts.strand_context_sums,
        fragment_count_per_context=test_context_counts.fragment_count_per_context,
    )
    _n_phred = train_context_counts.phred_context_sums.shape[-1] if train_context_counts.phred_context_sums is not None else 0
    _n_pos = train_context_counts.position_context_sums.shape[-1] if train_context_counts.position_context_sums is not None else 0
    _use_strand = train_context_counts.strand_context_sums is not None
    _use_frag = train_context_counts.fragment_count_per_context is not None
    k = num_free_parameters(
        train_context_counts.context_shape,
        scalar_run=train_context_counts.scalar_run,
        additive_context=train_context_counts.additive_context,
        max_run=train_context_counts.run_values.numel() - 1
        if train_context_counts.run_values is not None
        else DEFAULT_MAX_RUN,
        num_phred_lags=_n_phred,
        num_position_features=_n_pos,
        use_strand=_use_strand,
        use_fragment_overdispersion=_use_frag,
    )
    return FitResult(
        params=params,
        losses=losses,
        train_log_likelihood=train_ll,
        test_log_likelihood=test_ll,
        num_parameters=k,
        aic=aic(test_ll, k),
    )


def fit_bayesian_and_test(
    train_context_counts: ContextCounts,
    test_context_counts: ContextCounts,
    *,
    lr: float = 0.01,
    num_steps: int = 1000,
    clip_norm: float = 10.0,
    pseudo_count: float = 0.5,
    prior_scale: float = 2.0,
    seed: int = 42,
    progress_callback: TrainProgressCallback | None = None,
) -> BayesianFitResult:
    """Train a variational posterior and evaluate posterior-mean performance."""
    _cl = (
        _context_length_from_shape(train_context_counts.context_shape)
        if train_context_counts.additive_context
        else None
    )
    params_mean, params_stdev, inference_params, losses = train_bayesian_counts(
        train_context_counts.counts,
        run_values=train_context_counts.run_values,
        additive_context=train_context_counts.additive_context,
        context_length=_cl,
        context_indices=train_context_counts.context_indices,
        phred_context_sums=train_context_counts.phred_context_sums,
        position_context_sums=train_context_counts.position_context_sums,
        strand_context_sums=train_context_counts.strand_context_sums,
        fragment_count_per_context=train_context_counts.fragment_count_per_context,
        lr=lr,
        num_steps=num_steps,
        clip_norm=clip_norm,
        pseudo_count=pseudo_count,
        prior_scale=prior_scale,
        seed=seed,
        progress_callback=progress_callback,
    )
    train_ll = log_likelihood(
        train_context_counts.counts,
        params_mean,
        train_context_counts.run_values,
        train_context_counts.additive_context,
        context_indices=train_context_counts.context_indices,
        phred_context_sums=train_context_counts.phred_context_sums,
        position_context_sums=train_context_counts.position_context_sums,
        strand_context_sums=train_context_counts.strand_context_sums,
        fragment_count_per_context=train_context_counts.fragment_count_per_context,
    )
    test_ll = log_likelihood(
        test_context_counts.counts,
        params_mean,
        test_context_counts.run_values,
        test_context_counts.additive_context,
        context_indices=test_context_counts.context_indices,
        phred_context_sums=test_context_counts.phred_context_sums,
        position_context_sums=test_context_counts.position_context_sums,
        strand_context_sums=test_context_counts.strand_context_sums,
        fragment_count_per_context=test_context_counts.fragment_count_per_context,
    )
    return BayesianFitResult(
        params_mean=params_mean,
        params_stdev=params_stdev,
        inference_params=inference_params,
        losses=losses,
        train_log_likelihood=train_ll,
        test_log_likelihood=test_ll,
        train_elbo=-elbo_loss(
            train_context_counts.counts,
            params_mean,
            run_values=train_context_counts.run_values,
            additive_context=train_context_counts.additive_context,
            prior_scale=prior_scale,
            context_indices=train_context_counts.context_indices,
            phred_context_sums=train_context_counts.phred_context_sums,
            position_context_sums=train_context_counts.position_context_sums,
            strand_context_sums=train_context_counts.strand_context_sums,
            fragment_count_per_context=train_context_counts.fragment_count_per_context,
        ),
        test_elbo=-elbo_loss(
            test_context_counts.counts,
            params_mean,
            run_values=test_context_counts.run_values,
            additive_context=test_context_counts.additive_context,
            prior_scale=prior_scale,
            context_indices=test_context_counts.context_indices,
            phred_context_sums=test_context_counts.phred_context_sums,
            position_context_sums=test_context_counts.position_context_sums,
            strand_context_sums=test_context_counts.strand_context_sums,
            fragment_count_per_context=test_context_counts.fragment_count_per_context,
        ),
        prior_scale=prior_scale,
    )
