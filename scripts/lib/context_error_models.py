"""Context-dependent, non-HMM sequencing error models.

These models treat each aligned base observation from ``skiver dump --base`` as
conditionally independent given sequence-context covariates.  They are fitted by
maximum likelihood with Pyro parameters and a categorical log-likelihood.
"""
from __future__ import annotations

import csv
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.initialization import init_to_value

from .encoding import NUM_ERROR_TYPES, encode_error_type

logger = logging.getLogger(__name__)

BASES: Final[tuple[str, ...]] = ("A", "C", "G", "T", "N")
BASE_TO_IDX: Final[dict[str, int]] = {base: idx for idx, base in enumerate(BASES)}
NUM_CONTEXT_BASES: Final[int] = len(BASES)
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
        context = ["N"] * max(0, self.context_length - len(history))
        context.extend(history[-self.context_length:])
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
    """Return a stable base index, mapping gaps/unknowns to N."""
    return BASE_TO_IDX.get(base, BASE_TO_IDX["N"])


def _normalise_base(base: str) -> str:
    """Return A/C/G/T/N for context reconstruction."""
    return base if base in BASE_TO_IDX and base != "N" else "N"


def _parse_bool(value: str) -> bool:
    """Parse skiver TSV boolean text."""
    return value.lower() == "true"


def _update_run(history: list[str]) -> tuple[str, int]:
    """Return the immediate homopolymer run ending before the current base."""
    if not history:
        return "N", 0

    run_base = history[-1]
    run_length = 0
    for base in reversed(history):
        if base != run_base:
            break
        run_length += 1
    return run_base, run_length


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
            NUM_ERROR_TYPES,
            dtype=torch.float32,
        )
        run_values = torch.tensor(
            [model.run_value(length) for length in range(model.max_run + 1)],
            dtype=torch.float32,
        )
    else:
        counts = torch.zeros(*model.context_shape, NUM_ERROR_TYPES, dtype=torch.float32)
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

    if model.scalar_run:
        context_totals = counts.sum(dim=(-1, -2))
    else:
        context_totals = counts.sum(dim=-1)
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
        NUM_ERROR_TYPES,
        dtype=torch.float32,
    )
    hpoly_counts = torch.zeros(
        *hpoly_model.context_shape,
        max_run + 1,
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
    prev2_context_totals = prev2_counts.sum(dim=-1)
    hpoly_context_totals = hpoly_counts.sum(dim=(-1, -2))
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


def aggregate_context_length_screen_counts(
    prefixes: Iterable[str | Path],
    *,
    context_lengths: Sequence[int],
    passes_filter_only: bool = True,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 10_000,
) -> ContextLengthScreenCounts:
    """Aggregate previous-base context models in one pass over TSV files.

    Args:
        prefixes: Skiver dump prefixes. Each prefix must have a matching
            ``.base_observations.tsv`` file.
        context_lengths: Previous-base context lengths to aggregate.
        passes_filter_only: If true, ignore rows from outlier keys.
        progress_callback: Optional callback receiving path, scanned rows,
            accepted rows, and skipped rows since the previous callback.
        progress_interval: Row interval between progress callback invocations.

    Returns:
        Counts keyed by context length.
    """
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

    total_observations = 0
    skipped_rows = 0
    for prefix in prefixes:
        path = Path(f"{prefix}.base_observations.tsv")
        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            continue
        file_total, file_skipped = _aggregate_context_length_screen_file(
            path,
            models,
            count_tensors,
            passes_filter_only=passes_filter_only,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
        total_observations += file_total
        skipped_rows += file_skipped
        logger.info("Aggregated %d rows from %s", file_total, path)

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


def _aggregate_context_length_screen_file(
    path: Path,
    models: Sequence[PreviousBasesErrorModel],
    count_tensors: dict[int, torch.Tensor],
    *,
    passes_filter_only: bool,
    progress_callback: ProgressCallback | None,
    progress_interval: int,
) -> tuple[int, int]:
    """Aggregate one TSV file into all previous-base context length tensors."""
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

            row_prev_base = _normalise_base(row["prev_base"])
            obs_id = int(row["obs_id"])
            if obs_id != current_obs_id:
                current_obs_id = obs_id
                history = [row_prev_base]

            target = encode_error_type(
                row["true_base"],
                row["obs_base"],
                row["edit_op"],
            )
            for model in models:
                context_idx = model.context_index_from_history(history)
                count_tensors[model.context_length][(*context_idx, target)] += 1

            total_observations += 1
            accepted_since_callback += 1

            true_base = _normalise_base(row["true_base"])
            if true_base != "N":
                history.append(true_base)

            if scanned_since_callback >= progress_interval:
                flush_progress()

    flush_progress()
    return total_observations, skipped_rows


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

            row_prev_base = _normalise_base(row["prev_base"])
            obs_id = int(row["obs_id"])
            if obs_id != current_obs_id:
                current_obs_id = obs_id
                history = [row_prev_base]

            prev1_base = history[-1] if history else row_prev_base
            prev2_base = history[-2] if len(history) >= 2 else "N"
            run_base, run_length = _update_run(history)
            target = encode_error_type(
                row["true_base"],
                row["obs_base"],
                row["edit_op"],
            )

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
            prev2_counts[(*prev2_idx, target)] += 1
            hpoly_counts[(*hpoly_idx, run_bin, target)] += 1

            total_observations += 1
            accepted_since_callback += 1

            true_base = _normalise_base(row["true_base"])
            if true_base != "N":
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

            row_prev_base = _normalise_base(row["prev_base"])
            obs_id = int(row["obs_id"])
            if obs_id != current_obs_id:
                current_obs_id = obs_id
                history = [row_prev_base]

            prev1_base = history[-1] if history else row_prev_base
            prev2_base = history[-2] if len(history) >= 2 else "N"
            run_base, run_length = _update_run(history)

            target = encode_error_type(
                row["true_base"],
                row["obs_base"],
                row["edit_op"],
            )
            context_idx = model.context_index(
                prev2_base,
                prev1_base,
                run_base,
                run_length,
            )
            if model.scalar_run:
                run_bin = min(max(run_length, 0), model.max_run)
                counts[(*context_idx, run_bin, target)] += 1
            else:
                counts[(*context_idx, target)] += 1
            total_observations += 1
            accepted_since_callback += 1

            true_base = _normalise_base(row["true_base"])
            if true_base != "N":
                history.append(true_base)

            if scanned_since_callback >= progress_interval:
                flush_progress()

    flush_progress()
    return total_observations, skipped_rows


def context_error_model(
    counts: torch.Tensor,
    init_logits: torch.Tensor,
    run_values: torch.Tensor | None,
) -> None:
    """Pyro model for aggregated conditional categorical observations."""
    logits = pyro.param("logits", init_logits)
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
    log_probs = F.log_softmax(logits, dim=-1)
    pyro.factor("error_type_log_likelihood", (counts * log_probs).sum())


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


def bayesian_context_error_model(
    counts: torch.Tensor,
    init_logits: torch.Tensor,
    run_values: torch.Tensor | None,
    prior_scale: float,
) -> None:
    """Bayesian context error model with Normal priors over logit parameters."""
    logits = pyro.sample(
        "logits",
        dist.Normal(torch.zeros_like(init_logits), prior_scale).to_event(
            init_logits.dim()
        ),
    )
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
    log_probs = F.log_softmax(final_logits, dim=-1)
    pyro.factor("error_type_log_likelihood", (counts * log_probs).sum())


def empty_guide(
    counts: torch.Tensor,
    init_logits: torch.Tensor,
    run_values: torch.Tensor | None,
) -> None:
    """Empty guide for maximum-likelihood optimisation."""
    del counts, init_logits, run_values


def initialise_logits(counts: torch.Tensor, *, pseudo_count: float = 0.5) -> torch.Tensor:
    """Return stable empirical logits for optimisation initialisation."""
    if pseudo_count <= 0:
        raise ValueError("pseudo_count must be positive")
    return torch.log(counts + pseudo_count)


def train_counts(
    counts: torch.Tensor,
    *,
    run_values: torch.Tensor | None = None,
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
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    init_counts = counts.sum(dim=-2) if run_values is not None else counts
    init_logits = initialise_logits(init_counts, pseudo_count=pseudo_count)
    optimiser = pyro.optim.ClippedAdam({"lr": lr, "clip_norm": clip_norm})
    svi = SVI(context_error_model, empty_guide, optimiser, loss=Trace_ELBO())

    losses = []
    for step in range(num_steps):
        loss = float(svi.step(counts, init_logits, run_values))
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

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    init_counts = counts.sum(dim=-2) if run_values is not None else counts
    init_logits = initialise_logits(init_counts, pseudo_count=pseudo_count)
    init_values = {"logits": init_logits}
    if run_values is not None:
        init_values["run_slopes"] = torch.zeros_like(init_logits)
        init_values["run_step_unconstrained"] = torch.zeros(
            run_values.numel() - 1,
            dtype=init_logits.dtype,
        )
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
        loss = float(svi.step(counts, init_logits, run_values, prior_scale))
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
) -> float:
    """Return the conditional categorical log likelihood for counts."""
    logits = params["logits"]
    if run_values is not None:
        run_steps = F.softplus(params["run_step_unconstrained"])
        learned_run_values = torch.cat(
            [torch.zeros(1, dtype=logits.dtype), torch.cumsum(run_steps, dim=0)]
        )
        run_shape = (1,) * (logits.dim() - 1) + (run_values.numel(), 1)
        run_x = learned_run_values.reshape(run_shape)
        logits = logits.unsqueeze(-2) + params["run_slopes"].unsqueeze(-2) * run_x
    return float((counts * F.log_softmax(logits, dim=-1)).sum().item())


def elbo_loss(
    counts: torch.Tensor,
    params: dict[str, torch.Tensor],
    *,
    run_values: torch.Tensor | None = None,
    prior_scale: float = 2.0,
) -> float:
    """Return joint negative log posterior at a parameter point."""
    if prior_scale <= 0:
        raise ValueError("prior_scale must be positive")

    loss = -log_likelihood(counts, params, run_values)
    for value in params.values():
        loss -= float(
            dist.Normal(torch.zeros_like(value), prior_scale)
            .log_prob(value)
            .sum()
            .item()
        )
    return loss


def num_free_parameters(
    context_shape: Sequence[int],
    *,
    scalar_run: bool = False,
    max_run: int = DEFAULT_MAX_RUN,
) -> int:
    """Return free categorical parameters after row-wise softmax invariance."""
    multiplier = 2 if scalar_run else 1
    transform_params = max_run if scalar_run else 0
    return math.prod(context_shape) * multiplier * (NUM_ERROR_TYPES - 1) + transform_params


def aic(log_lik: float, num_parameters: int) -> float:
    """Return Akaike information criterion."""
    return 2.0 * num_parameters - 2.0 * log_lik


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
    params, losses = train_counts(
        train_context_counts.counts,
        run_values=train_context_counts.run_values,
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
    )
    test_ll = log_likelihood(
        test_context_counts.counts,
        params,
        test_context_counts.run_values,
    )
    k = num_free_parameters(
        train_context_counts.context_shape,
        scalar_run=train_context_counts.scalar_run,
        max_run=train_context_counts.run_values.numel() - 1
        if train_context_counts.run_values is not None
        else DEFAULT_MAX_RUN,
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
    params_mean, params_stdev, inference_params, losses = train_bayesian_counts(
        train_context_counts.counts,
        run_values=train_context_counts.run_values,
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
    )
    test_ll = log_likelihood(
        test_context_counts.counts,
        params_mean,
        test_context_counts.run_values,
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
            prior_scale=prior_scale,
        ),
        test_elbo=-elbo_loss(
            test_context_counts.counts,
            params_mean,
            run_values=test_context_counts.run_values,
            prior_scale=prior_scale,
        ),
        prior_scale=prior_scale,
    )
