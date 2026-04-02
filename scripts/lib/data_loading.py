"""Data loading and tensor construction for the profile HMM error model.

Reads skiver dump ``base_observations.tsv`` files and produces tensors
for error types, phred bins, and dinucleotide context indices.
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from .encoding import (
    bin_phred,
    encode_context,
    encode_error_type,
    error_type_to_class,
)

logger = logging.getLogger(__name__)


# ─── Per-sequence data ────────────────────────────────────────────────────────

class SequenceData:
    """Encoded observations for a single (key, value) occurrence.

    Attributes:
        error_type: Error type indices, shape ``[v]``.
        phred_bin: Phred bin indices, shape ``[v]``.
        context_idx: Dinucleotide context indices, shape ``[v]``.
        error_class: Coarse error class indices, shape ``[v]``.
        has_error: Whether any position has a non-match error type.
    """

    __slots__ = ("error_type", "phred_bin", "context_idx", "error_class",
                 "has_error")

    def __init__(
        self,
        error_type: list[int],
        phred_bin: list[int],
        context_idx: list[int],
        error_class: list[int],
        has_error: bool,
    ) -> None:
        self.error_type = error_type
        self.phred_bin = phred_bin
        self.context_idx = context_idx
        self.error_class = error_class
        self.has_error = has_error


# ─── Loading ──────────────────────────────────────────────────────────────────

def _has_prev_base_column(path: Path) -> bool:
    """Check whether the TSV file has a ``prev_base`` column."""
    with open(path, newline="") as fh:
        header = fh.readline().strip().split("\t")
    return "prev_base" in header


def load_base_observations(
    prefix: str | Path,
    *,
    passes_filter_only: bool = True,
) -> list[SequenceData]:
    """Load base_observations.tsv and return encoded sequence data.

    Handles both old-format (no ``prev_base`` column) and new-format files.
    For old-format files, prev_base defaults to the previous row's true_base
    within the same obs_id (or 'A' at t=1 as a fallback).

    Args:
        prefix: Output prefix used with ``skiver dump -o``.
        passes_filter_only: If true, skip observations from outlier keys.

    Returns:
        List of SequenceData, one per (key, value) occurrence.
    """
    path = Path(f"{prefix}.base_observations.tsv")
    if not path.exists():
        logger.warning("File not found: %s", path)
        return []

    has_prev_base = _has_prev_base_column(path)
    if not has_prev_base:
        logger.info("Old-format file (no prev_base column): %s", path)

    sequences: dict[int, SequenceData] = {}
    prev_true_base_by_obs: dict[int, str] = {}

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if passes_filter_only and row["passes_filter"] != "true":
                continue

            obs_id = int(row["obs_id"])
            true_base = row["true_base"]
            obs_base = row["obs_base"]
            edit_op = row["edit_op"]
            phred = int(row["phred"])

            # Determine prev_base.
            if has_prev_base:
                prev_base = row["prev_base"]
            else:
                prev_base = prev_true_base_by_obs.get(obs_id, "A")

            # For context encoding, if true_base is a gap (insertion),
            # use obs_base as a stand-in for the "position" context.
            context_true = true_base if true_base != "-" else obs_base

            et = encode_error_type(true_base, obs_base, edit_op)
            pb = bin_phred(phred)
            ci = encode_context(prev_base, context_true)
            ec = error_type_to_class(et)

            if obs_id not in sequences:
                sequences[obs_id] = SequenceData([], [], [], [], False)

            seq = sequences[obs_id]
            seq.error_type.append(et)
            seq.phred_bin.append(pb)
            seq.context_idx.append(ci)
            seq.error_class.append(ec)
            if et != 0:
                seq.has_error = True

            # Track prev_true_base for old-format fallback.
            if true_base != "-":
                prev_true_base_by_obs[obs_id] = true_base

    result = list(sequences.values())
    n_errors = sum(1 for s in result if s.has_error)
    logger.info(
        "Loaded %d sequences (%d with errors) from %s",
        len(result), n_errors, path,
    )
    return result


def load_multiple(
    prefixes: Sequence[str | Path],
    *,
    passes_filter_only: bool = True,
) -> list[SequenceData]:
    """Load and concatenate base observations from multiple prefixes.

    Args:
        prefixes: Output prefixes used with ``skiver dump -o``.
        passes_filter_only: If true, skip observations from outlier keys.

    Returns:
        Combined list of SequenceData from all prefixes.
    """
    all_seqs: list[SequenceData] = []
    for prefix in prefixes:
        all_seqs.extend(
            load_base_observations(prefix, passes_filter_only=passes_filter_only)
        )
    logger.info("Total sequences loaded: %d", len(all_seqs))
    return all_seqs


# ─── Stratified subsampling ──────────────────────────────────────────────────

def stratified_subsample(
    sequences: list[SequenceData],
    max_no_error_ratio: float = 50.0,
    seed: int = 42,
) -> list[SequenceData]:
    """Subsample sequences to reduce class imbalance.

    Keeps ALL sequences with at least one error.  Subsamples error-free
    sequences so that the ratio of error-free to error-containing sequences
    does not exceed ``max_no_error_ratio``.

    Args:
        sequences: Full list of loaded sequences.
        max_no_error_ratio: Maximum ratio of error-free to error sequences.
        seed: Random seed for reproducibility.

    Returns:
        Subsampled list of SequenceData.
    """
    with_error = [s for s in sequences if s.has_error]
    without_error = [s for s in sequences if not s.has_error]

    max_no_error = int(len(with_error) * max_no_error_ratio)
    if len(without_error) <= max_no_error:
        logger.info(
            "No subsampling needed: %d error, %d no-error",
            len(with_error), len(without_error),
        )
        return sequences

    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(without_error), generator=gen)[:max_no_error]
    sampled_no_error = [without_error[i] for i in indices.tolist()]

    logger.info(
        "Subsampled: %d error + %d no-error (from %d)",
        len(with_error), len(sampled_no_error), len(without_error),
    )
    return with_error + sampled_no_error


# ─── Tensor construction ─────────────────────────────────────────────────────

class BatchTensors:
    """Packed tensors for a batch of sequences.

    Attributes:
        error_type: Error type indices, shape ``[N, T]``.
        phred_bin: Phred bin indices, shape ``[N, T]``.
        context_idx: Dinucleotide context indices, shape ``[N, T]``.
        error_class: Coarse error class indices, shape ``[N, T]``.
        lengths: Actual sequence lengths, shape ``[N]``.
        v: Maximum sequence length (= value window size).
    """

    __slots__ = ("error_type", "phred_bin", "context_idx", "error_class",
                 "lengths", "v")

    def __init__(
        self,
        error_type: torch.Tensor,
        phred_bin: torch.Tensor,
        context_idx: torch.Tensor,
        error_class: torch.Tensor,
        lengths: torch.Tensor,
        v: int,
    ) -> None:
        self.error_type = error_type
        self.phred_bin = phred_bin
        self.context_idx = context_idx
        self.error_class = error_class
        self.lengths = lengths
        self.v = v


def build_tensors(sequences: list[SequenceData]) -> BatchTensors:
    """Convert a list of SequenceData into padded tensors.

    All sequences are padded (with zeros) to the maximum observed length.

    Args:
        sequences: Encoded sequence data from ``load_base_observations``.

    Returns:
        BatchTensors with all fields populated.

    Raises:
        ValueError: If sequences is empty.
    """
    if not sequences:
        raise ValueError("sequences must not be empty")

    lengths = [len(s.error_type) for s in sequences]
    v = max(lengths)
    n = len(sequences)

    error_type = torch.zeros(n, v, dtype=torch.long)
    phred_bin = torch.zeros(n, v, dtype=torch.long)
    context_idx = torch.zeros(n, v, dtype=torch.long)
    error_class = torch.zeros(n, v, dtype=torch.long)

    for i, seq in enumerate(sequences):
        seq_len = lengths[i]
        error_type[i, :seq_len] = torch.tensor(seq.error_type, dtype=torch.long)
        phred_bin[i, :seq_len] = torch.tensor(seq.phred_bin, dtype=torch.long)
        context_idx[i, :seq_len] = torch.tensor(seq.context_idx, dtype=torch.long)
        error_class[i, :seq_len] = torch.tensor(seq.error_class, dtype=torch.long)

    return BatchTensors(
        error_type=error_type,
        phred_bin=phred_bin,
        context_idx=context_idx,
        error_class=error_class,
        lengths=torch.tensor(lengths, dtype=torch.long),
        v=v,
    )
