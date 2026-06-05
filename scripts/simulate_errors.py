#!/usr/bin/env python3
"""Simulate sequencing reads by injecting errors from a context error model.

Reads FASTA or gzipped FASTA reference sequences and produces a simulated
output file with errors sampled position-by-position from a trained context
error model.  The simulation uses the true reference as context (not the
accumulating mutated read), matching the convention under which the models
were trained.

Usage::

    python scripts/simulate_errors.py \\
        --model context_error_models/additive_7_hq-illumina.pt \\
        --input reference.fasta \\
        --output simulated_reads.fasta

    python scripts/simulate_errors.py \\
        --model context_error_models/context_3_ont.pt \\
        --input reference.fasta.gz \\
        --output simulated_reads.fastq \\
        --fastq --n-copies 5 --seed 0
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from lib.encoding import BASE_TO_IDX, NUM_ERROR_TYPES

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_BASES: Final[str] = "ACGT"
_N_BASES: Final[int] = 4
_TABLE_L_LIMIT: Final[int] = 10   # precompute full table only when L ≤ this
_MAX_INS_RUN_DEFAULT: Final[int] = 10

# Error type index boundaries (matches lib/encoding.py)
_ERR_MATCH: Final[int] = 0
_ERR_SUB_START: Final[int] = 1    # 1–4: sub_to_A/C/G/T
_ERR_INS_START: Final[int] = 5    # 5–8: ins_A/C/G/T
_ERR_DEL: Final[int] = 9


# ── Model loading ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Model:
    """Precomputed model state for simulation.

    Attributes:
        context_length: Number of preceding reference bases used as context.
        logit_table: Per-context logits, shape [4^L, 10].  None when L exceeds
            _TABLE_L_LIMIT, in which case intercept and base_logits are used.
        intercept: Intercept logits [10] for additive on-the-fly composition.
        base_logits: Mean-centred position effects [L, 4, 10].
    """

    context_length: int
    logit_table: np.ndarray | None
    intercept: np.ndarray | None
    base_logits: np.ndarray | None


@dataclass
class SimulationTruth:
    """Sparse empirical counts accumulated during simulation."""

    context_length: int
    event_counts: Counter[tuple[int, int, int]]
    n_references: int = 0
    n_reads: int = 0
    n_input_bases: int = 0
    n_output_bases: int = 0

    def add_event(self, context_index: int, true_base: int, error_type: int) -> None:
        """Record one sampled error event."""
        self.event_counts[(context_index, true_base, error_type)] += 1

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable sparse truth summary."""
        return {
            "context_length": self.context_length,
            "n_references": self.n_references,
            "n_reads": self.n_reads,
            "n_input_bases": self.n_input_bases,
            "n_output_bases": self.n_output_bases,
            "error_type_names": [
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
            ],
            "counts": [
                {
                    "context_index": context_index,
                    "true_base": true_base,
                    "error_type": error_type,
                    "count": count,
                }
                for (context_index, true_base, error_type), count in sorted(
                    self.event_counts.items()
                )
            ],
        }


def _build_additive_table(
    intercept: np.ndarray,
    bl_centered: np.ndarray,
    context_length: int,
) -> np.ndarray:
    """Enumerate all 4^L contexts and return composed logits [4^L, 10].

    Args:
        intercept: Shape [10], float32.
        bl_centered: Mean-centred base logits, shape [L, 4, 10], float32.
        context_length: L.

    Returns:
        Float32 array of shape [4^L, 10].
    """
    n_ctx = _N_BASES ** context_length
    indices = np.arange(n_ctx, dtype=np.int32)
    table = np.broadcast_to(intercept, (n_ctx, NUM_ERROR_TYPES)).copy()
    for pos in range(context_length):
        div = _N_BASES ** (context_length - pos - 1)
        base_idx = (indices // div) % _N_BASES
        table += bl_centered[pos, base_idx, :]
    return table.astype(np.float32)


def load_model(path: Path, use_vi: bool = False) -> _Model:
    """Load a .pt context error model artifact.

    Args:
        path: Path to the trained model checkpoint.
        use_vi: Use variational-inference posterior-mean parameters instead of
            the maximum-likelihood point estimate.

    Returns:
        Precomputed model state ready for :func:`simulate_read`.

    Raises:
        KeyError: If the artifact is missing the requested inference block.
        ValueError: If the parameter structure is not recognised.
    """
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")

    context_length = int(artifact.get("context_length", 1))
    inf_key = "variational_inference" if use_vi else "maximum_likelihood"
    param_key = "params_mean" if use_vi else "params"
    params: dict = artifact[inf_key][param_key]

    # Load calibration offset if present (baked in at load time for zero simulation overhead).
    offset_key = "calibration_offset_vi" if use_vi else "calibration_offset_mle"
    cal = artifact.get("calibration") or {}
    calibration_offset = float(cal.get(offset_key, 0.0))

    if "logits" in params:
        # Combinatorial context: the logits tensor is the table directly.
        table = params["logits"].detach().cpu().numpy().reshape(-1, NUM_ERROR_TYPES)
        table = table.astype(np.float32)
        if calibration_offset != 0.0:
            table[:, 1:] += calibration_offset
        return _Model(context_length=context_length, logit_table=table,
                      intercept=None, base_logits=None)

    if "intercept_logits" in params:
        intercept = params["intercept_logits"].detach().cpu().numpy().astype(np.float32)
        bl = params["base_logits"].detach().cpu().numpy().astype(np.float32)
        bl_centered = (bl - bl.mean(axis=1, keepdims=True)).astype(np.float32)

        if context_length <= _TABLE_L_LIMIT:
            table = _build_additive_table(intercept, bl_centered, context_length)
            if calibration_offset != 0.0:
                table[:, 1:] += calibration_offset
            return _Model(context_length=context_length, logit_table=table,
                          intercept=None, base_logits=None)

        # L > _TABLE_L_LIMIT: compose on the fly during simulation; shift the intercept.
        if calibration_offset != 0.0:
            intercept = intercept.copy()
            intercept[1:] += calibration_offset
        return _Model(context_length=context_length, logit_table=None,
                      intercept=intercept, base_logits=bl_centered)

    raise ValueError(f"Unrecognised parameter keys in {path}: {sorted(params)}")


# ── Context encoding & logit lookup ───────────────────────────────────────────


def _context_index(context_bases: list[int], context_length: int) -> int:
    """Encode a list of base indices as a big-endian base-4 integer.

    Missing leading positions (near the sequence start) are treated as A (0).

    Args:
        context_bases: Base indices (0=A … 3=T), oldest first.  May be shorter
            than context_length.
        context_length: Full context length L.

    Returns:
        Integer in [0, 4^L).
    """
    idx = 0
    for _ in range(context_length - len(context_bases)):
        idx = idx * _N_BASES          # A-padding: multiply, no addition
    for b in context_bases:
        idx = idx * _N_BASES + b
    return idx


def _logits_at(model: _Model, context_bases: list[int]) -> np.ndarray:
    """Return logits[10] for the given preceding context.

    Args:
        model: Loaded model.
        context_bases: Preceding base indices, oldest first.

    Returns:
        Float32 logit array of shape [10].
    """
    ctx_idx = _context_index(context_bases, model.context_length)

    if model.logit_table is not None:
        return model.logit_table[ctx_idx]

    # On-the-fly additive composition for large models (L > _TABLE_L_LIMIT).
    if model.intercept is None or model.base_logits is None:
        raise RuntimeError("Model has no on-the-fly composition parameters")
    logits = model.intercept.copy()
    offset = model.context_length - len(context_bases)
    for i, b in enumerate(context_bases):
        logits += model.base_logits[offset + i, b, :]
    return logits


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Return softmax probabilities for a 1-D logit array."""
    shifted = logits - logits.max()
    exp_l = np.exp(shifted)
    return exp_l / exp_l.sum()


def probabilities_for_context(
    model: _Model,
    context_index: int,
    true_base_idx: int | None = None,
) -> np.ndarray:
    """Return valid sampling probabilities for a context and optional true base.

    Args:
        model: Loaded context error model.
        context_index: Big-endian base-4 context index.
        true_base_idx: Current true base index. When present, the impossible
            substitution-to-self category is masked before normalisation.

    Returns:
        Probability vector over the 10 error-type categories.
    """
    context_bases = []
    idx = context_index
    for _ in range(model.context_length):
        context_bases.append(idx % _N_BASES)
        idx //= _N_BASES
    logits = _logits_at(model, list(reversed(context_bases)))
    probs = _softmax(logits)
    if true_base_idx is not None:
        probs = probs.copy()
        probs[_ERR_SUB_START + true_base_idx] = 0.0
        total = probs.sum()
        if total <= 0.0:
            raise ValueError("All probabilities were masked for a context")
        probs /= total
    return probs


def _phred(probs: np.ndarray) -> int:
    """Compute integer Phred quality from a probability vector.

    Returns -10 * log10(P(error)), clamped to [0, 60].

    Args:
        probs: Softmax probabilities of length NUM_ERROR_TYPES.

    Returns:
        Integer Phred score.
    """
    p_err = max(float(1.0 - probs[_ERR_MATCH]), 1e-6)
    return min(60, int(-10.0 * math.log10(p_err)))


# ── Phred calibration ─────────────────────────────────────────────────────────


def load_phred_calibration(path: Path) -> np.ndarray:
    """Load a Phred calibration JSON and return P(Q | error_type).

    Args:
        path: Path to a calibration JSON produced by fit_phred_calibration.py.

    Returns:
        Float32 array of shape [10, 61] where entry [e, q] is the probability
        of observing Phred score q given error type e.

    Raises:
        ValueError: If the calibration file has an unexpected shape.
    """
    with open(path) as fh:
        cal = json.load(fh)
    probs = np.array(cal["probs"], dtype=np.float64)
    if probs.shape != (NUM_ERROR_TYPES, 61):
        raise ValueError(
            f"Calibration {path} has unexpected shape {probs.shape}; "
            f"expected ({NUM_ERROR_TYPES}, 61)"
        )
    # Normalize each row so it sums exactly to 1 (guards against float precision
    # drift introduced by JSON round-trip and dtype conversion).
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return (probs / row_sums).astype(np.float32)


def _sample_phred(
    error_type: int,
    calib: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Sample an integer Phred score from the calibration for an error type.

    Args:
        error_type: Error type index (0–9).
        calib: Calibration array [10, 61] from :func:`load_phred_calibration`.
        rng: NumPy random generator.

    Returns:
        Sampled Phred score in [0, 60], or -1 for deletion (error_type=9) or
        any row with no empirical observations.
    """
    if error_type == _ERR_DEL or calib[error_type].sum() == 0:
        return -1
    p = calib[error_type].astype(np.float64)
    p /= p.sum()
    return int(rng.choice(61, p=p))


# ── Core simulation ────────────────────────────────────────────────────────────


def simulate_read(
    reference: str,
    model: _Model,
    rng: np.random.Generator,
    *,
    max_ins_run: int = _MAX_INS_RUN_DEFAULT,
    emit_quality: bool = False,
    phred_calib: np.ndarray | None = None,
    truth: SimulationTruth | None = None,
) -> tuple[str, str | None]:
    """Simulate a single sequencing read from a reference sequence.

    For each reference base, an error type is sampled from the model's
    conditional distribution P(error | preceding k true bases).  Insertions
    do not advance the reference pointer; a cap on consecutive insertions at
    the same position prevents degenerate output.

    Args:
        reference: True reference sequence (ACGT string; other characters map
            to A).
        model: Loaded context error model.
        rng: NumPy random generator for reproducible sampling.
        max_ins_run: Maximum consecutive insertions at one reference position
            before the reference pointer is forcibly advanced.
        emit_quality: If True, compute and return a Phred quality string.
        phred_calib: Optional calibration array [10, 61] from
            :func:`load_phred_calibration`.  When provided, Phred scores are
            sampled from P(Q | error_type) after the error type is known.
            When None, Phred is derived from P(error | context) via
            ``-10·log10(1 − P(match))``.

    Returns:
        Tuple of (bases, quality_string_or_None).  The quality string uses
        Phred+33 ASCII encoding, matching FASTQ conventions.
    """
    ref_idx = [BASE_TO_IDX.get(b.upper(), 0) for b in reference]
    k = model.context_length
    ref_len = len(ref_idx)

    output_bases: list[str] = []
    output_phreds: list[int] = []
    ref_pos = 0

    # Resolve the quality-scoring strategy once before the inner loop.
    get_phred: Callable[[int, np.ndarray], int] | None = None
    if emit_quality and phred_calib is not None:
        _calib = phred_calib  # capture for closure

        def _calib_phred(et: int, p: np.ndarray) -> int:  # noqa: ARG001
            del p
            return _sample_phred(et, _calib, rng)

        get_phred = _calib_phred
    elif emit_quality:
        def _context_phred(et: int, p: np.ndarray) -> int:  # noqa: ARG001
            del et
            return _phred(p)

        get_phred = _context_phred

    while ref_pos < ref_len:
        ctx_start = max(0, ref_pos - k)
        context_bases = ref_idx[ctx_start:ref_pos]
        ctx_idx = _context_index(context_bases, k)

        logits = _logits_at(model, context_bases)
        probs = _softmax(logits)
        probs = probs.copy()
        probs[_ERR_SUB_START + ref_idx[ref_pos]] = 0.0
        prob_sum = probs.sum()
        if prob_sum <= 0.0:
            raise ValueError("All probabilities were masked for a context")
        probs /= prob_sum

        ins_count = 0
        while True:
            error_type = int(rng.choice(NUM_ERROR_TYPES, p=probs))
            if truth is not None:
                truth.add_event(ctx_idx, ref_idx[ref_pos], error_type)

            if error_type == _ERR_MATCH:
                output_bases.append(_BASES[ref_idx[ref_pos]])
                if get_phred is not None:
                    q = get_phred(error_type, probs)
                    if q >= 0:
                        output_phreds.append(q)
                ref_pos += 1
                break

            if _ERR_SUB_START <= error_type < _ERR_INS_START:
                output_bases.append(_BASES[error_type - _ERR_SUB_START])
                if get_phred is not None:
                    q = get_phred(error_type, probs)
                    if q >= 0:
                        output_phreds.append(q)
                ref_pos += 1
                break

            if _ERR_INS_START <= error_type < _ERR_DEL:
                output_bases.append(_BASES[error_type - _ERR_INS_START])
                if get_phred is not None:
                    q = get_phred(error_type, probs)
                    if q >= 0:
                        output_phreds.append(q)
                ins_count += 1
                if ins_count >= max_ins_run:
                    ref_pos += 1
                    break
                # Resample at the same reference position; context unchanged.
                continue

            # Deletion: consume the reference base, emit nothing.
            # No quality byte is emitted for deleted bases.
            ref_pos += 1
            break

    bases_str = "".join(output_bases)
    qual_str = (
        "".join(chr(q + 33) for q in output_phreds) if emit_quality else None
    )
    return bases_str, qual_str


# ── I/O helpers ────────────────────────────────────────────────────────────────


def _iter_fasta(path: Path, opener) -> Iterator[tuple[str, str]]:
    """Yield (name, sequence) pairs from an open-able FASTA path."""
    with opener(path, "rt") as fh:
        name, parts = "", []
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name:
                    yield name, "".join(parts)
                name, parts = line[1:].split()[0], []
            elif not line.startswith(";"):
                parts.append(line)
        if name:
            yield name, "".join(parts)


def _iter_fastq(path: Path, opener) -> Iterator[tuple[str, str]]:
    """Yield (name, sequence) pairs from an open-able FASTQ path."""
    with opener(path, "rt") as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().rstrip("\n")
            fh.readline()   # '+'
            fh.readline()   # quality
            if header.startswith("@") and seq:
                yield header[1:].rstrip("\n").split()[0], seq


def iter_sequences(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (name, sequence) pairs from a FASTA or FASTQ file.

    Supports plain and gzip-compressed files.  Format is detected from the
    first non-empty line (``>`` = FASTA, ``@`` = FASTQ).

    Args:
        path: Path to the input file.

    Yields:
        Tuples of (sequence_name, sequence_string).

    Raises:
        ValueError: If the file format cannot be determined.
    """
    opener = gzip.open if path.suffix == ".gz" else open

    first_char = ""
    with opener(path, "rt") as fh:
        for line in fh:
            if line.strip():
                first_char = line[0]
                break

    if first_char == ">":
        yield from _iter_fasta(path, opener)
    elif first_char == "@":
        yield from _iter_fastq(path, opener)
    else:
        raise ValueError(
            f"Cannot determine sequence format of {path!r} "
            f"(first non-empty character: {first_char!r})"
        )


def _write_fasta(fh, name: str, sequence: str, width: int = 80) -> None:
    """Write one FASTA record."""
    fh.write(f">{name}\n")
    for i in range(0, len(sequence), width):
        fh.write(sequence[i : i + width] + "\n")


def _write_fastq(fh, name: str, sequence: str, quality: str) -> None:
    """Write one FASTQ record."""
    fh.write(f"@{name}\n{sequence}\n+\n{quality}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Return the configured argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Simulate sequencing reads by sampling errors from a trained "
            "context error model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", required=True, type=Path, metavar="MODEL.pt",
        help="Trained context error model artifact (.pt file).",
    )
    p.add_argument(
        "--input", required=True, type=Path, metavar="REF.fasta[.gz]",
        help="Input reference sequences (FASTA or FASTQ, plain or gzipped).",
    )
    p.add_argument(
        "--output", required=True, type=Path, metavar="OUT",
        help="Output file (.fasta or .fastq).",
    )
    p.add_argument(
        "--fastq", action="store_true",
        help="Write FASTQ output with model-derived Phred quality scores.",
    )
    p.add_argument(
        "--n-copies", type=int, default=1, metavar="N",
        help="Number of simulated reads per reference entry.",
    )
    p.add_argument(
        "--use-vi", action="store_true",
        help="Use VI posterior-mean parameters instead of MLE.",
    )
    p.add_argument(
        "--seed", type=int, default=42, metavar="SEED",
        help="Random seed.",
    )
    p.add_argument(
        "--max-ins-run", type=int, default=_MAX_INS_RUN_DEFAULT, metavar="N",
        help="Max consecutive insertions per reference position before advancing.",
    )
    p.add_argument(
        "--phred-calibration", type=Path, default=None, metavar="CAL.json",
        help=(
            "Phred calibration JSON from fit_phred_calibration.py.  "
            "When provided, Phred scores are sampled from P(Q | error_type) "
            "conditioned on the sampled error type.  "
            "When absent, Phred = -10·log10(P(error | context))."
        ),
    )
    p.add_argument(
        "--truth-summary", type=Path, default=None, metavar="TRUTH.json",
        help="Write sparse generated error counts by context and error type.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the error simulator.

    Args:
        argv: Command-line arguments; defaults to sys.argv.

    Returns:
        Exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    logger.info("Loading model from %s", args.model)
    model = load_model(args.model, use_vi=args.use_vi)
    logger.info(
        "context_length=%d  table=%s  inference=%s",
        model.context_length,
        "precomputed" if model.logit_table is not None else "on-the-fly",
        "VI" if args.use_vi else "MLE",
    )

    phred_calib: np.ndarray | None = None
    if args.phred_calibration is not None:
        phred_calib = load_phred_calibration(args.phred_calibration)
        logger.info("Loaded Phred calibration from %s", args.phred_calibration)

    rng = np.random.default_rng(args.seed)
    emit_quality = args.fastq
    opener = gzip.open if args.output.suffix == ".gz" else open
    n_refs = n_reads = 0
    truth = (
        SimulationTruth(model.context_length, Counter())
        if args.truth_summary is not None
        else None
    )

    with opener(args.output, "wt") as out_fh:
        for ref_name, ref_seq in iter_sequences(args.input):
            n_refs += 1
            if truth is not None:
                truth.n_references += 1
                truth.n_input_bases += len(ref_seq)
            for copy_idx in range(args.n_copies):
                read_name = (
                    f"{ref_name}_sim{copy_idx + 1}" if args.n_copies > 1
                    else f"{ref_name}_sim"
                )
                bases, qual = simulate_read(
                    ref_seq, model, rng,
                    max_ins_run=args.max_ins_run,
                    emit_quality=emit_quality,
                    phred_calib=phred_calib,
                    truth=truth,
                )
                if emit_quality and qual is not None:
                    _write_fastq(out_fh, read_name, bases, qual)
                else:
                    _write_fasta(out_fh, read_name, bases)
                n_reads += 1
                if truth is not None:
                    truth.n_reads += 1
                    truth.n_output_bases += len(bases)

    logger.info(
        "Wrote %d read(s) from %d reference sequence(s) to %s",
        n_reads, n_refs, args.output,
    )
    if args.truth_summary is not None and truth is not None:
        args.truth_summary.parent.mkdir(parents=True, exist_ok=True)
        with open(args.truth_summary, "w") as handle:
            json.dump(truth.to_json_dict(), handle, indent=2)
            handle.write("\n")
        logger.info("Wrote truth summary to %s", args.truth_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
