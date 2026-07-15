#!/usr/bin/env python3
"""``skiver generate`` — apply a context error model to sequences.

This is the generic subprocess interface other tools (e.g. genome-blender) use
to request error application.  It reads FASTA records from a file or stdin and
writes FASTQ records (with a CIGAR carried in the header comment) to a file or
stdout, sampling errors from a trained context error model.

The model can be selected three ways (mutually exclusive):

* ``--model PATH.pt``          — a trained artifact
* ``--preset NAME``            — a bundled platform preset (``hq-illumina``,
                                 ``lq-illumina``, ``ont``, ``pacbio``, …)
* ``--components STR --params PARAMS.{json,pt}`` — a component string plus
                                 explicit parameter values

Output format (one record per input sequence, valid FASTQ)::

    @<name> cigar:<CIGAR>
    <observed bases>
    +
    <phred+33 quality>

For paired-end input pass ``--paired`` with R1/R2 **interleaved** (records named
``…/1`` and ``…/2``).  R2 mates are simulated with the reverse-strand covariate.

Usage::

    printf '>r1\\nACGTACGTACGT\\n' | skiver-generate --preset hq-illumina --seed 0
    skiver-generate --model m.pt --input reads.fasta --output out.fastq --seed 0
    skiver-generate --paired --input pairs.fasta --preset hq-illumina > out.fastq
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.error_application import (  # noqa: E402
    ErrorModel,
    apply_read,
    load_phred_calibration,
)

logger = logging.getLogger("skiver.generate")

_MAX_INS_RUN_DEFAULT = 10
_BATCH = 4096   # flush cadence; per-read sampling is already vectorised


# ── Input ────────────────────────────────────────────────────────────────────────


def _open_text(path: Path, mode: str):
    return gzip.open(path, mode) if path.suffix == ".gz" else open(path, mode)


def _iter_fasta(handle) -> Iterator[tuple[str, str]]:
    """Yield (name, sequence) from an open FASTA text handle."""
    name, parts = "", []
    for line in handle:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name:
                yield name, "".join(parts)
            name, parts = line[1:].split()[0] if len(line) > 1 else "", []
        elif line and not line.startswith(";"):
            parts.append(line)
    if name:
        yield name, "".join(parts)


def _read_records(input_path: Path | None) -> Iterator[tuple[str, str]]:
    """Yield (name, sequence) from a path or stdin (FASTA)."""
    if input_path is None:
        yield from _iter_fasta(sys.stdin)
    else:
        with _open_text(input_path, "rt") as fh:
            yield from _iter_fasta(fh)


def _is_forward(name: str, paired: bool) -> bool:
    """R2 mates (``…/2``) are reverse strand; everything else is forward."""
    return not (paired and name.endswith("/2"))


# ── Output ───────────────────────────────────────────────────────────────────────


def _write_fastq(handle, name: str, sequence: str, quality: str, cigar: str) -> None:
    """Write one FASTQ record with the CIGAR in the header comment."""
    handle.write(f"@{name} cigar:{cigar}\n{sequence}\n+\n{quality}\n")


# ── Model selection ──────────────────────────────────────────────────────────────


def _load_params(path: Path, *, use_vi: bool) -> dict:
    """Load an explicit parameter mapping from JSON or a torch ``.pt`` file.

    Accepts either a bare parameter mapping (``logits`` or ``intercept_logits``
    + ``base_logits`` …) or a full training artifact, from which the requested
    inference block's parameters are extracted.
    """
    if path.suffix in {".pt", ".pth"}:
        import torch

        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
    else:
        with open(path) as fh:
            obj = json.load(fh)

    # Unwrap a full artifact into its parameter mapping.
    if "maximum_likelihood" in obj or "variational_inference" in obj:
        block = "variational_inference" if use_vi else "maximum_likelihood"
        key = "params_mean" if use_vi else "params"
        return obj[block][key]
    return obj


def _build_model(args: argparse.Namespace) -> ErrorModel:
    if args.model is not None:
        model = ErrorModel.load(args.model, use_vi=args.use_vi)
    elif args.preset is not None:
        model = ErrorModel.preset(args.preset, use_vi=args.use_vi)
    else:
        params = _load_params(args.params, use_vi=args.use_vi)
        model = ErrorModel.from_spec(
            args.components, params, calibration_offset=args.calibration_offset,
        )
    if args.error_rate_scale != 1.0:
        model = _scale_error_rate(model, args.error_rate_scale)
    return model


def _scale_error_rate(model: ErrorModel, scale: float) -> ErrorModel:
    """Approximate a multiplicative error-rate scale.

    Adds ``log(scale)`` to every non-match logit once, so the change is shared
    across all reads instead of being recomputed per read.
    """
    import dataclasses

    delta = float(np.log(max(scale, 1e-12)))
    if model.logit_table is not None:
        tbl = model.logit_table.copy()
        tbl[:, 1:] += delta
        return dataclasses.replace(model, logit_table=tbl)
    assert model.intercept is not None
    inter = model.intercept.copy()
    inter[1:] += delta
    return dataclasses.replace(model, intercept=inter)


# ── CLI ──────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skiver-generate",
        description="Apply a Skiver context error model to sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--model", type=Path, metavar="MODEL.pt",
                     help="Trained context error model artifact.")
    sel.add_argument("--preset", type=str, metavar="NAME",
                     help="Bundled platform preset (e.g. hq-illumina, ont, pacbio).")
    sel.add_argument("--components", type=str, metavar="STR",
                     help="Component string, used with --params.")
    p.add_argument("--params", type=Path, metavar="PARAMS",
                   help="Parameter values (JSON or .pt) for --components.")
    p.add_argument("--calibration-offset", type=float, default=0.0,
                   help="Scalar added to every non-match logit (with --components).")

    p.add_argument("--input", type=Path, default=None, metavar="REF.fasta[.gz]",
                   help="Input FASTA (default: stdin).")
    p.add_argument("--output", type=Path, default=None, metavar="OUT.fastq[.gz]",
                   help="Output FASTQ (default: stdout).")
    p.add_argument("--paired", action="store_true",
                   help="Treat input as interleaved R1/R2 (…/1, …/2); R2 is reverse strand.")
    p.add_argument("--use-vi", action="store_true",
                   help="Use VI posterior-mean parameters instead of MLE.")
    p.add_argument("--error-rate-scale", type=float, default=1.0,
                   help="Multiplier applied to every non-match probability mass.")
    p.add_argument("--phred-calibration", type=Path, default=None, metavar="CAL.json",
                   help="P(Q|error_type) calibration JSON; else Phred from context.")
    p.add_argument("--max-ins-run", type=int, default=_MAX_INS_RUN_DEFAULT,
                   help="Max consecutive insertions per reference position.")
    p.add_argument("--no-quality", action="store_true",
                   help="Emit '*' quality strings instead of model-derived Phred.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.components is not None and args.params is None:
        _build_parser().error("--components requires --params")
    if not 0.0 <= args.error_rate_scale:
        _build_parser().error("--error-rate-scale must be non-negative")

    model = _build_model(args)
    logger.info(
        "model=%s context_length=%d strand=%s",
        model.name, model.context_length, model.strand_weights is not None,
    )

    phred_calib = (
        load_phred_calibration(args.phred_calibration)
        if args.phred_calibration is not None else None
    )
    rng = np.random.default_rng(args.seed)
    emit_quality = not args.no_quality
    out = _open_text(args.output, "wt") if args.output is not None else sys.stdout
    n = 0
    try:
        for name, seq in _read_records(args.input):
            rec = apply_read(
                model, name, seq, rng,
                is_forward=_is_forward(name, args.paired),
                max_ins_run=args.max_ins_run,
                emit_quality=emit_quality,
                phred_calib=phred_calib,
            )
            qual = rec.quality if rec.quality is not None else "*" * len(rec.sequence)
            _write_fastq(out, rec.name, rec.sequence, qual, rec.cigar)
            n += 1
            if n % _BATCH == 0 and out is not sys.stdout:
                out.flush()
    finally:
        if out is not sys.stdout:
            out.close()
    logger.info("Wrote %d read(s).", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
