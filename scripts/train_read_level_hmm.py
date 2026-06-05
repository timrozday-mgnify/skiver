#!/usr/bin/env python3
"""Train a read-level HMM from skiver dump --windows output.

Models slow-varying latent quality states across a read. Each kv-mer window
observation is binary (is_error = error_type > 0). The HMM has S states; each
state has a different error rate expressed as an offset from the marginal rate:

    logit P(is_error=1 | state_s) = logit(P_marginal) + emission_logit_offsets[s]

Training reads {prefix}.windows.bin produced by ``skiver dump --windows``.

Usage:
    python scripts/train_read_level_hmm.py \\
        ../skiver_run/sample/sample -o read_level_hmm.pt

    python scripts/train_read_level_hmm.py \\
        prefix1 prefix2 -o model.pt --states 2 --steps 2000
"""
from __future__ import annotations

import argparse
import logging
import math
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pyro
import torch
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────────

WINDOWS_BIN_MAGIC = b"SKIVRERR"
WINDOWS_BIN_HEADER_SIZE = 32
WINDOWS_BIN_RECORD_DTYPE = np.dtype(
    [("read_id", "<u8"), ("start_index", "<u4"), ("error_type", "u1")]
)

DEFAULT_NUM_STATES = 2
DEFAULT_LR = 0.005
DEFAULT_NUM_STEPS = 2000


# ─── Data loading ───────────────────────────────────────────────────────────────

def _parse_header(f: "BinaryIO") -> dict:
    """Read and validate the 32-byte windows.bin header."""
    magic = f.read(8)
    if magic != WINDOWS_BIN_MAGIC:
        raise ValueError(f"Not a windows.bin file (magic={magic!r})")
    version, k, v, _ = struct.unpack("BBBB", f.read(4))
    if version != 1:
        raise ValueError(f"Unsupported windows.bin version {version}")
    (n_records,) = struct.unpack("<Q", f.read(8))
    f.read(12)  # padding
    return {"version": version, "k": k, "v": v, "n_records": n_records}


def load_windows_bin(
    paths: Sequence[str | Path],
    *,
    reads_cap: int | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """Load one or more windows.bin files and group by read.

    Args:
        paths: Paths to ``{prefix}.windows.bin`` files.
        reads_cap: If given, keep only the first this many reads.

    Returns:
        Tuple of (reads, metadata) where:
        - reads: list of (positions, is_error) per read. ``positions`` is a
          sorted int32 array of start_index values; ``is_error`` is a bool array.
        - metadata: dict with keys ``k``, ``v``, ``n_records``, ``P_marginal``.
    """
    all_recs: list[np.ndarray] = []
    meta: dict | None = None

    for path in paths:
        path = Path(path)
        bin_path = path if path.suffix == ".bin" else Path(f"{path}.windows.bin")
        if not bin_path.exists():
            logger.warning("File not found: %s", bin_path)
            continue
        with open(bin_path, "rb") as f:
            hdr = _parse_header(f)
            if meta is None:
                meta = {"k": hdr["k"], "v": hdr["v"], "n_records": 0}
            recs = np.fromfile(f, dtype=WINDOWS_BIN_RECORD_DTYPE)
        meta["n_records"] = meta.get("n_records", 0) + len(recs)
        all_recs.append(recs)
        logger.info("Loaded %d records from %s", len(recs), bin_path)

    if not all_recs:
        raise FileNotFoundError("No windows.bin records loaded; check input paths")
    if meta is None:
        meta = {"k": 0, "v": 0, "n_records": 0}

    combined = np.concatenate(all_recs)
    # Sort by (read_id, start_index)
    order = np.lexsort((combined["start_index"], combined["read_id"]))
    combined = combined[order]

    # Estimate marginal error rate from data
    n_error = int((combined["error_type"] > 0).sum())
    n_total = len(combined)
    p_marginal = n_error / n_total if n_total > 0 else 0.05
    meta["P_marginal"] = float(p_marginal)

    # Group by read_id
    reads: list[tuple[np.ndarray, np.ndarray]] = []
    _, first_indices = np.unique(combined["read_id"], return_index=True)
    split_points = np.append(first_indices[1:], len(combined))

    for start, end in zip(first_indices, split_points):
        block = combined[start:end]
        positions = block["start_index"].astype(np.int32)
        is_error = (block["error_type"] > 0)
        reads.append((positions, is_error))

    if reads_cap is not None and len(reads) > reads_cap:
        reads = reads[:reads_cap]

    logger.info(
        "Grouped into %d reads; marginal error rate = %.4f",
        len(reads), p_marginal,
    )
    return reads, meta


# ─── Forward algorithm ──────────────────────────────────────────────────────────

def _build_a_powers(
    A: torch.Tensor,
    gaps: np.ndarray,
) -> dict[int, torch.Tensor]:
    """Precompute A^g for all unique gap values (iterative multiplication)."""
    unique_gaps = sorted(set(int(g) for g in gaps if g > 0))
    if not unique_gaps:
        return {}
    # Iterative: A^1, A^2, ..., A^max_gap via successive matrix multiply
    max_gap = unique_gaps[-1]
    A_pows: dict[int, torch.Tensor] = {}
    current = A
    for g in range(1, max_gap + 1):
        if g > 1:
            current = current @ A
        A_pows[g] = current
    return A_pows


def gap_aware_log_likelihood(
    reads: list[tuple[np.ndarray, np.ndarray]],
    initial_logits: torch.Tensor,
    transition_logits: torch.Tensor,
    emission_logit_offsets: torch.Tensor,
    P_marginal: float,
) -> torch.Tensor:
    """Compute sum of per-read log-likelihoods via gap-aware forward algorithm.

    Args:
        reads: List of (positions, is_error) pairs from load_windows_bin.
        initial_logits: Shape [S]; log-unnorm initial state distribution.
        transition_logits: Shape [S, S]; row = from-state.
        emission_logit_offsets: Shape [S]; additive offsets to logit(P_marginal).
        P_marginal: Background error rate scalar.

    Returns:
        Scalar tensor: sum of log P(observations) over all reads.
    """
    A = F.softmax(transition_logits, dim=-1)  # [S, S]
    log_initial = F.log_softmax(initial_logits, dim=-1)  # [S]

    base_logit = math.log(P_marginal / (1.0 - P_marginal + 1e-15) + 1e-15)
    p_error = torch.sigmoid(base_logit + emission_logit_offsets)  # [S]
    log_emit = torch.stack([
        torch.log((1.0 - p_error).clamp(min=1e-40)),  # is_error=False
        torch.log(p_error.clamp(min=1e-40)),            # is_error=True
    ], dim=0)  # [2, S]

    # Collect all gaps to precompute A^g
    all_gaps: list[int] = []
    for positions, _ in reads:
        if len(positions) > 1:
            gaps = np.diff(positions.astype(np.int64))
            all_gaps.extend(int(g) for g in gaps)

    A_pows = _build_a_powers(A, np.array(all_gaps, dtype=np.int64))

    total_log_lik = torch.zeros((), dtype=initial_logits.dtype)

    for positions, is_error in reads:
        n = len(positions)
        if n == 0:
            continue

        log_alpha = log_initial.clone()  # [S]

        for i in range(n):
            if i > 0:
                gap = int(positions[i]) - int(positions[i - 1])
                if gap > 0:
                    A_g = A_pows[gap]
                    log_A_g = A_g.clamp(min=1e-40).log()
                    # alpha_new[j] = logsumexp_i( alpha[i] + log_A[i,j] )
                    log_alpha = torch.logsumexp(
                        log_alpha.unsqueeze(1) + log_A_g, dim=0
                    )
                # gap=0 should not occur (duplicate position), skip transition

            emit_idx = int(bool(is_error[i]))
            log_alpha = log_alpha + log_emit[emit_idx]

        total_log_lik = total_log_lik + torch.logsumexp(log_alpha, dim=-1)

    return total_log_lik


# ─── Pyro model ─────────────────────────────────────────────────────────────────

def read_level_hmm_model(
    reads: list[tuple[np.ndarray, np.ndarray]],
    P_marginal: float,
    num_states: int,
) -> None:
    """Pyro model: read-level HMM with Bernoulli emissions.

    All parameters are ``pyro.param`` (MAP estimation; empty guide).
    """
    S = num_states

    initial_logits = pyro.param(
        "initial_logits",
        torch.zeros(S),
    )
    transition_logits = pyro.param(
        "transition_logits",
        torch.eye(S) * 3.0,  # sticky init: strong self-transitions
    )
    emission_logit_offsets = pyro.param(
        "emission_logit_offsets",
        torch.zeros(S),  # init at marginal rate
    )

    log_lik = gap_aware_log_likelihood(
        reads, initial_logits, transition_logits, emission_logit_offsets, P_marginal,
    )
    pyro.factor("log_lik", log_lik)


def read_level_hmm_guide(
    reads: list[tuple[np.ndarray, np.ndarray]],
    P_marginal: float,
    num_states: int,
) -> None:
    """Empty guide — all parameters declared as pyro.param."""


# ─── Training ───────────────────────────────────────────────────────────────────

def train(
    reads: list[tuple[np.ndarray, np.ndarray]],
    P_marginal: float,
    num_states: int = DEFAULT_NUM_STATES,
    lr: float = DEFAULT_LR,
    num_steps: int = DEFAULT_NUM_STEPS,
) -> dict[str, torch.Tensor]:
    """Train the read-level HMM via MAP estimation.

    Args:
        reads: Grouped per-read observations from load_windows_bin.
        P_marginal: Background error rate.
        num_states: Number of latent quality states.
        lr: Adam learning rate.
        num_steps: Number of SVI steps.

    Returns:
        Dictionary of trained parameter tensors.
    """
    pyro.clear_param_store()

    svi = SVI(
        read_level_hmm_model,
        read_level_hmm_guide,
        pyro.optim.Adam({"lr": lr}),
        loss=Trace_ELBO(),
    )

    logger.info(
        "Training read-level HMM: %d reads, %d states, P_marginal=%.4f, steps=%d",
        len(reads), num_states, P_marginal, num_steps,
    )

    for step in range(num_steps):
        loss = svi.step(reads, P_marginal, num_states)
        if step % 100 == 0 or step == num_steps - 1:
            logger.info("Step %4d / %d  loss = %.4f", step, num_steps, loss)

    return {
        name: value.detach().clone()
        for name, value in pyro.get_param_store().items()
    }


# ─── Reporting ──────────────────────────────────────────────────────────────────

def report(params: dict[str, torch.Tensor], P_marginal: float) -> None:
    """Print a human-readable summary of the trained HMM parameters."""
    initial = torch.softmax(params["initial_logits"], dim=-1)
    transition = torch.softmax(params["transition_logits"], dim=-1)
    offsets = params["emission_logit_offsets"]

    S = initial.shape[0]
    base_logit = math.log(P_marginal / (1.0 - P_marginal + 1e-15) + 1e-15)
    per_state_error_rate = torch.sigmoid(base_logit + offsets)

    print(f"\n=== Read-level HMM  (S={S}, P_marginal={P_marginal:.4f}) ===")

    print("\n--- Initial state probabilities ---")
    for s in range(S):
        print(f"  State {s}: {initial[s].item():.4f}")

    print("\n--- Transition matrix (row = from-state) ---")
    header = "       " + "".join(f"  S{j:<5d}" for j in range(S))
    print(header)
    for i in range(S):
        row = f"  S{i}   " + "".join(f"  {transition[i, j].item():.4f}" for j in range(S))
        print(row)

    print("\n--- Per-state error rate ---")
    for s in range(S):
        rate = per_state_error_rate[s].item()
        offset = offsets[s].item()
        print(f"  State {s}: error_rate = {rate:.6f}  (offset = {offset:+.3f})")


# ─── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a read-level HMM from skiver dump --windows output.",
    )
    parser.add_argument(
        "prefixes",
        nargs="+",
        help="One or more skiver dump output prefixes (the -o value used with dump).",
    )
    parser.add_argument(
        "-o", "--output",
        default="read_level_hmm.pt",
        help="Path to save the trained model (default: read_level_hmm.pt).",
    )
    parser.add_argument(
        "-s", "--states",
        type=int,
        default=DEFAULT_NUM_STATES,
        help=f"Number of latent states (default: {DEFAULT_NUM_STATES}).",
    )
    parser.add_argument(
        "--marginal-rate",
        type=float,
        default=None,
        help=(
            "Override the background error rate P_marginal. "
            "If omitted, estimated from the data as fraction of is_error windows."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Adam learning rate (default: {DEFAULT_LR}).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help=f"Number of SVI training steps (default: {DEFAULT_NUM_STEPS}).",
    )
    parser.add_argument(
        "--reads-cap",
        type=int,
        default=None,
        metavar="N",
        help="Limit training to the first N reads (memory control).",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help=(
            "Include windows from outlier keys (passes_filter=false). "
            "Note: windows.bin does not carry a filter flag; this option is "
            "reserved for future joint-training mode and has no effect here."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args(argv)

    paths = [Path(p) for p in args.prefixes]
    reads, meta = load_windows_bin(paths, reads_cap=args.reads_cap)

    if not reads:
        logger.error("No reads loaded; check input paths.")
        sys.exit(1)

    P_marginal = args.marginal_rate if args.marginal_rate is not None else meta["P_marginal"]
    if not (0.0 < P_marginal < 1.0):
        logger.error("P_marginal must be in (0, 1); got %f", P_marginal)
        sys.exit(1)

    params = train(
        reads,
        P_marginal,
        num_states=args.states,
        lr=args.lr,
        num_steps=args.steps,
    )

    output_path = Path(args.output)
    artifact = {
        "params": params,
        "P_marginal": P_marginal,
        "k": meta.get("k"),
        "v": meta.get("v"),
        "num_states": args.states,
    }
    torch.save(artifact, output_path)
    logger.info("Saved model to %s", output_path)

    report(params, P_marginal)


if __name__ == "__main__":
    main()
