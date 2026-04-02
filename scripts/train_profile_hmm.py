#!/usr/bin/env python3
"""Train a profile HMM error model from skiver dump outputs.

The profile HMM models context-dependent sequencing errors along a (k,v)-mer
value window.  Hidden states capture latent error-propensity regimes.
Emissions are factored into error type (10 categories, conditioned on
dinucleotide context and position) and Phred quality (8 bins, conditioned on
error class and position).

Usage:
    python scripts/train_profile_hmm.py \
        ../skiver_run/mimicc_example/250700000051_25Nov5669-DL133_S133_L001_R1 \
        -o profile_hmm.pt

    # Multiple prefixes:
    python scripts/train_profile_hmm.py \
        ../skiver_run/mimicc_example/250700000051_25Nov5669-DL133_S133_L001_R1 \
        ../skiver_run/mimicc_example/250700000051_25Nov5669-DL133_S133_L001_R2 \
        -o profile_hmm.pt --states 4 --steps 2000
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from lib.data_loading import (
    build_tensors,
    load_multiple,
    stratified_subsample,
)
from lib.profile_hmm import (
    DEFAULT_NUM_STATES,
    marginal_error_rate,
    pack_observations,
    stationary_distribution,
    train,
)

logger = logging.getLogger(__name__)


# ─── Reporting ────────────────────────────────────────────────────────────────

def report(params: dict[str, torch.Tensor]) -> None:
    """Print a human-readable summary of the trained profile HMM."""
    num_states = int(params["_num_states"].item())

    # Initial state distribution.
    pi = stationary_distribution(params)
    print("\n=== Initial state probabilities ===")
    for s in range(num_states):
        print(f"  State {s}: {pi[s]:.4f}")

    # Transition matrix.
    trans = torch.softmax(params["transition_logits"], dim=-1)  # [S, S]
    avg_trans = trans
    print("\n=== Transition matrix ===")
    header = "      " + "".join(f"  S{j:<5d}" for j in range(num_states))
    print(header)
    for i in range(num_states):
        row = f"  S{i}  " + "".join(
            f"  {avg_trans[i, j]:.4f}" for j in range(num_states)
        )
        print(row)

    # Per-state error rate.
    err_rates = marginal_error_rate(params)
    print("\n=== Per-state error rate ===")
    for s in range(num_states):
        print(f"  State {s}: {err_rates[s]:.6f}")

    # Weighted error rate.
    weighted = (pi * err_rates).sum()
    print(f"\n  Weighted average error rate: {weighted:.6f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a profile HMM error model from skiver dump outputs.",
    )
    parser.add_argument(
        "prefixes",
        nargs="+",
        help="One or more skiver dump output prefixes (the -o value).",
    )
    parser.add_argument(
        "-o", "--output",
        default="profile_hmm.pt",
        help="Path to save the trained model (default: profile_hmm.pt).",
    )
    parser.add_argument(
        "-s", "--states",
        type=int,
        default=DEFAULT_NUM_STATES,
        help=f"Number of hidden HMM states (default: {DEFAULT_NUM_STATES}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate (default: 0.005).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of SVI training steps (default: 2000).",
    )
    parser.add_argument(
        "--subsample-ratio",
        type=float,
        default=50.0,
        help="Max ratio of error-free to error-containing sequences (default: 50).",
    )
    parser.add_argument(
        "--no-subsample",
        action="store_true",
        help="Disable stratified subsampling.",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Include observations from keys that failed the outlier filter.",
    )
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=10.0,
        help="Gradient clip norm (default: 10.0).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args(argv)

    # Load data.
    sequences = load_multiple(
        args.prefixes,
        passes_filter_only=not args.include_outliers,
    )
    if not sequences:
        logger.error("No observation sequences loaded. Check input prefixes.")
        sys.exit(1)

    # Stratified subsampling.
    if not args.no_subsample:
        sequences = stratified_subsample(
            sequences, max_no_error_ratio=args.subsample_ratio,
        )

    # Build tensors.
    batch = build_tensors(sequences)
    packed_obs = pack_observations(batch.error_type, batch.phred_bin)

    logger.info(
        "Training data: %d sequences, v=%d, states=%d",
        packed_obs.shape[0], batch.v, args.states,
    )

    # Train.
    params = train(
        packed_obs,
        batch.context_idx,
        num_states=args.states,
        v=batch.v,
        lr=args.lr,
        num_steps=args.steps,
        clip_norm=args.clip_norm,
    )

    # Save.
    output_path = Path(args.output)
    torch.save(params, output_path)
    logger.info("Saved model to %s", output_path)

    # Report.
    report(params)


if __name__ == "__main__":
    main()
