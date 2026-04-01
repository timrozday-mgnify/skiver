#!/usr/bin/env python3
"""Train an HMM error model from skiver dump outputs.

The HMM models the sequencing error process along a (k,v)-mer value window.
Hidden states capture latent error-propensity regimes (e.g. "low-error" vs
"high-error" stretches).  Emissions are the observed (true_base, obs_base,
phred) tuples at each position, encoded as a single integer category.

Usage:
    python scripts/train_hmm_error_model.py \\
        ../skiver_run/mimicc_example/250700000051_25Nov5669-DL133_S133_L001_R1 \\
        -o hmm_error_model.pt

    # Multiple prefixes (e.g. R1 and R2):
    python scripts/train_hmm_error_model.py \\
        ../skiver_run/mimicc_example/250700000051_25Nov5669-DL133_S133_L001_R1 \\
        ../skiver_run/mimicc_example/250700000051_25Nov5669-DL133_S133_L001_R2 \\
        -o hmm_error_model.pt
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────────

BASE_TO_IDX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3, "-": 4}
NUM_BASES = 5  # A, C, G, T, gap
NUM_PHRED_BINS = 8  # phred scores binned: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35+

# Each observation is (true_base, obs_base, phred_bin) encoded as a single int.
# Total categories = NUM_BASES * NUM_BASES * NUM_PHRED_BINS.
NUM_OBS_CATEGORIES = NUM_BASES * NUM_BASES * NUM_PHRED_BINS

DEFAULT_NUM_STATES = 3
DEFAULT_LR = 0.005
DEFAULT_NUM_STEPS = 1000


# ─── Data loading ───────────────────────────────────────────────────────────────

def _bin_phred(phred: int) -> int:
    """Bin a phred score into NUM_PHRED_BINS categories."""
    if phred < 0:
        return 0
    return min(phred // 5, NUM_PHRED_BINS - 1)


def _encode_obs(true_base: str, obs_base: str, phred: int) -> int:
    """Encode a single base observation as a flat category index."""
    tb = BASE_TO_IDX.get(true_base, 4)
    ob = BASE_TO_IDX.get(obs_base, 4)
    pb = _bin_phred(phred)
    return tb * NUM_BASES * NUM_PHRED_BINS + ob * NUM_PHRED_BINS + pb


def load_base_observations(
    prefix: str | Path,
    *,
    passes_filter_only: bool = True,
) -> list[torch.Tensor]:
    """Load base_observations.tsv and return encoded observation sequences.

    Each returned tensor has shape ``[v]`` where *v* is the value length,
    containing integer-encoded observation categories.

    Args:
        prefix: Output prefix used with ``skiver dump -o``.
        passes_filter_only: If true, skip observations from outlier keys.

    Returns:
        List of 1-D long tensors, one per (key, value) occurrence.
    """
    path = Path(f"{prefix}.base_observations.tsv")
    if not path.exists():
        logger.warning("File not found: %s", path)
        return []

    # Group rows by obs_id, preserving order.
    sequences: dict[int, list[int]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if passes_filter_only and row["passes_filter"] != "true":
                continue
            obs_id = int(row["obs_id"])
            encoded = _encode_obs(
                row["true_base"],
                row["obs_base"],
                int(row["phred"]),
            )
            sequences.setdefault(obs_id, []).append(encoded)

    result = [torch.tensor(seq, dtype=torch.long) for seq in sequences.values()]
    logger.info("Loaded %d sequences from %s", len(result), path)
    return result


# ─── HMM model ──────────────────────────────────────────────────────────────────

def _init_emission_logits(num_states: int) -> torch.Tensor:
    """Create emission logits initialised to strongly favour match categories.

    Match categories (true_base == obs_base) get logit 0; mismatch categories
    get logit -10.  This reflects the prior that most bases are correct
    (error rate ~0.1%) and prevents the optimiser from starting in a region
    where every state looks high-error.

    Different states get slightly different mismatch penalties so the optimiser
    can break symmetry and specialise states by error propensity.
    """
    k = NUM_OBS_CATEGORIES
    logits = torch.full((num_states, k), -10.0)
    for cat in range(k):
        pb = cat % NUM_PHRED_BINS
        remainder = cat // NUM_PHRED_BINS
        ob = remainder % NUM_BASES
        tb = remainder // NUM_BASES
        if tb == ob:
            # Match: high logit, modulated by quality bin.
            logits[:, cat] = float(pb)  # higher quality → higher logit
        else:
            # Mismatch: low logit, slightly different per state.
            for s in range(num_states):
                logits[s, cat] = -8.0 + s * 1.0
    return logits


def hmm_model(
    data: torch.Tensor,
    lengths: torch.Tensor,
    num_states: int,
    max_length: int,
) -> None:
    """Pyro model: discrete HMM with Categorical emissions.

    Uses ``DiscreteHMM`` for efficient forward-algorithm computation.

    Args:
        data: Padded observation tensor of shape ``[N, max_length]``.
        lengths: Actual lengths per sequence, shape ``[N]``.
        num_states: Number of hidden states.
        max_length: Maximum sequence length (padding boundary).
    """
    s = num_states
    k = NUM_OBS_CATEGORIES

    # Learnable parameters with appropriate constraints.
    initial_logits = pyro.param(
        "initial_logits", torch.zeros(s),
    )
    transition_logits = pyro.param(
        "transition_logits",
        torch.eye(s) * 3.0,  # sticky initialisation
    )
    emission_logits = pyro.param(
        "emission_logits", _init_emission_logits(s),
    )

    obs_dist = dist.Categorical(logits=emission_logits)

    with pyro.plate("sequences", data.shape[0]):
        hmm = dist.DiscreteHMM(
            initial_logits=initial_logits,
            transition_logits=transition_logits,
            observation_dist=obs_dist,
            duration=max_length,
        )
        pyro.sample("obs", hmm, obs=data)


def hmm_guide(
    data: torch.Tensor,
    lengths: torch.Tensor,
    num_states: int,
    max_length: int,
) -> None:
    """Empty guide — all parameters are ``pyro.param``, no latent samples."""


# ─── Training ───────────────────────────────────────────────────────────────────

def _pad_sequences(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences into a single tensor.

    Returns:
        Tuple of (padded_data [N, max_len], lengths [N]).
    """
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, s in enumerate(sequences):
        padded[i, : s.shape[0]] = s
    return padded, lengths


def train(
    sequences: list[torch.Tensor],
    num_states: int = DEFAULT_NUM_STATES,
    lr: float = DEFAULT_LR,
    num_steps: int = DEFAULT_NUM_STEPS,
) -> dict[str, torch.Tensor]:
    """Train the HMM error model via MAP estimation (Baum-Welch equivalent).

    Args:
        sequences: Encoded observation sequences from ``load_base_observations``.
        num_states: Number of hidden states in the HMM.
        lr: Learning rate for Adam optimiser.
        num_steps: Number of SVI steps.

    Returns:
        Dictionary of learned parameter tensors.
    """
    data, lengths = _pad_sequences(sequences)
    max_length = int(lengths.max().item())

    pyro.clear_param_store()

    svi = SVI(
        hmm_model,
        hmm_guide,
        pyro.optim.Adam({"lr": lr}),
        loss=Trace_ELBO(),
    )

    logger.info(
        "Training HMM: %d sequences, max_length=%d, states=%d, steps=%d",
        len(sequences), max_length, num_states, num_steps,
    )

    for step in range(num_steps):
        loss = svi.step(data, lengths, num_states, max_length)
        if step % 50 == 0 or step == num_steps - 1:
            logger.info("Step %4d / %d  loss = %.4f", step, num_steps, loss)

    params = {
        name: value.detach().clone()
        for name, value in pyro.get_param_store().items()
    }
    return params


# ─── Reporting ──────────────────────────────────────────────────────────────────

def _decode_category(cat: int) -> tuple[str, str, str]:
    """Decode a flat category index back to (true_base, obs_base, phred_bin_label)."""
    idx_to_base = {v: k for k, v in BASE_TO_IDX.items()}
    pb = cat % NUM_PHRED_BINS
    remainder = cat // NUM_PHRED_BINS
    ob = remainder % NUM_BASES
    tb = remainder // NUM_BASES
    lo = pb * 5
    hi = "+" if pb == NUM_PHRED_BINS - 1 else str(lo + 4)
    return idx_to_base[tb], idx_to_base[ob], f"Q{lo}-{hi}"


def report(params: dict[str, torch.Tensor]) -> None:
    """Print a human-readable summary of the trained HMM parameters."""
    initial = torch.softmax(params["initial_logits"], dim=-1)
    transition = torch.softmax(params["transition_logits"], dim=-1)
    emission = torch.softmax(params["emission_logits"], dim=-1)

    num_states = initial.shape[0]

    print("\n=== Initial state probabilities ===")
    for s in range(num_states):
        print(f"  State {s}: {initial[s]:.4f}")

    print("\n=== Transition matrix ===")
    header = "      " + "".join(f"  S{j:<5d}" for j in range(num_states))
    print(header)
    for i in range(num_states):
        row = f"  S{i}  " + "".join(f"  {transition[i, j]:.4f}" for j in range(num_states))
        print(row)

    print("\n=== Top emission probabilities per state ===")
    for s in range(num_states):
        probs = emission[s]
        top_k = min(10, probs.shape[0])
        top_indices = torch.topk(probs, top_k).indices
        print(f"\n  State {s}:")
        for idx in top_indices:
            cat = idx.item()
            prob = probs[cat].item()
            tb, ob, ql = _decode_category(cat)
            match_str = "match" if tb == ob else f"{tb}>{ob}"
            print(f"    {match_str:8s} {ql:8s}  p={prob:.4f}")

    # Summary: per-state error rate
    print("\n=== Per-state error rate ===")
    for s in range(num_states):
        probs = emission[s]
        error_prob = 0.0
        for cat in range(NUM_OBS_CATEGORIES):
            tb, ob, _ = _decode_category(cat)
            if tb != ob:
                error_prob += probs[cat].item()
        print(f"  State {s}: error_rate = {error_prob:.6f}")


# ─── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an HMM sequencing error model from skiver dump outputs.",
    )
    parser.add_argument(
        "prefixes",
        nargs="+",
        help="One or more skiver dump output prefixes (the -o value).",
    )
    parser.add_argument(
        "-o", "--output",
        default="hmm_error_model.pt",
        help="Path to save the trained model parameters (default: hmm_error_model.pt).",
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
        default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR}).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help=f"Number of SVI training steps (default: {DEFAULT_NUM_STEPS}).",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Include observations from keys that failed the outlier filter.",
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

    # Load sequences from all prefixes.
    all_sequences: list[torch.Tensor] = []
    for prefix in args.prefixes:
        seqs = load_base_observations(
            prefix, passes_filter_only=not args.include_outliers,
        )
        all_sequences.extend(seqs)

    if not all_sequences:
        logger.error("No observation sequences loaded. Check input prefixes.")
        sys.exit(1)

    logger.info("Total sequences: %d", len(all_sequences))

    # Train.
    params = train(
        all_sequences,
        num_states=args.states,
        lr=args.lr,
        num_steps=args.steps,
    )

    # Save.
    output_path = Path(args.output)
    torch.save(params, output_path)
    logger.info("Saved model to %s", output_path)

    # Report.
    report(params)


if __name__ == "__main__":
    main()
