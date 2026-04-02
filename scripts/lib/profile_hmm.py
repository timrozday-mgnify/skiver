"""Profile HMM error model with factored, context-dependent emissions.

The model uses S latent quality-regime states with emissions factored into:
  - Error type: P(error_type | state, prev_base, true_base, position)
  - Phred quality: P(phred_bin | error_class, state, position)

Designed for use with Pyro's DiscreteHMM and trained via SVI.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from .encoding import (
    ERROR_CLASS_MATCH,
    ERROR_TYPE_MATCH,
    ERROR_TYPE_TO_CLASS_TENSOR,
    NUM_BASES,
    NUM_ERROR_CLASSES,
    NUM_ERROR_TYPES,
    NUM_PHRED_BINS,
)

# ─── Constants ────────────────────────────────────────────────────────────────

NUM_CONTEXTS = NUM_BASES * NUM_BASES  # 16 dinucleotide contexts
DEFAULT_NUM_STATES = 4


# ─── Custom emission distribution ────────────────────────────────────────────

class ProfileEmission(dist.TorchDistribution):
    """Factored emission: P(obs | state) = P(error_type | ...) * P(phred | ...).

    This distribution is used inside ``DiscreteHMM``. At each time step, the
    HMM marginalises over states by evaluating ``log_prob`` for each state.
    The observation is a packed integer: ``error_type * NUM_PHRED_BINS + phred_bin``.

    The logits are pre-gathered for the batch so that context dependence is
    already baked in. Shapes after gathering:

    - ``error_logits``: ``[N, T, S, NUM_ERROR_TYPES]``
    - ``phred_logits``: ``[N, T, S, NUM_ERROR_CLASSES, NUM_PHRED_BINS]``

    where N = batch size, T = sequence length, S = num states.
    """

    arg_constraints: dict[str, dist.constraints.Constraint] = {}
    support = dist.constraints.nonnegative_integer

    def __init__(
        self,
        error_logits: torch.Tensor,
        phred_logits: torch.Tensor,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self._error_logits = error_logits
        self._phred_logits = phred_logits
        # batch_shape = [N, T, S], event_shape = []
        batch_shape = error_logits.shape[:-1]
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log P(obs | state) for each state.

        Args:
            value: Packed observation indices. DiscreteHMM passes
                ``value.unsqueeze(-1)`` so shape is ``[N, T, 1]``.
                Encoding: ``error_type * NUM_PHRED_BINS + phred_bin``.

        Returns:
            Log probabilities, shape ``[N, T, S]``.
        """
        # DiscreteHMM calls log_prob(value.unsqueeze(-1)) → [N, T, 1].
        # Squeeze trailing size-1 dims to get [N, T].
        while value.dim() > 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)

        n, t = value.shape
        s = self._error_logits.shape[2]

        # Unpack observation.
        error_type = value.div(NUM_PHRED_BINS, rounding_mode="floor").long()
        phred_bin = (value % NUM_PHRED_BINS).long()

        # Error type log-prob: [N, T, S, 10] → gather at error_type → [N, T, S]
        error_log_probs = F.log_softmax(self._error_logits, dim=-1)
        et_idx = error_type.unsqueeze(-1).unsqueeze(-1).expand(n, t, s, 1)
        lp_error = error_log_probs.gather(-1, et_idx).squeeze(-1)

        # Error class for phred conditioning.
        error_class = ERROR_TYPE_TO_CLASS_TENSOR[error_type]  # [N, T]

        # Phred log-prob: [N, T, S, 3, 8] → gather class → [N, T, S, 8]
        #                                  → gather phred → [N, T, S]
        phred_log_probs_full = F.log_softmax(self._phred_logits, dim=-1)
        ec_idx = error_class.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            n, t, s, 1, NUM_PHRED_BINS,
        )
        phred_by_class = phred_log_probs_full.gather(-2, ec_idx).squeeze(-2)

        pb_idx = phred_bin.unsqueeze(-1).unsqueeze(-1).expand(n, t, s, 1)
        lp_phred = phred_by_class.gather(-1, pb_idx).squeeze(-1)

        return lp_error + lp_phred


# ─── Parameter initialization ────────────────────────────────────────────────

def _init_error_type_logits(num_states: int, v: int) -> torch.Tensor:
    """Initialise error-type logits to strongly favour match.

    Shape: ``[S, NUM_CONTEXTS, T, NUM_ERROR_TYPES]``.
    Match gets logit 0; errors get -8 + state_offset for symmetry breaking.
    """
    logits = torch.full(
        (num_states, NUM_CONTEXTS, v, NUM_ERROR_TYPES), -8.0,
    )
    for s in range(num_states):
        logits[s, :, :, ERROR_TYPE_MATCH] = 4.0 + s * 0.5
        # Slightly different error logits per state for symmetry breaking.
        logits[s, :, :, 1:] = -8.0 + s * 0.5
    return logits


def _init_phred_logits(num_states: int, v: int) -> torch.Tensor:
    """Initialise phred logits with a mild quality-increasing prior.

    Shape: ``[S, NUM_ERROR_CLASSES, T, NUM_PHRED_BINS]``.
    For matches, higher phred bins are slightly favoured.
    """
    logits = torch.zeros(num_states, NUM_ERROR_CLASSES, v, NUM_PHRED_BINS)
    for pb in range(NUM_PHRED_BINS):
        # Matches: higher phred → higher logit.
        logits[:, ERROR_CLASS_MATCH, :, pb] = float(pb) * 0.5
    return logits


# ─── Gather functions ─────────────────────────────────────────────────────────

def gather_error_logits(
    error_type_logits: torch.Tensor,
    context_idx: torch.Tensor,
) -> torch.Tensor:
    """Pre-gather error-type logits for the batch's context indices.

    Args:
        error_type_logits: Full parameter, shape ``[S, 16, T, 10]``.
        context_idx: Per-observation context indices, shape ``[N, T]``.

    Returns:
        Gathered logits, shape ``[N, T, S, 10]``.
    """
    s, _c, t, e = error_type_logits.shape
    n = context_idx.shape[0]

    # Rearrange to [16, S, T, 10] for easier gathering.
    param = error_type_logits.permute(1, 0, 2, 3)  # [16, S, T, 10]

    # Expand context_idx to [N, T, S, 10] for gathering along dim=0.
    ci = context_idx.unsqueeze(-1).unsqueeze(-1)  # [N, T, 1, 1]
    ci = ci.expand(n, t, s, e)

    # Flatten the context dim: reshape param to [16, S*T*10]
    # and context to [N, S*T*10], then gather.
    # Simpler: index_select per position would be slow. Use advanced indexing.
    # param[context_idx[n, t], :, t, :] → result[n, t, :, :]
    # Use gather on flattened representation.
    result = torch.zeros(n, t, s, e)
    for ti in range(t):
        # context_idx[:, ti] is [N], index into param[:, :, ti, :] which is [16, S, 10]
        ci_t = context_idx[:, ti]  # [N]
        param_t = param[:, :, ti, :]  # [16, S, 10]
        result[:, ti, :, :] = param_t[ci_t]  # [N, S, 10]

    return result


def gather_phred_logits(
    phred_logits: torch.Tensor,
) -> torch.Tensor:
    """Expand phred logits for the batch.

    Phred logits are not context-dependent, only (state, error_class, position).
    We just rearrange for the emission distribution.

    Args:
        phred_logits: Full parameter, shape ``[S, 3, T, 8]``.

    Returns:
        Expanded logits, shape ``[1, T, S, 3, 8]`` (broadcastable over N).
    """
    # [S, 3, T, 8] → [T, S, 3, 8] → [1, T, S, 3, 8]
    return phred_logits.permute(2, 0, 1, 3).unsqueeze(0)


# ─── Pyro model ───────────────────────────────────────────────────────────────

def profile_hmm_model(
    packed_obs: torch.Tensor,
    context_idx: torch.Tensor,
    num_states: int,
    v: int,
) -> None:
    """Pyro model: factored profile HMM with context-dependent emissions.

    Args:
        packed_obs: Packed observation tensor ``[N, T]``, encoding
            ``error_type * NUM_PHRED_BINS + phred_bin``.
        context_idx: Dinucleotide context indices ``[N, T]``.
        num_states: Number of hidden states S.
        v: Value window length (= T).
    """
    s = num_states

    initial_logits = pyro.param("initial_logits", torch.zeros(s))

    # Position-independent transitions: [S, S].
    # (Pyro's DiscreteHMM does not support time-varying transitions;
    # position dependence is captured in the emission distributions.)
    transition_logits = pyro.param(
        "transition_logits",
        torch.eye(s) * 3.0,  # sticky initialisation
    )

    # Context-dependent error-type logits: [S, 16, T, 10].
    error_type_logits = pyro.param(
        "error_type_logits",
        _init_error_type_logits(s, v),
    )

    # Phred logits: [S, 3, T, 8].
    phred_logits = pyro.param(
        "phred_logits",
        _init_phred_logits(s, v),
    )

    # Pre-gather for this batch.
    gathered_error = gather_error_logits(error_type_logits, context_idx)
    gathered_phred = gather_phred_logits(phred_logits).expand(
        packed_obs.shape[0], -1, -1, -1, -1,
    )

    obs_dist = ProfileEmission(gathered_error, gathered_phred)

    with pyro.plate("sequences", packed_obs.shape[0]):
        hmm = dist.DiscreteHMM(
            initial_logits=initial_logits,
            transition_logits=transition_logits,
            observation_dist=obs_dist,
            duration=v,
        )
        pyro.sample("obs", hmm, obs=packed_obs)


def profile_hmm_guide(
    packed_obs: torch.Tensor,
    context_idx: torch.Tensor,
    num_states: int,
    v: int,
) -> None:
    """Empty guide — all parameters are ``pyro.param``, no latent samples."""


# ─── Training ─────────────────────────────────────────────────────────────────

def pack_observations(
    error_type: torch.Tensor,
    phred_bin: torch.Tensor,
) -> torch.Tensor:
    """Pack error_type and phred_bin into a single integer for the emission.

    Args:
        error_type: Error type indices, shape ``[N, T]``.
        phred_bin: Phred bin indices, shape ``[N, T]``.

    Returns:
        Packed tensor, shape ``[N, T]``.
    """
    return error_type * NUM_PHRED_BINS + phred_bin


def train(
    packed_obs: torch.Tensor,
    context_idx: torch.Tensor,
    num_states: int = DEFAULT_NUM_STATES,
    v: int | None = None,
    lr: float = 0.005,
    num_steps: int = 2000,
    clip_norm: float = 10.0,
    log_every: int = 50,
) -> dict[str, torch.Tensor]:
    """Train the profile HMM via MAP estimation using SVI.

    Args:
        packed_obs: Packed observations ``[N, T]``.
        context_idx: Context indices ``[N, T]``.
        num_states: Number of hidden states.
        v: Value window length. If None, inferred from packed_obs.shape[1].
        lr: Learning rate.
        num_steps: Number of SVI steps.
        clip_norm: Gradient clip norm.
        log_every: Log loss every this many steps.

    Returns:
        Dictionary of learned parameter tensors.
    """
    if v is None:
        v = packed_obs.shape[1]

    pyro.clear_param_store()

    optimizer = pyro.optim.ClippedAdam({"lr": lr, "clip_norm": clip_norm})
    svi = pyro.infer.SVI(
        profile_hmm_model,
        profile_hmm_guide,
        optimizer,
        loss=pyro.infer.Trace_ELBO(),
    )

    losses = []
    for step in range(num_steps):
        loss = svi.step(packed_obs, context_idx, num_states, v)
        losses.append(loss)
        if step % log_every == 0 or step == num_steps - 1:
            avg_loss = loss / packed_obs.shape[0]
            logging.info(
                "Step %4d / %d  loss = %.4f  (per-seq = %.4f)",
                step, num_steps, loss, avg_loss,
            )

    params = {
        name: value.detach().clone()
        for name, value in pyro.get_param_store().items()
    }
    params["_losses"] = torch.tensor(losses)
    params["_num_states"] = torch.tensor(num_states)
    params["_v"] = torch.tensor(v)
    return params


# ─── Analysis utilities ───────────────────────────────────────────────────────

def marginal_error_rate(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute per-state error rate marginalised over contexts and positions.

    Args:
        params: Trained parameter dict.

    Returns:
        Tensor of shape ``[S]`` with per-state error rates.
    """
    logits = params["error_type_logits"]  # [S, 16, T, 10]
    probs = F.softmax(logits, dim=-1)
    # Error = everything except match (index 0).
    error_probs = 1.0 - probs[:, :, :, ERROR_TYPE_MATCH]  # [S, 16, T]
    return error_probs.mean(dim=(1, 2))  # [S]


def stationary_distribution(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute approximate stationary distribution from initial logits.

    Args:
        params: Trained parameter dict.

    Returns:
        Tensor of shape ``[S]`` with state probabilities.
    """
    return F.softmax(params["initial_logits"], dim=-1)
