"""Vectorised generative application of a trained context error model.

This is the single implementation of "apply a Skiver context error model to a
sequence".  It loads a trained ``.pt`` artifact (or builds a model from a
component string plus explicit parameter values, or a named platform preset),
then samples observed reads from reference sequences.

The hot loop is **numpy-vectorised per read**: the error type for every
reference position is drawn in one batched categorical sample rather than one
``rng.choice`` call per base, which is the dominant cost of the old
position-by-position simulator.  Insertions (which resample at the same
reference position) are handled by a short fallback only for the reads that
actually draw one.

Context convention matches training: the context for a position is taken from
the *true reference*, not the accumulating mutated read.  Missing leading
context near the sequence start is A-padded.

Error type categories (see :mod:`lib.encoding`)::

    0      = match
    1..4   = sub_to_A / C / G / T
    5..8   = ins_A / C / G / T
    9      = deletion
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from lib.context_error_models import parse_model_components
from lib.encoding import NUM_ERROR_TYPES

# ── Constants ───────────────────────────────────────────────────────────────────

_BASES: Final[str] = "ACGT"
_N_BASES: Final[int] = 4
_TABLE_L_LIMIT: Final[int] = 10   # precompute the full 4^L table only when L ≤ this
_MAX_INS_RUN_DEFAULT: Final[int] = 10

_ERR_MATCH: Final[int] = 0
_ERR_SUB_START: Final[int] = 1    # 1–4
_ERR_INS_START: Final[int] = 5    # 5–8
_ERR_DEL: Final[int] = 9

# Lookup tables for fast base <-> index <-> ASCII conversion.
_BASE_ASCII: Final[np.ndarray] = np.frombuffer(_BASES.encode("ascii"), dtype=np.uint8)
_CHAR_TO_IDX: Final[np.ndarray] = np.zeros(256, dtype=np.uint8)
for _i, _c in enumerate(_BASES):
    _CHAR_TO_IDX[ord(_c)] = _i
    _CHAR_TO_IDX[ord(_c.lower())] = _i


@dataclass(frozen=True)
class ResultRecord:
    """One simulated read.

    Attributes:
        name: Read name (carried through from the input record).
        sequence: Observed base string (ACGT).
        quality: Phred+33 quality string, or ``None`` when quality is disabled.
        cigar: Standard CIGAR string against the reference (e.g. ``30M1I19M``).
    """

    name: str
    sequence: str
    quality: str | None
    cigar: str


# ── Parameter helpers ───────────────────────────────────────────────────────────


def _to_np(value: Any) -> np.ndarray:
    """Convert a torch tensor / array-like to a float32 numpy array."""
    detach = getattr(value, "detach", None)
    if detach is not None:                       # torch.Tensor
        value = detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _build_additive_table(
    intercept: np.ndarray,
    bl_centered: np.ndarray,
    context_length: int,
) -> np.ndarray:
    """Enumerate all 4^L contexts and return composed logits ``[4^L, 10]``."""
    n_ctx = _N_BASES ** context_length
    indices = np.arange(n_ctx, dtype=np.int64)
    table = np.broadcast_to(intercept, (n_ctx, NUM_ERROR_TYPES)).astype(np.float32).copy()
    for pos in range(context_length):
        div = _N_BASES ** (context_length - pos - 1)
        base_idx = (indices // div) % _N_BASES
        table += bl_centered[pos, base_idx, :]
    return table.astype(np.float32)


def _reject_unsupported(params: dict[str, Any]) -> None:
    """Raise for parameters the generator cannot apply generatively.

    Phred-lag and homopolymer covariates are not generatively well defined
    (they depend on the read's own emitted quality / run structure), and the
    bundled platform presets do not use them.  Fail loudly rather than silently
    ignore a covariate the user trained.
    """
    if "phred_weights" in params:
        raise NotImplementedError(
            "PhredContext covariate is not supported in generative mode "
            "(quality is an output, not an input)."
        )
    if "position_weights" in params:
        raise NotImplementedError(
            "Position covariate is not yet supported in generative mode."
        )
    if "run_slopes" in params:
        raise NotImplementedError(
            "The legacy scalar-run homopolymer model is not supported in "
            "generative mode; retrain with the additive Homopolymer component."
        )
    # log_phi_unconstrained (fragment overdispersion) only affects the training
    # likelihood, not the marginal per-context distribution we sample from, so
    # it is safely ignored here.


# ── Model ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _HMMLayer:
    """Read-level latent-state error-rate modulation (see lib.hmm_latent_state).

    ``init_probs`` and ``transition`` are row-stochastic; ``scales[s]`` multiplies
    the local error probability when the read is in state ``s`` (``scales[0]==1``).
    """

    init_probs: np.ndarray      # [S]
    transition: np.ndarray      # [S, S]
    scales: np.ndarray          # [S]

    @classmethod
    def from_artifact(cls, block: dict[str, Any]) -> "_HMMLayer":
        params = block["params"]
        init_logits = _to_np(params["hmm_init_logits"]).reshape(-1).astype(np.float64)
        trans_logits = _to_np(params["hmm_transition_logits"]).astype(np.float64)
        scale_free = _to_np(params["hmm_scale_unconstrained"]).reshape(-1).astype(np.float64)
        init = np.exp(init_logits - init_logits.max())
        init /= init.sum()
        trans = np.exp(trans_logits - trans_logits.max(axis=1, keepdims=True))
        trans /= trans.sum(axis=1, keepdims=True)
        softplus = np.log1p(np.exp(-np.abs(scale_free))) + np.maximum(scale_free, 0.0)
        scales = np.concatenate([[1.0], softplus + 1e-3])
        return cls(init.astype(np.float64), trans.astype(np.float64), scales.astype(np.float64))

    def sample_states(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample a length-``n`` state trajectory (per-base, contiguous gaps)."""
        states = np.empty(n, dtype=np.int64)
        cdf_init = np.cumsum(self.init_probs)
        cdf_trans = np.cumsum(self.transition, axis=1)
        states[0] = int(np.searchsorted(cdf_init, rng.random()))
        for i in range(1, n):
            states[i] = int(np.searchsorted(cdf_trans[states[i - 1]], rng.random()))
        return states


@dataclass(frozen=True)
class ErrorModel:
    """A loaded context error model ready for simulation.

    Either ``logit_table`` is populated (precomputed ``[4^L, 10]``) or
    ``intercept`` + ``base_logits`` are populated for on-the-fly additive
    composition when ``4^L`` is too large to precompute.
    """

    context_length: int
    logit_table: np.ndarray | None
    intercept: np.ndarray | None
    base_logits: np.ndarray | None
    strand_weights: np.ndarray | None
    name: str
    homopolymer_weights: np.ndarray | None = None
    hmm: "_HMMLayer | None" = None

    # ---- constructors -------------------------------------------------------

    @classmethod
    def _from_params(
        cls,
        params: dict[str, Any],
        context_length: int,
        *,
        calibration_offset: float,
        name: str,
    ) -> "ErrorModel":
        _reject_unsupported(params)

        strand_weights: np.ndarray | None = None
        if "strand_weights" in params:
            strand_weights = _to_np(params["strand_weights"]).reshape(-1)
            if strand_weights.shape[0] != NUM_ERROR_TYPES:
                strand_weights = strand_weights[:NUM_ERROR_TYPES]

        homopolymer_weights: np.ndarray | None = None
        if "homopolymer_weights" in params:
            homopolymer_weights = _to_np(params["homopolymer_weights"]).reshape(-1)
            if homopolymer_weights.shape[0] != NUM_ERROR_TYPES:
                homopolymer_weights = homopolymer_weights[:NUM_ERROR_TYPES]

        if "logits" in params:
            table = _to_np(params["logits"]).reshape(-1, NUM_ERROR_TYPES).copy()
            if calibration_offset != 0.0:
                table[:, 1:] += calibration_offset
            return cls(context_length, table, None, None, strand_weights, name, homopolymer_weights)

        if "intercept_logits" in params:
            intercept = _to_np(params["intercept_logits"]).reshape(NUM_ERROR_TYPES).copy()
            bl = _to_np(params["base_logits"])
            bl_centered = (bl - bl.mean(axis=1, keepdims=True)).astype(np.float32)
            if context_length <= _TABLE_L_LIMIT:
                table = _build_additive_table(intercept, bl_centered, context_length)
                if calibration_offset != 0.0:
                    table[:, 1:] += calibration_offset
                return cls(context_length, table, None, None, strand_weights, name, homopolymer_weights)
            if calibration_offset != 0.0:
                intercept = intercept.copy()
                intercept[1:] += calibration_offset
            return cls(context_length, None, intercept, bl_centered, strand_weights, name, homopolymer_weights)

        raise ValueError(f"Unrecognised parameter keys: {sorted(params)}")

    @classmethod
    def load(cls, path: Path, *, use_vi: bool = False) -> "ErrorModel":
        """Load a trained ``.pt`` artifact.

        Args:
            path: Trained model checkpoint from
                ``scripts/train_context_error_models.py``.
            use_vi: Use the variational-inference posterior mean instead of the
                maximum-likelihood point estimate.
        """
        import torch

        try:
            artifact = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(path, map_location="cpu")

        context_length = int(artifact.get("context_length", 1))
        inf_key = "variational_inference" if use_vi else "maximum_likelihood"
        param_key = "params_mean" if use_vi else "params"
        params = artifact[inf_key][param_key]

        offset_key = "calibration_offset_vi" if use_vi else "calibration_offset_mle"
        cal = artifact.get("calibration") or {}
        calibration_offset = float(cal.get(offset_key, 0.0))

        model = cls._from_params(
            params,
            context_length,
            calibration_offset=calibration_offset,
            name=str(artifact.get("model_id", Path(path).stem)),
        )
        hmm_block = artifact.get("hmm_latent_state")
        if hmm_block:
            import dataclasses

            model = dataclasses.replace(model, hmm=_HMMLayer.from_artifact(hmm_block))
        return model

    @classmethod
    def from_spec(
        cls,
        components: str,
        params: dict[str, Any],
        *,
        calibration_offset: float = 0.0,
        name: str | None = None,
    ) -> "ErrorModel":
        """Build a model from a component string plus explicit parameter values.

        Args:
            components: Composable component string, e.g.
                ``"AdditiveContext(7)+Strand"`` (see
                :func:`lib.context_error_models.parse_model_components`).
            params: Mapping of parameter name to tensor / array (``logits`` or
                ``intercept_logits`` + ``base_logits``, optionally
                ``strand_weights``).
            calibration_offset: Scalar added to every non-match logit.
            name: Optional model name.
        """
        parsed = parse_model_components(components)
        return cls._from_params(
            params,
            parsed.context_length,
            calibration_offset=calibration_offset,
            name=name or components,
        )

    @classmethod
    def preset(cls, name: str, *, use_vi: bool = False) -> "ErrorModel":
        """Load a bundled platform preset by name (see :mod:`lib.presets`)."""
        from lib.presets import resolve_preset

        return cls.load(resolve_preset(name), use_vi=use_vi)

    # ---- per-read logits ----------------------------------------------------

    def _logits_for_reference(self, ref_idx: np.ndarray, is_forward: bool) -> np.ndarray:
        """Return per-position logits ``[L, 10]`` for a reference index array."""
        k = self.context_length
        n = ref_idx.shape[0]
        padded = np.empty(n + k, dtype=np.int64)
        padded[:k] = 0                      # A-padding for leading context
        padded[k:] = ref_idx
        windows = np.lib.stride_tricks.sliding_window_view(padded, k)[:n]  # [n, k]

        if self.logit_table is not None:
            weights = (_N_BASES ** np.arange(k - 1, -1, -1)).astype(np.int64)
            ctx_idx = windows @ weights
            logits = self.logit_table[ctx_idx].astype(np.float32, copy=True)
        else:
            assert self.intercept is not None and self.base_logits is not None
            logits = np.broadcast_to(self.intercept, (n, NUM_ERROR_TYPES)).astype(np.float32).copy()
            for j in range(k):
                logits += self.base_logits[j, windows[:, j], :]

        if self.homopolymer_weights is not None:
            logits += self._homopolymer_feature(ref_idx)[:, None] * self.homopolymer_weights
        if self.strand_weights is not None and is_forward:
            logits += self.strand_weights
        return logits

    @staticmethod
    def _homopolymer_feature(ref_idx: np.ndarray) -> np.ndarray:
        """Return ``log1p(run_length)`` of the run ending at the *previous* base.

        Matches the training feature, which uses the homopolymer run of the
        consensus history up to (but excluding) the current base.  Position 0 has
        no preceding base, so its feature is 0.
        """
        n = ref_idx.shape[0]
        feat = np.zeros(n, dtype=np.float32)
        if n < 2:
            return feat
        same = ref_idx[1:] == ref_idx[:-1]            # same[i] == (ref[i+1]==ref[i])
        run = np.ones(n, dtype=np.float32)            # run length ending at ref[i]
        for i in range(1, n):
            run[i] = run[i - 1] + 1.0 if same[i - 1] else 1.0
        feat[1:] = np.log1p(run[:-1])                 # run ending at previous base
        return feat


# ── Single-context probabilities (diagnostics / benchmarks) ─────────────────────


def _logits_for_context_index(model: ErrorModel, context_index: int) -> np.ndarray:
    """Return logits ``[10]`` for a fully specified big-endian base-4 context."""
    if model.logit_table is not None:
        return model.logit_table[context_index].astype(np.float32)
    assert model.intercept is not None and model.base_logits is not None
    k = model.context_length
    bases = np.empty(k, dtype=np.int64)
    idx = context_index
    for j in range(k - 1, -1, -1):
        bases[j] = idx % _N_BASES
        idx //= _N_BASES
    logits = model.intercept.astype(np.float32).copy()
    for j in range(k):
        logits = logits + model.base_logits[j, bases[j], :]
    return logits


def probabilities_for_context(
    model: ErrorModel,
    context_index: int,
    true_base_idx: int | None = None,
) -> np.ndarray:
    """Return the error-type probability vector ``[10]`` for one context.

    When ``true_base_idx`` is given, the impossible substitution-to-self
    category is masked before renormalisation.  The strand covariate is not
    applied here (this is the context-marginal distribution used by diagnostics
    and recovery benchmarks).
    """
    logits = _logits_for_context_index(model, context_index)
    shifted = logits - logits.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    if true_base_idx is not None:
        probs = probs.copy()
        probs[_ERR_SUB_START + true_base_idx] = 0.0
        total = probs.sum()
        if total <= 0.0:
            raise ValueError("All probabilities were masked for a context")
        probs /= total
    return probs


# ── Sampling ────────────────────────────────────────────────────────────────────


def _masked_probs(logits: np.ndarray, ref_idx: np.ndarray) -> np.ndarray:
    """Return softmax probabilities with the self-substitution masked out."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs[np.arange(ref_idx.shape[0]), _ERR_SUB_START + ref_idx] = 0.0
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _sample_rows(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorised categorical draw: one sample per row of ``probs``."""
    cdf = np.cumsum(probs, axis=1)
    u = rng.random(probs.shape[0])
    return (cdf > u[:, None]).argmax(axis=1)


def _apply_hmm_scaling(
    probs: np.ndarray, hmm: "_HMMLayer", rng: np.random.Generator
) -> np.ndarray:
    """Scale each position's error mass by its sampled latent-state factor.

    A per-base latent-state trajectory is drawn; in state ``s`` the local error
    probability ``1 - P(match)`` is multiplied by ``scales[s]`` (clamped to a
    valid probability) and the error categories are renormalised to the new mass.
    """
    n = probs.shape[0]
    states = hmm.sample_states(n, rng)
    sc = hmm.scales[states]                                  # [n]
    p_err = 1.0 - probs[:, _ERR_MATCH]
    new_err = np.clip(sc * p_err, 0.0, 1.0 - 1e-6)
    ratio = np.divide(new_err, p_err, out=np.ones_like(p_err), where=p_err > 0)
    out = probs.copy()
    out[:, 1:] *= ratio[:, None]
    out[:, _ERR_MATCH] = 1.0 - new_err
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


def _phred_from_probs(probs: np.ndarray) -> np.ndarray:
    """Context-derived integer Phred: ``-10·log10(1 − P(match))`` clamped [0,60]."""
    p_err = np.clip(1.0 - probs[:, _ERR_MATCH], 1e-6, 1.0)
    return np.clip(np.floor(-10.0 * np.log10(p_err)), 0, 60).astype(np.int64)


def _sample_phred_calib(
    error_types: np.ndarray,
    calib: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample integer Phred per emitted base from ``P(Q | error_type)``.

    Rows with no empirical mass yield ``-1`` (caller substitutes a default).
    """
    rows = calib[error_types]                        # [m, 61]
    totals = rows.sum(axis=1)
    safe = np.where(totals[:, None] > 0, rows, 1.0)
    safe = safe / safe.sum(axis=1, keepdims=True)
    cdf = np.cumsum(safe, axis=1)
    u = rng.random(error_types.shape[0])
    q = (cdf > u[:, None]).argmax(axis=1)
    return np.where(totals > 0, q, -1)


def _cigar_from_ops(ops: np.ndarray) -> str:
    """Run-length encode an array of CIGAR op characters (uint8 ASCII)."""
    if ops.shape[0] == 0:
        return ""
    change = np.empty(ops.shape[0], dtype=bool)
    change[0] = True
    change[1:] = ops[1:] != ops[:-1]
    starts = np.flatnonzero(change)
    lengths = np.diff(np.append(starts, ops.shape[0]))
    return "".join(f"{int(l)}{chr(int(ops[s]))}" for l, s in zip(lengths, starts))


_OP_M: Final[int] = ord("M")
_OP_I: Final[int] = ord("I")
_OP_D: Final[int] = ord("D")


def _apply_fast(
    ref_idx: np.ndarray,
    probs: np.ndarray,
    error_types: np.ndarray,
    rng: np.random.Generator,
    *,
    emit_quality: bool,
    phred_calib: np.ndarray | None,
) -> tuple[str, str | None, str]:
    """Assemble a read when no position drew an insertion (fully vectorised)."""
    keep = error_types != _ERR_DEL
    # Observed base index: ref base for match, (et-1) for substitution.
    obs_idx = np.where(error_types == _ERR_MATCH, ref_idx, error_types - _ERR_SUB_START)
    obs_kept = obs_idx[keep]
    sequence = _BASE_ASCII[obs_kept].tobytes().decode("ascii")

    ops = np.where(error_types == _ERR_DEL, _OP_D, _OP_M).astype(np.uint8)
    cigar = _cigar_from_ops(ops)

    quality: str | None = None
    if emit_quality:
        if phred_calib is not None:
            q = _sample_phred_calib(error_types[keep], phred_calib, rng)
            q = np.where(q < 0, _phred_from_probs(probs[keep]), q)
        else:
            q = _phred_from_probs(probs[keep])
        quality = (q + 33).astype(np.uint8).tobytes().decode("ascii")
    return sequence, quality, cigar


def _apply_with_insertions(
    ref_idx: np.ndarray,
    probs: np.ndarray,
    first_draw: np.ndarray,
    rng: np.random.Generator,
    *,
    max_ins_run: int,
    emit_quality: bool,
    phred_calib: np.ndarray | None,
) -> tuple[str, str | None, str]:
    """Sequential assembly for reads that drew at least one insertion.

    Reuses the precomputed per-position ``probs`` and the already-drawn
    ``first_draw`` as the first sample at each position; only insertion
    positions trigger additional resampling.
    """
    n = ref_idx.shape[0]
    out_bases: list[int] = []
    out_ops: list[int] = []
    out_q: list[int] = []

    def _quality(et: int, pos: int) -> int:
        if phred_calib is not None:
            q = int(_sample_phred_calib(np.array([et]), phred_calib, rng)[0])
            if q >= 0:
                return q
        return int(_phred_from_probs(probs[pos : pos + 1])[0])

    for pos in range(n):
        et = int(first_draw[pos])
        ins_count = 0
        while True:
            if et == _ERR_MATCH:
                out_bases.append(int(ref_idx[pos]))
                out_ops.append(_OP_M)
                if emit_quality:
                    out_q.append(_quality(et, pos))
                break
            if _ERR_SUB_START <= et < _ERR_INS_START:
                out_bases.append(et - _ERR_SUB_START)
                out_ops.append(_OP_M)
                if emit_quality:
                    out_q.append(_quality(et, pos))
                break
            if _ERR_INS_START <= et < _ERR_DEL:
                out_bases.append(et - _ERR_INS_START)
                out_ops.append(_OP_I)
                if emit_quality:
                    out_q.append(_quality(et, pos))
                ins_count += 1
                if ins_count >= max_ins_run:
                    break
                et = int(_sample_rows(probs[pos : pos + 1], rng)[0])
                continue
            # deletion
            out_ops.append(_OP_D)
            break

    sequence = _BASE_ASCII[np.array(out_bases, dtype=np.int64)].tobytes().decode("ascii") if out_bases else ""
    cigar = _cigar_from_ops(np.array(out_ops, dtype=np.uint8))
    quality = (
        (np.array(out_q, dtype=np.int64) + 33).astype(np.uint8).tobytes().decode("ascii")
        if emit_quality
        else None
    )
    return sequence, quality, cigar


def apply_read(
    model: ErrorModel,
    name: str,
    reference: str,
    rng: np.random.Generator,
    *,
    is_forward: bool = True,
    max_ins_run: int = _MAX_INS_RUN_DEFAULT,
    emit_quality: bool = True,
    phred_calib: np.ndarray | None = None,
) -> ResultRecord:
    """Simulate one read from a reference sequence."""
    raw = np.frombuffer(reference.encode("ascii", "replace"), dtype=np.uint8)
    ref_idx = _CHAR_TO_IDX[raw].astype(np.int64)
    if ref_idx.shape[0] == 0:
        return ResultRecord(name, "", "" if emit_quality else None, "")

    logits = model._logits_for_reference(ref_idx, is_forward)
    probs = _masked_probs(logits, ref_idx)
    if model.hmm is not None:
        probs = _apply_hmm_scaling(probs, model.hmm, rng)
    draw = _sample_rows(probs, rng)

    has_ins = bool(np.any((draw >= _ERR_INS_START) & (draw < _ERR_DEL)))
    if has_ins:
        seq, qual, cigar = _apply_with_insertions(
            ref_idx, probs, draw, rng,
            max_ins_run=max_ins_run, emit_quality=emit_quality, phred_calib=phred_calib,
        )
    else:
        seq, qual, cigar = _apply_fast(
            ref_idx, probs, draw, rng,
            emit_quality=emit_quality, phred_calib=phred_calib,
        )
    return ResultRecord(name, seq, qual, cigar)


def apply_batch(
    model: ErrorModel,
    records: list[tuple[str, str, bool]],
    rng: np.random.Generator,
    *,
    max_ins_run: int = _MAX_INS_RUN_DEFAULT,
    emit_quality: bool = True,
    phred_calib: np.ndarray | None = None,
) -> list[ResultRecord]:
    """Simulate a batch of reads.

    Args:
        model: Loaded error model.
        records: ``(name, sequence, is_forward)`` tuples.  ``is_forward`` selects
            the strand covariate (R2 mates pass ``False``).
        rng: NumPy generator for reproducible sampling.
        max_ins_run: Cap on consecutive insertions at one reference position.
        emit_quality: Produce Phred quality strings.
        phred_calib: Optional ``[10, 61]`` calibration from
            :func:`load_phred_calibration`.
    """
    return [
        apply_read(
            model, name, seq, rng,
            is_forward=is_forward, max_ins_run=max_ins_run,
            emit_quality=emit_quality, phred_calib=phred_calib,
        )
        for name, seq, is_forward in records
    ]


# ── Phred calibration ───────────────────────────────────────────────────────────


def load_phred_calibration(path: Path) -> np.ndarray:
    """Load a Phred calibration JSON and return ``P(Q | error_type)`` ``[10, 61]``."""
    with open(path) as fh:
        cal = json.load(fh)
    probs = np.array(cal["probs"], dtype=np.float64)
    if probs.shape != (NUM_ERROR_TYPES, 61):
        raise ValueError(
            f"Calibration {path} has unexpected shape {probs.shape}; "
            f"expected ({NUM_ERROR_TYPES}, 61)"
        )
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return (probs / row_sums).astype(np.float32)
