"""Observation encoding for the profile HMM error model.

Maps skiver dump outputs to integer-encoded tensors suitable for the
factored emission model: error_type (10 categories) + phred_bin (8 bins).
"""
from __future__ import annotations

import torch

# ─── Base encoding ────────────────────────────────────────────────────────────

BASE_TO_IDX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
GAP_CHAR = "-"
NUM_BASES = 4  # A, C, G, T (no gap in context indexing)
NUM_PHRED_BINS = 8  # 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35+

# ─── Error type encoding ─────────────────────────────────────────────────────
#
# 10 categories:
#   0 = match
#   1..4 = sub_to_A, sub_to_C, sub_to_G, sub_to_T
#   5..8 = ins_A, ins_C, ins_G, ins_T
#   9 = deletion

NUM_ERROR_TYPES = 10

ERROR_TYPE_MATCH = 0
ERROR_TYPE_SUB_A = 1
ERROR_TYPE_SUB_C = 2
ERROR_TYPE_SUB_G = 3
ERROR_TYPE_SUB_T = 4
ERROR_TYPE_INS_A = 5
ERROR_TYPE_INS_C = 6
ERROR_TYPE_INS_G = 7
ERROR_TYPE_INS_T = 8
ERROR_TYPE_DEL = 9

# Coarse error class for phred model conditioning.
NUM_ERROR_CLASSES = 3
ERROR_CLASS_MATCH = 0
ERROR_CLASS_MISMATCH = 1
ERROR_CLASS_INDEL = 2

_ERROR_TYPE_TO_CLASS = [
    ERROR_CLASS_MATCH,       # 0: match
    ERROR_CLASS_MISMATCH,    # 1: sub_to_A
    ERROR_CLASS_MISMATCH,    # 2: sub_to_C
    ERROR_CLASS_MISMATCH,    # 3: sub_to_G
    ERROR_CLASS_MISMATCH,    # 4: sub_to_T
    ERROR_CLASS_INDEL,       # 5: ins_A
    ERROR_CLASS_INDEL,       # 6: ins_C
    ERROR_CLASS_INDEL,       # 7: ins_G
    ERROR_CLASS_INDEL,       # 8: ins_T
    ERROR_CLASS_INDEL,       # 9: deletion
]


def error_type_to_class(error_type: int) -> int:
    """Map a fine error type (0..9) to a coarse error class (0..2)."""
    return _ERROR_TYPE_TO_CLASS[error_type]


ERROR_TYPE_TO_CLASS_TENSOR = torch.tensor(_ERROR_TYPE_TO_CLASS, dtype=torch.long)


# ─── Encoding functions ──────────────────────────────────────────────────────

def encode_error_type(true_base: str, obs_base: str, edit_op: str) -> int:
    """Encode a single base observation into an error type index (0..9).

    Args:
        true_base: Consensus base (ACGT or '-').
        obs_base: Observed base (ACGT or '-').
        edit_op: Edit operation string from skiver dump (e.g. 'C>T', '->A',
            'G>-', 'NA').

    Returns:
        Integer error type in [0, 9].
    """
    if edit_op == "NA" or true_base == obs_base:
        return ERROR_TYPE_MATCH

    if true_base == GAP_CHAR:
        # Insertion: true_base is gap, obs_base is inserted base.
        return ERROR_TYPE_INS_A + BASE_TO_IDX.get(obs_base, 0)

    if obs_base == GAP_CHAR:
        return ERROR_TYPE_DEL

    # Substitution: obs_base differs from true_base.
    return ERROR_TYPE_SUB_A + BASE_TO_IDX.get(obs_base, 0)


def encode_context(prev_base: str, true_base: str) -> int:
    """Encode dinucleotide context as a flat index (0..15).

    Args:
        prev_base: Previous base (ACGT).
        true_base: Current consensus base (ACGT). For insertions where
            true_base is '-', the caller should pass the next match-state
            base instead.

    Returns:
        Integer context index in [0, 15].
    """
    pb = BASE_TO_IDX.get(prev_base, 0)
    tb = BASE_TO_IDX.get(true_base, 0)
    return pb * NUM_BASES + tb


def bin_phred(phred: int) -> int:
    """Bin a Phred quality score into NUM_PHRED_BINS categories.

    Bins: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35+.
    Negative phred (e.g. -1 for deletions) maps to bin 0.
    """
    if phred < 0:
        return 0
    return min(phred // 5, NUM_PHRED_BINS - 1)
