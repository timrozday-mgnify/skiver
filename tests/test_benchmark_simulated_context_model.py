from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_simulated_context_model as bench
from lib.context_error_models import (
    NUM_TRUE_BASE_BINS,
    UNKNOWN_TRUE_BASE_BIN,
    _masked_log_probs_for_counts,
    aggregate_context_length_screen_counts,
)
from lib.context_h5_cache import (
    _aggregate_context_length_from_events,
    _row_cache_context_events,
)
from lib.encoding import ERROR_TYPE_MATCH, ERROR_TYPE_SUB_A, ERROR_TYPE_SUB_C, NUM_ERROR_TYPES


def test_reference_sequence_for_bam_name_accepts_genome_prefix() -> None:
    references = {"seq1": "ACGT"}

    assert bench._reference_sequence_for_bam_name("seq1", references) == "ACGT"
    assert (
        bench._reference_sequence_for_bam_name("genome_id:seq1", references)
        == "ACGT"
    )
    assert bench._normalise_bam_reference_name("genome_id:seq1", references) == "seq1"


def test_physical_truth_missing_bam_is_nonfatal(tmp_path: Path) -> None:
    fasta = tmp_path / "ref.fasta"
    fasta.write_text(">seq1\nACGT\n")

    result = bench._physical_truth_from_bam(tmp_path / "missing.bam", fasta)

    assert result["available"] is False
    assert "Missing BAM" in str(result["reason"])


def test_split_rate_recovery_compares_models_on_same_exposure() -> None:
    source = np.array([990.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    retrained = np.array([980.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    observed = np.array([970.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = bench._split_rate_recovery(
        split="train",
        source_counts=source,
        retrained_counts=retrained,
        observed_counts=observed,
        physical_truth={"available": False, "reason": "not tested"},
    )

    assert result["split"] == "train"
    assert math.isclose(result["source_model_error_rate"], 0.01)
    assert math.isclose(result["retrained_model_error_rate"], 0.02)
    assert math.isclose(result["model_error_rate_delta"], 0.01)
    assert math.isclose(result["model_error_rate_ratio"], 2.0)
    assert math.isclose(result["skiver_observed_window"]["error_rate"], 0.03)
    assert result["physical_bam_truth"]["available"] is False


def test_bam_query_to_fastq_pos_handles_reverse_reads() -> None:
    assert bench._bam_query_to_fastq_pos(0, 300, False) == 0
    assert bench._bam_query_to_fastq_pos(17, 300, False) == 17
    assert bench._bam_query_to_fastq_pos(0, 300, True) == 299
    assert bench._bam_query_to_fastq_pos(17, 300, True) == 282


def test_skiver_row_physical_pos_handles_reverse_value_rows() -> None:
    assert bench._skiver_row_physical_pos(is_forward=True, read_pos=200, t=9) == 200
    assert bench._skiver_row_physical_pos(is_forward=False, read_pos=200, t=9) == 183


def test_fastq_name_normalisation_and_mate_detection() -> None:
    path = Path("synthetic_train_R2.fastq")

    assert bench._normalise_fastq_name("@contig:0-300:+/2 read_1") == "contig:0-300:+"
    assert bench._mate_from_fastq_name("@contig:0-300:+/2 read_1", path, "R1") == "R2"
    assert bench._mate_from_fastq_name("@contig:0-300:+ read_1", path, "R1") == "R2"


def test_fastq_read_index_tracks_duplicate_name_occurrences(tmp_path: Path) -> None:
    fastq = tmp_path / "reads.fastq"
    fastq.write_text(
        "@dup/1\n"
        "AAAAAAAAAA\n"
        "+\n"
        "IIIIIIIIII\n"
        "@dup/1\n"
        "AAAAATAAAA\n"
        "+\n"
        "IIIIIIIIII\n"
    )

    index = bench._fastq_read_index([fastq])

    assert [row["occurrence_index"] for row in index] == [0, 1]


def test_nearest_distance_for_wiggle_matching() -> None:
    assert bench._nearest_distance([], 12) is None
    assert bench._nearest_distance([10, 50, 100], None) is None
    assert bench._nearest_distance([10, 50, 100], 8) == 2
    assert bench._nearest_distance([10, 50, 100], 70) == 20
    assert bench._nearest_distance([10, 50, 100], 130) == 30


def test_insertion_pair_reference_position_uses_neighbouring_alignment() -> None:
    pairs = [(0, 100), (1, None), (2, 101)]

    assert bench._nearest_ref_pos_from_aligned_pairs(pairs, 1) == 100
    assert bench._nearest_ref_pos_from_aligned_pairs([(0, None), (1, 9)], 0) == 9
    assert bench._nearest_ref_pos_from_aligned_pairs([(0, None)], 0) is None


def test_skiver_bam_matching_uses_duplicate_read_occurrences(tmp_path: Path) -> None:
    pysam = pytest.importorskip("pysam")
    reference = tmp_path / "ref.fa"
    reference.write_text(">seq1\nAAAAAAAAAAAAAAAAAAAA\n")
    fastq = tmp_path / "reads.fastq"
    fastq.write_text(
        "@dup/1\n"
        "AATAAAAAAAAAAAAAAAAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIII\n"
        "@dup/1\n"
        "AAAAAAATAAAAAAAAAAAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIII\n"
        "@dup/1\n"
        "AAAAAAAAAAAAAAAAAAAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIII\n"
        "@dup/1\n"
        "AAAAAAAAAAAAAAAAAAAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIII\n"
        "@dup/1\n"
        "AAAAAAAAATAAAAAAAAAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIII\n"
    )
    bam = tmp_path / "truth.bam"
    header = {
        "HD": {"VN": "1.0"},
        "SQ": [{"LN": 20, "SN": "seq1"}],
    }
    with pysam.AlignmentFile(bam, "wb", header=header) as handle:
        for sequence in [
            "AATAAAAAAAAAAAAAAAAA",
            "AAAAAAATAAAAAAAAAAAA",
            "AAAAAAAAAAAAAAAAAAAA",
            "AAAAAAAAAAAAAAAAAAAA",
            "AAAAAAAAATAAAAAAAAAA",
        ]:
            read = pysam.AlignedSegment()
            read.query_name = "dup"
            read.query_sequence = sequence
            read.flag = 0
            read.reference_id = 0
            read.reference_start = 0
            read.mapping_quality = 60
            read.cigar = [(0, 20)]
            read.query_qualities = pysam.qualitystring_to_array("I" * 20)
            handle.write(read)
    base_observations = tmp_path / "toy.base_observations.tsv"
    base_observations.write_text(
        "obs_id\tread_id\tt\ttrue_base\tobs_base\tprev_base\tedit_op\tphred\t"
        "read_pos\tdist_to_end\tis_forward\tpasses_filter\n"
        "0\t0\t1\tA\tT\tA\tA>T\t40\t2\t18\ttrue\ttrue\n"
        "1\t1\t1\tA\tA\tA\tNA\t40\t7\t13\ttrue\ttrue\n"
        "2\t2\t1\tA\tT\tA\tA>T\t40\t5\t15\ttrue\ttrue\n"
        "3\t3\t1\tA\tA\tA\tNA\t40\t6\t14\ttrue\ttrue\n"
    )

    result = bench._skiver_bam_error_match_split(
        split="toy",
        base_observations=base_observations,
        reads=[fastq],
        bam_path=bam,
        reference=reference,
    )

    assert result["duplicate_name_mate_keys"] == 1
    assert result["max_name_mate_occurrences"] == 5
    assert result["skiver_detected_errors"] == 2
    assert result["same_coordinate_true"] == 1
    assert result["strict_true_fraction"] == 0.5
    assert result["wiggle_window_bp"] == 50
    assert result["coordinate_set_confusion"] == {
        "universe": 100,
        "skiver_positive_coordinates": 2,
        "bam_truth_positive_coordinates": 3,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 2,
        "true_negative": 96,
        "precision": 0.5,
        "recall": 1 / 3,
        "specificity": 96 / 97,
        "jaccard": 0.25,
    }
    assert result["skiver_covered_coordinate_confusion"] == {
        "universe": 4,
        "skiver_positive_coordinates": 2,
        "bam_truth_positive_coordinates": 2,
        "total_bam_truth_positive_coordinates": 3,
        "out_of_scope_due_to_skiver_sparsity": 1,
        "skiver_covered_coordinate_fraction": 0.04,
        "bam_truth_coverage_fraction": 2 / 3,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
        "specificity": 0.5,
        "f1": 0.5,
        "jaccard": 1 / 3,
    }

    records = bench._covered_truth_quality_records(
        base_observations=base_observations,
        reads=[fastq],
        bam_path=bam,
        reference=reference,
    )
    assert records == [(4, 40), (4, 40), (0, 40), (0, 40)]
    phred_artifact = bench._quality_artifact_from_records(records, split="toy")
    counts = np.asarray(phred_artifact["counts"])
    assert counts[0, 40] == 2
    assert counts[4, 40] == 2


def test_context_counts_keep_true_base_axis_for_masked_training(tmp_path: Path) -> None:
    base_observations = tmp_path / "toy.base_observations.tsv"
    base_observations.write_text(
        "obs_id\tt\ttrue_base\tobs_base\tprev_base\tedit_op\tphred\tread_pos\t"
        "dist_to_end\tis_forward\tpasses_filter\n"
        "0\t1\tA\tA\tC\tNA\t40\t0\t10\ttrue\ttrue\n"
        "0\t2\tA\tC\tA\tA>C\t40\t1\t9\ttrue\ttrue\n"
        "0\t3\t-\tG\tA\t->G\t40\t2\t8\ttrue\ttrue\n"
    )

    counts = aggregate_context_length_screen_counts(
        [base_observations.with_suffix("").with_suffix("")],
        context_lengths=[1],
    ).by_length[1].counts

    assert tuple(counts.shape) == (4, NUM_TRUE_BASE_BINS, NUM_ERROR_TYPES)
    assert counts[1, 0, ERROR_TYPE_MATCH] == 1
    assert counts[0, 0, ERROR_TYPE_SUB_C] == 1
    assert counts[0, UNKNOWN_TRUE_BASE_BIN, 7] == 1


def test_context_counts_seed_v1_history_from_raw_key(tmp_path: Path) -> None:
    prefix = tmp_path / "toy"
    raw_observations = tmp_path / "toy.raw_observations.tsv"
    raw_observations.write_text(
        "obs_id\tkey_str\tconsensus_str\tobs_value_str\tedit_distance\tedit_op\t"
        "edit_position\tqual_str\tstart_index\tdist_to_read_end\tis_forward\t"
        "passes_filter\n"
        "0\tCA\tA\tC\t1\tA>C\t0\tI\t2\t8\ttrue\ttrue\n"
    )
    base_observations = tmp_path / "toy.base_observations.tsv"
    base_observations.write_text(
        "obs_id\tt\ttrue_base\tobs_base\tprev_base\tedit_op\tphred\tread_pos\t"
        "dist_to_end\tis_forward\tpasses_filter\n"
        "0\t1\tA\tC\tA\tA>C\t40\t2\t8\ttrue\ttrue\n"
    )

    counts = aggregate_context_length_screen_counts(
        [prefix],
        context_lengths=[2],
    ).by_length[2].counts

    context_ca = 1 * 4 + 0
    assert counts[context_ca, 0, ERROR_TYPE_SUB_C] == 1


def test_masked_log_probs_remove_only_impossible_self_substitutions() -> None:
    counts = torch.zeros(1, NUM_TRUE_BASE_BINS, NUM_ERROR_TYPES)
    logits = torch.zeros(1, NUM_ERROR_TYPES)

    log_probs = _masked_log_probs_for_counts(logits, counts)

    assert torch.isneginf(log_probs[0, 0, ERROR_TYPE_SUB_A])
    assert torch.isfinite(log_probs[0, 0, ERROR_TYPE_SUB_C])
    assert torch.isfinite(log_probs[0, UNKNOWN_TRUE_BASE_BIN, ERROR_TYPE_SUB_A])


def test_h5_context_cache_aggregation_keeps_true_base_axis() -> None:
    obs_starts = np.array([True, False, False])
    prev_bases = np.array([1, 0, 0], dtype=np.uint8)
    true_bases = np.array([0, 0, 4], dtype=np.uint8)
    targets = np.array([ERROR_TYPE_MATCH, ERROR_TYPE_SUB_C, 7], dtype=np.uint8)

    event_bases, row_event_ends, available_history = _row_cache_context_events(
        obs_starts,
        prev_bases,
        true_bases,
    )
    counts = _aggregate_context_length_from_events(
        event_bases,
        row_event_ends,
        available_history,
        true_bases,
        targets,
        1,
    )

    assert tuple(counts.shape) == (4, NUM_TRUE_BASE_BINS, NUM_ERROR_TYPES)
    assert counts[1, 0, ERROR_TYPE_MATCH] == 1
    assert counts[0, 0, ERROR_TYPE_SUB_C] == 1
    assert counts[0, UNKNOWN_TRUE_BASE_BIN, 7] == 1


def test_load_fasta_is_memoised(tmp_path: Path) -> None:
    """`_load_fasta` is cached so the (large) reference is parsed once and reused."""
    fasta = tmp_path / "ref.fasta"
    fasta.write_text(">seq1\nACGT\n>seq2\nTTGG\n")

    bench._load_fasta.cache_clear()
    first = bench._load_fasta(fasta)
    second = bench._load_fasta(fasta)

    assert first == {"seq1": "ACGT", "seq2": "TTGG"}
    assert first is second  # same object returned, not reparsed
    info = bench._load_fasta.cache_info()
    assert info.hits >= 1
