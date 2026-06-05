//! Core behaviour tests for skiver's library API.
//!
//! These pin down the pieces that the dead-code cleanup touches (kv-mer counting
//! and consensus determination, neighbour enumeration, seeding) plus a couple of
//! numeric helpers, so a refactor that changes observable behaviour is caught.

use rustc_hash::FxHashMap;

use skiver::huber::huber_ridge_fit_1d;
use skiver::kvmer::KVmerCountMap;
use skiver::seeding::fmh_seeds_masked;
use skiver::types::ValueInfo;
use skiver::utils::{_get_neighbors, _kmer_to_string};

// ── kmer string encoding ────────────────────────────────────────────────────

#[test]
fn kmer_to_string_decodes_2bit_bases() {
    // 2 bits per base, most-significant base first.
    assert_eq!(_kmer_to_string(0b00_01_10_11, 4), "ACGT");
    assert_eq!(_kmer_to_string(0b00, 1), "A");
    assert_eq!(_kmer_to_string(0b11_11_11, 3), "TTT");
}

// ── neighbour enumeration ───────────────────────────────────────────────────

#[test]
fn get_neighbors_substitutions_are_hamming_distance_one() {
    // Every substitution neighbour must differ from the value in exactly one
    // 2-bit base, and the value is never its own substitution neighbour.
    let value = 0b00_01_10_11; // ACGT
    let subs = substitution_neighbors(&_get_neighbors(value, 4));
    assert!(!subs.is_empty());
    for &nbr in subs.keys() {
        assert_ne!(nbr, value);
        let differing_bases = (0..4u8)
            .filter(|p| (nbr >> (p * 2)) & 0b11 != (value >> (p * 2)) & 0b11)
            .count();
        assert_eq!(differing_bases, 1, "neighbor {nbr:#b} should differ in one base");
    }
}

/// Helper: keep only substitution-op neighbours (exclude indels/ambiguous).
fn substitution_neighbors(
    neighbors: &FxHashMap<u64, skiver::types::NeighborInfo>,
) -> FxHashMap<u64, skiver::types::NeighborInfo> {
    use skiver::types::EditOperation::*;
    neighbors
        .iter()
        .filter(|(_, info)| {
            matches!(
                info.op,
                AC | AG | AT | CA | CG | CT | GA | GC | GT | TA | TC | TG
            )
        })
        .map(|(&k, &v)| (k, v))
        .collect()
}

// ── seeding ─────────────────────────────────────────────────────────────────

#[test]
fn fmh_seeds_masked_c1_extracts_every_window_forward() {
    // c = 1 keeps every window. For forward-only seeding, window `idx` covers
    // bases [i-k-v+1 ..= i] where i = k+v-1+idx, split into a k-mer key and a
    // v-mer value. Re-decoding key+value must reproduce that substring.
    let seq_str = "ACGTACGTAC";
    let seq = seq_str.as_bytes();
    let (k, v) = (2usize, 2usize);
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mut infos: Vec<ValueInfo> = Vec::new();
    fmh_seeds_masked(seq, &mut keys, &mut values, &mut infos, 1, k, v, false, 0);

    let n_windows = seq.len() - (k + v) + 1;
    assert_eq!(keys.len(), n_windows);
    assert_eq!(values.len(), n_windows);
    assert_eq!(infos.len(), n_windows);

    for (idx, (&key, &val)) in keys.iter().zip(values.iter()).enumerate() {
        let i = (k + v - 1) + idx;
        let start = i + 1 - k - v;
        let expected = &seq_str[start..=i];
        let got = format!(
            "{}{}",
            _kmer_to_string(key, k as u8),
            _kmer_to_string(val, v as u8)
        );
        assert_eq!(got, expected, "window {idx}");
        assert!(infos[idx].is_forward);
    }
}

#[test]
fn fmh_seeds_masked_skips_short_sequences() {
    let seq = b"ACG"; // shorter than k + v
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mut infos = Vec::new();
    fmh_seeds_masked(seq, &mut keys, &mut values, &mut infos, 1, 2, 2, false, 0);
    assert!(keys.is_empty());
}

// ── KVmerCountMap consensus / error spectrum ────────────────────────────────

/// Build a count map directly from (key, value) -> [total, forward] entries.
fn count_map_from(entries: &[(u64, u64, [u32; 2])], k: u8, v: u8) -> KVmerCountMap {
    let mut cm = KVmerCountMap::new(k, v, false);
    for &(key, value, counts) in entries {
        cm.counts.insert((key, value), counts);
    }
    cm
}

#[test]
fn build_full_stats_no_reference_consensus_and_counts() {
    let key = 0b00_01_10; // k=3: "ACG"
    let cons = 0b00_01_10_11; // v=4: "ACGT" (consensus, 10 obs)
    let err = 0b00_01_10_00; // "ACGA" — a 1-edit (T>A) neighbour of the consensus
    let cm = count_map_from(&[(key, cons, [10, 10]), (key, err, [2, 2])], 3, 4);

    let stats = cm.build_full_stats_no_reference(0);
    assert_eq!(stats.keys, vec![key]);
    assert_eq!(stats.consensus_values, vec![cons]);
    assert_eq!(stats.error_summary.consensus_counts, vec![10]);
    assert_eq!(stats.error_summary.total_counts, vec![12]);
    // The single error value is a 1-edit neighbour, so it is counted.
    assert_eq!(stats.error_summary.neighbor_counts, vec![2]);
    assert_eq!(stats.error_summary.second_counts, vec![2]);
}

#[test]
fn build_full_stats_no_reference_threshold_filters_low_count_keys() {
    let key = 0b00_01_10;
    let cons = 0b00_01_10_11;
    // total = 3; with threshold 3 the key is dropped (total <= threshold).
    let cm = count_map_from(&[(key, cons, [3, 3])], 3, 4);
    assert!(cm.build_full_stats_no_reference(3).keys.is_empty());
    assert_eq!(cm.build_full_stats_no_reference(2).keys, vec![key]);
}

#[test]
fn consensus_tie_break_picks_smallest_value() {
    // Two values tied on count: consensus must be the smaller u64, deterministically.
    let key = 0b00_01_10;
    let lo = 0b00_01_10_11; // "ACGT"
    let hi = 0b11_10_01_00; // "TGCA" (larger u64, also not a 1-edit neighbour of lo)
    assert!(lo < hi);
    let cm = count_map_from(&[(key, lo, [5, 5]), (key, hi, [5, 5])], 3, 4);

    let stats = cm.build_full_stats_no_reference(0);
    assert_eq!(stats.consensus_values, vec![lo]);

    // build_consensus_only must agree.
    let (keys, cons) = cm.build_consensus_only(0, None);
    assert_eq!(keys, vec![key]);
    assert_eq!(cons, vec![lo]);
}

// ── huber ridge fit ─────────────────────────────────────────────────────────

#[test]
fn huber_ridge_fit_recovers_a_line() {
    // y = 2x + 1, no regularisation -> exact recovery. Returns (slope, intercept).
    let x: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
    let (slope, intercept) = huber_ridge_fit_1d(&x, &y, 1.0, 0.0, 200, 1e-7);
    assert!((slope - 2.0).abs() < 1e-3, "slope={slope}");
    assert!((intercept - 1.0).abs() < 1e-3, "intercept={intercept}");
}

#[test]
fn huber_ridge_fit_rejects_degenerate_input() {
    assert_eq!(huber_ridge_fit_1d(&[1.0], &[1.0], 1.0, 0.5, 100, 1e-6), (0.0, 0.0));
}
