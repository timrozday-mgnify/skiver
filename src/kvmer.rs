use log::{error, info, warn};
use needletail::parse_fastx_file;
use rust_htslib::{bam, bam::Read as BamRead}; // Added rust-htslib
use serde::{Deserialize, Serialize};

use std::fs::File;
use std::io::{BufReader, BufWriter, Read as IoRead, Write};

use std::collections::HashMap;
use rustc_hash::FxHashMap;

use crate::summary::{ErrorSpectrumSummary, ErrorSummary, PhredScoreSummary, ReadPositionSummary};
use crate::{seeding::*, types::*};

/// kv-mer statistics for downstream analysis.
pub struct KVmerStats {
    pub k: u8,
    pub v: u8,
    pub keys: Vec<u64>,
    pub consensus_values: Vec<u64>,
    pub error_summary: ErrorSummary,
    pub error_spectrum: ErrorSpectrumSummary,
    pub phred_summary: PhredScoreSummary,
    pub read_position_summary: ReadPositionSummary,
}

const KVMER_MAGIC: &[u8; 8] = b"SKIVR002";

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct KVmerSet {
    pub key_size: u8,
    pub value_size: u8,
    pub kv_size: u8,
    pub num_kvmers: u32,

    /// key -> value -> list of per-observation metadata.
    /// The count of a (key, value) pair is `info_list.len()`.
    pub key_value_qual_map: HashMap<u64, HashMap<u64, Vec<ValueInfo>>>,

    // utilities to extract key and value from a kmer hash
    key_mask: u64,
    value_mask: u64,

    // whether both forward and reverse complement of the reads are included
    bidirectional: bool,

    // sequential read counter; not persisted in the sketch file
    #[serde(skip)]
    read_id_counter: u64,
}

// ─── Fragment ID helpers ──────────────────────────────────────────────────────

/// Compute a stable fragment ID from a read name by stripping known pair
/// suffixes and hashing the base name with FNV-1a.
///
/// For paired-end FASTQ reads named `read1/1` and `read1/2`, both produce the
/// same fragment ID. For BAM records the QNAME is identical for both ends of a
/// pair, so passing `record.qname()` directly yields the correct shared ID.
pub fn fragment_id_from_name(name: &[u8]) -> u64 {
    // Strip /1 or /2 suffix
    let base = if name.ends_with(b"/1") || name.ends_with(b"/2") {
        &name[..name.len() - 2]
    } else {
        name
    };
    // FNV-1a 64-bit hash
    let mut h: u64 = 14695981039346656037u64;
    for &b in base {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211u64);
    }
    h
}

/// Set `fragment_id` on all entries in `info_vec` to `fid`.
#[inline]
fn set_fragment_ids(info_vec: &mut Vec<ValueInfo>, fid: u64) {
    for info in info_vec.iter_mut() {
        info.fragment_id = fid;
    }
}

// ─── Shared seeding helpers ────────────────────────────────────────────────────

/// Extract (key, value, ValueInfo) triples from a sequence with quality scores.
/// Handles AVX2 dispatch and trimming internally.
fn extract_kv_with_qual(
    seq: &[u8],
    qual: &[u8],
    k: u8,
    v: u8,
    bidirectional: bool,
    c: usize,
    trim_front: usize,
    trim_back: usize,
    read_id: u64,
) -> (Vec<u64>, Vec<u64>, Vec<ValueInfo>) {
    let start = trim_front.min(seq.len());
    let end = seq.len().saturating_sub(trim_back);
    let seq_t = &seq[start..end];
    let qual_t = &qual[start..end];
    let mut key_vec = Vec::new();
    let mut value_vec = Vec::new();
    let mut info_vec = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            use crate::avx2_seeding::extract_markers_avx2_masked_with_qual;
            unsafe {
                extract_markers_avx2_masked_with_qual(
                    seq_t, qual_t, &mut key_vec, &mut value_vec, &mut info_vec,
                    c, k as usize, v as usize, bidirectional, read_id,
                );
            }
            return (key_vec, value_vec, info_vec);
        }
    }
    fmh_seeds_masked_with_qual(
        seq_t, qual_t, &mut key_vec, &mut value_vec, &mut info_vec,
        c, k as usize, v as usize, bidirectional, read_id,
    );
    (key_vec, value_vec, info_vec)
}

/// Extract (key, value, ValueInfo) triples from a sequence without quality scores (FASTA).
fn extract_kv_no_qual(
    seq: &[u8],
    k: u8,
    v: u8,
    bidirectional: bool,
    c: usize,
    trim_front: usize,
    trim_back: usize,
    read_id: u64,
) -> (Vec<u64>, Vec<u64>, Vec<ValueInfo>) {
    let start = trim_front.min(seq.len());
    let end = seq.len().saturating_sub(trim_back);
    let seq_t = &seq[start..end];
    let mut key_vec = Vec::new();
    let mut value_vec = Vec::new();
    let mut info_vec = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            use crate::avx2_seeding::extract_markers_avx2_masked;
            unsafe {
                extract_markers_avx2_masked(
                    seq_t, &mut key_vec, &mut value_vec, &mut info_vec,
                    c, k as usize, v as usize, bidirectional, read_id,
                );
            }
            return (key_vec, value_vec, info_vec);
        }
    }
    fmh_seeds_masked(
        seq_t, &mut key_vec, &mut value_vec, &mut info_vec,
        c, k as usize, v as usize, bidirectional, read_id,
    );
    (key_vec, value_vec, info_vec)
}

/// Like `extract_kv_no_qual` but fills caller-provided buffers (cleared first) so
/// they can be reused across reads — avoids a fresh allocation per read. Used by
/// the counting paths, which need only key/value/strand (no quality), so this is
/// also cheaper than the quality-extracting variant.
fn extract_kv_no_qual_into(
    seq: &[u8],
    k: u8,
    v: u8,
    bidirectional: bool,
    c: usize,
    trim_front: usize,
    trim_back: usize,
    read_id: u64,
    key_vec: &mut Vec<u64>,
    value_vec: &mut Vec<u64>,
    info_vec: &mut Vec<ValueInfo>,
) {
    key_vec.clear();
    value_vec.clear();
    info_vec.clear();
    let start = trim_front.min(seq.len());
    let end = seq.len().saturating_sub(trim_back);
    let seq_t = &seq[start..end];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            use crate::avx2_seeding::extract_markers_avx2_masked;
            unsafe {
                extract_markers_avx2_masked(
                    seq_t, key_vec, value_vec, info_vec,
                    c, k as usize, v as usize, bidirectional, read_id,
                );
            }
            return;
        }
    }
    fmh_seeds_masked(
        seq_t, key_vec, value_vec, info_vec,
        c, k as usize, v as usize, bidirectional, read_id,
    );
}

/// Increment `(key, value)` counts for one read's seeded markers, honouring an
/// optional key allow-list. Shared by the serial and threaded counting paths.
#[inline]
fn count_seeds_into(
    counts: &mut FxHashMap<(u64, u64), [u32; 2]>,
    keys: &[u64],
    values: &[u64],
    infos: &[ValueInfo],
    allowed_keys: Option<&FxHashMap<u64, u64>>,
) {
    for ((key, value), info) in keys.iter().zip(values.iter()).zip(infos.iter()) {
        if let Some(filter) = allowed_keys {
            if !filter.contains_key(key) {
                continue;
            }
        }
        let e = counts.entry((*key, *value)).or_insert([0u32; 2]);
        e[0] += 1;
        if info.is_forward {
            e[1] += 1;
        }
    }
}

// ─── KVmerCountMap ─────────────────────────────────────────────────────────────

/// Compact in-memory accumulator for pass 1 of the two-pass dump strategy.
/// Stores only (total_count, forward_count) per (key, value) pair — no ValueInfo.
pub struct KVmerCountMap {
    pub key_size: u8,
    pub value_size: u8,
    pub bidirectional: bool,
    /// (key, value) → [total_count, forward_count].
    ///
    /// The key and value are kept as a `(u64, u64)` tuple rather than packed into a
    /// single `u64`: with `k` and `v` each up to 32 bases (e.g. the pipeline's
    /// k=21, v=13 ⇒ 42 + 26 = 68 bits) the combined kv-mer does not fit in 64 bits,
    /// and a `u128` would be the same 16 bytes as the tuple — so 16 B is already the
    /// minimal key width for this regime.
    pub counts: FxHashMap<(u64, u64), [u32; 2]>,
    read_id_counter: u64,
}

impl KVmerCountMap {
    pub fn new(key_size: u8, value_size: u8, bidirectional: bool) -> Self {
        Self::with_capacity(key_size, value_size, bidirectional, 0)
    }

    /// Like [`new`](Self::new) but pre-allocates room for `capacity` distinct
    /// kv-mers. Reserving up front avoids the incremental hashbrown resizes whose
    /// peak transiently holds both the old and new backing tables (the
    /// "resize-doubling" spike); callers pass an estimate derived from the input
    /// size and subsampling rate.
    pub fn with_capacity(key_size: u8, value_size: u8, bidirectional: bool, capacity: usize) -> Self {
        KVmerCountMap {
            key_size,
            value_size,
            bidirectional,
            counts: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            read_id_counter: 0,
        }
    }

    /// Flatten the count map into `(key, value, counts)` triples sorted by `key`,
    /// so callers can group per key by scanning contiguous runs instead of building
    /// a `HashMap<u64, HashMap<u64, _>>` (one inner map per key — millions of tiny
    /// allocations at scale). One `Vec` replaces all of them.
    fn sorted_key_value_counts(&self) -> Vec<(u64, u64, [u32; 2])> {
        let mut entries: Vec<(u64, u64, [u32; 2])> = self
            .counts
            .iter()
            .map(|(&(key, value), &c)| (key, value, c))
            .collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        entries
    }

    /// Visit every key whose summed count exceeds `threshold`, supplying its
    /// `value → [total, forward]` counts to `f`.
    ///
    /// Memory-frugal grouping: instead of materialising a
    /// `HashMap<u64, FxHashMap<u64, [u32; 2]>>` (one inner map per key — millions of
    /// tiny allocations), this sorts a single `Vec` by key and reuses one scratch
    /// `HashMap` across keys, so `value_counts` keeps the exact type the
    /// `ErrorSummary` updaters already expect.
    ///
    /// When `reference_consensus` is `Some`, the visited keys and their consensus
    /// values come from the reference; otherwise every key is visited with its
    /// most-frequent value as consensus (ties broken by run order, as before).
    fn for_each_consensus_group<F>(
        &self,
        threshold: u32,
        reference_consensus: Option<&FxHashMap<u64, u64>>,
        mut f: F,
    ) where
        F: FnMut(u64, u64, &FxHashMap<u64, [u32; 2]>),
    {
        let entries = self.sorted_key_value_counts();
        let mut scratch: FxHashMap<u64, [u32; 2]> = FxHashMap::default();

        // Load one key's run into `scratch`; return (total_count, argmax_value).
        let fill = |scratch: &mut FxHashMap<u64, [u32; 2]>,
                    run: &[(u64, u64, [u32; 2])]|
         -> (u32, u64) {
            scratch.clear();
            let mut total: u32 = 0;
            let mut best_value: u64 = 0;
            let mut best_count: u32 = 0;
            for &(_, value, counts) in run {
                scratch.insert(value, counts);
                total = total.saturating_add(counts[0]);
                // Deterministic argmax: highest count, ties broken by smallest
                // value. (The previous `HashMap` + `max_by_key` broke ties by
                // arbitrary iteration order, making consensus nondeterministic.)
                if counts[0] > best_count || (counts[0] == best_count && value < best_value) {
                    best_count = counts[0];
                    best_value = value;
                }
            }
            (total, best_value)
        };

        if let Some(ref_consensus) = reference_consensus {
            for (&key, &consensus) in ref_consensus {
                let start = entries.partition_point(|e| e.0 < key);
                if start >= entries.len() || entries[start].0 != key {
                    continue;
                }
                let mut end = start;
                while end < entries.len() && entries[end].0 == key {
                    end += 1;
                }
                let (total, _) = fill(&mut scratch, &entries[start..end]);
                if total <= threshold {
                    continue;
                }
                f(key, consensus, &scratch);
            }
        } else {
            let mut i = 0;
            while i < entries.len() {
                let key = entries[i].0;
                let mut end = i;
                while end < entries.len() && entries[end].0 == key {
                    end += 1;
                }
                let (total, consensus) = fill(&mut scratch, &entries[i..end]);
                i = end;
                if total <= threshold {
                    continue;
                }
                f(key, consensus, &scratch);
            }
        }
    }

    /// Process one input file (FASTQ/BAM), accumulating only per-(key,value) counts.
    /// Per-read ValueInfo is allocated temporarily and dropped after each read.
    pub fn add_file(&mut self, file: &str, c: usize, trim_front: usize, trim_back: usize) {
        self.add_file_impl(file, c, trim_front, trim_back, None);
    }

    /// Like `add_file` but only inserts observations whose key is in `allowed_keys`.
    pub fn add_file_filtered(
        &mut self,
        file: &str,
        c: usize,
        trim_front: usize,
        trim_back: usize,
        allowed_keys: &FxHashMap<u64, u64>,
    ) {
        self.add_file_impl(file, c, trim_front, trim_back, Some(allowed_keys));
    }

    fn add_file_impl(
        &mut self,
        file: &str,
        c: usize,
        trim_front: usize,
        trim_back: usize,
        allowed_keys: Option<&FxHashMap<u64, u64>>,
    ) {
        let k = self.key_size;
        let v = self.value_size;
        let bidir = self.bidirectional;
        // Counting needs only key/value/strand, so quality is not extracted, and the
        // three seeding buffers are reused across reads (no per-read allocation).
        let (mut kbuf, mut vbuf, mut ibuf): (Vec<u64>, Vec<u64>, Vec<ValueInfo>) =
            (Vec::new(), Vec::new(), Vec::new());
        if file.ends_with(".bam") || file.ends_with(".sam") {
            match bam::Reader::from_path(file) {
                Ok(mut reader) => {
                    for record_result in reader.records() {
                        match record_result {
                            Ok(record) => {
                                let read_id = self.read_id_counter;
                                self.read_id_counter += 1;
                                let seq = record.seq().as_bytes();
                                extract_kv_no_qual_into(
                                    &seq, k, v, bidir, c, trim_front, trim_back, read_id,
                                    &mut kbuf, &mut vbuf, &mut ibuf,
                                );
                                count_seeds_into(&mut self.counts, &kbuf, &vbuf, &ibuf, allowed_keys);
                            }
                            Err(e) => warn!("Error reading BAM/SAM record: {}", e),
                        }
                    }
                }
                Err(e) => error!(
                    "{} is not a valid BAM/SAM file (Error: {}); skipping.",
                    file, e
                ),
            }
        } else {
            let reader = parse_fastx_file(file);
            if reader.is_err() {
                error!("{} is not a valid fasta/fastq file; skipping.", file);
                return;
            }
            let mut reader = reader.unwrap();
            while let Some(record) = reader.next() {
                match record {
                    Ok(record) => {
                        let read_id = self.read_id_counter;
                        self.read_id_counter += 1;
                        extract_kv_no_qual_into(
                            &record.seq(), k, v, bidir, c, trim_front, trim_back, read_id,
                            &mut kbuf, &mut vbuf, &mut ibuf,
                        );
                        count_seeds_into(&mut self.counts, &kbuf, &vbuf, &ibuf, allowed_keys);
                    }
                    Err(e) => warn!("Error reading record: {}", e),
                }
            }
        }
    }

    /// Merge counts from a KVmerSet (e.g. loaded from a .kvmer sketch file) into this map.
    pub fn merge_from_kvmer_set(&mut self, kvmer_set: &KVmerSet) {
        for (key, value_map) in &kvmer_set.key_value_qual_map {
            for (value, info_list) in value_map {
                let e = self
                    .counts
                    .entry((*key, *value))
                    .or_insert([0u32; 2]);
                e[0] += info_list.len() as u32;
                e[1] += info_list.iter().filter(|i| i.is_forward).count() as u32;
            }
        }
    }

    /// Build KVmerStats from accumulated counts.
    /// If `reference_consensus` is provided, use those consensus values and iterate
    /// only keys present in both the reference and this count map (mirrors
    /// `get_stats_with_reference`). Otherwise derive consensus from max count per key.
    pub fn build_stats(
        &self,
        threshold: u32,
        reference_consensus: Option<&FxHashMap<u64, u64>>,
    ) -> KVmerStats {
        let k = self.key_size;
        let v = self.value_size;

        let mut keys: Vec<u64> = Vec::new();
        let mut consensus_values: Vec<u64> = Vec::new();
        let mut error_summary = ErrorSummary::new(v as usize);
        let phred_summary = PhredScoreSummary::new();
        let read_position_summary = ReadPositionSummary::new();

        // Group flat counts per key without a HashMap-per-key (see
        // `for_each_consensus_group`). Consensus is the reference value when given,
        // else the most-frequent value. Only used in the non-`--use-all` path;
        // `--use-all` callers go through `build_consensus_only`.
        self.for_each_consensus_group(threshold, reference_consensus, |key, consensus, value_counts| {
            if error_summary.update_for_outlier_filter(consensus, v, value_counts) {
                keys.push(key);
                consensus_values.push(consensus);
            }
        });

        KVmerStats {
            k,
            v,
            keys,
            consensus_values,
            error_summary,
            error_spectrum: ErrorSpectrumSummary::new(v as usize),
            phred_summary,
            read_position_summary,
        }
    }

    /// Lightweight consensus-only build for `--use-all` paths where outlier filtering
    /// is skipped. Returns `(keys, consensus_values)` without building `ErrorSummary`.
    pub fn build_consensus_only(
        &self,
        threshold: u32,
        reference_consensus: Option<&FxHashMap<u64, u64>>,
    ) -> (Vec<u64>, Vec<u64>) {
        use crate::utils::_get_neighbors;
        let v = self.value_size;

        // Single pass over the flat (key, value) -> counts map to aggregate per-key
        // (total_count, argmax_value, argmax_count).
        let mut summary: FxHashMap<u64, (u32, u64, u32)> = FxHashMap::default();
        for (&(key, value), counts) in &self.counts {
            let e = summary.entry(key).or_insert((0u32, 0u64, 0u32));
            e.0 = e.0.saturating_add(counts[0]);
            // Deterministic argmax: highest count, ties broken by smallest value,
            // so the consensus does not depend on HashMap iteration order.
            if counts[0] > e.2 || (counts[0] == e.2 && value < e.1) {
                e.1 = value;
                e.2 = counts[0];
            }
        }

        let mut keys: Vec<u64> = Vec::new();
        let mut consensus_values: Vec<u64> = Vec::new();

        if let Some(ref_consensus) = reference_consensus {
            for (&key, &consensus) in ref_consensus {
                let Some(&(total, _, _)) = summary.get(&key) else {
                    continue;
                };
                if total <= threshold {
                    continue;
                }
                let neighbors = _get_neighbors(consensus, v);
                if neighbors.contains_key(&consensus) {
                    continue;
                }
                keys.push(key);
                consensus_values.push(consensus);
            }
        } else {
            for (&key, &(total, best_value, _)) in &summary {
                if total <= threshold {
                    continue;
                }
                let neighbors = _get_neighbors(best_value, v);
                if neighbors.contains_key(&best_value) {
                    continue;
                }
                keys.push(key);
                consensus_values.push(best_value);
            }
        }

        (keys, consensus_values)
    }

    /// Build full `KVmerStats` with all `ErrorSummary` fields populated, from the
    /// flat count map (no per-observation `ValueInfo`).
    ///
    /// `PhredScoreSummary` and `ReadPositionSummary` require per-observation quality
    /// and position data that flat counts cannot supply; their per-key vectors are
    /// populated with empty entries so that downstream `to_csv()` indexing is safe,
    /// but the resulting CSVs will contain only headers and no data rows.
    ///
    /// `reference_consensus` must be `Some`; the function panics otherwise.
    pub fn build_full_stats(
        &self,
        threshold: u32,
        reference_consensus: &FxHashMap<u64, u64>,
    ) -> KVmerStats {
        self.build_full_stats_impl(threshold, Some(reference_consensus))
    }

    /// Build KVmerStats using self-determined consensus (most-frequent value per key).
    /// Used by the non-reference `skiver analyze` path.  Phred and read-position
    /// summaries will be empty — no ValueInfo is stored.
    pub fn build_full_stats_no_reference(&self, threshold: u32) -> KVmerStats {
        self.build_full_stats_impl(threshold, None)
    }

    /// Shared body for [`build_full_stats`](Self::build_full_stats) and
    /// [`build_full_stats_no_reference`](Self::build_full_stats_no_reference).
    /// Groups counts per key via [`for_each_consensus_group`](Self::for_each_consensus_group)
    /// (no HashMap-per-key) and populates the full `ErrorSummary` + spectrum.
    fn build_full_stats_impl(
        &self,
        threshold: u32,
        reference_consensus: Option<&FxHashMap<u64, u64>>,
    ) -> KVmerStats {
        let k = self.key_size;
        let v = self.value_size;

        let mut keys: Vec<u64> = Vec::new();
        let mut consensus_values: Vec<u64> = Vec::new();
        let mut error_summary = ErrorSummary::new(v as usize);
        let mut error_spectrum = ErrorSpectrumSummary::new(v as usize);
        let mut phred_summary = PhredScoreSummary::new();
        let mut read_position_summary = ReadPositionSummary::new();

        self.for_each_consensus_group(threshold, reference_consensus, |key, consensus, value_counts| {
            if error_summary.update_from_counts(key, consensus, k, v, value_counts) {
                keys.push(key);
                consensus_values.push(consensus);
                error_spectrum.update(
                    error_summary.error_counts_per_key.last().unwrap().clone(),
                    error_summary.forward_error_counts_per_key.last().unwrap().clone(),
                );
                // Push empty entries so phred/position to_csv indexing remains in-bounds.
                phred_summary.correct_per_key.push(HashMap::new());
                phred_summary.error_per_key.push(HashMap::new());
                read_position_summary.correct_from_start_per_key.push(HashMap::new());
                read_position_summary.correct_from_end_per_key.push(HashMap::new());
                read_position_summary.error_from_start_per_key.push(HashMap::new());
                read_position_summary.error_from_end_per_key.push(HashMap::new());
            }
        });

        KVmerStats {
            k,
            v,
            keys,
            consensus_values,
            error_summary,
            error_spectrum,
            phred_summary,
            read_position_summary,
        }
    }

}

// ─── Reference consensus builder ───────────────────────────────────────────────

/// Build `{key → value}` directly from a FASTA/FASTQ/BAM file, retaining only
/// keys that have a single canonical value (i.e. the reference assigns one
/// v-mer per k-mer). Uses `u64::MAX` as an in-progress sentinel for keys with
/// multiple distinct values; sentinel entries are filtered out at the end.
///
/// Memory peak is O(unique keys) — does not allocate a `(key, value) → counts`
/// intermediate, in contrast to `KVmerCountMap::add_file`.
///
/// `v` must be < 32 so that no legitimate v-mer bit-pattern can equal the
/// sentinel `u64::MAX`.
pub fn build_reference_consensus_from_file(
    file: &str,
    k: u8,
    v: u8,
    bidirectional: bool,
    c: usize,
    trim_front: usize,
    trim_back: usize,
) -> FxHashMap<u64, u64> {
    debug_assert!(v < 32, "v must be < 32 to use u64::MAX as multi-value sentinel");
    const MULTI: u64 = u64::MAX;
    let mut state: FxHashMap<u64, u64> = FxHashMap::default();
    let mut read_id_counter: u64 = 0;

    let process = |key: u64, value: u64, state: &mut FxHashMap<u64, u64>| {
        use std::collections::hash_map::Entry;
        match state.entry(key) {
            Entry::Vacant(slot) => {
                slot.insert(value);
            }
            Entry::Occupied(mut slot) => {
                let existing = *slot.get();
                if existing != MULTI && existing != value {
                    slot.insert(MULTI);
                }
            }
        }
    };

    if file.ends_with(".bam") || file.ends_with(".sam") {
        match bam::Reader::from_path(file) {
            Ok(mut reader) => {
                for record_result in reader.records() {
                    match record_result {
                        Ok(record) => {
                            let read_id = read_id_counter;
                            read_id_counter += 1;
                            let seq = record.seq().as_bytes();
                            let qual = record.qual().to_vec();
                            let (key_vec, value_vec, _info_vec) = extract_kv_with_qual(
                                &seq, &qual, k, v, bidirectional, c, trim_front, trim_back, read_id,
                            );
                            for (key, value) in key_vec.iter().zip(value_vec.iter()) {
                                process(*key, *value, &mut state);
                            }
                        }
                        Err(e) => warn!("Error reading BAM/SAM record: {}", e),
                    }
                }
            }
            Err(e) => error!(
                "{} is not a valid BAM/SAM file (Error: {}); skipping.",
                file, e
            ),
        }
    } else {
        let reader = parse_fastx_file(file);
        if reader.is_err() {
            error!("{} is not a valid fasta/fastq file; skipping.", file);
            return state;
        }
        let mut reader = reader.unwrap();
        while let Some(record) = reader.next() {
            match record {
                Ok(record) => {
                    let read_id = read_id_counter;
                    read_id_counter += 1;
                    let seq = record.seq();
                    let (key_vec, value_vec, _info_vec) = if let Some(qual) = record.qual() {
                        extract_kv_with_qual(
                            &seq, qual, k, v, bidirectional, c, trim_front, trim_back, read_id,
                        )
                    } else {
                        extract_kv_no_qual(
                            &seq, k, v, bidirectional, c, trim_front, trim_back, read_id,
                        )
                    };
                    for (key, value) in key_vec.iter().zip(value_vec.iter()) {
                        process(*key, *value, &mut state);
                    }
                }
                Err(e) => warn!("Error reading record: {}", e),
            }
        }
    }

    state.retain(|_, val| *val != MULTI);
    state
}

// ─── Streaming second-pass function ────────────────────────────────────────────

/// Re-read a file and call `callback(key, consensus, obs_value, info)` for each
/// extracted kv-mer whose key is present in `consensus_map`.
/// `read_id_counter` is incremented per read and persists across multiple calls
/// to maintain consistent read_ids when streaming multiple files.
pub fn stream_file_observations<F>(
    file: &str,
    c: usize,
    k: u8,
    v: u8,
    bidirectional: bool,
    trim_front: usize,
    trim_back: usize,
    consensus_map: &FxHashMap<u64, u64>,
    read_id_counter: &mut u64,
    mut callback: F,
) where
    F: FnMut(u64, u64, u64, &ValueInfo),
{
    if file.ends_with(".bam") || file.ends_with(".sam") {
        match bam::Reader::from_path(file) {
            Ok(mut reader) => {
                for record_result in reader.records() {
                    match record_result {
                        Ok(record) => {
                            let read_id = *read_id_counter;
                            *read_id_counter += 1;
                            let fid = fragment_id_from_name(record.qname());
                            let seq = record.seq().as_bytes();
                            let qual = record.qual().to_vec();
                            let (key_vec, value_vec, mut info_vec) = extract_kv_with_qual(
                                &seq, &qual, k, v, bidirectional, c, trim_front, trim_back, read_id,
                            );
                            set_fragment_ids(&mut info_vec, fid);
                            for ((key, obs_value), info) in
                                key_vec.iter().zip(value_vec.iter()).zip(info_vec.iter())
                            {
                                if let Some(&consensus) = consensus_map.get(key) {
                                    callback(*key, consensus, *obs_value, info);
                                }
                            }
                        }
                        Err(e) => warn!("Error reading BAM/SAM record: {}", e),
                    }
                }
            }
            Err(e) => error!(
                "{} is not a valid BAM/SAM file (Error: {}); skipping.",
                file, e
            ),
        }
    } else {
        let reader = parse_fastx_file(file);
        if reader.is_err() {
            error!("{} is not a valid fasta/fastq file; skipping.", file);
            return;
        }
        let mut reader = reader.unwrap();
        while let Some(record) = reader.next() {
            match record {
                Ok(record) => {
                    let read_id = *read_id_counter;
                    *read_id_counter += 1;
                    let fid = fragment_id_from_name(record.id());
                    let seq = record.seq();
                    let (key_vec, value_vec, mut info_vec) = if let Some(qual) = record.qual() {
                        extract_kv_with_qual(
                            &seq, qual, k, v, bidirectional, c, trim_front, trim_back, read_id,
                        )
                    } else {
                        extract_kv_no_qual(&seq, k, v, bidirectional, c, trim_front, trim_back, read_id)
                    };
                    set_fragment_ids(&mut info_vec, fid);
                    for ((key, obs_value), info) in
                        key_vec.iter().zip(value_vec.iter()).zip(info_vec.iter())
                    {
                        if let Some(&consensus) = consensus_map.get(key) {
                            callback(*key, consensus, *obs_value, info);
                        }
                    }
                }
                Err(e) => warn!("Error reading record: {}", e),
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

impl KVmerSet {
    pub fn new(key_size: u8, value_size: u8, bidirectional: bool) -> Self {
        assert!(
            key_size <= 32 && value_size <= 32,
            "Currently, we only support k, v <= 32."
        );

        let v_mask = (1 << (value_size * 2)) - 1;
        let k_mask = ((1 << (key_size * 2)) - 1) << (value_size * 2);

        KVmerSet {
            key_size,
            value_size,
            kv_size: key_size + value_size,
            num_kvmers: 0,
            key_value_qual_map: HashMap::new(),
            key_mask: k_mask,
            value_mask: v_mask,
            bidirectional,
            read_id_counter: 0,
        }
    }

    /// Record a batch of (key, value, ValueInfo) triples, optionally dropping
    /// observations whose key is not in
    /// `allowed_keys`. Used by `add_file_to_kvmer_set_filtered` to avoid
    /// materialising observations the caller will discard later.
    fn add_kv_qual_vector_internal(
        &mut self,
        key_vec: &[u64],
        value_vec: &[u64],
        info_vec: &[ValueInfo],
        allowed_keys: Option<&FxHashMap<u64, u64>>,
    ) {
        assert!(
            key_vec.len() == value_vec.len() && key_vec.len() == info_vec.len(),
            "Key, value, and info vectors must have the same length."
        );
        let mut inserted: u32 = 0;
        for ((&key, &value), info) in key_vec.iter().zip(value_vec.iter()).zip(info_vec.iter()) {
            if let Some(filter) = allowed_keys {
                if !filter.contains_key(&key) {
                    continue;
                }
            }
            self.key_value_qual_map
                .entry(key)
                .or_insert_with(HashMap::new)
                .entry(value)
                .or_insert_with(Vec::new)
                .push(info.clone());
            inserted += 1;
        }
        self.num_kvmers += inserted;
    }

    fn extract_markers_masked(
        &self,
        string: &[u8],
        key_vec: &mut Vec<u64>,
        value_vec: &mut Vec<u64>,
        c: usize,
        trim_front: usize,
        trim_back: usize,
        value_info_vec: &mut Vec<ValueInfo>,
        read_id: u64,
    ) {
        let start = std::cmp::min(trim_front, string.len());
        let end = string.len().saturating_sub(trim_back);
        let string_trimmed = &string[start..end];
        // extract sketched kv-mers from the given sequence string
        #[cfg(any(target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                use crate::avx2_seeding::*;
                unsafe {
                    extract_markers_avx2_masked(
                        string_trimmed,
                        key_vec,
                        value_vec,
                        value_info_vec,
                        c,
                        self.key_size as usize,
                        self.value_size as usize,
                        self.bidirectional,
                        read_id,
                    );
                }
            } else {
                fmh_seeds_masked(
                    string_trimmed,
                    key_vec,
                    value_vec,
                    value_info_vec,
                    c,
                    self.key_size as usize,
                    self.value_size as usize,
                    self.bidirectional,
                    read_id,
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            fmh_seeds_masked(
                string_trimmed,
                key_vec,
                value_vec,
                value_info_vec,
                c,
                self.key_size as usize,
                self.value_size as usize,
                self.bidirectional,
                read_id,
            );
        }
    }

    /// Like `extract_markers_masked`, but also extracts quality scores and builds `ValueInfo`.
    fn extract_markers_masked_with_qual(
        &self,
        string: &[u8],
        qual: &[u8],
        key_vec: &mut Vec<u64>,
        value_vec: &mut Vec<u64>,
        info_vec: &mut Vec<ValueInfo>,
        c: usize,
        trim_front: usize,
        trim_back: usize,
        read_id: u64,
    ) {
        let start = std::cmp::min(trim_front, string.len());
        let end = string.len().saturating_sub(trim_back);
        let string_trimmed = &string[start..end];
        let qual_trimmed = &qual[start..end];
        #[cfg(any(target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                use crate::avx2_seeding::*;
                unsafe {
                    extract_markers_avx2_masked_with_qual(
                        string_trimmed,
                        qual_trimmed,
                        key_vec,
                        value_vec,
                        info_vec,
                        c,
                        self.key_size as usize,
                        self.value_size as usize,
                        self.bidirectional,
                        read_id,
                    );
                }
            } else {
                fmh_seeds_masked_with_qual(
                    string_trimmed,
                    qual_trimmed,
                    key_vec,
                    value_vec,
                    info_vec,
                    c,
                    self.key_size as usize,
                    self.value_size as usize,
                    self.bidirectional,
                    read_id,
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            fmh_seeds_masked_with_qual(
                string_trimmed,
                qual_trimmed,
                key_vec,
                value_vec,
                info_vec,
                c,
                self.key_size as usize,
                self.value_size as usize,
                self.bidirectional,
                read_id,
            );
        }
    }

    // MODIFIED: Added BAM/SAM support
    pub fn add_file_to_kvmer_set(
        &mut self,
        seq_file: &str,
        c: usize,
        trim_front: usize,
        trim_back: usize,
    ) {
        self.add_file_to_kvmer_set_impl(seq_file, c, trim_front, trim_back, None);
    }

    /// Like `add_file_to_kvmer_set`, but only inserts observations whose key
    /// is present in `allowed_keys`. Use this when the consumer only cares about
    /// keys that appear in a reference consensus map — dropping non-matching
    /// observations at load time avoids materialising them in `key_value_qual_map`.
    pub fn add_file_to_kvmer_set_filtered(
        &mut self,
        seq_file: &str,
        c: usize,
        trim_front: usize,
        trim_back: usize,
        allowed_keys: &FxHashMap<u64, u64>,
    ) {
        self.add_file_to_kvmer_set_impl(seq_file, c, trim_front, trim_back, Some(allowed_keys));
    }

    fn add_file_to_kvmer_set_impl(
        &mut self,
        seq_file: &str,
        c: usize,
        trim_front: usize,
        trim_back: usize,
        allowed_keys: Option<&FxHashMap<u64, u64>>,
    ) {
        let seq_file_clone = seq_file.to_string();

        if seq_file_clone.ends_with(".bam") || seq_file_clone.ends_with(".sam") {
            match bam::Reader::from_path(&seq_file_clone) {
                Ok(mut reader) => {
                    if !self.bidirectional {
                        // [FIXME] Correct the coverage estimation when using forward strand only with BAM/SAM input files
                        warn!(
                            "Using --forward-only with BAM/SAM input files may make the estimation of true coverage inaccurate."
                        )
                    }
                    for record_result in reader.records() {
                        match record_result {
                            Ok(record) => {
                                let read_id = self.read_id_counter;
                                self.read_id_counter += 1;
                                let fid = fragment_id_from_name(record.qname());
                                let seq = record.seq().as_bytes();
                                let qual = record.qual().to_vec();
                                let mut key_vec: Vec<u64> = Vec::new();
                                let mut value_vec: Vec<u64> = Vec::new();
                                let mut info_vec: Vec<ValueInfo> = Vec::new();
                                self.extract_markers_masked_with_qual(
                                    &seq,
                                    &qual,
                                    &mut key_vec,
                                    &mut value_vec,
                                    &mut info_vec,
                                    c,
                                    trim_front,
                                    trim_back,
                                    read_id,
                                );
                                set_fragment_ids(&mut info_vec, fid);
                                self.add_kv_qual_vector_internal(
                                    &key_vec,
                                    &value_vec,
                                    &info_vec,
                                    allowed_keys,
                                );
                            }
                            Err(e) => warn!("Error reading BAM/SAM record: {}", e),
                        }
                    }
                }
                Err(e) => error!(
                    "{} is not a valid BAM/SAM file (Error: {}); skipping.",
                    seq_file_clone, e
                ),
            }
        } else {
            let reader = parse_fastx_file(&seq_file_clone);
            if !reader.is_ok() {
                error!(
                    "{} is not a valid fasta/fastq file; skipping.",
                    seq_file_clone
                );
                return;
            }
            let mut reader = reader.unwrap();
            while let Some(record) = reader.next() {
                match record {
                    Ok(record) => {
                        let read_id = self.read_id_counter;
                        self.read_id_counter += 1;
                        let fid = fragment_id_from_name(record.id());
                        let mut key_vec: Vec<u64> = Vec::new();
                        let mut value_vec: Vec<u64> = Vec::new();
                        if let Some(qual) = record.qual() {
                            // FASTQ: record quality scores alongside k,v-mers.
                            let mut info_vec: Vec<ValueInfo> = Vec::new();
                            self.extract_markers_masked_with_qual(
                                &record.seq(),
                                qual,
                                &mut key_vec,
                                &mut value_vec,
                                &mut info_vec,
                                c,
                                trim_front,
                                trim_back,
                                read_id,
                            );
                            set_fragment_ids(&mut info_vec, fid);
                            self.add_kv_qual_vector_internal(
                                &key_vec,
                                &value_vec,
                                &info_vec,
                                allowed_keys,
                            );
                        } else {
                            // FASTA: no quality scores; record position/strand but empty qual.
                            let mut info_vec: Vec<ValueInfo> = Vec::new();
                            self.extract_markers_masked(
                                &record.seq(),
                                &mut key_vec,
                                &mut value_vec,
                                c,
                                trim_front,
                                trim_back,
                                &mut info_vec,
                                read_id,
                            );
                            set_fragment_ids(&mut info_vec, fid);
                            self.add_kv_qual_vector_internal(
                                &key_vec,
                                &value_vec,
                                &info_vec,
                                allowed_keys,
                            );
                        }
                    }
                    Err(e) => warn!("Error reading record: {}", e),
                }
            }
        }
    }

    pub fn dump(&self, output_dir: &str) {
        let mut writer = BufWriter::new(
            File::create(&output_dir).expect(&format!("{} path not valid; exiting ", output_dir)),
        );
        writer
            .write_all(KVMER_MAGIC)
            .expect("Failed to write sketch magic bytes");
        bincode::serialize_into(&mut writer, &self).unwrap();
        info!("Sketching complete.");
    }

    pub fn load(&mut self, input_file: &str) {
        let file = File::open(input_file).expect(&format!(
            "The sketch `{}` could not be opened. Exiting",
            input_file
        ));
        let mut reader = BufReader::with_capacity(10_000_000, file);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic).unwrap_or_else(|_| {
            panic!(
                "The sketch `{}` is too short to be a valid sketch file. \
                 It may be a legacy file (pre-read_id). Please regenerate with `skiver sketch`.",
                input_file
            )
        });
        if &magic != KVMER_MAGIC {
            panic!(
                "The sketch `{}` has unrecognised magic bytes ({:?}). \
                 It was likely created by an older version of skiver. \
                 Please regenerate with `skiver sketch`.",
                input_file, magic
            );
        }
        let that: KVmerSet = bincode::deserialize_from(reader)
            .expect(&format!(
                "The sketch `{}` is not a valid sketch. It may be generated by an older version of skiver. Please regenerate the sketch with the current version of skiver.",
                &input_file
            ));

        // load the data into self
        if self.key_size != that.key_size || self.value_size != that.value_size {
            warn!(
                "Key size or value size does not match when loading KVmerSet from file. Skipping input file {}.",
                input_file
            );
        } else {
            for (kmer, value_map) in that.key_value_qual_map {
                let entry = self
                    .key_value_qual_map
                    .entry(kmer)
                    .or_insert_with(HashMap::new);
                for (value, info_list) in value_map {
                    entry
                        .entry(value)
                        .or_insert_with(Vec::new)
                        .extend(info_list);
                }
            }
            self.num_kvmers += that.num_kvmers;
        }
    }
}
