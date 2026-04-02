use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};

use glob::glob;
use log::{error, info, warn};
use simple_logger::SimpleLogger;

use crate::cmdline::DumpArgs;
use crate::inference::ErrorAnalyzer;
use crate::kvmer::KVmerSet;
use crate::types::{EditOperation, NeighborInfo, SEQ_TO_CHAR, ValueInfo};
use crate::utils::{_get_neighbors, _kmer_to_string, estimate_c_from_raw_files,
                   is_fastx_file, is_sketch_file};

// ─── Classification ────────────────────────────────────────────────────────────

enum ObsClass {
    Consensus,
    Neighbor(NeighborInfo),
    MultiEdit,
}

fn classify_value(
    obs_value: u64,
    consensus: u64,
    neighbors: &std::collections::HashMap<u64, NeighborInfo>,
) -> ObsClass {
    if obs_value == consensus {
        ObsClass::Consensus
    } else if let Some(&ni) = neighbors.get(&obs_value) {
        ObsClass::Neighbor(ni)
    } else {
        ObsClass::MultiEdit
    }
}

// ─── Bit helpers ───────────────────────────────────────────────────────────────

/// Extract the base (0=A, 1=C, 2=G, 3=T) at 1-based left-to-right position `t`
/// from a kmer encoded with the leftmost base at the MSB.
#[inline]
fn base_at(kmer: u64, t: u8, v: u8) -> u8 {
    ((kmer >> ((v - t) * 2)) & 0b11) as u8
}

/// Return the preceding base character for position `t` (1-based) in the value.
/// At t=1, the preceding base is the last base of the key.
/// At t>1, the preceding base is the consensus base at t-1.
#[inline]
fn prev_base_ch(key: u64, k: u8, consensus: u64, v: u8, t: u8) -> char {
    let idx = if t == 1 {
        base_at(key, k, k) as usize
    } else {
        base_at(consensus, t - 1, v) as usize
    };
    SEQ_TO_CHAR[idx]
}

/// Return the integer Phred score (qual byte minus 33) at 0-based index `idx`,
/// or -1 if quality data is absent or the index is out of range.
#[inline]
fn phred_at(info: &ValueInfo, idx: usize) -> i32 {
    if info.qual.is_empty() || idx >= info.qual.len() {
        -1
    } else {
        (info.qual[idx].saturating_sub(33)) as i32
    }
}

/// Scan left-to-right and return the 1-based position of the first mismatch
/// between `consensus` and `obs_value`. Returns 0 if no mismatch is found.
fn first_disagreement_t(consensus: u64, obs_value: u64, v: u8) -> u8 {
    for t in 1..=v {
        if base_at(consensus, t, v) != base_at(obs_value, t, v) {
            return t;
        }
    }
    0
}

fn is_substitution(op: EditOperation) -> bool {
    matches!(
        op,
        EditOperation::AC | EditOperation::AG | EditOperation::AT
        | EditOperation::CA | EditOperation::CG | EditOperation::CT
        | EditOperation::GA | EditOperation::GC | EditOperation::GT
        | EditOperation::TA | EditOperation::TC | EditOperation::TG
    )
}

fn is_insertion(op: EditOperation) -> bool {
    matches!(op, EditOperation::_A | EditOperation::_C | EditOperation::_G | EditOperation::_T)
}

fn is_deletion(op: EditOperation) -> bool {
    matches!(op, EditOperation::A_ | EditOperation::C_ | EditOperation::G_ | EditOperation::T_)
}

// ─── Row writers ───────────────────────────────────────────────────────────────

fn write_raw_row(
    writer: &mut Option<BufWriter<File>>,
    obs_id: u64,
    key_str: &str,
    consensus_str: &str,
    obs_value: u64,
    v: u8,
    class: &ObsClass,
    info: &ValueInfo,
    passes_filter: bool,
) {
    let w = match writer.as_mut() {
        Some(w) => w,
        None => return,
    };

    let obs_value_str = _kmer_to_string(obs_value, v);

    let (edit_distance, edit_op, edit_position): (&str, String, String) = match class {
        ObsClass::Consensus => ("0", "NA".into(), "NA".into()),
        ObsClass::Neighbor(ni) => (
            "1",
            format!("{}", ni.op),
            format!("{}", ni.position),
        ),
        ObsClass::MultiEdit => ("2+", "NA".into(), "NA".into()),
    };

    let qual_str: String = if info.qual.is_empty() {
        "NA".into()
    } else {
        String::from_utf8(info.qual.clone()).unwrap_or_else(|_| "NA".into())
    };

    writeln!(
        w,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        obs_id, key_str, consensus_str, obs_value_str,
        edit_distance, edit_op, edit_position, qual_str,
        info.start_index, info.dist_to_read_end,
        info.is_forward, passes_filter,
    )
    .unwrap();
}

/// Compute the absolute read position (0-based from read start) for the base
/// at 1-based value position `t`.
///
/// For forward reads: read_pos = start_index + (t - 1).
/// For reverse-complement reads: the value was RC'd before storage, so
/// `start_index` is the 0-based position of the first (leftmost) base of the
/// RC value in the trimmed read, and the absolute position of the t-th base
/// increases by (t - 1) in the *reverse* read direction.
/// We report positions as they appear in the stored (possibly RC'd) value.
#[inline]
fn read_pos(info: &ValueInfo, t: u8) -> i64 {
    info.start_index as i64 + (t - 1) as i64
}

fn write_base_rows(
    writer: &mut Option<BufWriter<File>>,
    obs_id: u64,
    k: u8,
    v: u8,
    key: u64,
    consensus: u64,
    obs_value: u64,
    class: &ObsClass,
    info: &ValueInfo,
    passes_filter: bool,
) {
    let w = match writer.as_mut() {
        Some(w) => w,
        None => return,
    };

    match class {
        // 2+ edit distance or AMBIGUOUS: alignment is ambiguous, skip.
        ObsClass::MultiEdit => return,
        ObsClass::Neighbor(ni) if ni.op == EditOperation::AMBIGUOUS => return,

        // ── 0-edit: all positions match ──────────────────────────────────────
        ObsClass::Consensus => {
            for t in 1..=v {
                let b = base_at(consensus, t, v) as usize;
                let ch = SEQ_TO_CHAR[b];
                writeln!(
                    w,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    obs_id, t, ch, ch,
                    prev_base_ch(key, k, consensus, v, t),
                    "NA",
                    phred_at(info, (t - 1) as usize),
                    read_pos(info, t),
                    info.dist_to_read_end,
                    info.is_forward,
                    passes_filter,
                )
                .unwrap();
            }
        }

        ObsClass::Neighbor(ni) => {
            // NeighborInfo::position is 0-based from the LSB (rightmost base).
            // Convert to 1-based left-to-right: edit_t = v - position.
            let edit_t = v - ni.position;
            let op = ni.op;

            if is_substitution(op) {
                // ── Substitution ─────────────────────────────────────────────
                // All positions i != edit_t are matches; position edit_t differs.
                for t in 1..=v {
                    let true_b  = base_at(consensus, t, v) as usize;
                    let obs_b   = base_at(obs_value,  t, v) as usize;
                    let edit_op = if t == edit_t { format!("{}", op) } else { "NA".into() };
                    writeln!(
                        w,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        obs_id, t,
                        SEQ_TO_CHAR[true_b], SEQ_TO_CHAR[obs_b],
                        prev_base_ch(key, k, consensus, v, t),
                        edit_op,
                        phred_at(info, (t - 1) as usize),
                        read_pos(info, t),
                        info.dist_to_read_end,
                        info.is_forward,
                        passes_filter,
                    )
                    .unwrap();
                }

            } else if is_insertion(op) {
                // ── Insertion ────────────────────────────────────────────────
                // An extra base was inserted in the READ at left-to-right
                // position edit_t, causing the last consensus base to be dropped
                // from the v-length window.
                //
                // Alignment:
                //   t < edit_t  : cons[t]   vs obs[t]           (match)
                //   t = edit_t  : '-'        vs obs[t]           (insertion)
                //   t > edit_t  : cons[t-1]  vs obs[t]           (shifted match)
                for t in 1..=v {
                    let (true_ch, obs_ch, edit_op_str, phred) = if t < edit_t {
                        let b = base_at(consensus, t, v) as usize;
                        (SEQ_TO_CHAR[b], SEQ_TO_CHAR[b], "NA".into(),
                         phred_at(info, (t - 1) as usize))
                    } else if t == edit_t {
                        let obs_b = base_at(obs_value, t, v) as usize;
                        ('-', SEQ_TO_CHAR[obs_b], format!("{}", op),
                         phred_at(info, (t - 1) as usize))
                    } else {
                        // t > edit_t: true base is cons[t-1], obs base is obs[t]
                        let true_b = base_at(consensus, t - 1, v) as usize;
                        let obs_b  = base_at(obs_value, t, v) as usize;
                        (SEQ_TO_CHAR[true_b], SEQ_TO_CHAR[obs_b], "NA".into(),
                         phred_at(info, (t - 1) as usize))
                    };
                    writeln!(
                        w,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        obs_id, t, true_ch, obs_ch,
                        prev_base_ch(key, k, consensus, v, t),
                        edit_op_str,
                        phred,
                        read_pos(info, t),
                        info.dist_to_read_end,
                        info.is_forward,
                        passes_filter,
                    )
                    .unwrap();
                }

            } else if is_deletion(op) {
                // ── Deletion ─────────────────────────────────────────────────
                // A base was deleted from the READ at left-to-right position
                // edit_t. The v-length window then captures one extra base at
                // the end (appended unknown base at neighbor position v).
                //
                // Alignment (we emit v rows, not v+1):
                //   t < edit_t  : cons[t]   vs obs[t]            (match, qual[t-1])
                //   t = edit_t  : cons[t]   vs '-'               (deletion, phred=-1)
                //   t > edit_t  : cons[t]   vs obs[t-1]          (shifted match, qual[t-2])
                //   (obs[v] = appended unknown base; not emitted)
                for t in 1..=v {
                    let (true_ch, obs_ch, edit_op_str, phred) = if t < edit_t {
                        let b = base_at(consensus, t, v) as usize;
                        (SEQ_TO_CHAR[b], SEQ_TO_CHAR[b], "NA".into(),
                         phred_at(info, (t - 1) as usize))
                    } else if t == edit_t {
                        let true_b = base_at(consensus, t, v) as usize;
                        (SEQ_TO_CHAR[true_b], '-', format!("{}", op), -1i32)
                    } else {
                        // t > edit_t: obs base is at neighbor position t-1 (1-based),
                        // qual index is t-2 (0-based).
                        let true_b = base_at(consensus, t, v) as usize;
                        let obs_b  = base_at(obs_value, t - 1, v) as usize;
                        (SEQ_TO_CHAR[true_b], SEQ_TO_CHAR[obs_b], "NA".into(),
                         phred_at(info, (t - 2) as usize))
                    };
                    writeln!(
                        w,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        obs_id, t, true_ch, obs_ch,
                        prev_base_ch(key, k, consensus, v, t),
                        edit_op_str,
                        phred,
                        read_pos(info, t),
                        info.dist_to_read_end,
                        info.is_forward,
                        passes_filter,
                    )
                    .unwrap();
                }
            }
            // AMBIGUOUS already handled above.
        }
    }
}

fn write_survival_row(
    writer: &mut Option<BufWriter<File>>,
    obs_id: u64,
    key_str: &str,
    v: u8,
    consensus: u64,
    obs_value: u64,
    class: &ObsClass,
    info: &ValueInfo,
    passes_filter: bool,
) {
    let w = match writer.as_mut() {
        Some(w) => w,
        None => return,
    };

    let (first_error_t, censored) = match class {
        ObsClass::Consensus => (0u8, true),
        ObsClass::Neighbor(ni) => {
            // Convert 0-based LSB position to 1-based left-to-right.
            let t = v - ni.position;
            (t, false)
        }
        ObsClass::MultiEdit => {
            let t = first_disagreement_t(consensus, obs_value, v);
            (t, false)
        }
    };

    writeln!(
        w,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        obs_id, key_str, first_error_t, censored,
        info.start_index, info.dist_to_read_end,
        info.is_forward, passes_filter,
    )
    .unwrap();
}

// ─── Entry point ───────────────────────────────────────────────────────────────

pub fn dump(args: DumpArgs) {
    SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()
        .unwrap();

    let prefix = match &args.analyze.output_prefix {
        Some(p) => p.clone(),
        None => {
            error!("No output prefix provided. Use -o to specify the output prefix.");
            return;
        }
    };

    if !args.raw && !args.base && !args.survival {
        error!("No output format selected. Specify at least one of --raw, --base, --survival.");
        return;
    }

    // ── Expand input globs ────────────────────────────────────────────────────
    let mut raw_files: Vec<String> = Vec::new();
    let mut sketch_files: Vec<String> = Vec::new();
    for pattern in &args.analyze.files {
        for entry in glob(pattern).expect("Failed to read glob pattern") {
            match entry {
                Ok(path) => {
                    let s = path.to_str().unwrap().to_string();
                    if is_fastx_file(&s) {
                        raw_files.push(s);
                    } else if is_sketch_file(&s) {
                        sketch_files.push(s);
                    } else {
                        warn!("Unrecognized file format: {}. Skipping.", s);
                    }
                }
                Err(e) => warn!("Glob error: {:?}", e),
            }
        }
    }

    // ── Subsampling rate ──────────────────────────────────────────────────────
    let c = args.analyze.c.unwrap_or_else(|| {
        let refs: Vec<&str> = raw_files.iter().map(|s| s.as_str()).collect();
        let (auto_c, est_size) = estimate_c_from_raw_files(&refs);
        info!(
            "Estimated input size: {:.2} GB. Auto-determined subsampling rate: -c {}",
            est_size as f64 / (1u64 << 30) as f64,
            auto_c
        );
        auto_c
    });

    // ── Build KVmerSet ────────────────────────────────────────────────────────
    let mut kvmer_set = KVmerSet::new(
        args.analyze.k,
        args.analyze.v,
        !args.analyze.forward_only,
    );
    info!("Processing input files...");
    for f in &raw_files {
        kvmer_set.add_file_to_kvmer_set(f, c, args.analyze.trim_front, args.analyze.trim_back);
    }
    for f in &sketch_files {
        kvmer_set.load(f);
    }
    info!("Finished loading data.");

    // ── Compute stats and run outlier filter ──────────────────────────────────
    let lower_bound = args.analyze.lower_bound.unwrap_or(10);
    let stats = kvmer_set.get_stats(lower_bound, false);

    let inlier_indices: Vec<usize> = if args.analyze.use_all {
        (0..stats.keys.len()).collect()
    } else {
        let analyzer = ErrorAnalyzer::new(args.analyze.clone());
        analyzer.find_hazard_ratio_outliers(&stats)
    };

    let inlier_set: HashSet<usize> = inlier_indices.into_iter().collect();

    // ── Open output writers ───────────────────────────────────────────────────
    let open = |suffix: &str| -> BufWriter<File> {
        let path = format!("{}{}", prefix, suffix);
        BufWriter::with_capacity(
            1 << 20,
            File::create(&path).unwrap_or_else(|e| panic!("Cannot create {}: {}", path, e)),
        )
    };

    let mut raw_writer: Option<BufWriter<File>> = if args.raw {
        Some(open(".raw_observations.tsv"))
    } else {
        None
    };
    let mut base_writer: Option<BufWriter<File>> = if args.base {
        Some(open(".base_observations.tsv"))
    } else {
        None
    };
    let mut surv_writer: Option<BufWriter<File>> = if args.survival {
        Some(open(".survival_observations.tsv"))
    } else {
        None
    };

    // ── Write headers ─────────────────────────────────────────────────────────
    if let Some(ref mut w) = raw_writer {
        writeln!(
            w,
            "obs_id\tkey_str\tconsensus_str\tobs_value_str\tedit_distance\t\
             edit_op\tedit_position\tqual_str\tstart_index\tdist_to_read_end\t\
             is_forward\tpasses_filter"
        )
        .unwrap();
    }
    if let Some(ref mut w) = base_writer {
        writeln!(
            w,
            "obs_id\tt\ttrue_base\tobs_base\tprev_base\tedit_op\tphred\tread_pos\t\
             dist_to_end\tis_forward\tpasses_filter"
        )
        .unwrap();
    }
    if let Some(ref mut w) = surv_writer {
        writeln!(
            w,
            "obs_id\tkey_str\tfirst_error_t\tcensored\tstart_index\t\
             dist_to_read_end\tis_forward\tpasses_filter"
        )
        .unwrap();
    }

    // ── Main iteration ────────────────────────────────────────────────────────
    // Iterate over keys in the same order as stats.keys (which were populated
    // from key_value_qual_map during get_stats). We use the key hash to look up
    // the raw observation list directly.
    let v = args.analyze.v;
    let k = args.analyze.k;
    let bidirectional = !args.analyze.forward_only;

    let mut obs_id: u64 = 0;

    for (key_idx, (&key, &consensus)) in stats
        .keys
        .iter()
        .zip(stats.consensus_values.iter())
        .enumerate()
    {
        let passes_filter = inlier_set.contains(&key_idx);
        let key_str = _kmer_to_string(key, k);
        let consensus_str = _kmer_to_string(consensus, v);

        let value_map = match kvmer_set.key_value_qual_map.get(&key) {
            Some(m) => m,
            None => continue,
        };

        // Build the neighbor map once per key (reused for all values of this key).
        let neighbors = _get_neighbors(consensus, v, bidirectional);

        for (&obs_value, info_list) in value_map {
            let class = classify_value(obs_value, consensus, &neighbors);

            for info in info_list {
                write_raw_row(
                    &mut raw_writer,
                    obs_id,
                    &key_str,
                    &consensus_str,
                    obs_value,
                    v,
                    &class,
                    info,
                    passes_filter,
                );
                write_base_rows(
                    &mut base_writer,
                    obs_id,
                    k,
                    v,
                    key,
                    consensus,
                    obs_value,
                    &class,
                    info,
                    passes_filter,
                );
                write_survival_row(
                    &mut surv_writer,
                    obs_id,
                    &key_str,
                    v,
                    consensus,
                    obs_value,
                    &class,
                    info,
                    passes_filter,
                );

                obs_id += 1;
            }
        }
    }

    info!("Wrote {} total observations.", obs_id);
}
