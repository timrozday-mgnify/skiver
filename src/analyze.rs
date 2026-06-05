use crate::cmdline::AnalyzeArgs;
use crate::inference::*;
use crate::kvmer::*;
use crate::utils::*;

use glob::glob;
use log::{error, info, warn};
use simple_logger::SimpleLogger;
use rustc_hash::FxHashMap;
use std::fs;

pub fn analyze(args: AnalyzeArgs) {
    SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()
        .unwrap();
    // [TODO] Multithreaded version is under development.

    // Expand globs and categorize files before processing so we can auto-determine -c
    let mut raw_files: Vec<String> = Vec::new();
    let mut sketch_files: Vec<String> = Vec::new();
    for file in &args.files {
        for entry in glob(file).expect("Failed to read glob pattern") {
            match entry {
                Ok(path) => {
                    let file_str = path.to_str().unwrap().to_string();
                    if is_fastx_file(&file_str) {
                        raw_files.push(file_str);
                    } else if is_sketch_file(&file_str) {
                        sketch_files.push(file_str);
                    } else {
                        warn!(
                            "File format not recognized for file: {}. Skipping.",
                            file_str
                        );
                    }
                }
                Err(e) => warn!("Error reading file: {:?}", e),
            }
        }
    }

    let c = args.c.unwrap_or_else(|| {
        let raw_refs: Vec<&str> = raw_files.iter().map(|s| s.as_str()).collect();
        let (auto_c, est_file_size) = estimate_c_from_raw_files(&raw_refs);
        info!(
            "Total estimated input sequence file size (decompressed): {:.2} GB",
            est_file_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        info!("Auto-determined subsampling rate: -c {}", auto_c);
        auto_c
    });

    // Build the reference consensus FIRST so the query loader can filter
    // observations to only keys present in the reference.
    let ref_consensus: Option<FxHashMap<u64, u64>> = if let Some(reference) = &args.reference {
        if args.lower_bound.is_none() {
            info!("Reference is provided. Using default lower bound of 0.");
        }
        info!("Loading reference (file={})…", reference);
        let map = build_reference_consensus_from_file(
            reference,
            args.k,
            args.v,
            true,
            c,
            args.trim_front,
            args.trim_back,
        );
        info!("ref_consensus built: {} keys retained", map.len());
        Some(map)
    } else {
        None
    };

    let analyzer = ErrorAnalyzer::new(args.clone());

    let stats: KVmerStats = if let Some(rc) = &ref_consensus {
        // Reference-guided path: accumulate only (key,value) counts — no ValueInfo.
        // This reduces peak RSS from ~60 GB to ~8-10 GB for typical configurations.
        if !sketch_files.is_empty() {
            warn!("Sketch files (.kvmer) are not supported in reference-guided mode and will be skipped.");
        }
        let lower_bound = args.lower_bound.unwrap_or(0);
        // Reference-guided: only keys present in the reference are kept, so the
        // reference key count is a safe upper bound on distinct keys — pre-size to it.
        let mut kv_counts =
            KVmerCountMap::with_capacity(args.k, args.v, !args.forward_only, rc.len());
        info!("Processing query files...");
        for file_str in &raw_files {
            kv_counts.add_file_filtered(file_str, c, args.trim_front, args.trim_back, rc);
        }
        info!("Finished processing query files.");
        info!("Building stats from {} (key,value) pairs...", kv_counts.counts.len());
        kv_counts.build_full_stats(lower_bound, rc)
    } else {
        // Non-reference path: flat count map eliminates ValueInfo (~30× less memory).
        let lower_bound = args.lower_bound.unwrap_or(10);
        // Non-reference: no safe a-priori bound on distinct kv-mers, so don't
        // pre-size (an input-size estimate proved unreliable — both under- and
        // over-shooting). The big analyze win comes from the allocation-free
        // per-key grouping in `build_full_stats_*`, not from reserving here.
        let mut kv_counts = KVmerCountMap::new(args.k, args.v, !args.forward_only);
        info!("Processing query files...");
        for file_str in &raw_files {
            kv_counts.add_file(file_str, c, args.trim_front, args.trim_back);
        }
        for file_str in &sketch_files {
            let mut temp = KVmerSet::new(args.k, args.v, !args.forward_only);
            temp.load(file_str);
            kv_counts.merge_from_kvmer_set(&temp);
        }
        info!("Finished processing query files.");
        info!("Building stats from {} (key,value) pairs...", kv_counts.counts.len());
        kv_counts.build_full_stats_no_reference(lower_bound)
    };
    // if reference is set, the filter should be disabled
    // [FIXME] enable --use-all by default
    if args.reference.is_some() && !args.use_all {
        warn!("If reference is provided, --use-all is recommended.");
    }

    let spectrum = analyzer.analyze(&stats);
    let analysis_output = format!(
        "{}\n{}",
        header_str(),
        spectrum_to_str(&spectrum, !args.forward_only)
    );

    if let Some(prefix) = &args.output_prefix {
        fs::write(
            format!("{}.summary_error_rate.csv", prefix),
            &analysis_output,
        )
        .unwrap();
        info!("Output written to prefix {}.", prefix);
    } else {
        error!(
            "No output prefix provided. Use -o or --output-prefix to specify the output file prefix for the analysis results."
        );
    }
}
