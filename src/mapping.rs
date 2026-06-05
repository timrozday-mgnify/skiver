use crate::{seeding::*, types::*};
use log::{info, warn};
use needletail::parse_fastx_file;
use rust_htslib::{bam, bam::Read as BamRead}; // Added rust-htslib for BAM/SAM support
use serde::{Deserialize, Serialize};
use simple_logger::SimpleLogger;
use std::collections::HashMap;

pub fn map(args: crate::cmdline::MapArgs) {
    SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()
        .unwrap();
    let mut kmer_set = KmerSet::new(args.k, true);
    kmer_set.add_file_to_kmer_set(&args.reference, args.c, args.trim_front, args.trim_back);
    info!("Loaded reference file: {}", args.reference);

    let mut total_matched: u32 = 0;
    let mut total_kmers: u32 = 0;

    info!("Processing query files...");
    for file in &args.files {
        let (matched, total) = kmer_set.query_file(
            file,
            args.c,
            args.lower_bound,
            args.sample_rate,
            !args.forward_only,
            args.print_verbose,
            args.trim_front,
            args.trim_back,
        );
        total_matched += matched;
        total_kmers += total;
    }
    info!("Finished processing query files.");
    info!(
        "At k={}: estimated overall kmer match rate: {}/{} = {:.4}%",
        args.k,
        total_matched,
        total_kmers,
        total_matched as f64 / total_kmers as f64 * 100.
    );
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct KmerSet {
    pub key_size: u8,
    pub num_kmers: u32,
    pub kmer_map: HashMap<u64, u32>,
    // whether both forward and reverse complement of the reads are included
    bidirectional: bool,
}

/**
 * Utility class for testing read accuracy
 * used only for testing purposes
 */
impl KmerSet {
    pub fn new(key_size: u8, bidirectional: bool) -> Self {
        assert!(key_size <= 32, "Currently, we only support k <= 32.");
        KmerSet {
            key_size,
            num_kmers: 0,
            kmer_map: HashMap::new(),
            bidirectional,
        }
    }

    pub fn add_seed_vector(&mut self, seed_vec: &[u64]) {
        for &kmer in seed_vec {
            let entry = self.kmer_map.entry(kmer).or_insert(0);
            *entry += 1;
        }
        self.num_kmers += seed_vec.len() as u32;
    }

    pub fn query_seed_vector(&self, seed_vec: &[u64]) -> (u32, u32) {
        let mut count: u32 = 0;
        for &kmer in seed_vec {
            if self.kmer_map.contains_key(&kmer) {
                count += 1;
            }
        }
        (count, seed_vec.len() as u32)
    }

    /// Extract subsampled k-mer keys from `string` into `kmer_vec`.
    /// `KmerSet` is key-only (value length 0), so the seeding routines' value and
    /// `ValueInfo` outputs are discarded into local scratch buffers.
    fn extract_markers_masked(
        &self,
        string: &[u8],
        kmer_vec: &mut Vec<u64>,
        c: usize,
        bidirectional: bool,
        trim_front: usize,
        trim_back: usize,
    ) {
        let start = std::cmp::min(trim_front, string.len());
        let end = string.len().saturating_sub(trim_back);

        // KmerSet keeps only keys; discard value / ValueInfo outputs.
        let mut value_scratch: Vec<u64> = Vec::new();
        let mut value_info_scratch: Vec<ValueInfo> = Vec::new();
        #[cfg(any(target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                use crate::avx2_seeding::*;
                unsafe {
                    extract_markers_avx2_masked(
                        &string[start..end],
                        kmer_vec,
                        &mut value_scratch,
                        &mut value_info_scratch,
                        c,
                        self.key_size as usize,
                        0,
                        bidirectional,
                        0,
                    );
                }
                return;
            }
        }
        fmh_seeds_masked(
            &string[start..end],
            kmer_vec,
            &mut value_scratch,
            &mut value_info_scratch,
            c,
            self.key_size as usize,
            0,
            bidirectional,
            0,
        );
    }

    pub fn add_file_to_kmer_set(
        &mut self,
        seq_file: &str,
        c: usize,
        trim_front: usize,
        trim_back: usize,
    ) {
        let bidirectional = self.bidirectional;
        for_each_read_seq(seq_file, |seq| {
            let mut kmer_vec: Vec<u64> = Vec::new();
            self.extract_markers_masked(seq, &mut kmer_vec, c, bidirectional, trim_front, trim_back);
            self.add_seed_vector(&kmer_vec);
        });
    }

    /**
     * Estimate the kmer match rate for the given read file.
     * The rate is calculated by the average kmer match rate across all reads,
     * excluding reads that have zero matched kmers.
     */
    pub fn query_file(
        &self,
        seq_file: &str,
        c: usize,
        threshold: u32,
        sample_per_num_read: usize,
        bidirectional: bool,
        print_verbose: bool,
        trim_front: usize,
        trim_back: usize,
    ) -> (u32, u32) {
        let mut matched_kmers: u32 = 0;
        let mut total_kmers: u32 = 0;
        let mut read_count: usize = 0;

        if print_verbose {
            println!("total,matched");
        }

        for_each_read_seq(seq_file, |seq| {
            read_count += 1;
            if read_count % sample_per_num_read != 0 {
                return;
            }
            let mut kmer_vec: Vec<u64> = Vec::new();
            self.extract_markers_masked(seq, &mut kmer_vec, c, bidirectional, trim_front, trim_back);

            let (matched, total) = self.query_seed_vector(&kmer_vec);
            if print_verbose {
                println!("{},{}", total, matched);
            }
            if matched >= threshold {
                matched_kmers += matched;
                total_kmers += total;
            }
        });

        (matched_kmers, total_kmers)
    }
}

/// Invoke `f` with each read's sequence bytes from a FASTA/FASTQ/BAM/SAM file,
/// logging (and skipping) unreadable files or records. Shared by
/// [`KmerSet::add_file_to_kmer_set`] and [`KmerSet::query_file`].
fn for_each_read_seq<F: FnMut(&[u8])>(seq_file: &str, mut f: F) {
    if seq_file.ends_with(".bam") || seq_file.ends_with(".sam") {
        match bam::Reader::from_path(seq_file) {
            Ok(mut reader) => {
                for record_result in reader.records() {
                    match record_result {
                        Ok(record) => f(&record.seq().as_bytes()),
                        Err(e) => warn!("Error reading BAM/SAM record: {}", e),
                    }
                }
            }
            Err(e) => warn!(
                "{} is not a valid BAM/SAM file (Error: {}); skipping.",
                seq_file, e
            ),
        }
    } else {
        match parse_fastx_file(seq_file) {
            Ok(mut reader) => {
                while let Some(record) = reader.next() {
                    match record {
                        Ok(record) => f(record.seq().as_ref()),
                        Err(e) => warn!("Error reading Fastx record: {}", e),
                    }
                }
            }
            Err(_) => warn!("{} is not a valid fasta/fastq file; skipping.", seq_file),
        }
    }
}
