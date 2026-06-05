//! End-to-end golden tests for the `analyze` and `dump` subcommands.
//!
//! Runs the built binary on a small committed FASTQ fixture and asserts the
//! outputs match committed golden files byte-for-byte. Output is deterministic,
//! so any unintended change to the analyze/dump code paths is caught.
//!
//! Regenerate goldens (only after an intentional, reviewed change) with:
//!   target/release/skiver analyze tests/fixtures/reads.fastq -k 5 -v 4 -c 1 -e 1e-3 \
//!       -o tests/fixtures/golden/analyze
//!   target/release/skiver dump tests/fixtures/reads.fastq --base --use-all -k 5 -v 4 -c 1 \
//!       -o tests/fixtures/golden/dump_useall
//!   target/release/skiver dump tests/fixtures/reads.fastq --base -k 5 -v 4 -c 1 \
//!       -o tests/fixtures/golden/dump_filter

use std::path::{Path, PathBuf};
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_skiver");

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// A throwaway output directory unique to this process; removed on drop.
struct TempOut(PathBuf);
impl TempOut {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("skiver_golden_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        TempOut(dir)
    }
    fn prefix(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_string()
    }
}
impl Drop for TempOut {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(args: &[&str]) {
    let status = Command::new(BIN)
        .args(args)
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {BIN}: {e}"));
    assert!(status.success(), "command failed: skiver {}", args.join(" "));
}

fn assert_matches_golden(produced: &str, golden_name: &str) {
    let golden = fixtures().join("golden").join(golden_name);
    let got = std::fs::read_to_string(produced)
        .unwrap_or_else(|e| panic!("missing produced output {produced}: {e}"));
    let want = std::fs::read_to_string(&golden)
        .unwrap_or_else(|e| panic!("missing golden {}: {e}", golden.display()));
    assert_eq!(got, want, "output {produced} differs from golden {golden_name}");
}

#[test]
fn analyze_no_reference_matches_golden() {
    let out = TempOut::new("analyze");
    let reads = fixtures().join("reads.fastq");
    let prefix = out.prefix("analyze");
    run(&[
        "analyze", reads.to_str().unwrap(),
        "-k", "5", "-v", "4", "-c", "1", "-e", "1e-3", "-o", &prefix,
    ]);
    // Per-key Weibull params and the error spectrum exercise
    // build_full_stats_no_reference + ErrorSummary + the inference fit.
    // NB: summary_error_rate.csv is intentionally NOT golden-compared — its
    // bootstrap percentile columns use an unseeded RNG and vary run-to-run.
    assert_matches_golden(&format!("{prefix}.kvmer.csv"), "analyze.kvmer.csv");
    assert_matches_golden(&format!("{prefix}.summary_error_spectrum.csv"), "analyze.summary_error_spectrum.csv");
}

#[test]
fn dump_base_useall_matches_golden() {
    let out = TempOut::new("dump_useall");
    let reads = fixtures().join("reads.fastq");
    let prefix = out.prefix("dump_useall");
    run(&[
        "dump", reads.to_str().unwrap(),
        "--base", "--use-all", "-k", "5", "-v", "4", "-c", "1", "-o", &prefix,
    ]);
    assert_matches_golden(&format!("{prefix}.base_observations.tsv"), "dump_useall.base_observations.tsv");
}

#[test]
fn dump_base_filter_matches_golden() {
    let out = TempOut::new("dump_filter");
    let reads = fixtures().join("reads.fastq");
    let prefix = out.prefix("dump_filter");
    run(&[
        "dump", reads.to_str().unwrap(),
        "--base", "-k", "5", "-v", "4", "-c", "1", "-o", &prefix,
    ]);
    assert_matches_golden(&format!("{prefix}.base_observations.tsv"), "dump_filter.base_observations.tsv");
}
