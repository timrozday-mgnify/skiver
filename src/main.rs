use clap::Parser;

use skiver::analyze;
use skiver::cmdline::*;
use skiver::dump;
use skiver::mapping;
use skiver::sketch;

//Use this allocator when statically compiling
//instead of the default
//because the musl statically compiled binary
//uses a bad default allocator which makes the
//binary take 60% longer!!! Only affects
//static compilation though.
#[cfg(all(target_env = "musl", not(feature = "dhat-heap")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Heap profiling: when built with `--features dhat-heap`, dhat replaces the
// global allocator and records every allocation, writing dhat-heap.json on
// exit. View at https://nnethercote.github.io/dh_view/dh_view.html and sort by
// "At t-gmax (bytes)" to find peak-memory culprits.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _dhat_profiler = dhat::Profiler::new_heap();

    let cli = Cli::parse();
    match cli.mode {
        Mode::Sketch(sketch_args) => sketch::sketch(sketch_args),
        Mode::Analyze(analyze_args) => analyze::analyze(analyze_args),
        Mode::Dump(dump_args) => dump::dump(dump_args),
        Mode::Map(map_args) => mapping::map(map_args),
    }
}
