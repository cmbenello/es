use clap::Parser;
use es::benchmark::{
    BenchmarkConfig, BenchmarkRunner, GenSortInputProvider, SimpleVerifier, print_benchmark_summary,
};
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser)]
struct SortArgs {
    /// Input GenSort file path
    #[arg(short, long)]
    input: PathBuf,

    /// Max number of threads
    #[arg(short, long, default_value = "4")]
    threads: usize,

    /// Maximum total memory
    #[arg(short, long, default_value = "1024")]
    memory_mb: usize,

    /// Sketch size for quantile estimation (KLL)
    #[arg(long, default_value = "200")]
    sketch_size: usize,

    /// Sketch sampling interval (update the sketch every N records of the run)
    #[arg(long, default_value = "1000")]
    sketch_sampling_interval: usize,

    /// Run indexing interval (store every Nth key in the run index)
    #[arg(long, default_value = "1000")]
    run_indexing_interval: usize,

    /// Use OVC encoding for keys
    #[arg(long, default_value = "false")]
    ovc: bool,

    /// Directory for temporary files
    #[arg(short, long, default_value = ".")]
    dir: PathBuf,

    /// Verify sorted output
    #[arg(short, long)]
    verify: bool,

    /// Number of benchmark runs per configuration
    #[arg(long, default_value = "1")]
    benchmark_runs: usize,

    /// Number of warmup runs before benchmarking (not included in results)
    #[arg(long, default_value = "0")]
    warmup_runs: usize,

    #[arg(long, default_value = "1.0")]
    boundary_imbalance_factor: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = SortArgs::parse();

    // Create benchmark configuration
    let config = BenchmarkConfig {
        threads: args.threads,
        memory_mb: args.memory_mb,
        warmup_runs: args.warmup_runs,
        benchmark_runs: args.benchmark_runs,
        verify: args.verify,
        ovc: args.ovc,
        temp_dir: args.dir,
        sketch_size: args.sketch_size,
        sketch_sampling_interval: args.sketch_sampling_interval,
        run_indexing_interval: args.run_indexing_interval,
        boundary_imbalance_factor: args.boundary_imbalance_factor,
    };

    // Create input provider for GenSort files
    let input_provider = Box::new(GenSortInputProvider { path: args.input });

    // Create benchmark runner
    let mut runner = BenchmarkRunner::new(config, input_provider);

    // Set up verification if requested
    if args.verify {
        let verifier = SimpleVerifier::new(); // With sample output
        runner.set_verifier(Box::new(verifier));
    }

    // Run all benchmarks
    let results = runner.run_benchmarks()?;

    // Print comprehensive summary (identical to original output)
    print_benchmark_summary(&results);

    Ok(())
}
