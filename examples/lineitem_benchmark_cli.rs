use clap::Parser;
use es::benchmark::{
    BenchmarkConfig, BenchmarkRunner, CsvInputProvider, DetailedCsvVerifier,
    print_benchmark_summary,
};
use std::path::PathBuf;
use std::fs::File;

#[derive(Parser)]
#[command(name = "lineitem_benchmark")]
#[command(about = "TPC-H Lineitem CSV Sort Benchmark with policy-based optimization")]
struct Args {
    /// Input CSV file path
    #[arg(short, long)]
    input: PathBuf,

    /// Key column indices (comma-separated)
    #[arg(short = 'k', long, default_value = "8,9,13,14,15")]
    key_columns: String,

    /// Value column indices (comma-separated)
    #[arg(short = 'v', long, default_value = "0,3,11")]
    value_columns: String,

    /// Max number of threads
    #[arg(short, long, default_value = "4")]
    threads: usize,

    /// Maximum total memory in MB
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

    /// CSV delimiter character
    #[arg(long, default_value = ",")]
    delimiter: char,

    /// CSV has headers
    #[arg(long, default_value = "true")]
    headers: bool,

    /// Verify sorted output
    #[arg(long, default_value = "false")]
    verify: bool,

    /// Number of benchmark runs per configuration
    #[arg(long, default_value = "1")]
    benchmark_runs: usize,

    /// Number of warmup runs before benchmarking
    #[arg(long, default_value = "0")]
    warmup_runs: usize,

    #[arg(long, default_value = "1.0")]
    boundary_imbalance_factor: f64,
}

fn parse_columns(column_str: &str) -> Vec<usize> {
    column_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Parse column indices
    let key_columns = parse_columns(&args.key_columns);
    let value_columns = parse_columns(&args.value_columns);

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
        boundary_imbalance_factor: args.boundary_imbalance_factor
    };

    // Create input provider for CSV files
    let input_provider = Box::new(CsvInputProvider::new(
        args.input,
        key_columns.clone(),
        value_columns,
        args.delimiter,
        args.headers,
    ));

    // Create benchmark runner
    let mut runner = BenchmarkRunner::new(config, input_provider);

    // Set up verification if requested
    if args.verify {
        let verifier = DetailedCsvVerifier::new(key_columns);
        runner.set_verifier(Box::new(verifier));
    }

    // Run all benchmarks
    let results = runner.run_benchmarks()?;

    // Print comprehensive summary (identical to original output)
    print_benchmark_summary(&results);

    Ok(())
}
