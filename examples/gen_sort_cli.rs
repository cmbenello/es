use clap::{ArgAction, Parser};
use es::benchmark::{
    BenchmarkConfig, BenchmarkInputProvider, BenchmarkResult, BenchmarkRunner,
    GenSortInputProvider, SimpleVerifier, print_benchmark_summary,
};
use es::diskio::constants::DEFAULT_BUFFER_SIZE;
use es::sort::core::engine::PartitionType;
use std::path::PathBuf;

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum PartitionArg {
    KeyOnly,
    CountBalanced,
    SizeBalanced,
}

impl From<PartitionArg> for PartitionType {
    fn from(value: PartitionArg) -> Self {
        match value {
            PartitionArg::KeyOnly => PartitionType::KeyOnly,
            PartitionArg::CountBalanced => PartitionType::CountBalanced,
            PartitionArg::SizeBalanced => PartitionType::SizeBalanced,
        }
    }
}

#[derive(Parser)]
struct SortArgs {
    /// Configuration name for labeling the run
    #[arg(short, default_value = "no_name")]
    name: String,
    /// Input GenSort file path
    #[arg(short, long)]
    input: PathBuf,

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

    /// Cooldown seconds between runs
    #[arg(long, default_value = "0")]
    cooldown_seconds: u64,

    /// Use OVC (Offset Value Coding) format
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    ovc: bool,

    /// Threads for run generation
    #[arg(long, required_unless_present = "estimate_size")]
    run_gen_threads: Option<usize>,

    /// Threads for merge phase
    #[arg(long, required_unless_present = "estimate_size")]
    merge_threads: Option<usize>,

    /// Run size for run generation (MB)
    #[arg(long, required_unless_present = "estimate_size")]
    run_size_mb: Option<f64>,

    /// Merge fan-in (per-thread)
    #[arg(long, required_unless_present = "estimate_size")]
    merge_fanin: Option<usize>,

    /// Merge imbalance factor (>= 1.0)
    #[arg(long, default_value = "1.0")]
    imbalance_factor: f64,

    /// Merge partition type (`key-only`, `count-balanced`, `size-balanced`)
    #[arg(long, default_value = "size-balanced", value_name = "PARTITION")]
    partition_type: PartitionArg,

    /// Discard final output (no write) for benchmarking
    #[arg(long, default_value = "false")]
    discard_final_output: bool,

    /// Only estimate dataset size (MB) and exit
    #[arg(long, default_value = "false")]
    estimate_size: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = SortArgs::parse();

    // Create input provider for GenSort files
    let input_provider = GenSortInputProvider { path: args.input };

    // If only estimating size, compute and print, then exit
    if args.estimate_size {
        let estimated_mb = input_provider.estimate_data_size_mb()?;
        println!("Estimated data size: {:.2} MB", estimated_mb);
        return Ok(());
    }

    // Unwrap required args (clap enforces presence when not estimating)
    let run_gen_threads = args
        .run_gen_threads
        .expect("--run-gen-threads required unless --estimate-size");
    let merge_threads = args
        .merge_threads
        .expect("--merge-threads required unless --estimate-size");
    let run_size_mb = args
        .run_size_mb
        .expect("--run-size-mb required unless --estimate-size");
    let merge_fanin = args
        .merge_fanin
        .expect("--merge-fanin required unless --estimate-size");

    let partition_type: PartitionType = args.partition_type.into();

    // Create benchmark configuration
    let config = BenchmarkConfig {
        config_name: args.name,
        warmup_runs: args.warmup_runs,
        benchmark_runs: args.benchmark_runs,
        cooldown_seconds: args.cooldown_seconds,
        verify: args.verify,
        temp_dir: args.dir,
        run_gen_threads,
        use_ovc: args.ovc,
        run_size_mb,
        run_gen_memory_mb: run_size_mb * run_gen_threads as f64,
        merge_threads,
        merge_fanin,
        merge_memory_mb: (merge_fanin as f64)
            * (merge_threads as f64)
            * (DEFAULT_BUFFER_SIZE as f64 / 1024.0 / 1024.0),
        imbalance_factor: args.imbalance_factor,
        partition_type,
        discard_final_output: args.discard_final_output,
    };

    let input_provider = Box::new(input_provider);

    // Create benchmark runner
    let mut runner = BenchmarkRunner::new(config, input_provider);

    // Set up verification if requested
    if args.verify {
        let verifier = SimpleVerifier::new(); // With sample output
        runner.set_verifier(Box::new(verifier));
    }

    // Run single benchmark configuration
    let result: BenchmarkResult = runner.run_configuration()?;

    // Print comprehensive summary
    print_benchmark_summary(&result);

    Ok(())
}
