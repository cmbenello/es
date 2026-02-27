use clap::{ArgAction, Parser};

use es::benchmark::input::KvBinInputProvider;
use es::benchmark::{
    BenchmarkConfig, BenchmarkInputProvider, BenchmarkResult, BenchmarkRunner, SimpleVerifier,
    print_benchmark_summary,
};
use es::diskio::constants::DEFAULT_BUFFER_SIZE;
use es::sort::core::engine::PartitionType;
use es::sort_policy_sub::{PlannerConfig, plan_resource_efficient};
use std::path::{Path, PathBuf};

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
#[command(name = "kvbin_benchmark")]
#[command(about = "KVBin Sort Benchmark")]
struct Args {
    /// Configuration name for labeling the run
    #[arg(short, default_value = "no_name")]
    name: String,

    /// Input KVBin file path
    #[arg(short, long)]
    input: PathBuf,

    /// KVBin index file path (defaults to <input>.idx if present)
    #[arg(long)]
    index: Option<PathBuf>,

    /// Directory for temporary files
    #[arg(short, long, default_value = ".")]
    dir: PathBuf,

    /// Verify sorted output
    #[arg(long, default_value = "false")]
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

    /// Automatically derive run_gen_threads, merge_threads, run_size_mb and
    /// merge_fanin from the dataset size and the budget (--memory-mb,
    /// --max-threads).  The planner applies the two-regime resource-efficient
    /// policy described in the paper.  When set, the four manual tuning args
    /// below become optional.
    #[arg(long, default_value = "false")]
    use_planner: bool,

    /// Available memory budget in MB (used by --use-planner)
    #[arg(long)]
    memory_mb: Option<f64>,

    /// Maximum thread count (used by --use-planner)
    #[arg(long)]
    max_threads: Option<usize>,

    /// Threads for run generation
    #[arg(long, required_unless_present_any = ["estimate_size", "use_planner", "print_plan"])]
    run_gen_threads: Option<usize>,

    /// Threads for merge phase
    #[arg(long, required_unless_present_any = ["estimate_size", "use_planner", "print_plan"])]
    merge_threads: Option<usize>,

    /// Run size for run generation (MB)
    #[arg(long, required_unless_present_any = ["estimate_size", "use_planner", "print_plan"])]
    run_size_mb: Option<f64>,

    /// Merge fan-in (global per merge operation)
    #[arg(long, required_unless_present_any = ["estimate_size", "use_planner", "print_plan"])]
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

    /// Compute and print the planner's resource configuration, then exit
    /// without running the sort.  Requires --memory-mb and --max-threads.
    #[arg(long, default_value = "false")]
    print_plan: bool,
}

fn default_index_path(input: &Path) -> PathBuf {
    PathBuf::from(format!("{}.idx", input.display()))
}

fn resolve_index_path(input: &Path, index: Option<PathBuf>) -> PathBuf {
    index.unwrap_or_else(|| default_index_path(input))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if !args.input.exists() {
        return Err(format!("Input KVBin file not found: {}", args.input.display()).into());
    }

    let index_path = resolve_index_path(&args.input, args.index);
    if !args.estimate_size && !args.print_plan && !index_path.exists() {
        return Err(format!(
            "Index file not found: {} (pass --index if it has a different name)",
            index_path.display()
        )
        .into());
    }

    let input_provider = KvBinInputProvider::new(args.input.clone(), index_path);

    // If only estimating size, compute and print, then exit
    if args.estimate_size {
        let estimated_mb = input_provider.estimate_data_size_mb()?;
        println!("Estimated data size: {:.2} MB", estimated_mb);
        return Ok(());
    }

    // If only printing the plan, compute it and exit without sorting
    if args.print_plan {
        let memory_mb = args
            .memory_mb
            .ok_or("--memory-mb is required when --print-plan is set")?;
        let max_threads = args
            .max_threads
            .ok_or("--max-threads is required when --print-plan is set")?;
        let dataset_mb = input_provider.estimate_data_size_mb()?;
        println!("Estimated data size: {dataset_mb:.2} MB");
        let plan = plan_resource_efficient(&PlannerConfig {
            dataset_mb,
            memory_mb,
            max_threads,
            page_size_kb: DEFAULT_BUFFER_SIZE as f64 / 1024.0,
            imbalance_factor: args.imbalance_factor,
            ..PlannerConfig::default()
        });
        println!("{plan}");
        return Ok(());
    }

    // Resolve run configuration — either from explicit args or via the planner.
    let (run_gen_threads, merge_threads, run_size_mb, merge_fanin) = if args.use_planner {
        let memory_mb = args
            .memory_mb
            .ok_or("--memory-mb is required when --use-planner is set")?;
        let max_threads = args
            .max_threads
            .ok_or("--max-threads is required when --use-planner is set")?;
        let dataset_mb = input_provider.estimate_data_size_mb()?;
        println!("Planner: estimated dataset size = {dataset_mb:.2} MB");

        let plan = plan_resource_efficient(&PlannerConfig {
            dataset_mb,
            memory_mb,
            max_threads,
            page_size_kb: DEFAULT_BUFFER_SIZE as f64 / 1024.0,
            imbalance_factor: args.imbalance_factor,
            ..PlannerConfig::default()
        });
        println!("Planner: {plan}");
        (
            plan.run_gen_threads,
            plan.merge_threads,
            plan.run_size_mb,
            plan.merge_fanin,
        )
    } else {
        // Unwrap required args (clap enforces presence when not using planner)
        let run_gen_threads = args
            .run_gen_threads
            .expect("--run-gen-threads required unless --estimate-size or --use-planner");
        let merge_threads = args
            .merge_threads
            .expect("--merge-threads required unless --estimate-size or --use-planner");
        let run_size_mb = args
            .run_size_mb
            .expect("--run-size-mb required unless --estimate-size or --use-planner");
        let merge_fanin = args
            .merge_fanin
            .expect("--merge-fanin required unless --estimate-size or --use-planner");
        (run_gen_threads, merge_threads, run_size_mb, merge_fanin)
    };

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
        let verifier = SimpleVerifier::new();
        runner.set_verifier(Box::new(verifier));
    }

    // Run single benchmark configuration
    let result: BenchmarkResult = runner.run_configuration()?;

    // Print comprehensive summary
    print_benchmark_summary(&result);

    Ok(())
}
