use clap::Parser;

use es::benchmark::input::KvBinInputProvider;
use es::benchmark::{
    BenchmarkConfig, BenchmarkInputProvider, BenchmarkResult, BenchmarkRunner,
    LineitemCsvInputProvider, LineitemCsvVerifier, print_benchmark_summary,
};
use es::diskio::constants::DEFAULT_BUFFER_SIZE;
use es::kvbin::{binary_file_name, create_kvbin_from_input};
use es::sketch::SketchType;
use std::path::PathBuf;

// TPC-H lineitem column cardinality reference table
// Column Index | Column Name         | DataType      | Cardinality | Distinct Values    | Category
// -------------|-------------------- |---------------|-------------|--------------------|-----------
// 0            | l_orderkey          | Int64         | High        | ~1.5M * SF         | High (foreign key to Orders)
// 1            | l_partkey           | Int64         | High        | 200,000 * SF       | High (foreign key to Part)
// 2            | l_suppkey           | Int64         | Medium      | 10,000 * SF        | Medium (foreign key to Supplier)
// 3            | l_linenumber        | Int32         | Ultra-Low   | 7                  | Ultra-Low (values: 1-7)
// 4            | l_quantity          | Float64       | Low         | 50                 | Low (integer values: 1-50)
// 5            | l_extendedprice     | Float64       | High        | Continuous         | High (calculated: quantity * price)
// 6            | l_discount          | Float64       | Ultra-Low   | 11                 | Ultra-Low (values: 0.00-0.10, step 0.01)
// 7            | l_tax               | Float64       | Ultra-Low   | 9                  | Ultra-Low (values: 0.00-0.08, step 0.01)
// 8            | l_returnflag        | Utf8          | Ultra-Low   | 3                  | Ultra-Low (values: 'R', 'A', 'N')
// 9            | l_linestatus        | Utf8          | Binary      | 2                  | Binary (values: 'O', 'F')
// 10           | l_shipdate          | Date32        | Medium      | ~2,526             | Medium (range: 1992-01-02 to 1998-12-01)
// 11           | l_commitdate        | Date32        | Medium      | ~2,466             | Medium (similar range to shipdate)
// 12           | l_receiptdate       | Date32        | Medium      | ~2,554             | Medium (typically after shipdate)
// 13           | l_shipinstruct      | Utf8          | Ultra-Low   | 4                  | Ultra-Low (values: 'DELIVER IN PERSON', 'COLLECT COD', 'NONE', 'TAKE BACK RETURN')
// 14           | l_shipmode          | Utf8          | Ultra-Low   | 7                  | Ultra-Low (values: 'REG AIR', 'AIR', 'RAIL', 'SHIP', 'TRUCK', 'MAIL', 'FOB')
// 15           | l_comment           | Utf8          | High        | Nearly unique      | High (random text comments)

// Cardinality categories for sorting algorithm selection:
// - Binary (2 values):      Use simple partition - Column 9
// - Ultra-Low (≤11 values): Use counting sort - Columns 3, 6, 7, 8, 9, 13, 14
// - Low (12-100 values):    Use counting sort - Column 4
// - Medium (100-10K):       Use radix or quicksort - Columns 2, 10, 11, 12
// - High (>10K):           Use quicksort or parallel sort - Columns 0, 1, 5, 15

// Sorting-friendly columns (counting sort applicable): 3, 4, 6, 7, 8, 9, 13, 14
// String columns requiring expensive comparisons: 8, 9, 13, 14, 15
// Continuous numeric requiring comparison-based sort: 0, 1, 2, 5
// Date columns with temporal clustering: 10, 11, 12

// Scale Factor (SF) impact on row counts:
// SF=1:    ~6 million rows
// SF=10:   ~60 million rows
// SF=100:  ~600 million rows
// SF=1000: ~6 billion rows

#[derive(Parser)]
#[command(name = "lineitem_benchmark")]
#[command(about = "TPC-H Lineitem CSV Sort Benchmark")]
struct Args {
    #[arg(short, default_value = "no_name")]
    name: String,

    /// Input CSV file path
    #[arg(short, long)]
    input: PathBuf,

    /// Use KVBin binary format for input. Create if not existing.
    #[arg(long, default_value = "true")]
    use_binary_format: bool,

    /// Threads for run generation
    #[arg(long, required_unless_present = "estimate_size")]
    run_gen_threads: Option<usize>,

    /// Use OVC (Offset Value Coding) format
    #[arg(long, default_value = "true")]
    ovc: bool,

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

    /// Key column indices (comma-separated)
    #[arg(short = 'k', long, default_value = "8,9,13,14,15")]
    key_columns: String,

    /// Value column indices (comma-separated)
    #[arg(short = 'v', long, default_value = "0,3")]
    value_columns: String,

    /// Sketch type for quantile estimation (`kll` or `reservoir-sampling`)
    #[arg(long, default_value = "kll", value_name = "SKETCH")]
    sketch_type: SketchType,

    /// Sketch size for quantile estimation
    #[arg(long, default_value = "200")]
    sketch_size: usize,

    /// Sketch sampling interval (update the sketch every N records of the run)
    #[arg(long, default_value = "100")]
    sketch_sampling_interval: usize,

    /// Run indexing interval (store every Nth key in the run index)
    #[arg(long, default_value = "1000")]
    run_indexing_interval: usize,

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

    /// Only estimate dataset size (MB) and exit
    #[arg(long, default_value = "false")]
    estimate_size: bool,

    /// Cooldown seconds between runs
    #[arg(long, default_value = "0")]
    cooldown_seconds: u64,
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

    let input_provider = if args.use_binary_format {
        let (data_path, idx_path) = binary_file_name(&args.input, &key_columns, &value_columns)?;
        if data_path.exists() && idx_path.exists() {
            println!("Using existing KVBin file: {:?}", data_path);
            Box::new(KvBinInputProvider::new(data_path, idx_path))
                as Box<dyn BenchmarkInputProvider>
        } else {
            println!("Creating KVBin file: {:?}", data_path);
            let start = std::time::Instant::now();
            // Create KVBin input provider by converting from CSV
            let csv_provider = LineitemCsvInputProvider::new(
                args.input,
                key_columns.clone(),
                value_columns,
                args.delimiter,
                args.headers,
            );
            create_kvbin_from_input(csv_provider.create_sort_input()?, &data_path, &idx_path, 0)?;
            let duration = start.elapsed();
            println!("KVBin file created in {:.2?}", duration);
            Box::new(KvBinInputProvider::new(data_path, idx_path))
                as Box<dyn BenchmarkInputProvider>
        }
    } else {
        Box::new(LineitemCsvInputProvider::new(
            args.input,
            key_columns.clone(),
            value_columns,
            args.delimiter,
            args.headers,
        )) as Box<dyn BenchmarkInputProvider>
    };

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

    // Create benchmark configuration
    let config = BenchmarkConfig {
        config_name: args.name,
        warmup_runs: args.warmup_runs,
        benchmark_runs: args.benchmark_runs,
        cooldown_seconds: args.cooldown_seconds,
        verify: args.verify,
        temp_dir: args.dir,
        sketch_type: args.sketch_type,
        sketch_size: args.sketch_size,
        sketch_sampling_interval: args.sketch_sampling_interval,
        run_indexing_interval: args.run_indexing_interval,
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
    };

    // Create benchmark runner
    let mut runner = BenchmarkRunner::new(config, input_provider);

    // Set up verification if requested
    if args.verify {
        let verifier = LineitemCsvVerifier::new(key_columns);
        runner.set_verifier(Box::new(verifier));
    }

    // Run single benchmark configuration
    let result: BenchmarkResult = runner.run_configuration()?;

    // Print comprehensive summary
    print_benchmark_summary(&result);

    Ok(())
}
