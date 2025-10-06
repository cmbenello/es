use clap::Parser;
use es::benchmark::{
    BenchmarkConfig, BenchmarkRunner, CsvInputProvider, DetailedCsvVerifier,
    print_benchmark_summary,
};
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
#[command(about = "TPC-H Lineitem CSV Sort Benchmark with policy-based optimization")]
struct Args {
    /// Input CSV file path
    #[arg(short, long)]
    input: PathBuf,

    /// Key column indices (comma-separated)
    #[arg(short = 'k', long, default_value = "8,9,13,14,15")]
    key_columns: String,

    /// Value column indices (comma-separated)
    #[arg(short = 'v', long, default_value = "0,3")]
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

    /// Experiment type: "run_length", "thread_count", or "imbalance_factor"
    #[arg(long, default_value = "run_length")]
    experiment_type: String,
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

    // Validate experiment type
    let experiment_type = args.experiment_type.to_lowercase();
    if experiment_type != "run_length"
        && experiment_type != "thread_count"
        && experiment_type != "imbalance_factor"
    {
        eprintln!(
            "Error: experiment_type must be 'run_length', 'thread_count', or 'imbalance_factor'"
        );
        std::process::exit(1);
    }

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
        experiment_type,
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
