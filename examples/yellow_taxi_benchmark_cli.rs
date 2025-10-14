use clap::Parser;
use es::benchmark::{
    BenchmarkConfig, BenchmarkInputProvider, BenchmarkResult, BenchmarkRunner,
    YellowTaxiCsvInputProvider, YellowTaxiCsvVerifier, print_benchmark_summary,
};
use es::diskio::constants::DEFAULT_BUFFER_SIZE;
use std::path::PathBuf;

// NYC Yellow Taxi Dataset Column Reference
// Based on: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
//
// Column Index | Column Name                    | DataType      | Cardinality | Estimated Values       | Category
// -------------|--------------------------------|---------------|-------------|------------------------|-------------
// 0            | trip_id                        | Int64         | Ultra-High  | ~20M (unique per trip) | Ultra-High (primary key)
// 1            | vendor_id                      | String        | Ultra-Low   | 2-3                    | Ultra-Low (VTS, CMT, DDS)
// 2            | pickup_datetime                | DateTime      | High        | ~seconds precision     | High (continuous timestamps)
// 3            | dropoff_datetime               | DateTime      | High        | ~seconds precision     | High (continuous timestamps)
// 4            | store_and_fwd_flag             | Int32         | Binary      | 2                      | Binary (0, 1)
// 5            | rate_code                      | Int32         | Ultra-Low   | 6                      | Ultra-Low (1-6: standard, JFK, Newark, etc.)
// 6            | pickup_longitude               | Float64       | High        | Continuous             | High (GPS coordinates)
// 7            | pickup_latitude                | Float64       | High        | Continuous             | High (GPS coordinates)
// 8            | dropoff_longitude              | Float64       | High        | Continuous             | High (GPS coordinates)
// 9            | dropoff_latitude               | Float64       | High        | Continuous             | High (GPS coordinates)
// 10           | payment_type                   | Int32         | Ultra-Low   | 6                      | Ultra-Low (1-6: credit, cash, no charge, etc.)
// 11           | fare_amount                    | Float64       | High        | Continuous             | High (calculated fare)
// 12           | surcharge                      | Float64       | Low         | ~10-20                 | Low (fixed surcharge amounts)
// 13           | mta_tax                        | Float64       | Ultra-Low   | ~3                     | Ultra-Low (0.5, 0, etc.)
// 14           | tip_amount                     | Float64       | Medium      | Variable               | Medium (tips vary widely)
// 15           | tolls_amount                   | Float64       | Low         | ~50                    | Low (bridge/tunnel tolls)
// 16           | improvement_surcharge          | Float64       | Ultra-Low   | ~3                     | Ultra-Low (0.30, 0, etc.)
// 17           | total_amount                   | Float64       | High        | Continuous             | High (sum of all charges)
// 18           | congestion_surcharge           | Float64       | Ultra-Low   | ~4                     | Ultra-Low (2.50, 2.75, 0, etc.)
// 19           | airport_fee                    | Float64       | Ultra-Low   | ~3                     | Ultra-Low (1.25, 0, etc.)
// 20           | payment_type_desc              | String        | Ultra-Low   | 6                      | Ultra-Low (CSH, CRE, etc.)
// 21           | trip_type                      | Int32         | Binary      | 2                      | Binary (0, 1, 2)
// 22           | pickup_geom                    | String        | High        | ~unique per trip       | High (WKT geometry)
// 23           | dropoff_geom                   | String        | High        | ~unique per trip       | High (WKT geometry)
// 24           | cab_type                       | String        | Binary      | 1-2                    | Binary (yellow, green)
// 25-50        | Various location metadata      | Mixed         | Varies      | Borough, zone info     | Mixed cardinality
//
// Interesting sort keys for benchmarking:
// - Temporal clustering: pickup_datetime (2), dropoff_datetime (3)
// - Spatial clustering: pickup_longitude (6), pickup_latitude (7), dropoff_longitude (8), dropoff_latitude (9)
// - Low cardinality (counting sort friendly): vendor_id (1), rate_code (5), payment_type (10), payment_type_desc (20)
// - High cardinality: trip_id (0), fare_amount (11), total_amount (17)
// - Compound keys: (vendor_id, pickup_datetime), (payment_type, total_amount), (pickup_borough, dropoff_borough)
//
// Typical dataset sizes:
// - Monthly data: ~10-15M trips
// - Annual data: ~150-180M trips
// - Multi-year datasets: 500M+ trips

#[derive(Parser)]
#[command(name = "yellow_taxi_benchmark")]
#[command(about = "NYC Yellow Taxi CSV Sort Benchmark")]
struct Args {
    #[arg(short, default_value = "no_name")]
    name: String,

    /// Input CSV file path
    #[arg(short, long)]
    input: PathBuf,

    /// Only estimate dataset size (MB) and exit
    #[arg(long, default_value = "false")]
    estimate_size: bool,

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

    /// Key column indices (comma-separated)
    /// Examples:
    ///   "2"           - Sort by pickup_datetime (temporal)
    ///   "1,2"         - Sort by vendor_id, pickup_datetime (low card + temporal)
    ///   "6,7"         - Sort by pickup location (spatial)
    ///   "20,17"       - Sort by payment_type, total_amount (low card + continuous)
    ///   "34,44"       - Sort by pickup_borough, dropoff_borough (geographic)
    #[arg(short = 'k', long, default_value = "2,3")]
    key_columns: String,

    /// Value column indices (comma-separated)
    #[arg(short = 'v', long, default_value = "0")]
    value_columns: String,

    /// Sketch size for quantile estimation (KLL)
    #[arg(long, default_value = "200")]
    sketch_size: usize,

    /// Sketch sampling interval (update the sketch every N records of the run)
    #[arg(long, default_value = "100")]
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
    #[arg(long, default_value = "false")]
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

    // Create input provider for CSV files
    let input_provider = YellowTaxiCsvInputProvider::new(
        args.input,
        key_columns.clone(),
        value_columns,
        args.delimiter,
        args.headers,
    );

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
        ovc: args.ovc,
        temp_dir: args.dir,
        sketch_size: args.sketch_size,
        sketch_sampling_interval: args.sketch_sampling_interval,
        run_indexing_interval: args.run_indexing_interval,
        run_gen_threads,
        run_size_mb,
        run_gen_memory_mb: run_size_mb * run_gen_threads as f64,
        merge_threads,
        merge_fanin,
        merge_memory_mb: (merge_fanin as f64)
            * (merge_threads as f64)
            * (DEFAULT_BUFFER_SIZE as f64 / 1024.0 / 1024.0),
        imbalance_factor: args.imbalance_factor,
    };

    let input_provider = Box::new(input_provider);

    // Create benchmark runner
    let mut runner = BenchmarkRunner::new(config, input_provider);

    // Set up verification if requested
    if args.verify {
        let verifier = YellowTaxiCsvVerifier::new(key_columns);
        runner.set_verifier(Box::new(verifier));
    }

    // Run single benchmark configuration
    let result: BenchmarkResult = runner.run_configuration()?;

    // Print comprehensive summary
    print_benchmark_summary(&result);

    Ok(())
}
