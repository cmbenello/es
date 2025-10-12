use clap::Parser;

/// Optimized sort policy with fixed maximum threads and optimal run length selection
#[derive(Debug, Clone)]
pub struct PolicyResult {
    pub name: String,
    pub run_size_mb: f64,
    pub run_gen_threads: f64,
    pub merge_threads: f64,
    pub run_gen_memory_mb: f64,
    pub merge_memory_mb: f64,
    pub total_runs: f64,
}

impl Default for PolicyResult {
    fn default() -> Self {
        Self {
            name: String::new(),
            run_size_mb: 0.0,
            run_gen_threads: 0.0,
            merge_threads: 0.0,
            run_gen_memory_mb: 0.0,
            merge_memory_mb: 0.0,
            total_runs: 0.0,
        }
    }
}

impl std::fmt::Display for PolicyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]: Run Size: {:.2} MB, Run Gen Threads: {:.0}, Merge Threads: {:.0}, Run Gen Memory: {:.1} MB, Merge Memory: {:.1} MB, Total Runs: {:.0}",
            self.name,
            self.run_size_mb,
            self.run_gen_threads,
            self.merge_threads,
            self.run_gen_memory_mb,
            self.merge_memory_mb,
            self.total_runs,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SortConfig {
    pub memory_mb: f64,
    pub dataset_mb: f64,
    pub page_size_kb: f64,
    pub max_threads: f64,
}

impl Default for SortConfig {
    fn default() -> Self {
        Self {
            memory_mb: 32768.0,         // 32 GB
            dataset_mb: 200.0 * 1024.0, // 200 GB
            page_size_kb: 64.0,
            max_threads: 32.0,
        }
    }
}

/// Generic trait for sort policies that return optimal parameters
pub trait SortPolicy {
    fn name(&self) -> String;
    fn calculate(&self, config: SortConfig) -> PolicyResult;
}

/// Calculate feasible run length bounds
pub fn calculate_run_bounds(config: SortConfig) -> (f64, f64) {
    let m = config.memory_mb;
    let d = config.dataset_mb;
    let p = config.page_size_kb / 1024.0; // Convert to MB
    let t = config.max_threads;

    // Lower bound: From merge constraint
    let min_run_size = (d * p * t) / m;

    // Upper bound: From run generation constraint
    let max_run_size = m / t;

    (min_run_size, max_run_size)
}

/// Validate if configuration is feasible
pub fn is_configuration_feasible(config: SortConfig) -> bool {
    let (min_run, max_run) = calculate_run_bounds(config);
    min_run <= max_run
}

/// Policy that uses logarithmic interpolation with factor 0.0 (minimum feasible run length)
pub struct PolicyLog0RunLength;

impl SortPolicy for PolicyLog0RunLength {
    fn name(&self) -> String {
        "Log_0.0".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (min_run_size, _) = calculate_run_bounds(config);
        let run_gen_threads = config.max_threads;
        let merge_threads = config.max_threads;
        let total_runs = config.dataset_mb / min_run_size;

        PolicyResult {
            name: self.name(),
            run_size_mb: min_run_size,
            run_gen_threads,
            merge_threads,
            run_gen_memory_mb: min_run_size * run_gen_threads,
            merge_memory_mb: total_runs * merge_threads * (config.page_size_kb / 1024.0),
            total_runs,
        }
    }
}

/// Policy that uses logarithmic interpolation with factor 0.25
pub struct PolicyLog025RunLength;

impl SortPolicy for PolicyLog025RunLength {
    fn name(&self) -> String {
        "Log_0.25".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (min_run_size, max_run_size) = calculate_run_bounds(config);
        let run_size = min_run_size * (max_run_size / min_run_size).powf(0.25);
        let run_gen_threads = config.max_threads;
        let merge_threads = config.max_threads;
        let total_runs = config.dataset_mb / run_size;

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads,
            merge_threads,
            run_gen_memory_mb: run_size * run_gen_threads,
            merge_memory_mb: total_runs * merge_threads * (config.page_size_kb / 1024.0),
            total_runs,
        }
    }
}

/// Policy that uses logarithmic interpolation with factor 0.5 (geometric mean)
pub struct PolicyLog05RunLength;

impl SortPolicy for PolicyLog05RunLength {
    fn name(&self) -> String {
        "Log_0.5".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (min_run_size, max_run_size) = calculate_run_bounds(config);
        let run_size = (min_run_size * max_run_size).sqrt();
        let run_gen_threads = config.max_threads;
        let merge_threads = config.max_threads;
        let total_runs = config.dataset_mb / run_size;

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads,
            merge_threads,
            run_gen_memory_mb: run_size * run_gen_threads,
            merge_memory_mb: total_runs * merge_threads * (config.page_size_kb / 1024.0),
            total_runs,
        }
    }
}

/// Policy that uses logarithmic interpolation with factor 0.75
pub struct PolicyLog075RunLength;

impl SortPolicy for PolicyLog075RunLength {
    fn name(&self) -> String {
        "Log_0.75".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (min_run_size, max_run_size) = calculate_run_bounds(config);
        let run_size = min_run_size * (max_run_size / min_run_size).powf(0.75);
        let run_gen_threads = config.max_threads;
        let merge_threads = config.max_threads;
        let total_runs = config.dataset_mb / run_size;

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads,
            merge_threads,
            run_gen_memory_mb: run_size * run_gen_threads,
            merge_memory_mb: total_runs * merge_threads * (config.page_size_kb / 1024.0),
            total_runs,
        }
    }
}

/// Policy that uses logarithmic interpolation with factor 1.0 (maximum feasible run length)
pub struct PolicyLog1RunLength;

impl SortPolicy for PolicyLog1RunLength {
    fn name(&self) -> String {
        "Log_1.0".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (_, max_run_size) = calculate_run_bounds(config);
        let run_gen_threads = config.max_threads;
        let merge_threads = config.max_threads;
        let total_runs = config.dataset_mb / max_run_size;

        PolicyResult {
            name: self.name(),
            run_size_mb: max_run_size,
            run_gen_threads,
            merge_threads,
            run_gen_memory_mb: max_run_size * run_gen_threads,
            merge_memory_mb: total_runs * merge_threads * (config.page_size_kb / 1024.0),
            total_runs,
        }
    }
}

/// Get all available policies
pub fn get_all_policies() -> Vec<Box<dyn SortPolicy>> {
    vec![
        Box::new(PolicyLog0RunLength),
        Box::new(PolicyLog025RunLength),
        Box::new(PolicyLog05RunLength),
        Box::new(PolicyLog075RunLength),
        Box::new(PolicyLog1RunLength),
    ]
}

/// Calculate and return all policy results for a given configuration, sorted by run size
pub fn calculate_all_policies(config: SortConfig) -> Vec<PolicyResult> {
    if !is_configuration_feasible(config) {
        println!("Warning: Configuration is not feasible!");
        let (min_run, max_run) = calculate_run_bounds(config);
        println!(
            "Min run size: {:.2} MB, Max run size: {:.2} MB",
            min_run, max_run
        );
        return vec![];
    }

    let mut results: Vec<PolicyResult> = get_all_policies()
        .into_iter()
        .map(|policy| policy.calculate(config))
        .collect();

    results.sort_by(|a, b| a.run_size_mb.partial_cmp(&b.run_size_mb).unwrap());
    results
}

/// Print detailed analysis for all policies
pub fn print_detailed_analysis(config: SortConfig) {
    println!("=== Sort Policy Analysis ===");
    println!(
        "Memory: {:.0} MB ({:.1} GB)",
        config.memory_mb,
        config.memory_mb / 1024.0
    );
    println!(
        "Dataset: {:.0} MB ({:.1} GB)",
        config.dataset_mb,
        config.dataset_mb / 1024.0
    );
    println!("Page Size: {:.0} KB", config.page_size_kb);
    println!("Max Threads: {:.0}", config.max_threads);

    let (min_run, max_run) = calculate_run_bounds(config);
    println!("\nRun Length Bounds:");
    println!("  Minimum: {:.2} MB (merge constraint)", min_run);
    println!("  Maximum: {:.2} MB (run generation constraint)", max_run);
    println!(
        "  Feasible: {}",
        if min_run <= max_run { "Yes" } else { "No" }
    );

    if !is_configuration_feasible(config) {
        println!("\nConfiguration is not feasible - no valid run lengths exist!");
        return;
    }

    println!("\n{}", "=".repeat(120));
    println!(
        "{:<35} {:>12} {:>8} {:>8} {:>15} {:>15} {:>12}",
        "Policy", "Run Size", "RunGen", "Merge", "RunGen Mem", "Merge Mem", "Total Runs"
    );
    println!("{}", "=".repeat(120));

    for result in calculate_all_policies(config) {
        println!(
            "{:<35} {:>12.2} {:>8.0} {:>8.0} {:>15.1} {:>15.1} {:>12.0}",
            result.name,
            result.run_size_mb,
            result.run_gen_threads,
            result.merge_threads,
            result.run_gen_memory_mb,
            result.merge_memory_mb,
            result.total_runs
        );
    }
}

#[derive(Parser)]
struct Args {
    /// Total memory (MB)
    #[arg(long, default_value = "32768")]
    memory_mb: f64,
    /// Dataset size (MB)
    #[arg(long)]
    dataset_mb: f64,
    /// Page size (KB)
    #[arg(long, default_value = "64")]
    page_size_kb: f64,
    /// Max threads
    #[arg(long, default_value = "32")]
    max_threads: f64,
}

fn main() {
    let args = Args::parse();
    let cfg = SortConfig {
        memory_mb: args.memory_mb,
        dataset_mb: args.dataset_mb,
        page_size_kb: args.page_size_kb,
        max_threads: args.max_threads,
    };
    print_detailed_analysis(cfg);
    println!("\n=== Individual Policy Results ===");
    for res in calculate_all_policies(cfg) {
        println!("{}", res);
    }
}
