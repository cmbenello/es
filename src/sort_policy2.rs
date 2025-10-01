/// Optimized sort policy with fixed maximum threads and optimal run length selection
///
/// This module provides a generic trait-based approach for sort policies that fix
/// thread counts to maximum and optimize run length within feasible bounds.

#[derive(Debug, Clone)]
pub struct PolicyResult {
    pub name: String,
    pub run_size_mb: f64,
    pub run_gen_threads: f64,
    pub merge_threads: f64,
    pub run_gen_memory_mb: f64,
    pub merge_memory_mb: f64,
}

impl std::fmt::Display for PolicyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]: Run Size: {:.2} MB, Run Gen Threads: {:.0}, Merge Threads: {:.0}, Run Gen Memory: {:.1} MB, Merge Memory: {:.1} MB",
            self.name,
            self.run_size_mb,
            self.run_gen_threads,
            self.merge_threads,
            self.run_gen_memory_mb,
            self.merge_memory_mb
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
            memory_mb: 4096.0,
            dataset_mb: 32768.0,
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
    // fanin * page_size * threads <= memory
    // (dataset / run_size) * page_size * threads <= memory
    // run_size >= (dataset * page_size * threads) / memory
    let min_run_size = (d * p * t) / m;

    // Upper bound: From run generation constraint
    // run_size * threads <= memory
    let max_run_size = m / t;

    (min_run_size, max_run_size)
}

/// Validate if configuration is feasible
pub fn is_configuration_feasible(config: SortConfig) -> bool {
    let (min_run, max_run) = calculate_run_bounds(config);
    min_run <= max_run
}

/// Policy that uses 0% interpolation (minimum feasible run length)
pub struct Policy0PercentRunLength;

impl SortPolicy for Policy0PercentRunLength {
    fn name(&self) -> String {
        "0%_RunLength".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (min_run_size, _) = calculate_run_bounds(config);
        let run_gen_threads = config.max_threads;
        let merge_threads = config.max_threads;

        PolicyResult {
            name: self.name(),
            run_size_mb: min_run_size,
            run_gen_threads,
            merge_threads,
            run_gen_memory_mb: min_run_size * run_gen_threads,
            merge_memory_mb: (config.dataset_mb / min_run_size)
                * merge_threads
                * (config.page_size_kb / 1024.0),
        }
    }
}

/// Policy that uses 100% interpolation (maximum feasible run length)
pub struct Policy100PercentRunLength;

impl SortPolicy for Policy100PercentRunLength {
    fn name(&self) -> String {
        "100%_RunLength".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (_, max_run_size) = calculate_run_bounds(config);
        let run_gen_threads = config.max_threads;
        let merge_threads = config.max_threads;

        PolicyResult {
            name: self.name(),
            run_size_mb: max_run_size,
            run_gen_threads,
            merge_threads,
            run_gen_memory_mb: max_run_size * run_gen_threads,
            merge_memory_mb: (config.dataset_mb / max_run_size)
                * merge_threads
                * (config.page_size_kb / 1024.0),
        }
    }
}

/// Policy that uses 50% interpolation (arithmetic mean of the bounds)
pub struct Policy50PercentRunLength;

impl SortPolicy for Policy50PercentRunLength {
    fn name(&self) -> String {
        "50%_RunLength".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (min_run_size, max_run_size) = calculate_run_bounds(config);
        let run_size = (min_run_size + max_run_size) / 2.0;
        let run_gen_threads = config.max_threads;
        let merge_threads = config.max_threads;

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads,
            merge_threads,
            run_gen_memory_mb: run_size * run_gen_threads,
            merge_memory_mb: (config.dataset_mb / run_size)
                * merge_threads
                * (config.page_size_kb / 1024.0),
        }
    }
}

/// Policy that uses 75% interpolation
pub struct Policy75PercentRunLength;

impl SortPolicy for Policy75PercentRunLength {
    fn name(&self) -> String {
        "75%_RunLength".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (min_run_size, max_run_size) = calculate_run_bounds(config);
        // Use 75% towards max to reduce run gen memory pressure
        let run_size = min_run_size + 0.75 * (max_run_size - min_run_size);
        let run_gen_threads = config.max_threads;
        let merge_threads = config.max_threads;

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads,
            merge_threads,
            run_gen_memory_mb: run_size * run_gen_threads,
            merge_memory_mb: (config.dataset_mb / run_size)
                * merge_threads
                * (config.page_size_kb / 1024.0),
        }
    }
}

/// Policy that uses 25% interpolation
pub struct Policy25PercentRunLength;

impl SortPolicy for Policy25PercentRunLength {
    fn name(&self) -> String {
        "25%_RunLength".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (min_run_size, max_run_size) = calculate_run_bounds(config);
        // Use 25% towards max to reduce merge memory pressure
        let run_size = min_run_size + 0.25 * (max_run_size - min_run_size);
        let run_gen_threads = config.max_threads;
        let merge_threads = config.max_threads;

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads,
            merge_threads,
            run_gen_memory_mb: run_size * run_gen_threads,
            merge_memory_mb: (config.dataset_mb / run_size)
                * merge_threads
                * (config.page_size_kb / 1024.0),
        }
    }
}

/// Get all available policies
pub fn get_all_policies() -> Vec<Box<dyn SortPolicy>> {
    vec![
        Box::new(Policy0PercentRunLength),
        Box::new(Policy25PercentRunLength),
        Box::new(Policy50PercentRunLength),
        Box::new(Policy75PercentRunLength),
        Box::new(Policy100PercentRunLength),
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

    // Sort by run size (ascending)
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
        let total_runs = config.dataset_mb / result.run_size_mb;
        println!(
            "{:<35} {:>12.2} {:>8.0} {:>8.0} {:>15.1} {:>15.1} {:>12.0}",
            result.name,
            result.run_size_mb,
            result.run_gen_threads,
            result.merge_threads,
            result.run_gen_memory_mb,
            result.merge_memory_mb,
            total_runs
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_bounds_calculation() {
        let config = SortConfig {
            memory_mb: 1024.0,
            dataset_mb: 8192.0,
            page_size_kb: 64.0,
            max_threads: 16.0,
        };

        let (min_run, max_run) = calculate_run_bounds(config);

        // min_run = (8192 * 0.0625 * 16) / 1024 = 8.0 MB
        assert!((min_run - 8.0).abs() < 0.01);

        // max_run = 1024 / 16 = 64.0 MB
        assert!((max_run - 64.0).abs() < 0.01);

        assert!(is_configuration_feasible(config));
    }

    #[test]
    fn test_infeasible_configuration() {
        let config = SortConfig {
            memory_mb: 100.0,
            dataset_mb: 10000.0,
            page_size_kb: 64.0,
            max_threads: 32.0,
        };

        assert!(!is_configuration_feasible(config));
    }

    #[test]
    fn test_min_run_length_policy() {
        let config = SortConfig {
            memory_mb: 1024.0,
            dataset_mb: 4096.0,
            page_size_kb: 64.0,
            max_threads: 8.0,
        };

        let policy = Policy0PercentRunLength;
        let result = policy.calculate(config);

        let (min_run, _) = calculate_run_bounds(config);
        assert!((result.run_size_mb - min_run).abs() < 0.01);
        assert_eq!(result.run_gen_threads, 8.0);
        assert_eq!(result.merge_threads, 8.0);
    }

    #[test]
    fn test_max_run_length_policy() {
        let config = SortConfig {
            memory_mb: 1024.0,
            dataset_mb: 4096.0,
            page_size_kb: 64.0,
            max_threads: 8.0,
        };

        let policy = Policy100PercentRunLength;
        let result = policy.calculate(config);

        let (_, max_run) = calculate_run_bounds(config);
        assert!((result.run_size_mb - max_run).abs() < 0.01);
        assert_eq!(result.run_gen_threads, 8.0);
        assert_eq!(result.merge_threads, 8.0);
    }

    #[test]
    fn test_all_policies_feasible_config() {
        let config = SortConfig {
            memory_mb: 2048.0,
            dataset_mb: 8192.0,
            page_size_kb: 64.0,
            max_threads: 16.0,
        };

        let results = calculate_all_policies(config);
        assert_eq!(results.len(), 5);

        // All policies should fix threads to max
        for result in &results {
            assert_eq!(result.run_gen_threads, 16.0);
            assert_eq!(result.merge_threads, 16.0);
        }

        // All run sizes should be within bounds
        let (min_run, max_run) = calculate_run_bounds(config);
        for result in &results {
            assert!(result.run_size_mb >= min_run - 0.01);
            assert!(result.run_size_mb <= max_run + 0.01);
        }
    }

    #[test]
    fn test_memory_calculations() {
        let config = SortConfig {
            memory_mb: 1024.0,
            dataset_mb: 2048.0,
            page_size_kb: 64.0,
            max_threads: 8.0,
        };

        let policy = Policy50PercentRunLength;
        let result = policy.calculate(config);

        // Verify run gen memory = run_size * threads
        assert!(
            (result.run_gen_memory_mb - result.run_size_mb * result.run_gen_threads).abs() < 0.01
        );

        // Verify merge memory = (dataset/run_size) * threads * page_size
        let expected_merge_memory = (config.dataset_mb / result.run_size_mb)
            * result.merge_threads
            * (config.page_size_kb / 1024.0);
        assert!((result.merge_memory_mb - expected_merge_memory).abs() < 0.01);
    }

    #[test]
    fn test_print_all_policies() {
        let config = SortConfig {
            memory_mb: 32768.0,        // 32 GB
            dataset_mb: 80.0 * 1024.0, // 80 GB
            page_size_kb: 64.0,        // 64 KB
            max_threads: 32.0,         // 32 threads
        };

        println!("\n=== Testing All Policies ===");
        print_detailed_analysis(config);

        println!("\n=== Individual Policy Results ===");
        for result in calculate_all_policies(config) {
            println!("{}", result);
        }
    }
}
