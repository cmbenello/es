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

/// Policy that uses logarithmic interpolation with factor 0.25
pub struct PolicyLog025RunLength;

impl SortPolicy for PolicyLog025RunLength {
    fn name(&self) -> String {
        "Log_0.25".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (min_run_size, max_run_size) = calculate_run_bounds(config);
        // Logarithmic interpolation: min * (max/min)^0.25
        let run_size = min_run_size * (max_run_size / min_run_size).powf(0.25);
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

/// Policy that uses logarithmic interpolation with factor 0.5 (geometric mean)
pub struct PolicyLog05RunLength;

impl SortPolicy for PolicyLog05RunLength {
    fn name(&self) -> String {
        "Log_0.5".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (min_run_size, max_run_size) = calculate_run_bounds(config);
        // Logarithmic interpolation: min * (max/min)^0.5 = sqrt(min * max)
        let run_size = (min_run_size * max_run_size).sqrt();
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

/// Policy that uses logarithmic interpolation with factor 0.75
pub struct PolicyLog075RunLength;

impl SortPolicy for PolicyLog075RunLength {
    fn name(&self) -> String {
        "Log_0.75".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let (min_run_size, max_run_size) = calculate_run_bounds(config);
        // Logarithmic interpolation: min * (max/min)^0.75
        let run_size = min_run_size * (max_run_size / min_run_size).powf(0.75);
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

        let policy = PolicyLog0RunLength;
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

        let policy = PolicyLog1RunLength;
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

        let policy = PolicyLog05RunLength;
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
    fn test_logarithmic_interpolation() {
        let config = SortConfig {
            memory_mb: 1024.0,
            dataset_mb: 4096.0,
            page_size_kb: 64.0,
            max_threads: 8.0,
        };

        let (min_run, max_run) = calculate_run_bounds(config);

        // Test Log_0.5 gives geometric mean
        let policy_05 = PolicyLog05RunLength;
        let result_05 = policy_05.calculate(config);
        let geometric_mean = (min_run * max_run).sqrt();
        assert!((result_05.run_size_mb - geometric_mean).abs() < 0.01);

        // Test that log policies are ordered
        let policy_0 = PolicyLog0RunLength;
        let policy_025 = PolicyLog025RunLength;
        let policy_075 = PolicyLog075RunLength;
        let policy_1 = PolicyLog1RunLength;

        let result_0 = policy_0.calculate(config);
        let result_025 = policy_025.calculate(config);
        let result_075 = policy_075.calculate(config);
        let result_1 = policy_1.calculate(config);

        assert!(result_0.run_size_mb < result_025.run_size_mb);
        assert!(result_025.run_size_mb < result_05.run_size_mb);
        assert!(result_05.run_size_mb < result_075.run_size_mb);
        assert!(result_075.run_size_mb < result_1.run_size_mb);
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
