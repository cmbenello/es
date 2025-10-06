/// Sort policy with varying imbalance factors
///
/// This module provides policies that increase imbalance factor progressively
/// (1.0, 2.0, 4.0, 8.0, ...) until reaching max_imbalance_factor from config
/// and thread count constraints (imbalance_factor <= threads - 1).
use crate::sort_policy_run_length::{
    PolicyResult, SortConfig, SortPolicy, calculate_run_bounds, is_configuration_feasible,
};

/// Calculate run size as sqrt(data_size * page_size)
fn calculate_run_size(config: SortConfig) -> f64 {
    let d = config.dataset_mb;
    let p = config.page_size_kb / 1024.0; // Convert to MB
    (d * p).sqrt()
}

/// Generic policy for a specific imbalance factor
pub struct PolicyImbalanceFactor {
    imbalance_factor: f64,
}

impl PolicyImbalanceFactor {
    pub fn new(imbalance_factor: f64) -> Self {
        Self { imbalance_factor }
    }
}

impl SortPolicy for PolicyImbalanceFactor {
    fn name(&self) -> String {
        format!("Imbalance_{:.1}", self.imbalance_factor)
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let run_size = calculate_run_size(config);

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads: config.max_threads,
            merge_threads: config.max_threads,
            run_gen_memory_mb: run_size * config.max_threads,
            merge_memory_mb: (config.dataset_mb / run_size)
                * config.max_threads
                * (config.page_size_kb / 1024.0),
            imbalance_factor: self.imbalance_factor,
        }
    }
}

/// Get all imbalance factor policies up to maximum constraints
/// Generates factors: 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, ... until max_imbalance_factor or (threads - 1)
pub fn get_all_policies(config: SortConfig) -> Vec<Box<dyn SortPolicy>> {
    let mut policies: Vec<Box<dyn SortPolicy>> = Vec::new();

    // Maximum imbalance factor is capped by:
    // 1. config.imbalance_factor (from config)
    // 2. threads - 1 (since imbalance = max_entries / avg_entries, and max can be at most threads-1 more than avg)
    let max_imbalance = config.imbalance_factor.min(config.max_threads - 1.0);

    // Generate imbalance factors: 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, ...
    let mut factor = 1.0;
    while factor <= max_imbalance {
        policies.push(Box::new(PolicyImbalanceFactor::new(factor)));
        if factor == 1.0 {
            factor = 2.0;
        } else {
            factor += 2.0;
        }
    }

    policies
}

/// Calculate and return all policy results for a given configuration
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

    get_all_policies(config)
        .into_iter()
        .map(|policy| policy.calculate(config))
        .collect()
}

/// Print detailed analysis for all policies
pub fn print_detailed_analysis(config: SortConfig) {
    println!("=== Sort Policy Analysis (Imbalance Factor) ===");
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
    println!("Merge Threads: {:.0}", config.max_threads);

    let (min_run, max_run) = calculate_run_bounds(config);
    println!("\nRun Length Bounds:");
    println!("  Min Run Size: {:.2} MB", min_run);
    println!("  Max Run Size: {:.2} MB", max_run);
    let run_size = calculate_run_size(config);
    println!("\nCalculated Run Size:");
    println!(
        "  sqrt({:.0} MB * {:.3} MB) = {:.2} MB",
        config.dataset_mb,
        config.page_size_kb / 1024.0,
        run_size
    );

    println!("\n{}", "=".repeat(130));
    println!(
        "{:<35} {:>12} {:>8} {:>8} {:>15} {:>15} {:>12} {:>12}",
        "Policy",
        "Run Size",
        "RunGen",
        "Merge",
        "RunGen Mem",
        "Merge Mem",
        "Total Runs",
        "Imbalance"
    );
    println!("{}", "=".repeat(130));

    for result in calculate_all_policies(config) {
        let total_runs = config.dataset_mb / result.run_size_mb;
        println!(
            "{:<35} {:>12.2} {:>8.0} {:>8.0} {:>15.1} {:>15.1} {:>12.0} {:>12.1}",
            result.name,
            result.run_size_mb,
            result.run_gen_threads,
            result.merge_threads,
            result.run_gen_memory_mb,
            result.merge_memory_mb,
            total_runs,
            result.imbalance_factor
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy_result() {
        let result = PolicyResult::default();
        assert_eq!(result.imbalance_factor, 1.0);
    }

    #[test]
    fn test_imbalance_factor_policies() {
        let config = SortConfig {
            memory_mb: 1024.0,
            dataset_mb: 4096.0,
            page_size_kb: 64.0,
            max_threads: 32.0,
            imbalance_factor: 8.0,
        };

        let results = calculate_all_policies(config);

        // With max_threads=32 and config.imbalance_factor=8.0:
        // max_imbalance = min(8.0, 32-1) = 8.0
        // Factors: 1.0, 2.0, 4.0, 6.0, 8.0
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].imbalance_factor, 1.0);
        assert_eq!(results[1].imbalance_factor, 2.0);
        assert_eq!(results[2].imbalance_factor, 4.0);
        assert_eq!(results[3].imbalance_factor, 6.0);
        assert_eq!(results[4].imbalance_factor, 8.0);

        // All should have merge_threads = max_threads
        for result in &results {
            assert_eq!(result.merge_threads, 32.0);
        }
    }

    #[test]
    fn test_imbalance_factor_capped_by_threads() {
        let config = SortConfig {
            memory_mb: 1024.0,
            dataset_mb: 4096.0,
            page_size_kb: 64.0,
            max_threads: 8.0,
            imbalance_factor: 100.0, // Very high, should be capped
        };

        let results = calculate_all_policies(config);

        // max_imbalance = min(100.0, 8-1) = 7.0
        // Factors: 1.0, 2.0, 4.0, 6.0 (stops before 8.0 since 8.0 > 7.0)
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].imbalance_factor, 1.0);
        assert_eq!(results[1].imbalance_factor, 2.0);
        assert_eq!(results[2].imbalance_factor, 4.0);
        assert_eq!(results[3].imbalance_factor, 6.0);
    }

    #[test]
    fn test_imbalance_factor_capped_by_config() {
        let config = SortConfig {
            memory_mb: 1024.0,
            dataset_mb: 4096.0,
            page_size_kb: 64.0,
            max_threads: 32.0,
            imbalance_factor: 3.0, // Lower than threads-1
        };

        let policies = get_all_policies(config);
        let results: Vec<PolicyResult> =
            policies.into_iter().map(|p| p.calculate(config)).collect();

        // max_imbalance = min(3.0, 32-1) = 3.0
        // Factors: 1.0, 2.0 (stops before 4.0 since 4.0 > 3.0)
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].imbalance_factor, 1.0);
        assert_eq!(results[1].imbalance_factor, 2.0);
    }

    #[test]
    fn test_print_all_policies() {
        let config = SortConfig::default();

        println!("\n=== Testing Imbalance Factor Policies ===");
        print_detailed_analysis(config);

        println!("\n=== Individual Policy Results ===");
        for result in calculate_all_policies(config) {
            println!("{}", result);
        }
    }
}
