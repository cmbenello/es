/// Sort policy with varying imbalance factors
///
/// This module tests different imbalance factors (1.0, 2.0, 4.0, 6.0, 8.0)
/// with fixed memory (32 MB) and merge threads (32)
use crate::sort_policy_run_length::{
    PolicyResult, SortConfig, SortPolicy, calculate_run_bounds, is_configuration_feasible,
};

/// Calculate run size as sqrt(data_size * page_size)
fn calculate_run_size(config: SortConfig) -> f64 {
    let d = config.dataset_mb;
    let p = config.page_size_kb / 1024.0; // Convert to MB
    (d * p).sqrt()
}

/// Policy with imbalance factor 1.0
pub struct PolicyImbalance10;

impl SortPolicy for PolicyImbalance10 {
    fn name(&self) -> String {
        "Imbalance_1.0".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let run_size = calculate_run_size(config);
        let imbalance_factor = 1.0;

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads: config.max_threads,
            merge_threads: config.max_threads,
            run_gen_memory_mb: run_size * config.max_threads,
            merge_memory_mb: (config.dataset_mb / run_size)
                * config.max_threads
                * (config.page_size_kb / 1024.0),
            imbalance_factor,
        }
    }
}

/// Policy with imbalance factor 2.0
pub struct PolicyImbalance20;

impl SortPolicy for PolicyImbalance20 {
    fn name(&self) -> String {
        "Imbalance_2.0".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let run_size = calculate_run_size(config);
        let imbalance_factor = 2.0;

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads: config.max_threads,
            merge_threads: config.max_threads,
            run_gen_memory_mb: run_size * config.max_threads,
            merge_memory_mb: (config.dataset_mb / run_size)
                * config.max_threads
                * (config.page_size_kb / 1024.0),
            imbalance_factor,
        }
    }
}

/// Policy with imbalance factor 4.0
pub struct PolicyImbalance40;

impl SortPolicy for PolicyImbalance40 {
    fn name(&self) -> String {
        "Imbalance_4.0".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let run_size = calculate_run_size(config);
        let imbalance_factor = 4.0;

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads: config.max_threads,
            merge_threads: config.max_threads,
            run_gen_memory_mb: run_size * config.max_threads,
            merge_memory_mb: (config.dataset_mb / run_size)
                * config.max_threads
                * (config.page_size_kb / 1024.0),
            imbalance_factor,
        }
    }
}

/// Policy with imbalance factor 6.0
pub struct PolicyImbalance60;

impl SortPolicy for PolicyImbalance60 {
    fn name(&self) -> String {
        "Imbalance_6.0".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let run_size = calculate_run_size(config);
        let imbalance_factor = 6.0;

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads: config.max_threads,
            merge_threads: config.max_threads,
            run_gen_memory_mb: run_size * config.max_threads,
            merge_memory_mb: (config.dataset_mb / run_size)
                * config.max_threads
                * (config.page_size_kb / 1024.0),
            imbalance_factor,
        }
    }
}

/// Policy with imbalance factor 8.0
pub struct PolicyImbalance80;

impl SortPolicy for PolicyImbalance80 {
    fn name(&self) -> String {
        "Imbalance_8.0".to_string()
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let run_size = calculate_run_size(config);
        let imbalance_factor = 8.0;

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads: config.max_threads,
            merge_threads: config.max_threads,
            run_gen_memory_mb: run_size * config.max_threads,
            merge_memory_mb: (config.dataset_mb / run_size)
                * config.max_threads
                * (config.page_size_kb / 1024.0),
            imbalance_factor,
        }
    }
}

/// Get all available policies
pub fn get_all_policies() -> Vec<Box<dyn SortPolicy>> {
    vec![
        Box::new(PolicyImbalance10),
        Box::new(PolicyImbalance20),
        Box::new(PolicyImbalance40),
        Box::new(PolicyImbalance60),
        Box::new(PolicyImbalance80),
    ]
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

    get_all_policies()
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
        let config = SortConfig::default();

        let policies = get_all_policies();
        let results: Vec<PolicyResult> =
            policies.into_iter().map(|p| p.calculate(config)).collect();

        assert_eq!(results.len(), 5);
        assert_eq!(results[0].imbalance_factor, 1.0);
        assert_eq!(results[1].imbalance_factor, 2.0);
        assert_eq!(results[2].imbalance_factor, 4.0);
        assert_eq!(results[3].imbalance_factor, 6.0);
        assert_eq!(results[4].imbalance_factor, 8.0);

        // All should have merge_threads = 32
        for result in &results {
            assert_eq!(result.merge_threads, 32.0);
        }
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
