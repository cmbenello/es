/// Sort policy that varies thread count with fixed run size
///
/// This module provides policies that increase thread count progressively
/// (1, 2, 4, 8, 16, 32, ...) until reaching memory constraints,
/// while keeping run size fixed at sqrt(data_size * page_size).
use super::sort_policy_run_length::{PolicyResult, SortConfig, SortPolicy};

/// Calculate run size as sqrt(data_size * page_size)
fn calculate_run_size(config: SortConfig) -> f64 {
    let d = config.dataset_mb;
    let p = config.page_size_kb / 1024.0; // Convert to MB
    (d * p).sqrt()
}

/// Check if a thread count is feasible given memory constraints
fn is_thread_count_feasible(config: SortConfig, threads: f64) -> bool {
    let run_size = calculate_run_size(config);
    let p = config.page_size_kb / 1024.0; // Convert to MB

    // Run generation constraint: run_size * threads <= memory
    let run_gen_memory = run_size * threads;

    // Merge constraint: fanin * page_size * threads <= memory
    let fanin = config.dataset_mb / run_size;
    let merge_memory = fanin * p * threads;

    run_gen_memory <= config.memory_mb && merge_memory <= config.memory_mb
}

/// Generic policy for a specific thread count
pub struct PolicyThreadCount {
    threads: f64,
}

impl PolicyThreadCount {
    pub fn new(threads: f64) -> Self {
        Self { threads }
    }
}

impl SortPolicy for PolicyThreadCount {
    fn name(&self) -> String {
        format!("Thread_{}", self.threads as i32)
    }

    fn calculate(&self, config: SortConfig) -> PolicyResult {
        let run_size = calculate_run_size(config);

        PolicyResult {
            name: self.name(),
            run_size_mb: run_size,
            run_gen_threads: self.threads,
            merge_threads: self.threads,
            run_gen_memory_mb: run_size * self.threads,
            merge_memory_mb: (config.dataset_mb / run_size)
                * self.threads
                * (config.page_size_kb / 1024.0),
            imbalance_factor: config.imbalance_factor,
        }
    }
}

/// Get all thread count policies up to max_threads
pub fn get_all_thread_policies(config: SortConfig) -> Vec<Box<dyn SortPolicy>> {
    let mut policies: Vec<Box<dyn SortPolicy>> = Vec::new();

    // Progressive doubling: 1, 2, 4, 8, ... up to max_threads (inclusive)
    let mut threads = 1.0;
    while threads <= config.max_threads {
        policies.push(Box::new(PolicyThreadCount::new(threads)));
        threads *= 2.0;
    }

    policies
}

/// Calculate and return all thread count policy results for a given configuration
pub fn calculate_all_thread_policies(config: SortConfig) -> Vec<PolicyResult> {
    get_all_thread_policies(config)
        .into_iter()
        .map(|policy| policy.calculate(config))
        .collect()
}

/// Print detailed analysis for all thread count policies
pub fn print_thread_count_analysis(config: SortConfig) {
    println!("=== Thread Count Policy Analysis ===");
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

    let run_size = calculate_run_size(config);
    println!("\nFixed Run Size:");
    println!(
        "  sqrt({:.0} MB * {:.3} MB) = {:.2} MB",
        config.dataset_mb,
        config.page_size_kb / 1024.0,
        run_size
    );

    println!("\n{}", "=".repeat(120));
    println!(
        "{:<35} {:>12} {:>8} {:>8} {:>15} {:>15} {:>12}",
        "Policy", "Run Size", "RunGen", "Merge", "RunGen Mem", "Merge Mem", "Total Runs"
    );
    println!("{}", "=".repeat(120));

    for result in calculate_all_thread_policies(config) {
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
    fn test_run_size_calculation() {
        let config = SortConfig {
            memory_mb: 1024.0,
            dataset_mb: 1024.0,
            page_size_kb: 64.0,
            max_threads: 32.0,
            imbalance_factor: 1.0,
        };

        let run_size = calculate_run_size(config);
        // sqrt(1024 * 0.0625) = sqrt(64) = 8.0 MB
        assert!((run_size - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_all_thread_policies_same_run_size() {
        let config = SortConfig {
            memory_mb: 2048.0,
            dataset_mb: 8192.0,
            page_size_kb: 64.0,
            max_threads: 32.0,
            imbalance_factor: 1.0,
        };

        let results = calculate_all_thread_policies(config);
        assert_eq!(results.len(), 6);

        let expected_run_size = calculate_run_size(config);

        // All policies should have the same run size
        for result in &results {
            assert!((result.run_size_mb - expected_run_size).abs() < 0.01);
        }
    }

    #[test]
    fn test_thread_counts() {
        let config = SortConfig {
            memory_mb: 4096.0,
            dataset_mb: 4096.0,
            page_size_kb: 64.0,
            max_threads: 32.0,
            imbalance_factor: 1.0,
        };

        let results = calculate_all_thread_policies(config);
        let expected_threads = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];

        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.run_gen_threads, expected_threads[i]);
            assert_eq!(result.merge_threads, expected_threads[i]);
        }
    }

    #[test]
    fn test_memory_calculations() {
        let config = SortConfig {
            memory_mb: 1024.0,
            dataset_mb: 2048.0,
            page_size_kb: 64.0,
            max_threads: 32.0,
            imbalance_factor: 1.0,
        };

        let policy = PolicyThreadCount::new(8.0);
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
    fn test_policies_up_to_max_threads() {
        let config = SortConfig {
            memory_mb: 4096.0,
            dataset_mb: 4096.0,
            page_size_kb: 64.0,
            max_threads: 16.0,
            imbalance_factor: 1.0,
        };

        let results = calculate_all_thread_policies(config);
        // Should have: 1, 2, 4, 8, 16 = 5 policies
        assert_eq!(results.len(), 5);

        let last_thread_count = results.last().unwrap().run_gen_threads;
        assert_eq!(last_thread_count, 16.0);
    }

    #[test]
    fn test_print_thread_count_analysis() {
        let config = SortConfig {
            memory_mb: 32768.0,         // 32 GB
            dataset_mb: 200.0 * 1024.0, // 200 GB
            page_size_kb: 64.0,         // 64 KB
            max_threads: 32.0,          // 32 threads
            imbalance_factor: 1.0,
        };

        println!("\n=== Testing Thread Count Policies ===");
        print_thread_count_analysis(config);

        println!("\n=== Individual Policy Results ===");
        for result in calculate_all_thread_policies(config) {
            println!("{}", result);
        }
    }
}
