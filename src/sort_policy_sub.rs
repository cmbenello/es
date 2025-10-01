/// External sorting policy parameter calculator
///
/// This module computes run generation threads, merge threads, and run sizes
/// for different external sorting policies.

#[derive(Debug, Clone)]
pub struct PolicyParameters {
    pub name: String,
    pub run_size_mb: f64,
    pub run_gen_threads: f64,
    pub merge_threads: f64,
    pub run_gen_memory_mb: f64,
    pub merge_memory_mb: f64,
}

impl std::fmt::Display for PolicyParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]: Run Size: {:.1} MB, Run Gen Threads: {:.1}, Merge Threads: {:.1}, Run Gen Memory: {:.1} MB, Merge Memory: {:.1} MB",
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
    /// Total memory in MB
    pub memory_mb: f64,
    /// Dataset size in MB
    pub dataset_mb: f64,
    /// Page size in KB
    pub page_size_kb: f64,
    /// Maximum thread count (used as fixed threads for policies 2 and 3)
    pub max_threads: f64,
}

impl Default for SortConfig {
    fn default() -> Self {
        Self {
            memory_mb: 4096.0,   // 4 GB
            dataset_mb: 32768.0, // 32 GB
            page_size_kb: 64.0,  // 64 KB
            max_threads: 32.0,   // Maximum/fixed thread count
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Policy {
    /// Policy 1: Tiny runs to maximize parallelism in run generation, serial merge
    TinyRunsMaxParallelGenSerialMerge,
    /// Policy 2: Huge runs with serial generation, maximize merge parallelism
    HugeRunsSerialGenMaxParallelMerge,
    /// Policy 3: Medium runs with balanced parallelism at constraint intersection
    MediumRunsBalancedParallelism,
    /// Policy 4: Generation-bound runs with max parallel generation, limited parallel merge
    GenBoundRunsMaxParallelGenLimitedParallelMerge,
    /// Policy 5: Merge-bound runs with limited parallel generation, max parallel merge
    MergeBoundRunsLimitedParallelGenMaxParallelMerge,
}

impl Policy {
    /// Get all policies
    pub fn all() -> Vec<Policy> {
        vec![
            Policy::TinyRunsMaxParallelGenSerialMerge,
            Policy::HugeRunsSerialGenMaxParallelMerge,
            Policy::MediumRunsBalancedParallelism,
            Policy::GenBoundRunsMaxParallelGenLimitedParallelMerge,
            Policy::MergeBoundRunsLimitedParallelGenMaxParallelMerge,
        ]
    }

    /// Get policy name
    pub fn name(&self) -> String {
        match self {
            Policy::TinyRunsMaxParallelGenSerialMerge => {
                "TinyRuns_MaxParallelGen_SerialMerge".to_string()
            }
            Policy::HugeRunsSerialGenMaxParallelMerge => {
                "HugeRuns_SerialGen_MaxParallelMerge".to_string()
            }
            Policy::MediumRunsBalancedParallelism => "MediumRuns_BalancedParallelism".to_string(),
            Policy::GenBoundRunsMaxParallelGenLimitedParallelMerge => {
                "GenBoundRuns_MaxParallelGen_LimitedParallelMerge".to_string()
            }
            Policy::MergeBoundRunsLimitedParallelGenMaxParallelMerge => {
                "MergeBoundRuns_LimitedParallelGen_MaxParallelMerge".to_string()
            }
        }
    }
}

/// Calculate policy parameters
pub fn calculate_policy_parameters(policy: Policy, config: SortConfig) -> PolicyParameters {
    let m = config.memory_mb;
    let d = config.dataset_mb;
    let p = config.page_size_kb / 1024.0; // Convert KB to MB
    let t_max = config.max_threads;

    match policy {
        Policy::TinyRunsMaxParallelGenSerialMerge => {
            // Policy 1: Tiny runs to maximize parallelism in run generation, serial merge
            // Run size = D × P × T / M where T=1
            let run_size = (d * p * 1.0) / m;
            let run_gen_threads = (m / run_size).min(t_max);
            let merge_threads = (m / ((d / run_size) * p)).min(t_max);

            PolicyParameters {
                name: policy.name(),
                run_size_mb: run_size,
                run_gen_threads,
                merge_threads,
                run_gen_memory_mb: run_size * run_gen_threads,
                merge_memory_mb: (d / run_size) * merge_threads * p,
            }
        }

        Policy::HugeRunsSerialGenMaxParallelMerge => {
            // Policy 2: Huge runs with serial generation, maximize merge parallelism
            // Run size = M / T where T=1
            let run_size = m / 1.0;
            let run_gen_threads = 1.0;
            let merge_threads = (m / ((d / run_size) * p)).min(t_max);

            PolicyParameters {
                name: policy.name(),
                run_size_mb: run_size,
                run_gen_threads,
                merge_threads,
                run_gen_memory_mb: run_size * run_gen_threads,
                merge_memory_mb: (d / run_size) * merge_threads * p,
            }
        }

        Policy::MediumRunsBalancedParallelism => {
            // Policy 3: Medium runs with balanced parallelism at constraint intersection
            // T = M / sqrt(D × P)
            let t = m / (d * p).sqrt();
            let run_size = m / t;
            let run_gen_threads = t.min(t_max);
            let merge_threads = t.min(t_max);

            PolicyParameters {
                name: policy.name(),
                run_size_mb: run_size,
                run_gen_threads,
                merge_threads,
                run_gen_memory_mb: run_size * t,
                merge_memory_mb: (d / run_size) * t * p,
            }
        }

        Policy::GenBoundRunsMaxParallelGenLimitedParallelMerge => {
            // Policy 4: Generation-bound runs with max parallel generation, limited parallel merge
            // Run size = M / T where T=max_threads
            let run_size = m / t_max;
            let run_gen_threads = t_max;
            let merge_threads = (m / ((d / run_size) * p)).min(t_max);

            PolicyParameters {
                name: policy.name(),
                run_size_mb: run_size,
                run_gen_threads,
                merge_threads,
                run_gen_memory_mb: run_size * run_gen_threads,
                merge_memory_mb: (d / run_size) * merge_threads * p,
            }
        }

        Policy::MergeBoundRunsLimitedParallelGenMaxParallelMerge => {
            // Policy 5: Merge-bound runs with limited parallel generation, max parallel merge
            // Run size = D × P × T / M where T=max_threads
            let run_size = (d * p * t_max) / m;
            let run_gen_threads = (m / run_size).min(t_max);
            let merge_threads = t_max;

            PolicyParameters {
                name: policy.name(),
                run_size_mb: run_size,
                run_gen_threads,
                merge_threads,
                run_gen_memory_mb: run_size * run_gen_threads,
                merge_memory_mb: (d / run_size) * merge_threads * p,
            }
        }
    }
}

/// Get all policy parameters for a given configuration
pub fn get_all_policies(config: SortConfig) -> Vec<(Policy, PolicyParameters)> {
    Policy::all()
        .into_iter()
        .map(|p| (p, calculate_policy_parameters(p, config)))
        .collect()
}

pub fn get_policy_by_name(
    name_prefix: &str,
    config: SortConfig,
) -> Option<(Policy, PolicyParameters)> {
    Policy::all()
        .into_iter()
        .find(|p| p.name().starts_with(name_prefix))
        .map(|p| (p, calculate_policy_parameters(p, config)))
}

/// Check if a policy configuration is feasible
pub fn is_feasible(params: &PolicyParameters, memory_limit_mb: f64) -> bool {
    params.run_gen_memory_mb <= memory_limit_mb && params.merge_memory_mb <= memory_limit_mb
}

/// Get the memory bottleneck phase
pub fn get_bottleneck(params: &PolicyParameters) -> &'static str {
    if params.run_gen_memory_mb > params.merge_memory_mb {
        "Run Generation"
    } else {
        "Merge Phase"
    }
}

/// Calculate number of runs
pub fn calculate_run_count(dataset_mb: f64, run_size_mb: f64) -> f64 {
    dataset_mb / run_size_mb
}

/// Pretty print policy parameters
pub fn print_policy_params(policy: Policy, params: &PolicyParameters, config: &SortConfig) {
    println!("\n{} ({})", params.name, format!("{:?}", policy));
    println!("  Run Size: {:.1} MB", params.run_size_mb);

    // Show if run gen threads are capped
    let uncapped_run_gen = config.memory_mb / params.run_size_mb;
    if uncapped_run_gen > config.max_threads && params.run_gen_threads == config.max_threads {
        println!(
            "  Run Generation: {:.1} threads (capped from {:.0}) × {:.1} MB = {:.1} MB",
            params.run_gen_threads, uncapped_run_gen, params.run_size_mb, params.run_gen_memory_mb
        );
    } else {
        println!(
            "  Run Generation: {:.1} threads × {:.1} MB = {:.1} MB",
            params.run_gen_threads, params.run_size_mb, params.run_gen_memory_mb
        );
    }

    // Show if merge threads are capped
    let uncapped_merge = config.memory_mb
        / ((config.dataset_mb / params.run_size_mb) * (config.page_size_kb / 1024.0));
    if uncapped_merge > config.max_threads && params.merge_threads == config.max_threads {
        println!(
            "  Merge Phase: {:.1} threads (capped from {:.0}) × ({:.0}/{:.1} runs) × {:.1} KB = {:.1} MB",
            params.merge_threads,
            uncapped_merge,
            config.dataset_mb,
            params.run_size_mb,
            config.page_size_kb,
            params.merge_memory_mb
        );
    } else {
        println!(
            "  Merge Phase: {:.1} threads × ({:.0}/{:.1} runs) × {:.1} KB = {:.1} MB",
            params.merge_threads,
            config.dataset_mb,
            params.run_size_mb,
            config.page_size_kb,
            params.merge_memory_mb
        );
    }

    println!(
        "  Total Runs: {:.0}",
        calculate_run_count(config.dataset_mb, params.run_size_mb)
    );
    println!("  Bottleneck: {}", get_bottleneck(params));
    println!(
        "  Feasible: {}",
        if is_feasible(params, config.memory_mb) {
            "Yes"
        } else {
            "No"
        }
    );
}

/// Print comparison table
pub fn print_comparison_table(config: &SortConfig) {
    println!(
        "\nPolicy Comparison for M={:.0}GB, D={:.0}GB, Max T={:.0}",
        config.memory_mb / 1024.0,
        config.dataset_mb / 1024.0,
        config.max_threads
    );
    println!("{}", "=".repeat(140));
    println!(
        "{:<40} {:>15} {:>20} {:>20} {:>20} {:>20}",
        "Policy",
        "Run Size (MB)",
        "Run Gen Threads",
        "Merge Threads",
        "Run Gen Memory (MB)",
        "Merge Memory (MB)"
    );
    println!("{}", "=".repeat(140));

    for (_policy, params) in get_all_policies(*config) {
        println!(
            "{:<40} {:>15.1} {:>20.1} {:>20.1} {:>20.1} {:>20.1}",
            params.name,
            params.run_size_mb,
            params.run_gen_threads,
            params.merge_threads,
            params.run_gen_memory_mb,
            params.merge_memory_mb
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy1() {
        let config = SortConfig {
            memory_mb: 1024.0,   // 1 GB
            dataset_mb: 32768.0, // 32 GB
            page_size_kb: 64.0,  // 64 KB
            max_threads: 32.0,
        };

        let params = calculate_policy_parameters(Policy::TinyRunsMaxParallelGenSerialMerge, config);

        // Expected: run_size = 32768 * 64 / 1024 / 1024 = 2 MB
        assert!((params.run_size_mb - 2.0).abs() < 0.01);
        assert!((params.run_gen_threads - 32.0).abs() < 0.01);
    }

    #[test]
    fn test_policy2() {
        let config = SortConfig {
            memory_mb: 1024.0,
            dataset_mb: 32768.0,
            page_size_kb: 64.0,
            max_threads: 32.0,
        };

        let params = calculate_policy_parameters(
            Policy::GenBoundRunsMaxParallelGenLimitedParallelMerge,
            config,
        );

        // Expected: run_size = 1024 / 32 = 32 MB
        assert!((params.run_size_mb - 32.0).abs() < 0.01);
        assert_eq!(params.run_gen_threads, 32.0);
    }

    #[test]
    fn test_all_policies() {
        let config = SortConfig {
            memory_mb: 1024.0,
            dataset_mb: 1024.0 * 100.0, // 128 GB
            page_size_kb: 64.0,
            max_threads: 40.0,
        };
        print_comparison_table(&config);
    }
}
