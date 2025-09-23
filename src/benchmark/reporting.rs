use super::types::BenchmarkResult;
use std::fmt::Write;

/// Print a comprehensive benchmark summary with main table and detailed I/O statistics
pub fn print_benchmark_summary(results: &[BenchmarkResult]) {
    // Main summary table
    println!("\n{}", "=".repeat(200));
    println!("Benchmark Results Summary");
    println!("{}", "=".repeat(200));
    println!(
        "{:<50} {:<8} {:<12} {:<10} {:<10} {:<10} {:<6} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10} {:<12}",
        "Policy",
        "Threads",
        "Memory",
        "Run Size",
        "Gen Thr",
        "Merge Thr",
        "Runs",
        "Total (s)",
        "RunGen (s)",
        "Merge (s)",
        "Entries",
        "Throughput",
        "Read MB",
        "Write MB",
        "Imbalance"
    );
    println!(
        "{:<50} {:<8} {:<12} {:<10} {:<10} {:<10} {:<6} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10} {:<12}",
        "", "", "", "(MB)", "", "", "", "", "", "", "", "(M entries/s)", "", "", "Factor"
    );
    println!("{}", "-".repeat(200));

    for result in results {
        println!(
            "{:<50} {:<8} {:<12} {:<10.1} {:<10} {:<10} {:<6} {:<10.2} {:<12.2} {:<10.2} {:<12} {:<16.2} {:<10.1} {:<10.1} {:<12}",
            result.policy_name,
            result.threads,
            result.memory_str,
            result.run_size_mb,
            result.run_gen_threads,
            result.merge_threads,
            result.runs,
            result.total_time,
            result.run_gen_time,
            result.merge_time,
            result.entries,
            result.throughput,
            result.read_mb,
            result.write_mb,
            if result.imbalance_factor > 1.0 {
                format!("{:.5}x", result.imbalance_factor)
            } else {
                "N/A".to_string()
            },
        );
    }
    println!("{}", "=".repeat(200));

    // Detailed I/O Statistics
    println!("\nDetailed I/O Statistics Summary:");
    println!("{}", "-".repeat(180));
    println!(
        "{:<50} {:<8} {:<12} {:<20} {:<20} {:<20} {:<20} {:<15}",
        "Policy",
        "Threads",
        "Memory",
        "Run Gen Reads",
        "Run Gen Writes",
        "Merge Reads",
        "Merge Writes",
        "Read Amp."
    );
    println!(
        "{:<50} {:<8} {:<12} {:<20} {:<20} {:<20} {:<20} {:<15}",
        "", "", "", "(ops / MB)", "(ops / MB)", "(ops / MB)", "(ops / MB)", "Factor"
    );
    println!("{}", "-".repeat(180));

    for result in results {
        println!(
            "{:<50} {:<8} {:<12} {:<20} {:<20} {:<20} {:<20} {:<15}",
            result.policy_name,
            result.threads,
            result.memory_str,
            format!(
                "{} / {:.1}",
                result.run_gen_read_ops, result.run_gen_read_mb
            ),
            format!(
                "{} / {:.1}",
                result.run_gen_write_ops, result.run_gen_write_mb
            ),
            format!("{} / {:.1}", result.merge_read_ops, result.merge_read_mb),
            format!("{} / {:.1}", result.merge_write_ops, result.merge_write_mb),
            format!("{:.3}x", result.read_amplification),
        );
    }
    println!("{}", "-".repeat(180));
}

/// Convert bytes to human-readable format (B, KB, MB, GB)
pub fn bytes_to_human_readable(bytes: usize) -> String {
    const GB: usize = 1024 * 1024 * 1024;
    const MB: usize = 1024 * 1024;
    const KB: usize = 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Convert bytes (as f64) to human-readable format
pub fn bytes_f64_to_human_readable(bytes: f64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const KB: f64 = 1024.0;

    if bytes >= GB {
        format!("{:.2} GB", bytes / GB)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes / MB)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes / KB)
    } else {
        format!("{:.0} B", bytes)
    }
}

/// Format throughput in entries per second
pub fn format_throughput(entries: usize, time_seconds: f64) -> String {
    if time_seconds > 0.0 {
        let throughput = entries as f64 / time_seconds;
        if throughput >= 1_000_000.0 {
            format!("{:.2} M entries/s", throughput / 1_000_000.0)
        } else if throughput >= 1_000.0 {
            format!("{:.2} K entries/s", throughput / 1_000.0)
        } else {
            format!("{:.2} entries/s", throughput)
        }
    } else {
        "N/A".to_string()
    }
}

/// Print just the main summary table (without I/O details)
pub fn print_main_summary_table(results: &[BenchmarkResult]) {
    println!("\n{}", "=".repeat(200));
    println!("Benchmark Results Summary");
    println!("{}", "=".repeat(200));
    println!(
        "{:<50} {:<8} {:<12} {:<10} {:<10} {:<10} {:<6} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10} {:<12}",
        "Policy",
        "Threads",
        "Memory",
        "Run Size",
        "Gen Thr",
        "Merge Thr",
        "Runs",
        "Total (s)",
        "RunGen (s)",
        "Merge (s)",
        "Entries",
        "Throughput",
        "Read MB",
        "Write MB",
        "Imbalance"
    );
    println!(
        "{:<50} {:<8} {:<12} {:<10} {:<10} {:<10} {:<6} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10} {:<12}",
        "", "", "", "(MB)", "", "", "", "", "", "", "", "(M entries/s)", "", "", "Factor"
    );
    println!("{}", "-".repeat(200));

    for result in results {
        println!(
            "{:<50} {:<8} {:<12} {:<10.1} {:<10} {:<10} {:<6} {:<10.2} {:<12.2} {:<10.2} {:<12} {:<16.2} {:<10.1} {:<10.1} {:<12}",
            result.policy_name,
            result.threads,
            result.memory_str,
            result.run_size_mb,
            result.run_gen_threads,
            result.merge_threads,
            result.runs,
            result.total_time,
            result.run_gen_time,
            result.merge_time,
            result.entries,
            result.throughput,
            result.read_mb,
            result.write_mb,
            if result.imbalance_factor > 1.0 {
                format!("{:.5}x", result.imbalance_factor)
            } else {
                "N/A".to_string()
            },
        );
    }
    println!("{}", "=".repeat(200));
}

/// Print just the detailed I/O statistics table
pub fn print_io_statistics_table(results: &[BenchmarkResult]) {
    println!("\nDetailed I/O Statistics Summary:");
    println!("{}", "-".repeat(180));
    println!(
        "{:<50} {:<8} {:<12} {:<20} {:<20} {:<20} {:<20} {:<15}",
        "Policy",
        "Threads",
        "Memory",
        "Run Gen Reads",
        "Run Gen Writes",
        "Merge Reads",
        "Merge Writes",
        "Read Amp."
    );
    println!(
        "{:<50} {:<8} {:<12} {:<20} {:<20} {:<20} {:<20} {:<15}",
        "", "", "", "(ops / MB)", "(ops / MB)", "(ops / MB)", "(ops / MB)", "Factor"
    );
    println!("{}", "-".repeat(180));

    for result in results {
        println!(
            "{:<50} {:<8} {:<12} {:<20} {:<20} {:<20} {:<20} {:<15}",
            result.policy_name,
            result.threads,
            result.memory_str,
            format!(
                "{} / {:.1}",
                result.run_gen_read_ops, result.run_gen_read_mb
            ),
            format!(
                "{} / {:.1}",
                result.run_gen_write_ops, result.run_gen_write_mb
            ),
            format!("{} / {:.1}", result.merge_read_ops, result.merge_read_mb),
            format!("{} / {:.1}", result.merge_write_ops, result.merge_write_mb),
            format!("{:.3}x", result.read_amplification),
        );
    }
    println!("{}", "-".repeat(180));
}

/// Generate a CSV format of the benchmark results
pub fn benchmark_results_to_csv(results: &[BenchmarkResult]) -> String {
    let mut csv = String::new();

    // CSV Header
    writeln!(csv, "policy_name,threads,memory_mb,memory_str,run_size_mb,run_gen_threads,merge_threads,runs,total_time,run_gen_time,merge_time,entries,throughput,read_mb,write_mb,run_gen_read_ops,run_gen_read_mb,run_gen_write_ops,run_gen_write_mb,merge_read_ops,merge_read_mb,merge_write_ops,merge_write_mb,imbalance_factor,read_amplification").unwrap();

    // CSV Data
    for result in results {
        writeln!(csv, "{},{},{},{},{:.1},{},{},{},{:.2},{:.2},{:.2},{},{:.2},{:.1},{:.1},{},{:.1},{},{:.1},{},{:.1},{},{:.1},{:.5},{:.3}",
            result.policy_name,
            result.threads,
            result.memory_mb,
            result.memory_str,
            result.run_size_mb,
            result.run_gen_threads,
            result.merge_threads,
            result.runs,
            result.total_time,
            result.run_gen_time,
            result.merge_time,
            result.entries,
            result.throughput,
            result.read_mb,
            result.write_mb,
            result.run_gen_read_ops,
            result.run_gen_read_mb,
            result.run_gen_write_ops,
            result.run_gen_write_mb,
            result.merge_read_ops,
            result.merge_read_mb,
            result.merge_write_ops,
            result.merge_write_mb,
            result.imbalance_factor,
            result.read_amplification
        ).unwrap();
    }

    csv
}

/// Find the best performing result by throughput
pub fn find_best_performer(results: &[BenchmarkResult]) -> Option<&BenchmarkResult> {
    results
        .iter()
        .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
}

/// Print a compact one-line summary for each result
pub fn print_compact_summary(results: &[BenchmarkResult]) {
    println!("\n=== COMPACT SUMMARY ===");
    for result in results {
        println!(
            "{}: {:.2} M entries/s ({:.2}s total, {} entries, {} threads, {})",
            result.policy_name,
            result.throughput,
            result.total_time,
            result.entries,
            result.threads,
            result.memory_str
        );
    }
}

/// Print performance analysis comparing results
pub fn print_performance_analysis(results: &[BenchmarkResult]) {
    if results.is_empty() {
        return;
    }

    println!("\n=== PERFORMANCE ANALYSIS ===");

    if let Some(best) = find_best_performer(results) {
        println!(
            "Best performing policy: {} ({:.2} M entries/s)",
            best.policy_name, best.throughput
        );
    }

    // Calculate average metrics
    let avg_throughput = results.iter().map(|r| r.throughput).sum::<f64>() / results.len() as f64;
    let avg_total_time = results.iter().map(|r| r.total_time).sum::<f64>() / results.len() as f64;
    let avg_read_amp =
        results.iter().map(|r| r.read_amplification).sum::<f64>() / results.len() as f64;

    println!("Average throughput: {:.2} M entries/s", avg_throughput);
    println!("Average total time: {:.2}s", avg_total_time);
    println!("Average read amplification: {:.3}x", avg_read_amp);

    // Find extremes
    let fastest = results
        .iter()
        .min_by(|a, b| a.total_time.partial_cmp(&b.total_time).unwrap());
    let slowest = results
        .iter()
        .max_by(|a, b| a.total_time.partial_cmp(&b.total_time).unwrap());

    if let (Some(fast), Some(slow)) = (fastest, slowest) {
        if fast.policy_name != slow.policy_name {
            let speedup = slow.total_time / fast.total_time;
            println!(
                "Speedup: {:.2}x ({} vs {})",
                speedup, fast.policy_name, slow.policy_name
            );
        }
    }
}
