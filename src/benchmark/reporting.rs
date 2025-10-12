use super::types::BenchmarkResult;
use crate::SortStats;

fn summarize(
    result: &BenchmarkResult,
) -> (
    String, // name
    f64,    // run_size_mb
    usize,  // gen_thr
    usize,  // merge_thr
    usize,  // bench_runs
    usize,  // rg_runs_avg
    f64,    // total_s
    f64,    // run_gen_s
    f64,    // merge_s
    usize,  // entries
    f64,    // throughput_mps
    f64,    // read_mb
    f64,    // write_mb
) {
    let cfg = &result.config;
    let svec: &Vec<SortStats> = &result.stats;
    let runs = svec.len().max(1) as f64;

    let avg_total = svec
        .iter()
        .map(|s| {
            s.run_gen_stats.time_ms as f64 / 1000.0
                + s.per_merge_stats
                    .iter()
                    .map(|m| m.time_ms as f64 / 1000.0)
                    .sum::<f64>()
        })
        .sum::<f64>()
        / runs;
    let avg_run_gen = svec
        .iter()
        .map(|s| s.run_gen_stats.time_ms as f64 / 1000.0)
        .sum::<f64>()
        / runs;
    let avg_merge = svec
        .iter()
        .map(|s| {
            s.per_merge_stats
                .iter()
                .map(|m| m.time_ms as f64 / 1000.0)
                .sum::<f64>()
        })
        .sum::<f64>()
        / runs;

    let rg_read_mb = svec
        .iter()
        .map(|s| {
            s.run_gen_stats
                .io_stats
                .as_ref()
                .map(|io| io.read_bytes as f64 / (1024.0 * 1024.0))
                .unwrap_or(0.0)
        })
        .sum::<f64>()
        / runs;
    let m_read_mb = svec
        .iter()
        .map(|s| {
            s.per_merge_stats
                .iter()
                .filter_map(|m| m.io_stats.as_ref())
                .map(|io| io.read_bytes as f64 / (1024.0 * 1024.0))
                .sum::<f64>()
        })
        .sum::<f64>()
        / runs;
    let rg_write_mb = svec
        .iter()
        .map(|s| {
            s.run_gen_stats
                .io_stats
                .as_ref()
                .map(|io| io.write_bytes as f64 / (1024.0 * 1024.0))
                .unwrap_or(0.0)
        })
        .sum::<f64>()
        / runs;
    let m_write_mb = svec
        .iter()
        .map(|s| {
            s.per_merge_stats
                .iter()
                .filter_map(|m| m.io_stats.as_ref())
                .map(|io| io.write_bytes as f64 / (1024.0 * 1024.0))
                .sum::<f64>()
        })
        .sum::<f64>()
        / runs;

    let read_mb = rg_read_mb + m_read_mb;
    let write_mb = rg_write_mb + m_write_mb;

    // Average number of runs generated during run generation
    let avg_rg_runs = svec
        .iter()
        .map(|s| s.run_gen_stats.num_runs as f64)
        .sum::<f64>()
        / runs;
    let avg_rg_runs_usize = avg_rg_runs.round() as usize;

    let entries = svec
        .iter()
        .find_map(|s| {
            s.per_merge_stats
                .last()
                .map(|m| m.merge_entry_num.iter().sum::<u64>() as usize)
        })
        .unwrap_or(0);
    let throughput_mps = if avg_total > 0.0 {
        entries as f64 / avg_total / 1_000_000.0
    } else {
        0.0
    };

    (
        cfg.config_name.clone(),
        cfg.run_size_mb,
        cfg.run_gen_threads,
        cfg.merge_threads,
        svec.len(),
        avg_rg_runs_usize,
        avg_total,
        avg_run_gen,
        avg_merge,
        entries,
        throughput_mps,
        read_mb,
        write_mb,
    )
}

/// Print merge operations summary table
fn print_merge_operations_table(result: &BenchmarkResult, max_merges: usize) {
    let merges_to_show = max_merges.min(5);
    let start_merge = max_merges.saturating_sub(merges_to_show);

    println!("\n{}", "=".repeat(160));
    println!(
        "Merge Operations Summary (showing last {} of {} merge passes)",
        merges_to_show, max_merges
    );
    println!("{}", "=".repeat(160));

    // Build header dynamically based on number of merges to show
    let mut header = format!("{:<20}", "Config");
    for mi in (start_merge..max_merges).map(|i| i + 1) {
        header.push_str(&format!(
            " {:<10} {:<10} {:<10} {:<12} {:<12}",
            format!("M{}PAvg", mi),
            format!("M{}PMax", mi),
            format!("M{}Imbal", mi),
            format!("M{}Slow", mi),
            format!("M{}Fast", mi)
        ));
    }
    println!("{}", header);
    println!("{}", "-".repeat(160));

    // Print each benchmark run on one line
    for (run_idx, sort_stats) in result.stats.iter().enumerate() {
        let name = format!("{}[{}]", result.config.config_name, run_idx + 1);
        let mut line = format!("{:<20}", name);

        for mi in start_merge..max_merges {
            if mi < sort_stats.per_merge_stats.len() {
                let m = &sort_stats.per_merge_stats[mi];
                let total_entries: u64 = m.merge_entry_num.iter().sum();
                let fanin = m.merge_entry_num.len();
                let part_avg = if fanin > 0 {
                    total_entries / fanin as u64
                } else {
                    0
                };
                let part_max = m.merge_entry_num.iter().max().copied().unwrap_or(0);
                let imbalance = if part_avg > 0 {
                    part_max as f64 / part_avg as f64
                } else {
                    0.0
                };

                // For single run, slowest and fastest are the same
                let time_s = m.time_ms as f64 / 1000.0;

                line.push_str(&format!(
                    " {:<10} {:<10} {:<10.2} {:<12.2} {:<12.2}",
                    part_avg, part_max, imbalance, time_s, time_s
                ));
            } else {
                // No merge data for this pass
                line.push_str(&format!(
                    " {:<10} {:<10} {:<10} {:<12} {:<12}",
                    "-", "-", "-", "-", "-"
                ));
            }
        }
        println!("{}", line);
    }

    // Print aggregate statistics
    println!("{}", "-".repeat(160));
    let aggregate_name = format!("{}[avg]", result.config.config_name);
    let mut agg_line = format!("{:<20}", aggregate_name);

    for mi in start_merge..max_merges {
        let mut part_avgs: Vec<u64> = Vec::new();
        let mut part_maxs: Vec<u64> = Vec::new();
        let mut imbalances: Vec<f64> = Vec::new();
        let mut times_s: Vec<f64> = Vec::new();

        for s in &result.stats {
            if mi < s.per_merge_stats.len() {
                let m = &s.per_merge_stats[mi];
                let total_entries: u64 = m.merge_entry_num.iter().sum();
                let fanin = m.merge_entry_num.len();
                let part_avg = if fanin > 0 {
                    total_entries / fanin as u64
                } else {
                    0
                };
                let part_max = m.merge_entry_num.iter().max().copied().unwrap_or(0);
                let imbalance = if part_avg > 0 {
                    part_max as f64 / part_avg as f64
                } else {
                    0.0
                };

                part_avgs.push(part_avg);
                part_maxs.push(part_max);
                imbalances.push(imbalance);
                times_s.push(m.time_ms as f64 / 1000.0);
            }
        }

        if !times_s.is_empty() {
            let avg_part_avg = part_avgs.iter().sum::<u64>() / part_avgs.len() as u64;
            let avg_part_max = part_maxs.iter().sum::<u64>() / part_maxs.len() as u64;
            let avg_imbalance = imbalances.iter().sum::<f64>() / imbalances.len() as f64;
            let slowest_time = times_s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let fastest_time = times_s.iter().cloned().fold(f64::INFINITY, f64::min);

            agg_line.push_str(&format!(
                " {:<10} {:<10} {:<10.2} {:<12.2} {:<12.2}",
                avg_part_avg, avg_part_max, avg_imbalance, slowest_time, fastest_time
            ));
        } else {
            agg_line.push_str(&format!(
                " {:<10} {:<10} {:<10} {:<12} {:<12}",
                "-", "-", "-", "-", "-"
            ));
        }
    }
    println!("{}", agg_line);
    println!("{}", "=".repeat(160));
}

/// Print detailed I/O statistics table
fn print_detailed_io_statistics_table(result: &BenchmarkResult, max_merges: usize) {
    println!("\n{}", "=".repeat(160));
    println!("Detailed I/O Statistics");
    println!("{}", "=".repeat(160));

    // Build header dynamically based on number of merges
    let mut io_header = format!(
        "{:<20} {:<12} {:<12}",
        "Config", "RG Read MB", "RG Write MB"
    );
    for mi in 0..max_merges {
        io_header.push_str(&format!(
            " {:<12} {:<12}",
            format!("M{}Read MB", mi + 1),
            format!("M{}Write MB", mi + 1)
        ));
    }
    println!("{}", io_header);
    println!("{}", "-".repeat(160));

    // Print each benchmark run
    for (run_idx, sort_stats) in result.stats.iter().enumerate() {
        let rg_read_mb = sort_stats
            .run_gen_stats
            .io_stats
            .as_ref()
            .map(|io| io.read_bytes as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        let rg_write_mb = sort_stats
            .run_gen_stats
            .io_stats
            .as_ref()
            .map(|io| io.write_bytes as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);

        let name = format!("{}[{}]", result.config.config_name, run_idx + 1);
        let mut io_line = format!("{:<20} {:<12.1} {:<12.1}", name, rg_read_mb, rg_write_mb);

        for mi in 0..max_merges {
            if mi < sort_stats.per_merge_stats.len() {
                let m = &sort_stats.per_merge_stats[mi];
                let merge_read_mb = m
                    .io_stats
                    .as_ref()
                    .map(|io| io.read_bytes as f64 / (1024.0 * 1024.0))
                    .unwrap_or(0.0);
                let merge_write_mb = m
                    .io_stats
                    .as_ref()
                    .map(|io| io.write_bytes as f64 / (1024.0 * 1024.0))
                    .unwrap_or(0.0);

                io_line.push_str(&format!(
                    " {:<12.1} {:<12.1}",
                    merge_read_mb, merge_write_mb
                ));
            } else {
                io_line.push_str(&format!(" {:<12} {:<12}", "-", "-"));
            }
        }

        println!("{}", io_line);
    }

    // Print aggregate statistics
    println!("{}", "-".repeat(160));
    let aggregate_name = format!("{}[avg]", result.config.config_name);

    let num_runs = result.stats.len() as f64;
    let avg_rg_read_mb = result
        .stats
        .iter()
        .map(|s| {
            s.run_gen_stats
                .io_stats
                .as_ref()
                .map(|io| io.read_bytes as f64 / (1024.0 * 1024.0))
                .unwrap_or(0.0)
        })
        .sum::<f64>()
        / num_runs;
    let avg_rg_write_mb = result
        .stats
        .iter()
        .map(|s| {
            s.run_gen_stats
                .io_stats
                .as_ref()
                .map(|io| io.write_bytes as f64 / (1024.0 * 1024.0))
                .unwrap_or(0.0)
        })
        .sum::<f64>()
        / num_runs;

    let mut agg_io_line = format!(
        "{:<20} {:<12.1} {:<12.1}",
        aggregate_name, avg_rg_read_mb, avg_rg_write_mb
    );

    for mi in 0..max_merges {
        let avg_merge_read_mb = result
            .stats
            .iter()
            .map(|s| {
                if mi < s.per_merge_stats.len() {
                    s.per_merge_stats[mi]
                        .io_stats
                        .as_ref()
                        .map(|io| io.read_bytes as f64 / (1024.0 * 1024.0))
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / num_runs;
        let avg_merge_write_mb = result
            .stats
            .iter()
            .map(|s| {
                if mi < s.per_merge_stats.len() {
                    s.per_merge_stats[mi]
                        .io_stats
                        .as_ref()
                        .map(|io| io.write_bytes as f64 / (1024.0 * 1024.0))
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / num_runs;

        agg_io_line.push_str(&format!(
            " {:<12.1} {:<12.1}",
            avg_merge_read_mb, avg_merge_write_mb
        ));
    }

    println!("{}", agg_io_line);
    println!("{}", "=".repeat(160));
}

pub fn print_benchmark_summary(result: &BenchmarkResult) {
    println!("\n{}", "=".repeat(160));
    println!("Benchmark Results Summary");
    println!("{}", "=".repeat(160));
    println!(
        "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10}",
        "Config",
        "Run Size",
        "RG Runs",
        "Gen Thr",
        "Merge Thr",
        "Total (s)",
        "RunGen (s)",
        "Merge (s)",
        "Entries",
        "Throughput",
        "Read MB",
        "Write MB",
    );
    println!(
        "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10}",
        "", "(MB)", "", "", "", "", "", "", "", "(M entries/s)", "", "",
    );
    println!("{}", "-".repeat(160));

    // Print individual runs
    for (run_idx, sort_stats) in result.stats.iter().enumerate() {
        let cfg = &result.config;

        let total_s = sort_stats.run_gen_stats.time_ms as f64 / 1000.0
            + sort_stats
                .per_merge_stats
                .iter()
                .map(|m| m.time_ms as f64 / 1000.0)
                .sum::<f64>();
        let run_gen_s = sort_stats.run_gen_stats.time_ms as f64 / 1000.0;
        let merge_s = sort_stats
            .per_merge_stats
            .iter()
            .map(|m| m.time_ms as f64 / 1000.0)
            .sum::<f64>();

        let read_mb = sort_stats
            .run_gen_stats
            .io_stats
            .as_ref()
            .map(|io| io.read_bytes as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
            + sort_stats
                .per_merge_stats
                .iter()
                .filter_map(|m| m.io_stats.as_ref())
                .map(|io| io.read_bytes as f64 / (1024.0 * 1024.0))
                .sum::<f64>();
        let write_mb = sort_stats
            .run_gen_stats
            .io_stats
            .as_ref()
            .map(|io| io.write_bytes as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
            + sort_stats
                .per_merge_stats
                .iter()
                .filter_map(|m| m.io_stats.as_ref())
                .map(|io| io.write_bytes as f64 / (1024.0 * 1024.0))
                .sum::<f64>();

        let entries = sort_stats
            .per_merge_stats
            .last()
            .map(|m| m.merge_entry_num.iter().sum::<u64>() as usize)
            .unwrap_or(0);
        let throughput_mps = if total_s > 0.0 {
            entries as f64 / total_s / 1_000_000.0
        } else {
            0.0
        };

        let name = format!("{}[{}]", cfg.config_name, run_idx + 1);

        println!(
            "{:<20} {:<10.1} {:<10} {:<10} {:<10} {:<12.2} {:<10.2} {:<10.2} {:<12} {:<16.2} {:<10.1} {:<10.1}",
            name,
            cfg.run_size_mb,
            sort_stats.run_gen_stats.num_runs,
            cfg.run_gen_threads,
            cfg.merge_threads,
            total_s,
            run_gen_s,
            merge_s,
            entries,
            throughput_mps,
            read_mb,
            write_mb,
        );
    }

    // Print aggregate line
    let (
        name,
        run_size_mb,
        gen_thr,
        merge_thr,
        _bench_runs,
        rg_runs,
        total_s,
        run_gen_s,
        merge_s,
        entries,
        throughput_mps,
        read_mb,
        write_mb,
    ) = summarize(result);
    let aggregate_name = format!("{}[avg]", name);
    println!(
        "{:<20} {:<10.1} {:<10} {:<10} {:<10} {:<12.2} {:<10.2} {:<10.2} {:<12} {:<16.2} {:<10.1} {:<10.1}",
        aggregate_name,
        run_size_mb,
        rg_runs,
        gen_thr,
        merge_thr,
        total_s,
        run_gen_s,
        merge_s,
        entries,
        throughput_mps,
        read_mb,
        write_mb,
    );
    println!("{}", "=".repeat(160));

    // Print merge operation details and I/O statistics
    let max_merges = result
        .stats
        .iter()
        .map(|s| s.per_merge_stats.len())
        .max()
        .unwrap_or(0);

    if max_merges > 0 {
        print_merge_operations_table(result, max_merges);
        print_detailed_io_statistics_table(result, max_merges);
    }
}

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
