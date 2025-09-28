use std::path::PathBuf;

#[derive(Default)]
pub struct BenchmarkStats {
    pub total_time: f64,
    pub run_gen_time: f64,
    pub merge_time: f64,
    pub runs_count: usize,
    pub run_gen_read_ops: u64,
    pub run_gen_read_mb: f64,
    pub run_gen_write_ops: u64,
    pub run_gen_write_mb: f64,
    pub merge_read_ops: u64,
    pub merge_read_mb: f64,
    pub merge_write_ops: u64,
    pub merge_write_mb: f64,
    pub imbalance_sum: f64,
    pub imbalance_count: usize,
}

#[derive(Clone)]
pub struct BenchmarkResult {
    pub policy_name: String,
    pub threads: usize,
    pub memory_mb: usize,
    pub memory_str: String,
    pub run_size_mb: f64,
    pub run_gen_threads: usize,
    pub merge_threads: usize,
    pub runs: usize,
    pub total_time: f64,
    pub run_gen_time: f64,
    pub merge_time: f64,
    pub entries: usize,
    pub throughput: f64,
    pub read_mb: f64,
    pub write_mb: f64,
    pub run_gen_read_ops: u64,
    pub run_gen_read_mb: f64,
    pub run_gen_write_ops: u64,
    pub run_gen_write_mb: f64,
    pub merge_read_ops: u64,
    pub merge_read_mb: f64,
    pub merge_write_ops: u64,
    pub merge_write_mb: f64,
    pub imbalance_factor: f64,
    pub read_amplification: f64,
}

pub struct BenchmarkConfig {
    pub threads: usize,
    pub memory_mb: usize,
    pub warmup_runs: usize,
    pub benchmark_runs: usize,
    pub verify: bool,
    pub ovc: bool,
    pub temp_dir: PathBuf,
    pub sketch_size: usize,
    pub sketch_sampling_interval: usize,
    pub run_indexing_interval: usize,
    pub boundary_imbalance_factor: f64,
}

impl BenchmarkConfig {
    pub fn new(threads: usize, memory_mb: usize, temp_dir: PathBuf) -> Self {
        Self {
            threads,
            memory_mb,
            temp_dir,
            warmup_runs: 0,
            benchmark_runs: 1,
            verify: false,
            ovc: false,
            sketch_size: 200,
            sketch_sampling_interval: 1000,
            run_indexing_interval: 1000,
            boundary_imbalance_factor: 1.0
        }
    }
}

impl BenchmarkStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn accumulate_from_output(&mut self, output: &dyn crate::SortOutput) {
        if let Some(rg_time) = output.stats().run_generation_time_ms {
            self.run_gen_time += rg_time as f64 / 1000.0;
        }
        if let Some(m_time) = output.stats().merge_time_ms {
            self.merge_time += m_time as f64 / 1000.0;
        }

        self.total_time += self.run_gen_time + self.merge_time;
        self.runs_count += output.stats().num_runs;

        if let Some(ref io) = output.stats().run_generation_io_stats {
            self.run_gen_read_ops += io.read_ops;
            self.run_gen_read_mb += io.read_bytes as f64 / 1_000_000.0;
            self.run_gen_write_ops += io.write_ops;
            self.run_gen_write_mb += io.write_bytes as f64 / 1_000_000.0;
        }
        if let Some(ref io) = output.stats().merge_io_stats {
            self.merge_read_ops += io.read_ops;
            self.merge_read_mb += io.read_bytes as f64 / 1_000_000.0;
            self.merge_write_ops += io.write_ops;
            self.merge_write_mb += io.write_bytes as f64 / 1_000_000.0;
        }

        if output.stats().merge_entry_num.len() > 1 {
            let min_entries = *output.stats().merge_entry_num.iter().min().unwrap_or(&0);
            let max_entries = *output.stats().merge_entry_num.iter().max().unwrap_or(&0);
            if min_entries > 0 {
                let imbalance = max_entries as f64 / min_entries as f64;
                self.imbalance_sum += imbalance;
                self.imbalance_count += 1;
            }
        } else if output.stats().merge_entry_num.len() == 1 {
            self.imbalance_sum += 1.0;
            self.imbalance_count += 1;
        }
    }

    pub fn to_benchmark_result(
        &self,
        policy_name: String,
        config: &BenchmarkConfig,
        run_size_mb: f64,
        run_gen_threads: usize,
        merge_threads: usize,
        entries: usize,
        valid_runs: usize,
    ) -> BenchmarkResult {
        let runs_f64 = valid_runs as f64;
        let avg_total = self.total_time / runs_f64;
        let avg_run_gen = self.run_gen_time / runs_f64;
        let avg_merge = self.merge_time / runs_f64;
        let avg_runs = self.runs_count / valid_runs;

        let rg_read_ops = (self.run_gen_read_ops as f64 / runs_f64) as u64;
        let rg_read_mb = self.run_gen_read_mb / runs_f64;
        let rg_write_ops = (self.run_gen_write_ops as f64 / runs_f64) as u64;
        let rg_write_mb = self.run_gen_write_mb / runs_f64;
        let m_read_ops = (self.merge_read_ops as f64 / runs_f64) as u64;
        let m_read_mb = self.merge_read_mb / runs_f64;
        let m_write_ops = (self.merge_write_ops as f64 / runs_f64) as u64;
        let m_write_mb = self.merge_write_mb / runs_f64;

        let total_read_mb = rg_read_mb + m_read_mb;
        let total_write_mb = rg_write_mb + m_write_mb;

        let avg_imbalance_factor = if self.imbalance_count > 0 {
            self.imbalance_sum / self.imbalance_count as f64
        } else {
            1.0
        };

        let read_amplification = if rg_write_mb > 0.0 {
            m_read_mb / rg_write_mb
        } else {
            1.0
        };

        BenchmarkResult {
            policy_name,
            threads: config.threads,
            memory_mb: config.memory_mb,
            memory_str: bytes_to_human_readable(config.memory_mb * 1024 * 1024),
            run_size_mb,
            run_gen_threads,
            merge_threads,
            runs: avg_runs,
            total_time: avg_total,
            run_gen_time: avg_run_gen,
            merge_time: avg_merge,
            entries,
            throughput: entries as f64 / avg_total / 1_000_000.0,
            read_mb: total_read_mb,
            write_mb: total_write_mb,
            run_gen_read_ops: rg_read_ops,
            run_gen_read_mb: rg_read_mb,
            run_gen_write_ops: rg_write_ops,
            run_gen_write_mb: rg_write_mb,
            merge_read_ops: m_read_ops,
            merge_read_mb: m_read_mb,
            merge_write_ops: m_write_ops,
            merge_write_mb: m_write_mb,
            imbalance_factor: avg_imbalance_factor,
            read_amplification,
        }
    }
}

fn bytes_to_human_readable(bytes: usize) -> String {
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
