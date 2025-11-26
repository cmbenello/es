use crate::sketch::SketchType;
use crate::{RunGenerationAlgorithm, SortStats};
use std::path::PathBuf;

#[derive(Clone)]
pub struct BenchmarkResult {
    pub config: BenchmarkConfig,
    pub stats: Vec<SortStats>,
}

#[derive(Clone)]
pub struct BenchmarkConfig {
    pub config_name: String,
    pub warmup_runs: usize,
    pub benchmark_runs: usize,
    pub cooldown_seconds: u64,
    pub verify: bool,
    pub ovc: bool,
    pub temp_dir: PathBuf,
    pub sketch_type: SketchType,
    pub sketch_size: usize,
    pub sketch_sampling_interval: usize,
    pub run_indexing_interval: usize,
    pub run_gen_threads: usize,
    pub run_gen_algorithm: RunGenerationAlgorithm,
    pub run_size_mb: f64,
    pub run_gen_memory_mb: f64,
    pub merge_threads: usize,
    pub merge_fanin: usize,
    pub merge_memory_mb: f64,
    pub imbalance_factor: f64,
}
