// Simple External Sort Library

#![allow(clippy::needless_range_loop)]

// Core traits from sorter.rs
pub trait Sorter {
    fn sort(&mut self, sort_input: Box<dyn SortInput>) -> Result<Box<dyn SortOutput>, String>;
}

// Input/Output traits
pub trait SortInput {
    fn create_parallel_scanners(
        &self,
        num_scanners: usize,
        io_tracker: Option<IoStatsTracker>,
    ) -> Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>>;
}

pub trait SortOutput {
    fn iter(&self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>>;

    /// Get statistics about the sort operation
    fn stats(&self) -> SortStats;
}

/// Statistics about a sort operation
#[derive(Clone, Debug)]
pub struct SortStats {
    pub run_gen_stats: RunGenerationStats,
    pub per_merge_stats: Vec<MergeStats>,
}

impl SortStats {
    pub fn new(run_gen_stats: RunGenerationStats, per_merge_stats: Vec<MergeStats>) -> Self {
        Self {
            run_gen_stats,
            per_merge_stats,
        }
    }
}

impl std::fmt::Display for SortStats {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Clean display implementation
        writeln!(f, "SortStats:")?;
        let rg = &self.run_gen_stats;
        writeln!(f, "  Number of runs: {}", rg.num_runs)?;
        writeln!(f, "  (R) time: {} ms", rg.time_ms)?;
        writeln!(
            f,
            "  (R) breakdown: load={} ms, sort={} ms, store={} ms",
            rg.load_time_ms, rg.sort_time_ms, rg.store_time_ms
        )?;
        if let Some(io) = &rg.io_stats {
            writeln!(f, "  (R) I/O stats: {}", io)?;
        }

        let total_merge_time_ms: u128 = self.per_merge_stats.iter().map(|m| m.time_ms).sum();
        writeln!(f, "  (M) time: {} ms", total_merge_time_ms)?;

        // Display read amplification (sparse indexing effectiveness)
        if let Some(run_gen_io) = &self.run_gen_stats.io_stats {
            let merge_read_bytes: u64 = self
                .per_merge_stats
                .iter()
                .filter_map(|m| m.io_stats.as_ref())
                .map(|io| io.read_bytes)
                .sum();
            if run_gen_io.write_bytes > 0 {
                let read_amplification = merge_read_bytes as f64 / run_gen_io.write_bytes as f64;
                let excess_read_pct = (read_amplification - 1.0) * 100.0;
                writeln!(f, "  Read amplification:")?;
                writeln!(
                    f,
                    "    Run generation writes: {} bytes ({:.2} GB)",
                    run_gen_io.write_bytes,
                    run_gen_io.write_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                )?;
                writeln!(
                    f,
                    "    Merge phase reads: {} bytes ({:.2} GB)",
                    merge_read_bytes,
                    merge_read_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                )?;
                writeln!(
                    f,
                    "    Read amplification factor: {:.2}x",
                    read_amplification
                )?;
                if excess_read_pct > 0.0 {
                    writeln!(
                        f,
                        "    Excess reads: {:.1}% (sparse indexing overhead)",
                        excess_read_pct
                    )?;
                } else if excess_read_pct < 0.0 {
                    writeln!(f, "    Read reduction: {:.1}%", -excess_read_pct)?;
                } else {
                    writeln!(f, "    Perfect read efficiency (1.0x)")?;
                }
            }
        }

        // Display thread timing statistics if available
        if !self.per_merge_stats.is_empty() {
            for (i, merge_stat) in self.per_merge_stats.iter().enumerate() {
                if !merge_stat.per_thread_times_ms.is_empty() {
                    let min_time = *merge_stat.per_thread_times_ms.iter().min().unwrap_or(&0);
                    let max_time = *merge_stat.per_thread_times_ms.iter().max().unwrap_or(&0);
                    let avg_time = merge_stat.per_thread_times_ms.iter().sum::<u128>() as f64
                        / merge_stat.per_thread_times_ms.len() as f64;
                    let time_imbalance = max_time as f64 / avg_time;

                    if self.per_merge_stats.len() > 1 {
                        writeln!(f, "  Thread timing (merge pass {}):", i + 1)?;
                    } else {
                        writeln!(f, "  Thread timing:")?;
                    }
                    writeln!(f, "    Min thread time: {} ms", min_time)?;
                    writeln!(f, "    Max thread time: {} ms", max_time)?;
                    writeln!(f, "    Avg thread time: {:.0} ms", avg_time)?;
                    writeln!(
                        f,
                        "    Time imbalance factor (max/avg): {:.2}x",
                        time_imbalance
                    )?;

                    // Show all thread times if not too many threads
                    if merge_stat.per_thread_times_ms.len() <= 32 {
                        writeln!(f, "    Thread times: {:?}", merge_stat.per_thread_times_ms)?;
                    }
                }
            }
        }

        // Display per-merge statistics if available (from multi-level merge)
        if self.per_merge_stats.len() > 1 {
            let multi_merge_stats = &self.per_merge_stats;
            writeln!(f, "  Multi-level merge details:")?;
            writeln!(f, "    Total merge passes: {}", multi_merge_stats.len())?;

            for (i, merge_stat) in multi_merge_stats.iter().enumerate() {
                writeln!(f, "    Merge pass {}:", i + 1)?;
                writeln!(f, "      Time: {} ms", merge_stat.time_ms)?;
                writeln!(f, "      Output runs: {}", merge_stat.output_runs)?;

                if let Some(ref io) = merge_stat.io_stats {
                    writeln!(
                        f,
                        "      I/O: read={:.2} MiB, write={:.2} MiB",
                        io.read_bytes as f64 / (1024.0 * 1024.0),
                        io.write_bytes as f64 / (1024.0 * 1024.0)
                    )?;
                }

                if merge_stat.merge_entry_num.len() > 1 {
                    let total: u64 = merge_stat.merge_entry_num.iter().sum();
                    let avg = total as f64 / merge_stat.merge_entry_num.len() as f64;
                    let max = *merge_stat.merge_entry_num.iter().max().unwrap_or(&0);
                    let imbalance = max as f64 / avg;
                    writeln!(
                        f,
                        "      Partitions: {}, imbalance: {:.2}x",
                        merge_stat.merge_entry_num.len(),
                        imbalance
                    )?;
                }
            }
        }

        Ok(())
    }
}

/// Information about a single run
#[derive(Clone, Debug)]
pub struct RunInfo {
    pub entries: usize,
    pub file_size: u64,
}

impl std::fmt::Display for RunInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "entries={}, file_size={}", self.entries, self.file_size)
    }
}

/// Statistics from the run generation phase
#[derive(Clone, Debug)]
pub struct RunGenerationStats {
    pub num_runs: usize,
    pub runs_info: Vec<RunInfo>,
    pub time_ms: u128,
    pub io_stats: Option<IoStats>,
    pub load_time_ms: u128,
    pub sort_time_ms: u128,
    pub store_time_ms: u128,
}

/// Statistics from the merge phase
#[derive(Clone, Debug)]
pub struct MergeStats {
    pub output_runs: usize,
    pub merge_entry_num: Vec<u64>,
    pub time_ms: u128,
    pub io_stats: Option<IoStats>,
    pub per_thread_times_ms: Vec<u128>,
}

// Input implementation
pub struct InMemInput {
    pub data: Vec<(Vec<u8>, Vec<u8>)>,
}

impl SortInput for InMemInput {
    fn create_parallel_scanners(
        &self,
        num_scanners: usize,
        _io_tracker: Option<IoStatsTracker>,
    ) -> Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>> {
        if self.data.is_empty() {
            return vec![];
        }

        let chunk_size = self.data.len().div_ceil(num_scanners);
        let mut scanners = Vec::new();

        // Clone the data for each scanner since we can't move out of &self
        let data = self.data.clone();
        for chunk in data.chunks(chunk_size) {
            let chunk_vec = chunk.to_vec();
            scanners.push(Box::new(chunk_vec.into_iter())
                as Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>);
        }

        scanners
    }
}

// Output implementation that materializes all data
pub struct InMemOutput {
    pub data: Vec<(Vec<u8>, Vec<u8>)>,
    pub stats: SortStats,
}

impl SortOutput for InMemOutput {
    fn iter(&self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>> {
        Box::new(self.data.clone().into_iter())
    }

    fn stats(&self) -> SortStats {
        self.stats.clone()
    }
}

// Shared directory info for cleanup coordination
pub struct TempDirInfo {
    path: PathBuf,
    should_delete: bool,
}

impl TempDirInfo {
    /// Create a new TempDirInfo with the specified path
    pub fn new(path: impl AsRef<Path>, should_delete: bool) -> Self {
        // Create the directory if it doesn't exist
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            std::fs::create_dir_all(&path).expect("Failed to create temp directory");
        }
        Self {
            path,
            should_delete,
        }
    }
}

impl Drop for TempDirInfo {
    fn drop(&mut self) {
        if self.should_delete {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

impl AsRef<Path> for TempDirInfo {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

// Implementations
pub mod benchmark;
pub mod diskio;
pub mod input_reader;
pub mod kll;
pub mod order_preserving_encoding;
pub mod ovc;
pub mod rand;
pub mod replacement_selection;
pub mod sort;
pub mod sort_stats;

use std::path::{Path, PathBuf};

// Export the main types
pub use diskio::aligned_reader::{AlignedChunkReader, AlignedReader};
pub use diskio::aligned_writer::AlignedWriter;
pub use diskio::file::{file_size_fd, pread_fd, pwrite_fd}; // , GlobalFileManager};
pub use diskio::io_stats::{IoStats, IoStatsTracker};
pub use input_reader::csv_input_direct::{CsvDirectConfig, CsvInputDirect};
pub use input_reader::gensort_input_direct::GenSortInputDirect;
pub use sort::engine::RunGenerationAlgorithm;
pub use sort::ovc::sorter::{ExternalSorterWithOVC, RunsOutputWithOVC};
pub use sort::plain::run::RunImpl;
pub use sort::sorter::{ExternalSorter, RunsOutput};
