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
    fn stats(&self) -> SortStats {
        // Default implementation returns unknown stats
        SortStats {
            num_runs: 0,
            runs_info: vec![],
            run_generation_time_ms: None,
            merge_entry_num: vec![],
            merge_time_ms: None,
            run_generation_io_stats: None,
            merge_io_stats: None,
        }
    }
}

/// Statistics about a sort operation
#[derive(Clone, Debug)]
pub struct SortStats {
    pub num_runs: usize,
    pub runs_info: Vec<RunInfo>,
    pub run_generation_time_ms: Option<u128>,
    pub merge_entry_num: Vec<u64>,
    pub merge_time_ms: Option<u128>,
    pub run_generation_io_stats: Option<IoStats>,
    pub merge_io_stats: Option<IoStats>,
}

impl std::fmt::Display for SortStats {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Clean display implementation
        writeln!(f, "SortStats:")?;
        writeln!(f, "  Number of runs: {}", self.num_runs)?;
        writeln!(
            f,
            "  (R) time: {} ms",
            self.run_generation_time_ms.unwrap_or(0)
        )?;
        writeln!(f, "  (M) time: {} ms", self.merge_time_ms.unwrap_or(0))?;
        if let Some(io_stats) = &self.run_generation_io_stats {
            writeln!(f, "  (R) I/O stats: {}", io_stats)?;
        }
        if let Some(io_stats) = &self.merge_io_stats {
            writeln!(f, "  (M) I/O stats: {}", io_stats)?;
        }

        // Display read amplification (sparse indexing effectiveness)
        if let (Some(run_gen_io), Some(merge_io)) =
            (&self.run_generation_io_stats, &self.merge_io_stats)
        {
            if run_gen_io.write_bytes > 0 {
                let read_amplification = merge_io.read_bytes as f64 / run_gen_io.write_bytes as f64;
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
                    merge_io.read_bytes,
                    merge_io.read_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
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
                    writeln!(
                        f,
                        "    Read reduction: {:.1}% (possible data compression or skipping)",
                        -excess_read_pct
                    )?;
                } else {
                    writeln!(f, "    Perfect read efficiency (1.0x)")?;
                }
            }
        }

        // Display partition imbalance if merge was parallelized
        if self.merge_entry_num.len() > 1 {
            writeln!(f, "  Partition imbalance:")?;

            let total_entries: u64 = self.merge_entry_num.iter().sum();
            let avg_entries = total_entries as f64 / self.merge_entry_num.len() as f64;
            let min_entries = *self.merge_entry_num.iter().min().unwrap_or(&0);
            let max_entries = *self.merge_entry_num.iter().max().unwrap_or(&0);

            // Calculate standard deviation
            let variance = self
                .merge_entry_num
                .iter()
                .map(|&x| {
                    let diff = x as f64 - avg_entries;
                    diff * diff
                })
                .sum::<f64>()
                / self.merge_entry_num.len() as f64;
            let std_dev = variance.sqrt();

            // Calculate coefficient of variation (CV) as a percentage
            let cv = if avg_entries > 0.0 {
                (std_dev / avg_entries) * 100.0
            } else {
                0.0
            };

            // Calculate imbalance factor (max/min)
            let imbalance_factor = if min_entries > 0 {
                max_entries as f64 / min_entries as f64
            } else {
                0.0
            };

            writeln!(f, "    Partitions: {}", self.merge_entry_num.len())?;
            writeln!(f, "    Total entries: {}", total_entries)?;
            writeln!(f, "    Avg per partition: {:.0}", avg_entries)?;
            writeln!(
                f,
                "    Min entries: {} ({:.1}% of avg)",
                min_entries,
                if avg_entries > 0.0 {
                    (min_entries as f64 / avg_entries) * 100.0
                } else {
                    0.0
                }
            )?;
            writeln!(
                f,
                "    Max entries: {} ({:.1}% of avg)",
                max_entries,
                if avg_entries > 0.0 {
                    (max_entries as f64 / avg_entries) * 100.0
                } else {
                    0.0
                }
            )?;
            writeln!(f, "    Std deviation: {:.0}", std_dev)?;
            writeln!(f, "    Coefficient of variation: {:.1}%", cv)?;
            writeln!(
                f,
                "    Imbalance factor (max/min): {:.2}x",
                imbalance_factor
            )?;

            // Show distribution if not too many partitions
            if self.merge_entry_num.len() <= 32 {
                writeln!(f, "    Distribution: {:?}", self.merge_entry_num)?;
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
}

/// Statistics from the merge phase
#[derive(Clone, Debug)]
pub struct MergeStats {
    pub output_runs: usize,
    pub merge_entry_num: Vec<u64>,
    pub time_ms: u128,
    pub io_stats: Option<IoStats>,
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
    pub stats: Option<SortStats>,
}

impl SortOutput for InMemOutput {
    fn iter(&self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>> {
        Box::new(self.data.clone().into_iter())
    }

    fn stats(&self) -> SortStats {
        self.stats.clone().unwrap_or_else(|| {
            // Default stats for in-memory output
            SortStats {
                num_runs: 1,
                runs_info: vec![RunInfo {
                    entries: self.data.len(),
                    file_size: 0, // No file for in-memory data
                }],
                run_generation_time_ms: None,
                merge_entry_num: vec![],
                merge_time_ms: None,
                run_generation_io_stats: None,
                merge_io_stats: None,
            }
        })
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
pub mod sort;
pub mod sort_policy_run_length;
pub mod sort_policy_sub;
pub mod sort_policy_thread_count;
pub mod sort_stats;
pub mod sort_with_ovc;

use std::path::{Path, PathBuf};

// Export the main types
pub use diskio::aligned_reader::{AlignedChunkReader, AlignedReader};
pub use diskio::aligned_writer::AlignedWriter;
pub use diskio::file::{file_size_fd, pread_fd, pwrite_fd}; // , GlobalFileManager};
pub use diskio::io_stats::{IoStats, IoStatsTracker};
pub use input_reader::csv_input_direct::{CsvDirectConfig, CsvInputDirect};
pub use input_reader::gensort_input_direct::GenSortInputDirect;
pub use sort::run::RunImpl;
pub use sort::sorter::{ExternalSorter, RunsOutput};
pub use sort_with_ovc::sorter_with_ovc::{ExternalSorterWithOVC, RunsOutputWithOVC};
