use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::file::SharedFd;
use crate::diskio::io_stats::IoStatsTracker;
use crate::kll::Sketch;
use crate::sort::merge::MergeIterator;
use crate::sort::run::RunImpl;
use crate::sort::sort_buffer::SortBuffer;
use crate::{
    MergeStats, RunGenerationStats, RunInfo, SortInput, SortOutput, SortStats, Sorter, TempDirInfo,
};

// Abstraction for inputs to the merge function
pub enum MergeableRun {
    Single(RunImpl),
    RangePartitioned(Vec<RunImpl>),
}

impl MergeableRun {
    pub fn scan_range_with_io_tracker(
        &self,
        lower_bound: &[u8],
        upper_bound: &[u8],
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send> {
        match self {
            MergeableRun::Single(run) => {
                run.scan_range_with_io_tracker(lower_bound, upper_bound, io_tracker)
            }
            MergeableRun::RangePartitioned(runs) => {
                // [lower_bound, upper_bound) may span multiple partitions
                // Filter out empty runs (those with no entries)
                let non_empty_runs: Vec<&RunImpl> =
                    runs.iter().filter(|r| r.start_key().is_some()).collect();

                if non_empty_runs.is_empty() {
                    return Box::new(std::iter::empty());
                }

                // Find the last partition whose start key < lower_bound
                let start_partition = if lower_bound.is_empty() {
                    0
                } else {
                    let mut ok = -1;
                    let mut ng = non_empty_runs.len() as isize;
                    while (ng - ok).abs() > 1 {
                        let mid = (ok + ng) / 2;
                        if non_empty_runs[mid as usize].start_key().unwrap() < lower_bound {
                            ok = mid;
                        } else {
                            ng = mid;
                        }
                    }
                    if ok == -1 { 0 } else { ok as usize }
                };

                // Find the first partition whose start key >= upper_bound
                let end_partition = if upper_bound.is_empty() {
                    non_empty_runs.len()
                } else {
                    let mut ok = non_empty_runs.len() as isize;
                    let mut ng = -1;
                    while (ng - ok).abs() > 1 {
                        let mid = (ok + ng) / 2;
                        if upper_bound <= non_empty_runs[mid as usize].start_key().unwrap() {
                            ok = mid;
                        } else {
                            ng = mid;
                        }
                    }
                    ok as usize
                };

                let mut iterators = Vec::with_capacity(end_partition - start_partition);
                for i in start_partition..end_partition {
                    let run = non_empty_runs[i];
                    iterators.push(run.scan_range_with_io_tracker(
                        lower_bound,
                        upper_bound,
                        io_tracker.clone(),
                    ));
                }

                // Chain all iterators together
                Box::new(ChainedRangeIterator::new(iterators))
            }
        }
    }

    pub fn total_entries(&self) -> usize {
        match self {
            MergeableRun::Single(run) => run.total_entries(),
            MergeableRun::RangePartitioned(runs) => runs.iter().map(|r| r.total_entries()).sum(),
        }
    }

    /// Convert MergeableRun into Vec<RunImpl>
    /// - Single(run) becomes vec![run]
    /// - RangePartitioned(runs) returns runs directly
    pub fn into_runs(self) -> Vec<RunImpl> {
        match self {
            MergeableRun::Single(run) => vec![run],
            MergeableRun::RangePartitioned(runs) => runs,
        }
    }
}

// Helper iterator that chains multiple iterators for range-partitioned runs
struct ChainedRangeIterator {
    iterators: Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>>,
    current: usize,
}

impl ChainedRangeIterator {
    fn new(iterators: Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>>) -> Self {
        Self {
            iterators,
            current: 0,
        }
    }
}

impl Iterator for ChainedRangeIterator {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current < self.iterators.len() {
            if let Some(item) = self.iterators[self.current].next() {
                return Some(item);
            }
            self.current += 1;
        }
        None
    }
}

// Output implementation that chains run iterators without materializing
pub struct RunsOutput {
    pub runs: Vec<RunImpl>,
    pub stats: SortStats,
}

impl SortOutput for RunsOutput {
    fn iter(&self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>> {
        // Create a chained iterator from all runs
        let mut iterators: Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>>> = Vec::new();

        for run in &self.runs {
            iterators.push(run.scan_range(&[], &[]));
        }

        // Chain all iterators together
        Box::new(ChainedIterator {
            iterators,
            current: 0,
        })
    }

    fn stats(&self) -> SortStats {
        self.stats.clone()
    }
}

// Helper iterator that chains multiple iterators
struct ChainedIterator {
    iterators: Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>>>,
    current: usize,
}

impl Iterator for ChainedIterator {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current < self.iterators.len() {
            if let Some(item) = self.iterators[self.current].next() {
                return Some(item);
            }
            self.current += 1;
        }
        None
    }
}

// External sorter following sorter.rs pattern
pub struct ExternalSorter {
    run_gen_threads: usize,
    merge_threads: usize,
    max_memory: usize,
    sketch_size: usize,
    sketch_sampling_interval: usize,
    run_indexing_interval: usize,
    temp_dir_info: Arc<TempDirInfo>,
    boundary_imbalance_factor: f64,
}

impl ExternalSorter {
    /// Create a new ExternalSorter with temporary files in the current directory
    pub fn new(num_threads: usize, max_memory: usize) -> Self {
        Self::new_with_threads_and_dir(num_threads, num_threads, max_memory, ".")
    }

    /// Create a new ExternalSorter with temporary files in the specified directory
    /// Uses same thread count for both run generation and merge phases
    pub fn new_with_dir(
        num_threads: usize,
        max_memory: usize,
        base_dir: impl AsRef<std::path::Path>,
    ) -> Self {
        Self::new_with_threads_and_dir(num_threads, num_threads, max_memory, base_dir)
    }

    /// Create a new ExternalSorter with separate thread counts for run generation and merge
    /// The temporary directory will be automatically cleaned up when the sorter is dropped
    pub fn new_with_threads_and_dir(
        run_gen_threads: usize,
        merge_threads: usize,
        max_memory: usize,
        base_dir: impl AsRef<std::path::Path>,
    ) -> Self {
        // Use a unique temp directory for each instance to avoid conflicts
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_dir = base_dir.as_ref().join(format!(
            "external_sort_{}_{}",
            std::process::id(),
            timestamp
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();

        Self {
            run_gen_threads,
            merge_threads,
            max_memory,
            sketch_size: 200,
            sketch_sampling_interval: 1000,
            run_indexing_interval: 1000,
            temp_dir_info: Arc::new(TempDirInfo {
                path: temp_dir,
                should_delete: true,
            }),
            boundary_imbalance_factor: 1.0,
        }
    }

    pub fn run_generation(
        sort_input: Box<dyn SortInput>,
        num_threads: usize,
        per_thread_mem: usize,
        sketch_size: usize,
        sketch_sampling_interval: usize,
        run_indexing_interval: usize,
        dir: impl AsRef<Path>,
    ) -> Result<(Vec<RunImpl>, Sketch<Vec<u8>>, RunGenerationStats), String> {
        // Start timing for run generation
        let run_generation_start = Instant::now();

        // Create IO tracker for run generation phase
        let run_generation_io_tracker = Arc::new(IoStatsTracker::new());

        // Create parallel scanners for input data with IO tracking
        let scanners = sort_input
            .create_parallel_scanners(num_threads, Some((*run_generation_io_tracker).clone()));

        println!(
            "Starting sort with {} parallel scanners for run generation",
            scanners.len()
        );

        if scanners.is_empty() {
            return Ok((
                vec![],
                Sketch::new(sketch_size),
                RunGenerationStats {
                    num_runs: 0,
                    runs_info: vec![],
                    time_ms: 0,
                    io_stats: None,
                },
            ));
        }

        // Run Generation Phase (following sorter.rs pattern)
        let mut handles = vec![];

        for (thread_id, scanner) in scanners.into_iter().enumerate() {
            let io_tracker = Arc::clone(&run_generation_io_tracker);
            let dir = dir.as_ref().to_path_buf();

            let handle = thread::spawn(move || {
                let mut local_runs = Vec::new();
                let mut sort_buffer = SortBuffer::new(per_thread_mem);
                let mut sketch = Sketch::new(sketch_size);

                let run_path = dir.join(format!("intermediate_{}.dat", thread_id));
                let fd = Arc::new(
                    SharedFd::new_from_path(&run_path)
                        .expect("Failed to open run file with Direct I/O"),
                );
                let mut run_writer = Option::Some(
                    AlignedWriter::from_fd_with_tracker(fd, Some((*io_tracker).clone()))
                        .expect("Failed to create run writer"),
                );
                let mut cnt = 0;

                for (key, value) in scanner {
                    if !sort_buffer.has_space(&key, &value) {
                        let mut output_run = RunImpl::from_writer_with_indexing_interval(
                            run_writer.take().unwrap(),
                            run_indexing_interval,
                        )
                        .expect("Failed to create run");

                        for (k, v) in sort_buffer.sorted_iter() {
                            if cnt % sketch_sampling_interval == 0 {
                                sketch.update(k.clone());
                            }
                            cnt += 1;
                            output_run.append(k, v);
                        }

                        // Finalize the run and get writer back
                        run_writer = Some(output_run.finalize_write());

                        // Create a read-only run with proper metadata
                        local_runs.push(output_run);

                        sort_buffer.reset();
                        if !sort_buffer.append(key, value) {
                            panic!("Failed to append data to sort buffer, it should have space");
                        }
                    } else {
                        sort_buffer.append(key, value);
                    }
                }

                // Final buffer
                if !sort_buffer.is_empty() {
                    let mut output_run = RunImpl::from_writer_with_indexing_interval(
                        run_writer.take().unwrap(),
                        run_indexing_interval,
                    )
                    .expect("Failed to create run");

                    for (k, v) in sort_buffer.sorted_iter() {
                        if cnt % sketch_sampling_interval == 0 {
                            sketch.update(k.clone());
                        }
                        cnt += 1;
                        output_run.append(k, v);
                    }

                    // Finalize the run and get writer back
                    run_writer = Some(output_run.finalize_write());

                    // Create a read-only run with proper metadata
                    local_runs.push(output_run);
                }

                drop(run_writer); // Ensure the writer is closed

                (local_runs, sketch)
            });

            handles.push(handle);
        }

        // Collect runs from all threads
        let mut output_runs = Vec::new();
        let mut sketch = Sketch::new(sketch_size); // Combine sketches from all threads
        for handle in handles {
            let (runs, thread_sketch) = handle.join().unwrap();
            output_runs.extend(runs);
            sketch.merge(&thread_sketch);
        }

        let initial_runs_count = output_runs.len();

        // Capture run generation time
        let run_generation_time = run_generation_start.elapsed();
        let run_generation_time_ms = run_generation_time.as_millis();
        println!(
            "Generated {} runs in {} ms",
            initial_runs_count, run_generation_time_ms
        );

        // Capture initial run info (before any merging)
        let initial_runs_info: Vec<RunInfo> = output_runs
            .iter()
            .map(|run| RunInfo {
                entries: run.total_entries(),
                file_size: run.total_bytes() as u64,
            })
            .collect();

        // Capture run generation IO stats
        let run_generation_io_stats = Some(run_generation_io_tracker.get_detailed_stats());

        Ok((
            output_runs,
            sketch,
            RunGenerationStats {
                num_runs: initial_runs_count,
                runs_info: initial_runs_info,
                time_ms: run_generation_time_ms,
                io_stats: run_generation_io_stats,
            },
        ))
    }

    pub fn merge(
        output_runs: Vec<MergeableRun>,
        num_threads: usize,
        sketch: Sketch<Vec<u8>>,
        dir: impl AsRef<Path>,
        imbalance_factor: Option<f64>,
    ) -> Result<(MergeableRun, MergeStats), String> {
        let imbalance_factor = imbalance_factor.unwrap_or(1.0);
        println!(
            "CDF boundary imbalance factor for merge: {:.4}. One thread will handle {:.4} times the average load.",
            imbalance_factor, imbalance_factor
        );
        // If no runs or single run, return early
        if output_runs.is_empty() {
            let merge_stats = MergeStats {
                output_runs: 0,
                merge_entry_num: vec![],
                time_ms: 0,
                io_stats: None,
            };
            return Ok((MergeableRun::RangePartitioned(vec![]), merge_stats));
        }

        if output_runs.len() == 1 {
            let entry_count = output_runs[0].total_entries();
            let merge_stats = MergeStats {
                output_runs: 1,
                merge_entry_num: vec![entry_count as u64],
                time_ms: 0,
                io_stats: None,
            };
            let run = output_runs.into_iter().next().unwrap();
            return Ok((run, merge_stats));
        }

        // Start timing for merge phase
        let merge_start = Instant::now();

        // Create IO tracker for merge phase
        let merge_io_tracker = Arc::new(IoStatsTracker::new());

        // Parallel Merge Phase for many runs
        // let desired_threads = num_threads;
        let cdf = Arc::new(sketch.cdf());
        let merge_threads = if cdf.size() < num_threads {
            1
        } else {
            num_threads
        };

        println!(
            "Merging {} runs using {} threads",
            output_runs.len(),
            merge_threads
        );

        // Share runs across threads using Arc
        let runs_arc = Arc::new(output_runs);

        // Create merge tasks
        let mut merge_handles = vec![];

        // Create imbalance portions
        // Imbalance factor is defined as the ratio of the largest portion to the average portion
        let k = merge_threads as f64;
        let r = if imbalance_factor <= 0.0 {
            1.0
        } else {
            imbalance_factor
        };
        // each = portion size for all non-first threads
        let each = (1.0 - r * (1.0 / k)) / (k - 1.0);
        // When imbalance_factor = 1.0, each = 1/k
        // When imbalance_factor = 1.3, each = (1 - 1.3/k) / (k-1)

        for thread_id in 0..merge_threads {
            let runs = Arc::clone(&runs_arc);
            let dir = dir.as_ref().to_path_buf();
            let io_tracker = Arc::clone(&merge_io_tracker);
            let cdf = Arc::clone(&cdf);

            let handle = thread::spawn(move || {
                let tid = thread_id as f64;

                // Determine the key range for this thread
                let lower_bound = if thread_id == 0 {
                    vec![]
                } else {
                    cdf.query(r / k + (tid - 1.0) * each)
                };

                let upper_bound = if thread_id < merge_threads - 1 {
                    cdf.query(r / k + tid * each)
                } else {
                    vec![]
                };

                // Create iterators for this key range from all runs
                let iterators: Vec<_> = runs
                    .iter()
                    .map(|run| {
                        run.scan_range_with_io_tracker(
                            &lower_bound,
                            &upper_bound,
                            Some((*io_tracker).clone()),
                        )
                    })
                    .collect();

                // Create output run for this thread
                let run_path = dir.join(format!("merge_output_{}.dat", thread_id));
                let fd = Arc::new(
                    SharedFd::new_from_path(&run_path)
                        .expect("Failed to open merge output file with Direct I/O"),
                );
                let writer = AlignedWriter::from_fd_with_tracker(fd, Some((*io_tracker).clone()))
                    .expect("Failed to create run writer");
                let mut output_run =
                    RunImpl::from_writer(writer).expect("Failed to create merge output run");

                // Merge this range directly into the output run
                let merge_iter = MergeIterator::new(iterators);

                for (key, value) in merge_iter {
                    output_run.append(key, value);
                }

                // Finalize to flush
                let writer = output_run.finalize_write();
                drop(writer);

                // Return a read-only run
                output_run
            });

            merge_handles.push(handle);
        }

        // Collect merge results as runs
        let mut output_runs = Vec::new();

        for handle in merge_handles {
            output_runs.push(handle.join().unwrap());
        }

        // Capture merge time
        let merge_time = merge_start.elapsed();
        let merge_time_ms = merge_time.as_millis();

        // Capture merge IO stats
        let merge_io_stats = Some(merge_io_tracker.get_detailed_stats());

        println!("Merge phase took {} ms", merge_time_ms);

        let merge_entry_num: Vec<u64> = output_runs
            .iter()
            .map(|run| run.total_entries() as u64)
            .collect();

        let merge_stats = MergeStats {
            output_runs: output_runs.len(),
            merge_entry_num,
            time_ms: merge_time_ms,
            io_stats: merge_io_stats,
        };

        Ok((MergeableRun::RangePartitioned(output_runs), merge_stats))
    }
}

impl Sorter for ExternalSorter {
    /// Sort method that runs generation and merging phases
    fn sort(&mut self, sort_input: Box<dyn SortInput>) -> Result<Box<dyn SortOutput>, String> {
        // Run generation phase
        let (runs, sketch, run_gen_stats) = ExternalSorter::run_generation(
            sort_input,
            self.run_gen_threads,
            self.max_memory / self.run_gen_threads,
            self.sketch_size,
            self.sketch_sampling_interval,
            self.run_indexing_interval,
            self.temp_dir_info.as_ref(),
        )?;

        // Merge phase - convert runs to MergeableRun::Single
        let mergeable_runs: Vec<MergeableRun> = runs
            .into_iter()
            .map(|run| MergeableRun::Single(run))
            .collect();

        let (merged_run, merge_stats) = ExternalSorter::merge(
            mergeable_runs,
            self.merge_threads,
            sketch,
            self.temp_dir_info.as_ref(),
            Some(self.boundary_imbalance_factor),
        )?;

        // Convert MergeableRun back to Vec<RunImpl> for output
        let final_runs = merged_run.into_runs();

        // Combine stats
        let sort_stats: SortStats = SortStats {
            num_runs: run_gen_stats.num_runs,
            runs_info: run_gen_stats.runs_info,
            run_generation_time_ms: Some(run_gen_stats.time_ms),
            merge_entry_num: merge_stats.merge_entry_num,
            merge_time_ms: Some(merge_stats.time_ms),
            run_generation_io_stats: run_gen_stats.io_stats,
            merge_io_stats: merge_stats.io_stats,
        };

        // Create output
        Ok(Box::new(RunsOutput {
            runs: final_runs,
            stats: sort_stats,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskio::aligned_writer::AlignedWriter;
    use crate::diskio::file::SharedFd;
    use tempfile::TempDir;

    #[test]
    fn test_range_partitioned_scan_with_io_tracker() {
        // Create a temporary directory for test files
        let temp_dir = TempDir::new().unwrap();

        // Create 3 partitions with sorted, non-overlapping data
        // Partition 0: keys [a00..a99]
        let path0 = temp_dir.path().join("partition0.dat");
        let fd0 = Arc::new(SharedFd::new_from_path(&path0).unwrap());
        let writer0 = AlignedWriter::from_fd(fd0.clone()).unwrap();
        let mut run0 = RunImpl::from_writer(writer0).unwrap();
        for i in 0..100 {
            let key = format!("a{:02}", i).into_bytes();
            let value = format!("value_{}", i).into_bytes();
            run0.append(key, value);
        }
        run0.finalize_write();

        // Partition 1: keys [b00..b99]
        let path1 = temp_dir.path().join("partition1.dat");
        let fd1 = Arc::new(SharedFd::new_from_path(&path1).unwrap());
        let writer1 = AlignedWriter::from_fd(fd1.clone()).unwrap();
        let mut run1 = RunImpl::from_writer(writer1).unwrap();
        for i in 0..100 {
            let key = format!("b{:02}", i).into_bytes();
            let value = format!("value_{}", i + 100).into_bytes();
            run1.append(key, value);
        }
        run1.finalize_write();

        // Partition 2: keys [c00..c99]
        let path2 = temp_dir.path().join("partition2.dat");
        let fd2 = Arc::new(SharedFd::new_from_path(&path2).unwrap());
        let writer2 = AlignedWriter::from_fd(fd2.clone()).unwrap();
        let mut run2 = RunImpl::from_writer(writer2).unwrap();
        for i in 0..100 {
            let key = format!("c{:02}", i).into_bytes();
            let value = format!("value_{}", i + 200).into_bytes();
            run2.append(key, value);
        }
        run2.finalize_write();

        let mergeable_run = MergeableRun::RangePartitioned(vec![run0, run1, run2]);

        // Test 1: Scan range within a single partition
        let io_tracker1 = IoStatsTracker::new();
        let iter1 =
            mergeable_run.scan_range_with_io_tracker(b"a10", b"a20", Some(io_tracker1.clone()));
        let results1: Vec<_> = iter1.collect();
        assert_eq!(results1.len(), 10); // a10..a19
        assert_eq!(results1[0].0, b"a10");
        assert_eq!(results1[9].0, b"a19");
        let (read_ops1, read_bytes1) = io_tracker1.get_read_stats();
        assert!(read_ops1 > 0, "Should track IO operations");
        assert!(read_bytes1 > 0, "Should track IO bytes");

        // Test 2: Scan range across multiple partitions
        let io_tracker2 = IoStatsTracker::new();
        let iter2 =
            mergeable_run.scan_range_with_io_tracker(b"a90", b"c10", Some(io_tracker2.clone()));
        let results2: Vec<_> = iter2.collect();
        // Should get: a90..a99 (10) + b00..b99 (100) + c00..c09 (10) = 120
        assert_eq!(results2.len(), 120);
        assert_eq!(results2[0].0, b"a90");
        assert_eq!(results2[9].0, b"a99");
        assert_eq!(results2[10].0, b"b00");
        assert_eq!(results2[109].0, b"b99");
        assert_eq!(results2[110].0, b"c00");
        assert_eq!(results2[119].0, b"c09");
        let (read_ops2, read_bytes2) = io_tracker2.get_read_stats();
        assert!(read_ops2 > 0, "Should track IO operations");
        assert!(read_bytes2 > 0, "Should track IO bytes");

        // Test 3: Empty range
        let iter3 = mergeable_run.scan_range_with_io_tracker(b"d00", b"d99", None);
        let results3: Vec<_> = iter3.collect();
        assert_eq!(results3.len(), 0);

        // Test 4: Full scan (empty bounds)
        let io_tracker4 = IoStatsTracker::new();
        let iter4 = mergeable_run.scan_range_with_io_tracker(&[], &[], Some(io_tracker4.clone()));
        let results4: Vec<_> = iter4.collect();
        assert_eq!(results4.len(), 300); // All records
        let (read_ops4, read_bytes4) = io_tracker4.get_read_stats();
        assert!(read_ops4 > 0, "Should track IO operations");
        assert!(read_bytes4 > 0, "Should track IO bytes");
    }
}
