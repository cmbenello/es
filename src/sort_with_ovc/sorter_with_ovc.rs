use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::file::SharedFd;
use crate::kll::Sketch;
use crate::ovc::offset_value_coding_u64::OVCU64;
use crate::sort::merge::MergeIterator;
use crate::sort::run::RunImpl;
use crate::sort_with_ovc::merge_with_ovc::MergeWithOVC;
use crate::sort_with_ovc::run_with_ovc::RunWithOVC;
use crate::sort_with_ovc::sort_buffer_with_ovc::SortBufferOVC;
use crate::{
    IoStats, IoStatsTracker, MergeStats, RunGenerationStats, RunInfo, SortInput, SortOutput,
    SortStats, Sorter, TempDirInfo,
};

// Output implementation that chains run iterators without materializing
pub struct RunsOutputWithOVC {
    pub runs: Vec<RunWithOVC>,
    pub stats: SortStats,
}

impl SortOutput for RunsOutputWithOVC {
    fn iter(&self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>> {
        // Create a chained iterator from all runs
        let mut iterators: Vec<Box<dyn Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)>>> = Vec::new();

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
    iterators: Vec<Box<dyn Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)>>>,
    current: usize,
}

impl Iterator for ChainedIterator {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current < self.iterators.len() {
            if let Some((_, key, value)) = self.iterators[self.current].next() {
                return Some((key, value));
            }
            self.current += 1;
        }
        None
    }
}

// External sorter following sorter.rs pattern
pub struct ExternalSorterWithOVC {
    run_gen_threads: usize,
    merge_threads: usize,
    max_memory: usize,
    sketch_size: usize,
    sketch_sampling_interval: usize,
    run_indexing_interval: usize,
    temp_dir_info: Arc<TempDirInfo>,
}

impl ExternalSorterWithOVC {
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
    ) -> Result<(Vec<RunWithOVC>, Sketch<Vec<u8>>, RunGenerationStats), String> {
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
                let mut sort_buffer = SortBufferOVC::new(per_thread_mem);
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
                        let mut output_run = RunWithOVC::from_writer_with_indexing_interval(
                            run_writer.take().unwrap(),
                            run_indexing_interval,
                        )
                        .expect("Failed to create run");

                        for (ovc, key, value) in sort_buffer.sorted_iter() {
                            if cnt % sketch_sampling_interval == 0 {
                                sketch.update(key.clone());
                            }
                            cnt += 1;
                            output_run.append(ovc, key, value);
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
                    let mut output_run = RunWithOVC::from_writer_with_indexing_interval(
                        run_writer.take().unwrap(),
                        run_indexing_interval,
                    )
                    .expect("Failed to create run");

                    for (ovc, key, value) in sort_buffer.sorted_iter() {
                        if cnt % sketch_sampling_interval == 0 {
                            sketch.update(key.clone());
                        }
                        cnt += 1;
                        output_run.append(ovc, key, value);
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
        output_runs: Vec<RunWithOVC>,
        num_threads: usize,
        sketch: Sketch<Vec<u8>>,
        dir: impl AsRef<Path>,
    ) -> Result<(Vec<RunWithOVC>, MergeStats), String> {
        // If no runs or single run, return early
        if output_runs.is_empty() {
            let merge_stats = MergeStats {
                output_runs: 0,
                merge_entry_num: vec![],
                time_ms: 0,
                io_stats: None,
            };
            return Ok((vec![], merge_stats));
        }

        if output_runs.len() == 1 {
            let merge_stats = MergeStats {
                output_runs: 1,
                merge_entry_num: vec![output_runs[0].total_entries() as u64],
                time_ms: 0,
                io_stats: None,
            };
            return Ok((output_runs, merge_stats));
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

        for thread_id in 0..merge_threads {
            let runs = Arc::clone(&runs_arc);
            let dir = dir.as_ref().to_path_buf();
            let io_tracker = Arc::clone(&merge_io_tracker);
            let cdf = Arc::clone(&cdf);

            let handle = thread::spawn(move || {
                // Determine the key range for this thread
                let lower_bound = if thread_id == 0 {
                    vec![]
                } else {
                    cdf.query(thread_id as f64 / merge_threads as f64)
                };

                let upper_bound = if thread_id < merge_threads - 1 {
                    cdf.query((thread_id + 1) as f64 / merge_threads as f64)
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
                    RunWithOVC::from_writer(writer).expect("Failed to create merge output run");

                // Merge this range directly into the output run
                let merge_iter = MergeWithOVC::new(iterators);

                for (ovc, key, value) in merge_iter {
                    output_run.append(ovc, key, value);
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

        Ok((output_runs, merge_stats))
    }
}

impl Sorter for ExternalSorterWithOVC {
    /// Sort method that runs generation and merging phases
    fn sort(&mut self, sort_input: Box<dyn SortInput>) -> Result<Box<dyn SortOutput>, String> {
        // Run generation phase
        let (runs, sketch, run_gen_stats) = ExternalSorterWithOVC::run_generation(
            sort_input,
            self.run_gen_threads,
            self.max_memory / self.run_gen_threads,
            self.sketch_size,
            self.sketch_sampling_interval,
            self.run_indexing_interval,
            self.temp_dir_info.as_ref(),
        )?;

        // Merge phase
        let (merged_runs, merge_stats) = ExternalSorterWithOVC::merge(
            runs,
            self.merge_threads,
            sketch,
            self.temp_dir_info.as_ref(),
        )?;

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
        Ok(Box::new(RunsOutputWithOVC {
            runs: merged_runs,
            stats: sort_stats,
        }))
    }
}
