use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::file::SharedFd;
use crate::kll::Sketch;
use crate::ovc::offset_value_coding_u64::OVCU64;
use crate::rs::replacement_selection::{
    ReplacementSelectionSink, compute_ovc_delta, run_replacement_selection,
};
use crate::sort::run_generation::{
    RunGenerationThreadResult, RunSummary, Scanner, execute_run_generation,
};
use crate::sort::sort_buffer::SortBuffer;
use crate::sort::sorter::{fill_sort_buffer, RunGenerationAlgorithm};
use crate::sort_with_ovc::merge_with_ovc::MergeWithOVC;
use crate::sort_with_ovc::run_with_ovc::RunWithOVC;
use crate::{
    IoStatsTracker, MergeStats, RunGenerationStats, SortInput, SortOutput, SortStats, Sorter,
    TempDirInfo,
};

// Abstraction for inputs to the merge function
pub enum MergeableRunWithOVC {
    Single(RunWithOVC),
    RangePartitioned(Vec<RunWithOVC>),
}

impl MergeableRunWithOVC {
    pub fn scan_range_with_io_tracker(
        &self,
        lower_bound: &[u8],
        upper_bound: &[u8],
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)> + Send> {
        match self {
            MergeableRunWithOVC::Single(run) => {
                run.scan_range_with_io_tracker(lower_bound, upper_bound, io_tracker)
            }
            MergeableRunWithOVC::RangePartitioned(runs) => {
                // [lower_bound, upper_bound) may span multiple partitions
                // Filter out empty runs (those with no entries)
                let non_empty_runs: Vec<&RunWithOVC> =
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
                Box::new(ChainedRangeIteratorWithOVC::new(iterators))
            }
        }
    }

    pub fn total_entries(&self) -> usize {
        match self {
            MergeableRunWithOVC::Single(run) => run.total_entries(),
            MergeableRunWithOVC::RangePartitioned(runs) => {
                runs.iter().map(|r| r.total_entries()).sum()
            }
        }
    }

    pub fn total_bytes(&self) -> usize {
        match self {
            MergeableRunWithOVC::Single(run) => run.total_bytes(),
            MergeableRunWithOVC::RangePartitioned(runs) => {
                runs.iter().map(|r| r.total_bytes()).sum()
            }
        }
    }

    /// Convert MergeableRun into Vec<RunImpl>
    /// - Single(run) becomes vec![run]
    /// - RangePartitioned(runs) returns runs directly
    pub fn into_runs(self) -> Vec<RunWithOVC> {
        match self {
            MergeableRunWithOVC::Single(run) => vec![run],
            MergeableRunWithOVC::RangePartitioned(runs) => runs,
        }
    }
}

impl RunSummary for MergeableRunWithOVC {
    fn total_entries(&self) -> usize {
        self.total_entries()
    }

    fn total_bytes(&self) -> usize {
        self.total_bytes()
    }
}

// Helper iterator that chains multiple iterators for range-partitioned runs
struct ChainedRangeIteratorWithOVC {
    iterators: Vec<Box<dyn Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)> + Send>>,
    current: usize,
}

impl ChainedRangeIteratorWithOVC {
    fn new(iterators: Vec<Box<dyn Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)> + Send>>) -> Self {
        Self {
            iterators,
            current: 0,
        }
    }
}

impl Iterator for ChainedRangeIteratorWithOVC {
    type Item = (OVCU64, Vec<u8>, Vec<u8>);

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
pub struct RunsOutputWithOVC {
    pub run: MergeableRunWithOVC,
    pub stats: SortStats,
}

impl SortOutput for RunsOutputWithOVC {
    fn iter(&self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>> {
        Box::new(
            self.run
                .scan_range_with_io_tracker(&[], &[], None)
                .map(|(_, key, value)| (key, value)),
        )
    }

    fn stats(&self) -> SortStats {
        self.stats.clone()
    }
}

struct RunWriterSinkWithOVC {
    run_writer: Option<AlignedWriter>,
    run_indexing_interval: usize,
    sketch: Sketch<Vec<u8>>,
    sketch_sampling_interval: usize,
    records_seen: u64,
    current_run: Option<RunWithOVC>,
    runs: Vec<MergeableRunWithOVC>,
    prev_key: Option<Vec<u8>>,
}

impl RunWriterSinkWithOVC {
    fn new(
        run_writer: AlignedWriter,
        run_indexing_interval: usize,
        sketch_size: usize,
        sketch_sampling_interval: usize,
    ) -> Self {
        Self {
            run_writer: Some(run_writer),
            run_indexing_interval,
            sketch: Sketch::new(sketch_size),
            sketch_sampling_interval,
            records_seen: 0,
            current_run: None,
            runs: Vec::new(),
            prev_key: None,
        }
    }

    fn finalize_active_run(&mut self) {
        if let Some(mut run) = self.current_run.take() {
            let writer = run.finalize_write();
            self.run_writer = Some(writer);
            self.runs.push(MergeableRunWithOVC::Single(run));
        }
    }

    fn into_parts(mut self) -> (Vec<MergeableRunWithOVC>, Sketch<Vec<u8>>) {
        self.finalize_active_run();
        self.run_writer.take();
        (self.runs, self.sketch)
    }
}

impl ReplacementSelectionSink for RunWriterSinkWithOVC {
    fn start_run(&mut self) {
        if self.current_run.is_some() {
            self.prev_key = None;
            return;
        }
        let writer = self
            .run_writer
            .take()
            .expect("Run writer should be available when starting a run");
        let run =
            RunWithOVC::from_writer_with_indexing_interval(writer, self.run_indexing_interval)
                .expect("Failed to create run");
        self.current_run = Some(run);
        self.prev_key = None;
    }

    fn push_record(&mut self, key: &[u8], value: &[u8]) {
        let run = self
            .current_run
            .as_mut()
            .expect("run must be started before pushing records");

        if self.records_seen % self.sketch_sampling_interval as u64 == 0 {
            self.sketch.update(key.to_vec());
        }
        self.records_seen += 1;

        let ovc = compute_ovc_delta(self.prev_key.as_deref(), key);
        let key_vec = key.to_vec();
        self.prev_key = Some(key_vec.clone());
        let value_vec = value.to_vec();

        run.append(ovc, &key_vec, &value_vec);
    }

    fn finish_run(&mut self) {
        self.finalize_active_run();
        self.prev_key = None;
    }
}
// External sorter following sorter.rs pattern
pub struct ExternalSorterWithOVC {
    run_gen_threads: usize,
    run_size: usize,
    merge_threads: usize,
    merge_fanin: usize,
    sketch_size: usize,
    sketch_sampling_interval: usize,
    run_indexing_interval: usize,
    imbalance_factor: f64,
    temp_dir_info: Arc<TempDirInfo>,
    run_gen_algorithm: RunGenerationAlgorithm,
}

impl ExternalSorterWithOVC {
    /// Create a new ExternalSorter with separate thread counts for run generation and merge
    /// The temporary directory will be automatically cleaned up when the sorter is dropped
    pub fn new(
        run_gen_threads: usize,
        run_size: usize,
        merge_threads: usize,
        merge_fanin: usize,
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
            run_size,
            merge_threads,
            merge_fanin,
            sketch_size: 200,
            sketch_sampling_interval: 1000,
            run_indexing_interval: 1000,
            temp_dir_info: Arc::new(TempDirInfo {
                path: temp_dir,
                should_delete: true,
            }),
            imbalance_factor: 1.0,
            run_gen_algorithm: RunGenerationAlgorithm::ReplacementSelection,
        }
    }

    pub fn set_run_generation_algorithm(&mut self, algorithm: RunGenerationAlgorithm) {
        self.run_gen_algorithm = algorithm;
    }

    pub fn run_generation_algorithm(&self) -> RunGenerationAlgorithm {
        self.run_gen_algorithm
    }

    pub fn run_generation(
        sort_input: Box<dyn SortInput>,
        num_threads: usize,
        run_size: usize,
        sketch_size: usize,
        sketch_sampling_interval: usize,
        run_indexing_interval: usize,
        dir: impl AsRef<Path>,
        algorithm: RunGenerationAlgorithm,
    ) -> Result<
        (
            Vec<MergeableRunWithOVC>,
            Sketch<Vec<u8>>,
            RunGenerationStats,
        ),
        String,
    > {
        match algorithm {
            RunGenerationAlgorithm::ReplacementSelection => {
                let worker = move |thread_id: usize,
                                   scanner: Scanner,
                                   io_tracker: Arc<IoStatsTracker>,
                                   dir: std::path::PathBuf| {
                    let run_path = dir.join(format!("intermediate_{}.dat", thread_id));
                    let fd = Arc::new(
                        SharedFd::new_from_path(&run_path)
                            .expect("Failed to open run file with Direct I/O"),
                    );
                    let run_writer =
                        AlignedWriter::from_fd_with_tracker(fd, Some((*io_tracker).clone()))
                            .expect("Failed to create run writer");

                    let mut sink = RunWriterSinkWithOVC::new(
                        run_writer,
                        run_indexing_interval,
                        sketch_size,
                        sketch_sampling_interval,
                    );
                    let thread_start = Instant::now();
                    let _ = run_replacement_selection(scanner, &mut sink, run_size);
                    let (local_runs, sketch) = sink.into_parts();
                    let total_time_ms = thread_start.elapsed().as_millis();

                    println!(
                        "Run gen thread {} ({:?}) breakdown: sort={} ms",
                        thread_id,
                        RunGenerationAlgorithm::ReplacementSelection,
                        total_time_ms
                    );

                    RunGenerationThreadResult {
                        runs: local_runs,
                        sketch,
                        load_ms: 0,
                        sort_ms: total_time_ms,
                        store_ms: 0,
                    }
                };

                execute_run_generation(sort_input, num_threads, sketch_size, dir, worker)
            }
            RunGenerationAlgorithm::LoadSortStore => {
                let worker = move |thread_id: usize,
                                   scanner: Scanner,
                                   io_tracker: Arc<IoStatsTracker>,
                                   dir: std::path::PathBuf| {
                    let run_path = dir.join(format!("intermediate_{}.dat", thread_id));
                    let fd = Arc::new(
                        SharedFd::new_from_path(&run_path)
                            .expect("Failed to open run file with Direct I/O"),
                    );
                    let run_writer =
                        AlignedWriter::from_fd_with_tracker(fd, Some((*io_tracker).clone()))
                            .expect("Failed to create run writer");

                    let mut sink = RunWriterSinkWithOVC::new(
                        run_writer,
                        run_indexing_interval,
                        sketch_size,
                        sketch_sampling_interval,
                    );
                    let mut scanner = scanner;
                    let mut buffer = SortBuffer::new(run_size);
                    let mut pending_record: Option<(Vec<u8>, Vec<u8>)> = None;
                    let mut load_duration = Duration::default();
                    let mut sort_duration = Duration::default();
                    let mut store_duration = Duration::default();

                    loop {
                        let load_start = Instant::now();
                        fill_sort_buffer(
                            &mut buffer,
                            scanner.as_mut(),
                            &mut pending_record,
                            run_size,
                        );
                        load_duration += load_start.elapsed();

                        if buffer.is_empty() {
                            break;
                        }

                        let sort_start = Instant::now();
                        let mut sorted_iter = buffer.sorted_iter();
                        sort_duration += sort_start.elapsed();

                        sink.start_run();
                        let store_start = Instant::now();
                        for (key, value) in sorted_iter {
                            sink.push_record(&key, &value);
                        }
                        sink.finish_run();
                        store_duration += store_start.elapsed();

                        buffer.reset();
                    }

                    let (local_runs, sketch) = sink.into_parts();
                    let load_ms = load_duration.as_millis();
                    let sort_ms = sort_duration.as_millis();
                    let store_ms = store_duration.as_millis();

                    println!(
                        "Run gen thread {} ({:?}) breakdown: load={} ms, sort={} ms, store={} ms",
                        thread_id,
                        RunGenerationAlgorithm::LoadSortStore,
                        load_ms,
                        sort_ms,
                        store_ms
                    );

                    RunGenerationThreadResult {
                        runs: local_runs,
                        sketch,
                        load_ms,
                        sort_ms,
                        store_ms,
                    }
                };

                execute_run_generation(sort_input, num_threads, sketch_size, dir, worker)
            }
        }
    }

    /// Performs multi-level merge if needed based on memory constraints
    /// Merges runs level by level until a single MergeableRun is produced
    pub fn multi_merge(
        mut runs: Vec<MergeableRunWithOVC>,
        fanin: usize,
        num_threads: usize,
        sketch: &Sketch<Vec<u8>>,
        imbalance_factor: f64,
        dir: impl AsRef<Path>,
    ) -> Result<(MergeableRunWithOVC, Vec<MergeStats>), String> {
        let dir = dir.as_ref();
        let mut per_merge_stats = Vec::new();

        if fanin >= runs.len() {
            // Can merge all runs in one pass
            println!("Merging all {} runs in single pass", runs.len());
            let (merged_run, stats) =
                Self::merge(runs, num_threads, &sketch, imbalance_factor, dir)?;

            // Collect stats
            per_merge_stats.push(stats);
            return Ok((merged_run, per_merge_stats));
        }

        // Multi-level merge using optimal merge pattern (Knuth Vol 3, pp. 365-366):
        // Repeatedly merge the F shortest runs to minimize total I/O.

        let n = runs.len();
        let f = fanin;

        // Add dummy runs to make (N-1) divisible by (F-1)
        // This ensures we can always merge F runs at a time
        let dummy_count = if n > 1 {
            (f - 1 - ((n - 1) % (f - 1))) % (f - 1)
        } else {
            0
        };

        // Add dummies without extra logging (match non-OVC behavior)
        runs.extend((0..dummy_count).map(|_| MergeableRunWithOVC::RangePartitioned(vec![])));

        let total_runs = runs.len();
        let expected_merges = if total_runs > 1 {
            (total_runs - 1) / (f - 1)
        } else {
            0
        };

        println!(
            "Multi-level merge: {} runs (including {} dummies), F={}, expected {} merge operations",
            total_runs, dummy_count, f, expected_merges
        );

        let mut merge_count = 0;

        // Repeatedly merge F shortest runs until only one remains
        while runs.len() > 1 {
            merge_count += 1;

            // Sort runs by size (ascending) to find F shortest
            runs.sort_by_key(|r| r.total_entries());

            // Take F shortest runs
            let batch_size = std::cmp::min(f, runs.len());
            let batch: Vec<MergeableRunWithOVC> = runs.drain(..batch_size).collect();

            let total_entries: usize = batch.iter().map(|r| r.total_entries()).sum();
            println!(
                "Merge {}/{}: merging {} shortest runs ({} total entries)",
                merge_count, expected_merges, batch_size, total_entries
            );

            let (merged_run, stats) =
                Self::merge(batch, num_threads, sketch, imbalance_factor, dir)?;

            // Collect per-merge stats
            per_merge_stats.push(stats);

            // Add merged run back to the pool
            runs.push(merged_run);
        }

        // Should have exactly one run remaining
        if runs.is_empty() {
            return Err("No runs remaining after merge".to_string());
        }

        let final_run = runs.into_iter().next().unwrap();
        Ok((final_run, per_merge_stats))
    }

    pub fn merge(
        output_runs: Vec<MergeableRunWithOVC>,
        num_threads: usize,
        sketch: &Sketch<Vec<u8>>,
        imbalance_factor: f64,
        dir: impl AsRef<Path>,
    ) -> Result<(MergeableRunWithOVC, MergeStats), String> {
        // Match non-OVC: do not print CDF boundary imbalance factor
        // If no runs or single run, return early
        if output_runs.is_empty() {
            let merge_stats = MergeStats {
                output_runs: 0,
                merge_entry_num: vec![],
                time_ms: 0,
                io_stats: None,
                per_thread_times_ms: vec![],
            };
            return Ok((MergeableRunWithOVC::RangePartitioned(vec![]), merge_stats));
        }

        if output_runs.len() == 1 {
            let merge_stats = MergeStats {
                output_runs: 1,
                merge_entry_num: vec![output_runs[0].total_entries() as u64],
                time_ms: 0,
                io_stats: None,
                per_thread_times_ms: vec![],
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
                let thread_start = std::time::Instant::now();
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
                // Use a unique file name per thread. Use time with thread ID to avoid conflicts.
                let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
                let ts = format!("{}{:09}", now.as_secs(), now.subsec_nanos()); // zero-pad nanos to 9
                let run_path = dir.join(format!("merge_output_{}_{}.dat", thread_id, ts));
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
                    output_run.append(ovc, &key, &value);
                }

                // Finalize to flush
                let writer = output_run.finalize_write();
                drop(writer);

                let thread_time_ms = thread_start.elapsed().as_millis();

                // Return a read-only run and thread time
                (output_run, thread_time_ms)
            });

            merge_handles.push(handle);
        }

        // Collect merge results as runs
        let mut output_runs = Vec::new();
        let mut per_thread_times_ms = Vec::new();

        for handle in merge_handles {
            let (run, thread_time) = handle.join().unwrap();
            output_runs.push(run);
            per_thread_times_ms.push(thread_time);
        }

        // Capture merge time
        let merge_time = merge_start.elapsed();
        let merge_time_ms = merge_time.as_millis();

        // Capture merge IO stats
        let merge_io_stats = Some(merge_io_tracker.get_detailed_stats());

        // Build single-line output with all merge statistics (match non-OVC)
        let merge_entry_num: Vec<u64> = output_runs
            .iter()
            .map(|run| run.total_entries() as u64)
            .collect();
        let total_entries = merge_entry_num.iter().sum::<u64>();
        let num_runs = output_runs.len();

        let mut output = format!(
            "Merge phase took {} ms | Merged {} entries across {} runs",
            merge_time_ms, total_entries, num_runs
        );

        if !per_thread_times_ms.is_empty() {
            let (min_time, max_time) = (
                *per_thread_times_ms.iter().min().unwrap(),
                *per_thread_times_ms.iter().max().unwrap(),
            );
            output.push_str(&format!(
                " | Thread timing: min={} ms, max={} ms",
                min_time, max_time
            ));
        }

        if merge_entry_num.len() > 1 {
            let avg = total_entries as f64 / merge_entry_num.len() as f64;
            let max_entries = *merge_entry_num.iter().max().unwrap();
            output.push_str(&format!(
                " | Partition sizes: imbalance={:.2}x",
                max_entries as f64 / avg
            ));
        }

        println!("{}", output);

        let merge_stats = MergeStats {
            output_runs: output_runs.len(),
            merge_entry_num,
            time_ms: merge_time_ms,
            io_stats: merge_io_stats,
            per_thread_times_ms,
        };

        Ok((
            MergeableRunWithOVC::RangePartitioned(output_runs),
            merge_stats,
        ))
    }
}

impl Sorter for ExternalSorterWithOVC {
    /// Sort method that runs generation and merging phases
    fn sort(&mut self, sort_input: Box<dyn SortInput>) -> Result<Box<dyn SortOutput>, String> {
        // Run generation phase
        let (runs, sketch, run_gen_stats) = ExternalSorterWithOVC::run_generation(
            sort_input,
            self.run_gen_threads,
            self.run_size,
            self.sketch_size,
            self.sketch_sampling_interval,
            self.run_indexing_interval,
            self.temp_dir_info.as_ref(),
            self.run_gen_algorithm,
        )?;

        // Use multi-level merge with memory-aware policy
        let (merged_run, merge_stats) = ExternalSorterWithOVC::multi_merge(
            runs,
            self.merge_fanin,
            self.merge_threads,
            &sketch,
            self.imbalance_factor,
            self.temp_dir_info.as_ref(),
        )?;

        // Create output
        Ok(Box::new(RunsOutputWithOVC {
            run: merged_run,
            stats: SortStats::new(run_gen_stats, merge_stats),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskio::aligned_writer::AlignedWriter;
    use crate::diskio::file::SharedFd;
    use crate::rand::small_thread_rng;
    use crate::rs::replacement_selection::compute_ovc_delta;
    use std::sync::Arc;
    use tempfile::{TempDir, tempdir};

    #[test]
    fn test_compute_ovc_delta_basic_cases() {
        let a = b"a111".to_vec();
        let b = b"a111".to_vec();
        let c = b"a211".to_vec();

        let initial = compute_ovc_delta(None, &a);
        assert!(initial.is_initial_value());

        let duplicate = compute_ovc_delta(Some(&a), &b);
        assert!(duplicate.is_duplicate_value());

        let normal = compute_ovc_delta(Some(&b), &c);
        assert!(normal.is_normal_value());
        assert_eq!(normal.offset(), 0);
    }

    #[test]
    fn test_run_writer_sink_with_ovc_persists_runs() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("run.dat");
        let fd = Arc::new(SharedFd::new_from_path(&path).unwrap());
        let writer = AlignedWriter::from_fd(fd).unwrap();

        let mut sink = RunWriterSinkWithOVC::new(writer, 1, 32, 1);
        sink.start_run();
        let records = vec![
            (b"aa".to_vec(), b"v1".to_vec()),
            (b"aa".to_vec(), b"v2".to_vec()),
            (b"ab".to_vec(), b"v3".to_vec()),
        ];
        for (k, v) in &records {
            sink.push_record(k, v);
        }
        sink.finish_run();

        let (runs, _sketch) = sink.into_parts();
        assert_eq!(runs.len(), 1);

        let run = match runs.into_iter().next().unwrap() {
            MergeableRunWithOVC::Single(run) => run,
            MergeableRunWithOVC::RangePartitioned(_) => panic!("expected single run"),
        };

        let items: Vec<_> = run.scan_range(&[], &[]).collect();
        assert_eq!(items.len(), records.len());
        assert!(items[0].0.is_initial_value());
        assert!(items[1].0.is_duplicate_value());
        assert!(items[2].0.is_normal_value());
        for (idx, (_, key, value)) in items.into_iter().enumerate() {
            assert_eq!(key, records[idx].0);
            assert_eq!(value, records[idx].1);
        }
    }

    #[test]
    fn test_range_partitioned_scan_with_io_tracker() {
        // Create a temporary directory for test files
        let temp_dir = TempDir::new().unwrap();

        // Create 3 partitions with sorted, non-overlapping data
        // Partition 0: keys [a00..a99]
        let path0 = temp_dir.path().join("partition0.dat");
        let fd0 = Arc::new(SharedFd::new_from_path(&path0).unwrap());
        let writer0 = AlignedWriter::from_fd(fd0.clone()).unwrap();
        let mut run0 = RunWithOVC::from_writer(writer0).unwrap();
        for i in 0..100 {
            let key = format!("a{:02}", i).into_bytes();
            let value = format!("value_{}", i).into_bytes();
            let ovc = OVCU64::initial_value();
            run0.append(ovc, &key, &value);
        }
        run0.finalize_write();

        // Partition 1: keys [b00..b99]
        let path1 = temp_dir.path().join("partition1.dat");
        let fd1 = Arc::new(SharedFd::new_from_path(&path1).unwrap());
        let writer1 = AlignedWriter::from_fd(fd1.clone()).unwrap();
        let mut run1 = RunWithOVC::from_writer(writer1).unwrap();
        for i in 0..100 {
            let key = format!("b{:02}", i).into_bytes();
            let value = format!("value_{}", i + 100).into_bytes();
            let ovc = OVCU64::initial_value();
            run1.append(ovc, &key, &value);
        }
        run1.finalize_write();

        // Partition 2: keys [c00..c99]
        let path2 = temp_dir.path().join("partition2.dat");
        let fd2 = Arc::new(SharedFd::new_from_path(&path2).unwrap());
        let writer2 = AlignedWriter::from_fd(fd2.clone()).unwrap();
        let mut run2 = RunWithOVC::from_writer(writer2).unwrap();
        for i in 0..100 {
            let key = format!("c{:02}", i).into_bytes();
            let value = format!("value_{}", i + 200).into_bytes();
            let ovc = OVCU64::initial_value();
            run2.append(ovc, &key, &value);
        }
        run2.finalize_write();

        let mergeable_run = MergeableRunWithOVC::RangePartitioned(vec![run0, run1, run2]);

        // Test 1: Scan range within a single partition
        let io_tracker1 = IoStatsTracker::new();
        let iter1 =
            mergeable_run.scan_range_with_io_tracker(b"a10", b"a20", Some(io_tracker1.clone()));
        let results1: Vec<_> = iter1.collect();
        assert_eq!(results1.len(), 10); // a10..a19
        assert_eq!(results1[0].1, b"a10");
        assert_eq!(results1[9].1, b"a19");
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
        assert_eq!(results2[0].1, b"a90");
        assert_eq!(results2[9].1, b"a99");
        assert_eq!(results2[10].1, b"b00");
        assert_eq!(results2[109].1, b"b99");
        assert_eq!(results2[110].1, b"c00");
        assert_eq!(results2[119].1, b"c09");
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

    #[test]
    fn test_multi_level_merge_small_fanout() {
        use crate::InMemInput;
        use rand::seq::SliceRandom;

        // Test parameters (mirror non-OVC test)
        let num_records = 100000;
        let num_threads_run_gen = 2;
        let run_size = 512; // bytes
        let sketch_size = 200;
        let sketch_sampling_interval = 1000;
        let run_indexing_interval = 1000;
        let fanin = 100;
        let num_threads = 4;
        let imbalance_factor = 1.0;

        let temp_dir = TempDir::new().unwrap();
        let mut data: Vec<_> = (0..num_records)
            .rev()
            .map(|i| {
                (
                    format!("key_{:05}", i).into_bytes(),
                    format!("value_{}", i).into_bytes(),
                )
            })
            .collect();
        data.shuffle(&mut small_thread_rng());

        let (runs, sketch, _run_gen_stats) = ExternalSorterWithOVC::run_generation(
            Box::new(InMemInput { data }),
            num_threads_run_gen,
            run_size,
            sketch_size,
            sketch_sampling_interval,
            run_indexing_interval,
            temp_dir.path(),
            RunGenerationAlgorithm::ReplacementSelection,
        )
        .unwrap();

        let (merged_run, per_merge_stats) = ExternalSorterWithOVC::multi_merge(
            runs,
            fanin,
            num_threads,
            &sketch,
            imbalance_factor,
            temp_dir.path(),
        )
        .unwrap();

        println!("Multi-level merge: {} passes", per_merge_stats.len());
        let final_runs = merged_run.into_runs();
        assert_eq!(
            final_runs.len(),
            num_threads,
            "Expected {} final runs, got {}",
            num_threads,
            final_runs.len()
        );

        let (mut prev_key, mut count): (Option<Vec<u8>>, usize) = (None, 0);
        for run in &final_runs {
            let mut iter = run.scan_range(&[], &[]);
            while let Some((_ovc, key, _value)) = iter.next() {
                if let Some(prev) = &prev_key {
                    assert!(
                        key.as_slice() >= prev.as_slice(),
                        "Data not sorted: {:?} should be >= {:?}",
                        String::from_utf8_lossy(&key),
                        String::from_utf8_lossy(prev)
                    );
                }
                prev_key = Some(key);
                count += 1;
            }
        }
        assert_eq!(
            count, num_records,
            "Expected {} entries, got {}",
            num_records, count
        );

        for (i, stat) in per_merge_stats.iter().enumerate() {
            assert!(
                stat.time_ms > 0 && stat.output_runs > 0 && stat.io_stats.is_some(),
                "Merge pass {} missing required stats",
                i + 1
            );
        }
        assert_eq!(
            per_merge_stats.last().unwrap().output_runs,
            num_threads,
            "Final merge should produce exactly {} runs",
            num_threads
        );
    }
}
