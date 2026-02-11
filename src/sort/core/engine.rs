use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::constants::DEFAULT_BUFFER_SIZE;
use crate::diskio::file::SharedFd;
use crate::diskio::io_stats::IoStatsTracker;
use crate::rand::small_thread_rng;
use crate::sort::core::run_format::{
    IndexingInterval, KeyRunIdOffsetBound, MultiSparseIndexes, RUN_ID_COUNTER, SparseIndexRef,
    cmp_key_run_offset,
};
use crate::sort::run_sink::RunSink;
use crate::{
    MergeStats, RunGenerationStats, RunInfo, SortInput, SortOutput, SortStats, TempDirInfo,
};
use rand::Rng;

pub type Scanner = Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>;

pub struct RunGenerationThreadResult<R> {
    pub runs: Vec<R>,
    pub load_ms: u128,
    pub sort_ms: u128,
    pub store_ms: u128,
}

pub trait RunSummary {
    fn total_entries(&self) -> usize;
    fn total_bytes(&self) -> usize;
}

#[derive(Debug, Clone, Copy)]
pub enum PartitionType {
    KeyOnly,
    CountBalanced,
    SizeBalanced,
}

impl Default for PartitionType {
    fn default() -> Self {
        Self::SizeBalanced
    }
}

const INDEX_BUDGET_PCT: usize = 5;
const SPARSE_INDEX_WARN_AVG_ENTRIES_PER_RUN: f64 = 10.0;

fn merge_memory_budget_bytes(num_threads: usize, fanin: usize) -> usize {
    num_threads
        .saturating_mul(fanin)
        .saturating_mul(DEFAULT_BUFFER_SIZE)
}

pub fn execute_run_generation<R, F>(
    sort_input: Box<dyn SortInput>,
    num_threads: usize,
    dir: impl AsRef<Path>,
    worker: F,
) -> Result<(Vec<R>, RunGenerationStats), String>
where
    R: RunSummary + Send + 'static,
    F: Fn(usize, Scanner, Arc<IoStatsTracker>, PathBuf) -> RunGenerationThreadResult<R>
        + Send
        + Sync
        + 'static,
{
    let dir = dir.as_ref().to_path_buf();
    let run_generation_start = Instant::now();
    let run_generation_io_tracker = Arc::new(IoStatsTracker::new());

    let scanners = sort_input
        .create_parallel_scanners(num_threads, Some((*run_generation_io_tracker).clone()));

    println!(
        "Starting sort with {} parallel scanners for run generation",
        scanners.len()
    );

    if scanners.is_empty() {
        return Ok((
            vec![],
            RunGenerationStats {
                num_runs: 0,
                runs_info: vec![],
                time_ms: 0,
                io_stats: None,
                load_time_ms: 0,
                sort_time_ms: 0,
                store_time_ms: 0,
            },
        ));
    }

    let worker = Arc::new(worker);
    let mut handles = vec![];
    for (thread_id, scanner) in scanners.into_iter().enumerate() {
        let worker = Arc::clone(&worker);
        let io_tracker = Arc::clone(&run_generation_io_tracker);
        let dir = dir.clone();
        let handle = thread::spawn(move || worker(thread_id, scanner, io_tracker, dir));
        handles.push(handle);
    }

    let mut output_runs = Vec::new();
    let mut total_load_ms: u128 = 0;
    let mut total_sort_ms: u128 = 0;
    let mut total_store_ms: u128 = 0;
    let mut threads_count: u128 = 0;

    for handle in handles {
        let result = handle
            .join()
            .map_err(|_| "Run generation worker panicked".to_string())?;
        output_runs.extend(result.runs);
        total_load_ms += result.load_ms;
        total_sort_ms += result.sort_ms;
        total_store_ms += result.store_ms;
        threads_count += 1;
    }

    let initial_runs_count = output_runs.len();
    let run_generation_time_ms = run_generation_start.elapsed().as_millis();
    println!(
        "Generated {} runs in {} ms",
        initial_runs_count, run_generation_time_ms
    );

    let initial_runs_info: Vec<RunInfo> = output_runs
        .iter()
        .map(|run| RunInfo {
            entries: run.total_entries(),
            file_size: run.total_bytes() as u64,
        })
        .collect();

    let (avg_load_ms, avg_sort_ms, avg_store_ms) = if threads_count > 0 {
        (
            total_load_ms / threads_count,
            total_sort_ms / threads_count,
            total_store_ms / threads_count,
        )
    } else {
        (0, 0, 0)
    };

    Ok((
        output_runs,
        RunGenerationStats {
            num_runs: initial_runs_count,
            runs_info: initial_runs_info,
            time_ms: run_generation_time_ms,
            io_stats: Some(run_generation_io_tracker.get_detailed_stats()),
            load_time_ms: avg_load_ms,
            sort_time_ms: avg_sort_ms,
            store_time_ms: avg_store_ms,
        },
    ))
}

pub trait SortHooks: Clone + Send + Sync + 'static {
    type MergeableRun: RunSummary + Send + Sync + 'static;
    type Sink: RunSink<MergeableRun = Self::MergeableRun> + Send;

    fn create_sink(
        &self,
        writer: AlignedWriter,
        run_indexing_interval: IndexingInterval,
    ) -> Self::Sink;

    fn dummy_run(&self) -> Self::MergeableRun;

    fn combine_runs(&self, runs: Vec<Self::MergeableRun>) -> Self::MergeableRun;

    fn merge_range(
        &self,
        runs: Arc<Vec<Self::MergeableRun>>,
        thread_id: usize,
        run_id: u32,
        run_indexing_interval: IndexingInterval,
        lower_inc: Option<(&[u8], u32, usize)>,
        upper_exc: Option<(&[u8], u32, usize)>,
        dir: &Path,
        io_tracker: Arc<IoStatsTracker>,
        discard_output: bool,
    ) -> Result<(Self::MergeableRun, u128), String>;

    fn sparse_indexes<'a>(&self, run: &'a Self::MergeableRun) -> MultiSparseIndexes<'a>;

    fn into_output(&self, run: Self::MergeableRun, stats: SortStats) -> Box<dyn SortOutput>;

    /// Run replacement selection for this sorter
    fn run_replacement_selection(
        &self,
        scanner: Scanner,
        sink: &mut Self::Sink,
        run_gen_mem: usize,
    ) -> crate::replacement_selection::ReplacementSelectionStats;
}

pub struct SorterCore<H: SortHooks> {
    hooks: H,
    run_gen_threads: usize,
    run_gen_mem: usize,
    merge_threads: usize,
    merge_fanin: usize,
    run_indexing_interval: IndexingInterval,
    imbalance_factor: f64,
    partition_type: PartitionType,
    temp_dir_info: Arc<TempDirInfo>,
    discard_final_output: bool,
}

impl<H: SortHooks> SorterCore<H> {
    pub fn with_hooks(
        hooks: H,
        run_gen_threads: usize,
        run_gen_mem: usize,
        merge_threads: usize,
        merge_fanin: usize,
        run_indexing_interval: IndexingInterval,
        base_dir: impl AsRef<Path>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let random = small_thread_rng().random::<u32>();
        let temp_dir = base_dir
            .as_ref()
            .join(format!("external_sort_{}_{}", random, timestamp));
        std::fs::create_dir_all(&temp_dir).unwrap();

        Self {
            hooks,
            run_gen_threads,
            run_gen_mem,
            merge_threads,
            merge_fanin,
            run_indexing_interval,
            imbalance_factor: 1.0,
            partition_type: PartitionType::default(),
            temp_dir_info: Arc::new(TempDirInfo {
                path: temp_dir,
                should_delete: true,
            }),
            discard_final_output: false,
        }
    }

    pub fn temp_dir(&self) -> &Path {
        self.temp_dir_info.as_ref().as_ref()
    }

    pub fn set_imbalance_factor(&mut self, factor: f64) {
        self.imbalance_factor = factor;
    }

    pub fn set_partition_type(&mut self, partition_type: PartitionType) {
        self.partition_type = partition_type;
    }

    pub fn set_discard_final_output(&mut self, discard: bool) {
        self.discard_final_output = discard;
    }

    fn run_generation_indexing_interval(&self) -> IndexingInterval {
        let stride = self.run_indexing_interval.value().max(1);
        match self.partition_type {
            PartitionType::SizeBalanced => IndexingInterval::bytes(stride),
            _ => IndexingInterval::records(stride),
        }
    }

    fn run_generation_internal(
        &self,
        sort_input: Box<dyn SortInput>,
        run_indexing_interval: IndexingInterval,
        estimated_total_data_bytes: Option<usize>,
    ) -> Result<(Vec<H::MergeableRun>, RunGenerationStats), String> {
        run_generation_with_hooks(
            &self.hooks,
            sort_input,
            self.run_gen_threads,
            self.run_gen_mem,
            run_indexing_interval,
            estimated_total_data_bytes,
            self.temp_dir_info.as_ref().as_ref(),
        )
    }

    fn multi_merge_internal(
        &self,
        runs: Vec<H::MergeableRun>,
        run_indexing_interval: IndexingInterval,
    ) -> Result<(H::MergeableRun, Vec<MergeStats>), String> {
        let total_data_bytes: usize = runs.iter().map(|run| run.total_bytes()).sum();
        let merge_memory_bytes = merge_memory_budget_bytes(self.merge_threads, self.merge_fanin);
        let total_index_budget = merge_memory_bytes.saturating_mul(INDEX_BUDGET_PCT) / 100;
        let threads = self.merge_threads.max(1);
        let thread_index_budget = total_index_budget.div_ceil(threads);
        let estimated_thread_data_bytes = total_data_bytes.div_ceil(threads);

        let merge_indexing_interval = match run_indexing_interval {
            IndexingInterval::Bytes { stride, .. } => IndexingInterval::bytes(stride),
            IndexingInterval::Records { stride, .. } => IndexingInterval::records(stride),
        }
        .with_budget_bytes(thread_index_budget)
        .with_estimated_total_data_bytes(estimated_thread_data_bytes);

        multi_merge_with_hooks(
            &self.hooks,
            runs,
            self.merge_fanin,
            self.merge_threads,
            self.imbalance_factor,
            self.partition_type,
            merge_indexing_interval,
            self.temp_dir_info.as_ref().as_ref(),
            self.discard_final_output,
        )
    }

    pub fn run_generation(
        sort_input: Box<dyn SortInput>,
        num_threads: usize,
        run_gen_mem: usize,
        run_indexing_interval: usize,
        dir: impl AsRef<Path>,
    ) -> Result<(Vec<H::MergeableRun>, RunGenerationStats), String>
    where
        H: Default,
    {
        let hooks = H::default();
        run_generation_with_hooks(
            &hooks,
            sort_input,
            num_threads,
            run_gen_mem,
            IndexingInterval::records(run_indexing_interval),
            None,
            dir,
        )
    }

    pub fn multi_merge(
        runs: Vec<H::MergeableRun>,
        fanin: usize,
        num_threads: usize,
        imbalance_factor: f64,
        partition_type: PartitionType,
        dir: impl AsRef<Path>,
    ) -> Result<(H::MergeableRun, Vec<MergeStats>), String>
    where
        H: Default,
    {
        let total_data_bytes: usize = runs.iter().map(|run| run.total_bytes()).sum();
        let merge_memory_bytes = merge_memory_budget_bytes(num_threads, fanin);
        let total_index_budget = merge_memory_bytes.saturating_mul(INDEX_BUDGET_PCT) / 100;
        let threads = num_threads.max(1);
        let thread_index_budget = total_index_budget.div_ceil(threads);
        let estimated_thread_data_bytes = total_data_bytes.div_ceil(threads);

        let merge_indexing_interval = match partition_type {
            PartitionType::SizeBalanced => IndexingInterval::bytes(1000),
            _ => IndexingInterval::records(1000),
        }
        .with_budget_bytes(thread_index_budget)
        .with_estimated_total_data_bytes(estimated_thread_data_bytes);

        multi_merge_with_hooks(
            &H::default(),
            runs,
            fanin,
            num_threads,
            imbalance_factor,
            partition_type,
            merge_indexing_interval,
            dir.as_ref(),
            false, // Default: don't discard output
        )
    }
}

impl<H: SortHooks + Default> SorterCore<H> {
    pub fn with_indexing_interval(
        run_gen_threads: usize,
        run_gen_mem: usize,
        merge_threads: usize,
        merge_fanin: usize,
        run_indexing_interval: usize,
        base_dir: impl AsRef<Path>,
    ) -> Self {
        Self::with_hooks(
            H::default(),
            run_gen_threads,
            run_gen_mem,
            merge_threads,
            merge_fanin,
            IndexingInterval::records(run_indexing_interval),
            base_dir,
        )
    }
}

impl<H: SortHooks> crate::Sorter for SorterCore<H> {
    fn sort(&mut self, sort_input: Box<dyn SortInput>) -> Result<Box<dyn SortOutput>, String> {
        let run_indexing_interval = self.run_generation_indexing_interval();
        let estimated_total_data_bytes = sort_input.estimated_size_bytes().map(|value| {
            if value > usize::MAX as u64 {
                usize::MAX
            } else {
                value as usize
            }
        });
        let (runs, run_gen_stats) = self.run_generation_internal(
            sort_input,
            run_indexing_interval,
            estimated_total_data_bytes,
        )?;
        let (merged_run, merge_stats) = self.multi_merge_internal(runs, run_indexing_interval)?;
        let output = self
            .hooks
            .into_output(merged_run, SortStats::new(run_gen_stats, merge_stats));
        Ok(output)
    }
}

fn run_generation_with_hooks<H: SortHooks>(
    hooks: &H,
    sort_input: Box<dyn SortInput>,
    num_threads: usize,
    run_gen_mem: usize,
    run_indexing_interval: IndexingInterval,
    estimated_total_data_bytes: Option<usize>,
    dir: impl AsRef<Path>,
) -> Result<(Vec<H::MergeableRun>, RunGenerationStats), String> {
    let dir = dir.as_ref();
    let hooks = hooks.clone();
    let worker_hooks = hooks.clone();
    let worker_count = num_threads.max(1);
    let thread_index_budget = run_gen_mem.saturating_mul(INDEX_BUDGET_PCT).div_ceil(100);
    let estimated_thread_data_bytes =
        estimated_total_data_bytes.map(|value| value.div_ceil(worker_count));

    let worker =
        move |thread_id: usize, scanner: Scanner, io_tracker: Arc<IoStatsTracker>, dir: PathBuf| {
            let run_path = dir.join(format!("intermediate_{}.dat", thread_id));
            let fd = Arc::new(
                SharedFd::new_from_path(&run_path, true)
                    .expect("Failed to open run file with Direct I/O"),
            );
            let run_writer = AlignedWriter::from_fd_with_tracker(fd, Some((*io_tracker).clone()))
                .expect("Failed to create run writer");
            let mut thread_interval = run_indexing_interval.with_budget_bytes(thread_index_budget);
            if let Some(estimated_bytes) = estimated_thread_data_bytes {
                thread_interval = thread_interval.with_estimated_total_data_bytes(estimated_bytes);
            }
            let mut sink = worker_hooks.create_sink(run_writer, thread_interval);
            let thread_start = Instant::now();
            let _ = worker_hooks.run_replacement_selection(scanner, &mut sink, run_gen_mem);
            let local_runs = sink.finalize();
            let total_time_ms = thread_start.elapsed().as_millis();
            println!(
                "Run gen thread {} {} ms, {} runs generated",
                thread_id,
                total_time_ms,
                local_runs.len()
            );
            RunGenerationThreadResult {
                runs: local_runs,
                load_ms: 0,
                sort_ms: total_time_ms,
                store_ms: 0,
            }
        };

    let (runs, stats) = execute_run_generation(sort_input, num_threads, dir, worker)?;
    let (sparse_entries, avg_sparse_entries_per_run) = sparse_index_entry_stats(&hooks, &runs);
    println!(
        "Run generation sparse index: {} entries | avg sparse entries/run: {:.2}",
        sparse_entries, avg_sparse_entries_per_run
    );
    warn_if_sparse_index_too_sparse("run generation", avg_sparse_entries_per_run);
    Ok((runs, stats))
}

fn multi_merge_with_hooks<H: SortHooks>(
    hooks: &H,
    mut runs: Vec<H::MergeableRun>,
    fanin: usize,
    num_threads: usize,
    imbalance_factor: f64,
    partition_type: PartitionType,
    run_indexing_interval: IndexingInterval,
    dir: &Path,
    discard_final_output: bool,
) -> Result<(H::MergeableRun, Vec<MergeStats>), String> {
    let dir = dir.as_ref();
    let mut per_merge_stats = Vec::new();

    if runs.is_empty() {
        let merge_stats = MergeStats {
            output_runs: 0,
            merge_entry_num: vec![],
            merge_entry_bytes: vec![],
            time_ms: 0,
            io_stats: None,
            per_thread_times_ms: vec![],
            per_thread_io_stats: vec![],
        };
        return Ok((hooks.dummy_run(), vec![merge_stats]));
    }

    if runs.len() == 1 {
        let run = runs.into_iter().next().unwrap();
        let merge_stats = MergeStats {
            output_runs: 1,
            merge_entry_num: vec![run.total_entries() as u64],
            merge_entry_bytes: vec![run.total_bytes() as u64],
            time_ms: 0,
            io_stats: None,
            per_thread_times_ms: vec![],
            per_thread_io_stats: vec![],
        };
        return Ok((run, vec![merge_stats]));
    }

    if fanin >= runs.len() {
        // This is the final (and only) merge
        let (merged_run, stats) = merge_once_with_hooks(
            hooks,
            runs,
            num_threads,
            imbalance_factor,
            partition_type,
            run_indexing_interval,
            dir,
            discard_final_output,
        )?;
        per_merge_stats.push(stats);
        return Ok((merged_run, per_merge_stats));
    }

    let n = runs.len();
    let f = fanin;
    let dummy_count = if n > 1 {
        (f - 1 - ((n - 1) % (f - 1))) % (f - 1)
    } else {
        0
    };
    runs.extend((0..dummy_count).map(|_| hooks.dummy_run()));
    let total_runs = runs.len();
    let expected_merges = total_runs.saturating_sub(1) / (f - 1);
    println!(
        "Multi-level merge: {} runs (including {} dummies), F={}, expected {} merge operations",
        total_runs, dummy_count, f, expected_merges
    );

    let mut merge_count = 0;
    while runs.len() > 1 {
        merge_count += 1;
        runs.sort_by_key(|r| r.total_entries());

        let batch_size = std::cmp::min(f, runs.len());
        let batch: Vec<H::MergeableRun> = runs.drain(..batch_size).collect();
        let total_entries: usize = batch.iter().map(|r| r.total_entries()).sum();
        // Check if this will be the final merge (runs will have 0 after drain, 1 after push)
        let is_final_merge = runs.is_empty();
        let should_discard = is_final_merge && discard_final_output;
        let (merged_run, stats) = merge_once_with_hooks(
            hooks,
            batch,
            num_threads,
            imbalance_factor,
            partition_type,
            run_indexing_interval,
            dir,
            should_discard,
        )?;
        println!(
            "Merge {}/{}: merging {} shortest runs ({} total entries)",
            merge_count, expected_merges, batch_size, total_entries
        );
        per_merge_stats.push(stats);
        runs.push(merged_run);
    }

    if runs.is_empty() {
        return Err("No runs remaining after merge".to_string());
    }

    let final_run = runs.into_iter().next().unwrap();
    Ok((final_run, per_merge_stats))
}

fn merge_once_with_hooks<H: SortHooks>(
    hooks: &H,
    output_runs: Vec<H::MergeableRun>,
    num_threads: usize,
    imbalance_factor: f64,
    partition_type: PartitionType,
    run_indexing_interval: IndexingInterval,
    dir: &Path,
    discard_output: bool,
) -> Result<(H::MergeableRun, MergeStats), String> {
    if output_runs.is_empty() {
        let merge_stats = MergeStats {
            output_runs: 0,
            merge_entry_num: vec![],
            merge_entry_bytes: vec![],
            time_ms: 0,
            io_stats: None,
            per_thread_times_ms: vec![],
            per_thread_io_stats: vec![],
        };
        return Ok((hooks.dummy_run(), merge_stats));
    }

    if output_runs.len() == 1 {
        let entry_count = output_runs[0].total_entries() as u64;
        let entry_bytes = output_runs[0].total_bytes() as u64;
        let merge_stats = MergeStats {
            output_runs: 1,
            merge_entry_num: vec![entry_count],
            merge_entry_bytes: vec![entry_bytes],
            time_ms: 0,
            io_stats: None,
            per_thread_times_ms: vec![],
            per_thread_io_stats: vec![],
        };
        let run = output_runs.into_iter().next().unwrap();
        return Ok((run, merge_stats));
    }

    let (sparse_entries, avg_sparse_entries_per_run) =
        sparse_index_entry_stats(hooks, &output_runs);
    println!(
        "Merge input sparse index: {} entries | avg sparse entries/run: {:.2}",
        sparse_entries, avg_sparse_entries_per_run
    );
    warn_if_sparse_index_too_sparse("merge input", avg_sparse_entries_per_run);

    let merge_start = Instant::now();
    let merge_threads = num_threads;

    println!(
        "Merging {} runs using {} threads",
        output_runs.len(),
        merge_threads
    );

    let runs_arc = Arc::new(output_runs);
    let mut merge_handles = vec![];

    let run_id_base = RUN_ID_COUNTER.fetch_add(merge_threads as u32, Ordering::AcqRel);

    let partition_by_size = matches!(partition_type, PartitionType::SizeBalanced);
    let total_bytes: usize = runs_arc.iter().map(|run| run.total_bytes()).sum();

    for thread_id in 0..merge_threads {
        let runs = Arc::clone(&runs_arc);
        let dir = dir.to_path_buf();
        let thread_io_tracker = Arc::new(IoStatsTracker::new());
        let io_tracker = Arc::clone(&thread_io_tracker);
        let hooks = hooks.clone();
        let run_id = run_id_base + thread_id as u32;

        let handle = thread::spawn(move || {
            let indexes: Vec<_> = runs.iter().map(|run| hooks.sparse_indexes(run)).collect();
            let total_sparse_entries: usize = indexes.iter().map(|index| index.len()).sum();

            let lower = if partition_by_size {
                let lower_target = partition_size_target_at(
                    merge_threads,
                    imbalance_factor,
                    thread_id,
                    total_bytes,
                );
                partition_boundary_by_size_target(&indexes, lower_target)
            } else {
                let lower_target = partition_entry_target_at(
                    merge_threads,
                    imbalance_factor,
                    thread_id,
                    total_sparse_entries,
                );
                partition_boundary_by_entry_target(&indexes, lower_target)
            };

            let upper = if partition_by_size {
                let upper_target = partition_size_target_at(
                    merge_threads,
                    imbalance_factor,
                    thread_id + 1,
                    total_bytes,
                );
                partition_boundary_by_size_target(&indexes, upper_target)
            } else {
                let upper_target = partition_entry_target_at(
                    merge_threads,
                    imbalance_factor,
                    thread_id + 1,
                    total_sparse_entries,
                );
                partition_boundary_by_entry_target(&indexes, upper_target)
            };

            let lower_inc = match partition_type {
                PartitionType::KeyOnly => lower.as_ref().map(|b| (b.key.as_slice(), 0, 0)),
                _ => lower
                    .as_ref()
                    .map(|b| (b.key.as_slice(), b.run_id, b.offset)),
            };
            let upper_exc = match partition_type {
                PartitionType::KeyOnly => upper.as_ref().map(|b| (b.key.as_slice(), 0, 0)),
                _ => upper
                    .as_ref()
                    .map(|b| (b.key.as_slice(), b.run_id, b.offset)),
            };

            let format_bound = |bound: Option<(&[u8], u32, usize)>| -> String {
                match bound {
                    None => "None".to_string(),
                    Some((key, run_id, offset)) => {
                        let key_str = String::from_utf8_lossy(key);
                        format!("(\"{}\", run={}, offset={})", key_str, run_id, offset)
                    }
                }
            };

            println!(
                "Thread {}: partitioning with lower {} and upper {}",
                thread_id,
                format_bound(lower_inc),
                format_bound(upper_exc)
            );

            hooks
                .merge_range(
                    runs,
                    thread_id,
                    run_id,
                    run_indexing_interval,
                    lower_inc,
                    upper_exc,
                    &dir,
                    io_tracker,
                    discard_output,
                )
                .map(|(run, thread_time)| {
                    let io_stats = thread_io_tracker.get_detailed_stats();
                    (thread_id, run, thread_time, io_stats)
                })
        });

        merge_handles.push(handle);
    }

    let mut output_runs = Vec::new();
    let mut per_thread_times_ms = Vec::new();
    let mut per_thread_io_stats = Vec::new();

    for handle in merge_handles {
        let result = handle
            .join()
            .map_err(|_| "Merge worker panicked".to_string())?;
        let (thread_id, run, thread_time, io_stats) = result?;
        output_runs.push((thread_id, run));
        per_thread_times_ms.push((thread_id, thread_time));
        per_thread_io_stats.push((thread_id, io_stats));
    }

    output_runs.sort_by_key(|(thread_id, _)| *thread_id);
    per_thread_times_ms.sort_by_key(|(thread_id, _)| *thread_id);
    per_thread_io_stats.sort_by_key(|(thread_id, _)| *thread_id);

    let output_runs: Vec<_> = output_runs.into_iter().map(|(_, run)| run).collect();
    let per_thread_times_ms: Vec<_> = per_thread_times_ms
        .into_iter()
        .map(|(_, time)| time)
        .collect();
    let per_thread_io_stats: Vec<_> = per_thread_io_stats
        .into_iter()
        .map(|(_, stats)| stats)
        .collect();

    let merge_time_ms = merge_start.elapsed().as_millis();
    let merge_io_stats = if per_thread_io_stats.is_empty() {
        None
    } else {
        let mut merged = crate::IoStats {
            read_ops: 0,
            read_bytes: 0,
            write_ops: 0,
            write_bytes: 0,
        };
        for stats in &per_thread_io_stats {
            merged.read_ops = merged.read_ops.saturating_add(stats.read_ops);
            merged.read_bytes = merged.read_bytes.saturating_add(stats.read_bytes);
            merged.write_ops = merged.write_ops.saturating_add(stats.write_ops);
            merged.write_bytes = merged.write_bytes.saturating_add(stats.write_bytes);
        }
        Some(merged)
    };

    let merge_entry_num: Vec<u64> = output_runs
        .iter()
        .map(|run| run.total_entries() as u64)
        .collect();
    let merge_entry_bytes: Vec<u64> = output_runs
        .iter()
        .map(|run| run.total_bytes() as u64)
        .collect();
    let merge_bytes: u64 = merge_entry_bytes.iter().sum();
    let total_entries = merge_entry_num.iter().sum::<u64>();
    let num_runs = output_runs.len();

    println!(
        "Merge phase took {} ms | Merged {} entries ({} bytes) across {} runs",
        merge_time_ms, total_entries, merge_bytes, num_runs
    );

    if !per_thread_times_ms.is_empty() {
        let (min_time, max_time) = (
            *per_thread_times_ms.iter().min().unwrap(),
            *per_thread_times_ms.iter().max().unwrap(),
        );
        println!("Thread timing: min={} ms, max={} ms", min_time, max_time);
    }

    if !merge_entry_num.is_empty() {
        let min_entries = *merge_entry_num.iter().min().unwrap();
        let max_entries = *merge_entry_num.iter().max().unwrap();
        let avg_entries = total_entries as f64 / merge_entry_num.len() as f64;
        let min_bytes = *merge_entry_bytes.iter().min().unwrap();
        let max_bytes = *merge_entry_bytes.iter().max().unwrap();
        let avg_bytes = merge_bytes as f64 / merge_entry_bytes.len() as f64;
        let per_thread_time_ms: Vec<u64> = (0..merge_entry_num.len())
            .map(|idx| {
                per_thread_times_ms
                    .get(idx)
                    .copied()
                    .unwrap_or(0)
                    .min(u64::MAX as u128) as u64
            })
            .collect();
        let per_thread_io_bytes: Vec<u64> = (0..merge_entry_num.len())
            .map(|idx| {
                per_thread_io_stats
                    .get(idx)
                    .map(|stats| stats.total_bytes())
                    .unwrap_or(0)
            })
            .collect();
        let min_time = *per_thread_time_ms.iter().min().unwrap_or(&0);
        let max_time = *per_thread_time_ms.iter().max().unwrap_or(&0);
        let avg_time = if per_thread_time_ms.is_empty() {
            0.0
        } else {
            per_thread_time_ms.iter().sum::<u64>() as f64 / per_thread_time_ms.len() as f64
        };
        let min_io = *per_thread_io_bytes.iter().min().unwrap_or(&0);
        let max_io = *per_thread_io_bytes.iter().max().unwrap_or(&0);
        let avg_io = if per_thread_io_bytes.is_empty() {
            0.0
        } else {
            per_thread_io_bytes.iter().sum::<u64>() as f64 / per_thread_io_bytes.len() as f64
        };

        let format_ratio = |num: u64, den: u64| -> String {
            if den == 0 {
                "n/a".to_string()
            } else {
                format!("{:.2}x", num as f64 / den as f64)
            }
        };

        let mut rows = Vec::new();
        for (idx, (entries, bytes)) in merge_entry_num
            .iter()
            .zip(merge_entry_bytes.iter())
            .enumerate()
        {
            let time_ms = per_thread_time_ms.get(idx).copied().unwrap_or(0);
            let io_bytes = per_thread_io_bytes.get(idx).copied().unwrap_or(0);
            rows.push(vec![
                format!("T{}", idx),
                entries.to_string(),
                bytes.to_string(),
                time_ms.to_string(),
                io_bytes.to_string(),
            ]);
        }
        rows.push(vec![
            "Min".to_string(),
            min_entries.to_string(),
            min_bytes.to_string(),
            min_time.to_string(),
            min_io.to_string(),
        ]);
        rows.push(vec![
            "Avg".to_string(),
            format!("{:.2}", avg_entries),
            format!("{:.2}", avg_bytes),
            format!("{:.2}", avg_time),
            format!("{:.2}", avg_io),
        ]);
        rows.push(vec![
            "Max".to_string(),
            max_entries.to_string(),
            max_bytes.to_string(),
            max_time.to_string(),
            max_io.to_string(),
        ]);
        rows.push(vec![
            "Min/Max".to_string(),
            format_ratio(min_entries, max_entries),
            format_ratio(min_bytes, max_bytes),
            format_ratio(min_time, max_time),
            format_ratio(min_io, max_io),
        ]);

        let headers = vec![
            "thread".to_string(),
            "entry cnt".to_string(),
            "size".to_string(),
            "time ms".to_string(),
            "io bytes".to_string(),
        ];
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        for row in &rows {
            for (idx, cell) in row.iter().enumerate() {
                widths[idx] = widths[idx].max(cell.len());
            }
        }

        let format_row = |cells: &[String]| -> String {
            let mut parts = Vec::with_capacity(cells.len());
            for (idx, cell) in cells.iter().enumerate() {
                let width = widths[idx];
                if idx == 0 {
                    parts.push(format!("{:<width$}", cell, width = width));
                } else {
                    parts.push(format!("{:>width$}", cell, width = width));
                }
            }
            parts.join(" | ")
        };

        println!("Partition metrics (entries, bytes, time ms, io bytes):");
        println!("{}", format_row(&headers));
        for row in rows {
            println!("{}", format_row(&row));
        }
    }

    let merge_stats = MergeStats {
        output_runs: output_runs.len(),
        merge_entry_num,
        merge_entry_bytes,
        time_ms: merge_time_ms,
        io_stats: merge_io_stats,
        per_thread_times_ms,
        per_thread_io_stats,
    };

    Ok((hooks.combine_runs(output_runs), merge_stats))
}

fn sparse_index_entry_stats<H: SortHooks>(hooks: &H, runs: &[H::MergeableRun]) -> (usize, f64) {
    let mut total_entries = 0usize;
    for run in runs {
        let indexes = hooks.sparse_indexes(run);
        total_entries = total_entries.saturating_add(indexes.len());
    }
    let avg_sparse_entries_per_run = if runs.is_empty() {
        0.0
    } else {
        total_entries as f64 / runs.len() as f64
    };
    (total_entries, avg_sparse_entries_per_run)
}

fn warn_if_sparse_index_too_sparse(context: &str, avg_sparse_entries_per_run: f64) {
    if avg_sparse_entries_per_run <= SPARSE_INDEX_WARN_AVG_ENTRIES_PER_RUN {
        println!(
            "WARNING: {} sparse index density is low (avg sparse entries/run {:.2} <= {:.0})",
            context, avg_sparse_entries_per_run, SPARSE_INDEX_WARN_AVG_ENTRIES_PER_RUN
        );
    }
}

/// Returns the cumulative ratio at an interior boundary without allocating a
/// prefixes array.
///
/// For `num_partitions = k` and interior boundary index `i in [1, k-1]`:
///
/// ```text
/// heavy = min(max(r, 1) / k, 1)
/// each  = (1 - heavy) / (k - 1)
/// ratio(i) = heavy + (i - 1) * each
/// ```
fn partition_target_ratio_at(
    num_partitions: usize,
    imbalance_factor: f64,
    boundary_idx: usize,
) -> Option<f64> {
    if num_partitions <= 1 || boundary_idx == 0 || boundary_idx >= num_partitions {
        return None;
    }

    let k = num_partitions as f64;
    let mut r = if imbalance_factor <= 0.0 {
        1.0
    } else {
        imbalance_factor
    };
    if r < 1.0 {
        r = 1.0;
    }

    let heavy = (r / k).min(1.0);
    let each = (1.0 - heavy) / (k - 1.0);
    let ratio = heavy + each * (boundary_idx as f64 - 1.0);
    Some(ratio.min(1.0))
}

fn partition_target_at(
    num_partitions: usize,
    imbalance_factor: f64,
    boundary_idx: usize,
    total: usize,
) -> Option<usize> {
    if total == 0 {
        return None;
    }
    let ratio = partition_target_ratio_at(num_partitions, imbalance_factor, boundary_idx)?;
    let target = ((total as f64) * ratio).floor() as usize;
    (target < total).then_some(target)
}

fn partition_entry_target_at(
    num_partitions: usize,
    imbalance_factor: f64,
    boundary_idx: usize,
    total_entries: usize,
) -> Option<usize> {
    partition_target_at(
        num_partitions,
        imbalance_factor,
        boundary_idx,
        total_entries,
    )
}

fn partition_size_target_at(
    num_partitions: usize,
    imbalance_factor: f64,
    boundary_idx: usize,
    total_bytes: usize,
) -> Option<usize> {
    partition_target_at(num_partitions, imbalance_factor, boundary_idx, total_bytes)
}

fn bytes_before(index: &MultiSparseIndexes<'_>, pos: usize) -> usize {
    if pos == 0 || index.is_empty() {
        return 0;
    }

    let len = index.len();
    if pos >= len {
        return index.total_bytes();
    }

    index.global_offset_at(pos).unwrap_or(0)
}

fn choose_pivot_largest_range<'a>(
    indexes: &'a [MultiSparseIndexes<'a>],
    lo: &[usize],
    hi: &[usize],
) -> Option<SparseIndexRef<'a>> {
    let mut best_idx: Option<usize> = None;
    let mut best_len = 0usize;

    for idx in 0..indexes.len() {
        let len = hi[idx].saturating_sub(lo[idx]);
        if len > best_len {
            best_len = len;
            best_idx = Some(idx);
        }
    }

    let idx = best_idx?;
    if best_len == 0 {
        return None;
    }
    let mid = lo[idx] + best_len / 2;
    indexes[idx].get(mid)
}

fn choose_pivot_largest_range_by_bytes<'a>(
    indexes: &'a [MultiSparseIndexes<'a>],
    lo: &[usize],
    hi: &[usize],
) -> Option<SparseIndexRef<'a>> {
    let mut best_idx: Option<usize> = None;
    let mut best_bytes = 0usize;

    for idx in 0..indexes.len() {
        let len = hi[idx].saturating_sub(lo[idx]);
        if len == 0 {
            continue;
        }
        let bytes_hi = bytes_before(&indexes[idx], hi[idx]);
        let bytes_lo = bytes_before(&indexes[idx], lo[idx]);
        let bytes = bytes_hi.saturating_sub(bytes_lo);
        if bytes > best_bytes {
            best_bytes = bytes;
            best_idx = Some(idx);
        }
    }

    let idx = best_idx?;
    if best_bytes == 0 {
        return None;
    }
    let len = hi[idx].saturating_sub(lo[idx]);
    let mid = lo[idx] + len / 2;
    indexes[idx].get(mid)
}

/// Count-based boundary search over multiple sparse-index arrays.
///
/// This is the "double binary search" pattern:
///
/// ```text
/// outer search (value domain):
///   choose pivot from the midpoint of the largest active entry range
///
/// inner search (index domain, for each array):
///   pos[k] = upper_bound(indexes[k], pivot) in [lo[k], hi[k])
///   rank   = sum(pos[k])
///
/// update:
///   if rank < target_rank: lo = pos
///   else:                  hi = pos
/// ```
///
/// After narrowing, we evaluate a small candidate set and return the smallest
/// `(key, run_id, offset)` whose global rank reaches `target_rank`.
fn select_boundary_by_count(
    indexes: &[MultiSparseIndexes<'_>],
    target_count: usize,
) -> Option<KeyRunIdOffsetBound> {
    let total: usize = indexes.iter().map(|index| index.len()).sum();
    if total == 0 || target_count >= total {
        return None;
    }

    let target_rank = target_count + 1;
    let k = indexes.len();
    let mut lo = vec![0usize; k];
    let mut hi: Vec<usize> = indexes.iter().map(|index| index.len()).collect();
    let mut pos = Vec::with_capacity(k);

    while (0..k).any(|idx| hi[idx].saturating_sub(lo[idx]) > 1) {
        let pivot = match choose_pivot_largest_range(indexes, &lo, &hi) {
            Some(pivot) => pivot,
            None => return None,
        };

        let mut count = 0usize;
        pos.clear();
        for idx in 0..k {
            let p = indexes[idx].upper_bound_range(
                pivot.key(),
                pivot.run_id,
                pivot.file_offset(),
                lo[idx],
                hi[idx],
            );
            pos.push(p);
            count += p;
        }

        let mut made_progress = false;
        if count < target_rank {
            for idx in 0..k {
                if lo[idx] != pos[idx] {
                    made_progress = true;
                }
                lo[idx] = pos[idx];
            }
        } else {
            for idx in 0..k {
                if hi[idx] != pos[idx] {
                    made_progress = true;
                }
                hi[idx] = pos[idx];
            }
        }

        if !made_progress {
            break;
        }
    }

    let total_candidates: usize = hi
        .iter()
        .zip(lo.iter())
        .map(|(h, l)| h.saturating_sub(*l))
        .sum();
    debug_assert!(
        total_candidates <= 2 * k,
        "candidate window too large after count narrowing: {}",
        total_candidates
    );

    let mut candidates = Vec::with_capacity(k.saturating_mul(2));
    for idx in 0..k {
        for pos in lo[idx]..hi[idx] {
            if let Some(entry) = indexes[idx].get(pos) {
                candidates.push(entry);
            }
        }
    }

    if candidates.is_empty() {
        for idx in 0..k {
            if lo[idx] < indexes[idx].len() {
                if let Some(entry) = indexes[idx].get(lo[idx]) {
                    candidates.push(entry);
                }
            }
        }
    }

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|a, b| cmp_key_run_offset(a.as_key_run_offset(), b.as_key_run_offset()));

    for candidate in candidates {
        let mut count = 0usize;
        for index in indexes {
            count += index.upper_bound(candidate.key(), candidate.run_id, candidate.file_offset());
        }
        if count >= target_rank {
            return Some(KeyRunIdOffsetBound::from_components((
                candidate.key(),
                candidate.run_id,
                candidate.file_offset(),
            )));
        }
    }

    None
}

/// Byte-size variant of boundary search.
///
/// The control flow is identical to `select_boundary_by_count`, except the
/// objective is cumulative bytes before a pivot, and the pivot is chosen from
/// the midpoint of the largest active byte range:
///
/// `bytes = sum_k bytes_before(indexes[k], upper_bound(indexes[k], pivot))`
///
/// where `bytes_before` maps sparse-index position to an approximate file
/// offset inside each merged input.
fn select_boundary_by_size(
    indexes: &[MultiSparseIndexes<'_>],
    target_bytes: usize,
) -> Option<KeyRunIdOffsetBound> {
    let total_bytes: usize = indexes.iter().map(|index| index.total_bytes()).sum();
    if total_bytes == 0 || target_bytes >= total_bytes {
        return None;
    }

    let k = indexes.len();
    let mut lo = vec![0usize; k];
    let mut hi: Vec<usize> = indexes.iter().map(|index| index.len()).collect();
    let mut pos = Vec::with_capacity(k);
    let mut use_count_pivot = false;

    while (0..k).any(|idx| hi[idx].saturating_sub(lo[idx]) > 1) {
        let pivot = match if use_count_pivot {
            choose_pivot_largest_range(indexes, &lo, &hi)
        } else {
            choose_pivot_largest_range_by_bytes(indexes, &lo, &hi)
        } {
            Some(pivot) => pivot,
            None => {
                if use_count_pivot {
                    return None;
                }
                use_count_pivot = true;
                continue;
            }
        };

        let mut bytes = 0usize;
        pos.clear();
        for idx in 0..k {
            let p = indexes[idx].upper_bound_range(
                pivot.key(),
                pivot.run_id,
                pivot.file_offset(),
                lo[idx],
                hi[idx],
            );
            pos.push(p);
            bytes += bytes_before(&indexes[idx], p);
        }

        let mut made_progress = false;
        if bytes < target_bytes {
            for idx in 0..k {
                if lo[idx] != pos[idx] {
                    made_progress = true;
                }
                lo[idx] = pos[idx];
            }
        } else {
            for idx in 0..k {
                if hi[idx] != pos[idx] {
                    made_progress = true;
                }
                hi[idx] = pos[idx];
            }
        }

        if !made_progress {
            if !use_count_pivot {
                use_count_pivot = true;
                continue;
            }
            break;
        }
    }

    let total_candidates: usize = hi
        .iter()
        .zip(lo.iter())
        .map(|(h, l)| h.saturating_sub(*l))
        .sum();
    debug_assert!(
        total_candidates <= 2 * k,
        "candidate window too large after size narrowing: {}",
        total_candidates
    );

    let mut candidates = Vec::with_capacity(k.saturating_mul(2));
    for idx in 0..k {
        for pos in lo[idx]..hi[idx] {
            if let Some(entry) = indexes[idx].get(pos) {
                candidates.push(entry);
            }
        }
    }

    if candidates.is_empty() {
        for idx in 0..k {
            if lo[idx] < indexes[idx].len() {
                if let Some(entry) = indexes[idx].get(lo[idx]) {
                    candidates.push(entry);
                }
            }
        }
    }

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|a, b| cmp_key_run_offset(a.as_key_run_offset(), b.as_key_run_offset()));

    for candidate in candidates {
        let mut bytes = 0usize;
        for index in indexes {
            let pos = index.upper_bound(candidate.key(), candidate.run_id, candidate.file_offset());
            bytes += bytes_before(index, pos);
        }
        if bytes >= target_bytes {
            return Some(KeyRunIdOffsetBound::from_components((
                candidate.key(),
                candidate.run_id,
                candidate.file_offset(),
            )));
        }
    }

    None
}

#[cfg(test)]
fn partition_by_key_range_and_entry_count(
    indexes: &[MultiSparseIndexes<'_>],
    num_partitions: usize,
    imbalance_factor: f64,
) -> Vec<Option<KeyRunIdOffsetBound>> {
    let mut bounds = Vec::with_capacity(num_partitions + 1);
    bounds.push(None);

    if num_partitions <= 1 {
        bounds.push(None);
        return bounds;
    }

    let total_entries: usize = indexes.iter().map(|index| index.len()).sum();

    for idx in 1..num_partitions {
        let target =
            partition_entry_target_at(num_partitions, imbalance_factor, idx, total_entries);
        let bound = partition_boundary_by_entry_target(indexes, target);
        bounds.push(bound);
    }

    bounds.push(None);
    bounds
}

#[cfg(test)]
fn partition_by_key_range_and_size(
    indexes: &[MultiSparseIndexes<'_>],
    num_partitions: usize,
    imbalance_factor: f64,
) -> Vec<Option<KeyRunIdOffsetBound>> {
    let mut bounds = Vec::with_capacity(num_partitions + 1);
    bounds.push(None);

    if num_partitions <= 1 {
        bounds.push(None);
        return bounds;
    }

    let total_bytes: usize = indexes.iter().map(|index| index.total_bytes()).sum();

    for idx in 1..num_partitions {
        let target = partition_size_target_at(num_partitions, imbalance_factor, idx, total_bytes);
        let bound = partition_boundary_by_size_target(indexes, target);
        bounds.push(bound);
    }

    bounds.push(None);
    bounds
}

fn partition_boundary_by_entry_target(
    indexes: &[MultiSparseIndexes<'_>],
    target: Option<usize>,
) -> Option<KeyRunIdOffsetBound> {
    target.and_then(|target_count| select_boundary_by_count(indexes, target_count))
}

fn partition_boundary_by_size_target(
    indexes: &[MultiSparseIndexes<'_>],
    target: Option<usize>,
) -> Option<KeyRunIdOffsetBound> {
    target.and_then(|target_bytes| select_boundary_by_size(indexes, target_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskio::aligned_writer::AlignedWriter;
    use crate::diskio::file::SharedFd;
    use crate::ovc::offset_value_coding_32::OVCU32;
    use crate::sort::ovc::run::RunWithOVC;
    use crate::sort::ovc::sorter::MergeableRunWithOVC;
    use crate::sort::plain::run::Run;
    use crate::sort::plain::sorter::MergeableRun;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn build_run(temp_dir: &TempDir, name: &str, keys: &[&[u8]], value_len: usize) -> Run {
        let path = temp_dir.path().join(name);
        let fd = Arc::new(SharedFd::new_from_path(&path, true).unwrap());
        let writer = AlignedWriter::from_fd(fd).unwrap();
        let mut run =
            Run::from_writer_with_indexing_interval(writer, IndexingInterval::records(1)).unwrap();
        let value = vec![b'v'; value_len];
        for key in keys {
            run.append(key, &value);
        }
        run.finalize_write();
        run
    }

    fn build_ovc_run(
        temp_dir: &TempDir,
        name: &str,
        keys: &[&[u8]],
        value_len: usize,
    ) -> RunWithOVC {
        let path = temp_dir.path().join(name);
        let fd = Arc::new(SharedFd::new_from_path(&path, true).unwrap());
        let writer = AlignedWriter::from_fd(fd).unwrap();
        let mut run =
            RunWithOVC::from_writer_with_indexing_interval(writer, IndexingInterval::records(1))
                .unwrap();
        let value = vec![b'v'; value_len];
        for key in keys {
            run.append(OVCU32::initial_value(), key, &value);
        }
        run.finalize_write();
        run
    }

    fn build_run_with_interval(
        temp_dir: &TempDir,
        name: &str,
        keys: &[&[u8]],
        value_len: usize,
        indexing_interval: usize,
    ) -> Run {
        let path = temp_dir.path().join(name);
        let fd = Arc::new(SharedFd::new_from_path(&path, true).unwrap());
        let writer = AlignedWriter::from_fd(fd).unwrap();
        let mut run = Run::from_writer_with_indexing_interval(
            writer,
            IndexingInterval::records(indexing_interval),
        )
        .unwrap();
        let value = vec![b'v'; value_len];
        for key in keys {
            run.append(key, &value);
        }
        run.finalize_write();
        run
    }

    fn build_run_with_prefix(
        temp_dir: &TempDir,
        name: &str,
        prefix: &str,
        count: usize,
        value_len: usize,
    ) -> Run {
        let keys: Vec<Vec<u8>> = (0..count)
            .map(|i| format!("{}{:05}", prefix, i).into_bytes())
            .collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        build_run(temp_dir, name, &key_refs, value_len)
    }

    fn build_run_with_prefix_and_interval(
        temp_dir: &TempDir,
        name: &str,
        prefix: &str,
        count: usize,
        value_len: usize,
        indexing_interval: usize,
    ) -> Run {
        let keys: Vec<Vec<u8>> = (0..count)
            .map(|i| format!("{}{:05}", prefix, i).into_bytes())
            .collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        build_run_with_interval(temp_dir, name, &key_refs, value_len, indexing_interval)
    }

    fn flatten_sorted_entries(indexes: &[MultiSparseIndexes<'_>]) -> Vec<KeyRunIdOffsetBound> {
        let mut entries = Vec::new();
        for index in indexes {
            for i in 0..index.len() {
                let entry = index.get(i).unwrap();
                entries.push(KeyRunIdOffsetBound::from_components((
                    entry.key(),
                    entry.run_id,
                    entry.file_offset(),
                )));
            }
        }
        entries.sort_by(|a, b| {
            cmp_key_run_offset((&a.key, a.run_id, a.offset), (&b.key, b.run_id, b.offset))
        });
        entries
    }

    fn expected_boundary_by_count(
        indexes: &[MultiSparseIndexes<'_>],
        target_count: usize,
    ) -> Option<KeyRunIdOffsetBound> {
        let entries = flatten_sorted_entries(indexes);
        if target_count >= entries.len() {
            None
        } else {
            Some(entries[target_count].clone())
        }
    }

    fn bytes_before_at(index: &MultiSparseIndexes<'_>, pos: usize) -> usize {
        if pos == 0 {
            0
        } else if pos >= index.len() {
            index.total_bytes()
        } else {
            index.global_offset_at(pos).unwrap()
        }
    }

    fn expected_boundary_by_size(
        indexes: &[MultiSparseIndexes<'_>],
        target_bytes: usize,
    ) -> Option<KeyRunIdOffsetBound> {
        let total_bytes: usize = indexes.iter().map(|index| index.total_bytes()).sum();
        if total_bytes == 0 || target_bytes >= total_bytes {
            return None;
        }

        let candidates = flatten_sorted_entries(indexes);
        for cand in candidates {
            let mut bytes = 0usize;
            for index in indexes {
                let pos = index.upper_bound(&cand.key, cand.run_id, cand.offset);
                bytes += bytes_before_at(index, pos);
            }
            if bytes >= target_bytes {
                return Some(cand);
            }
        }

        None
    }

    fn expected_bounds_by_count(
        indexes: &[MultiSparseIndexes<'_>],
        num_partitions: usize,
        imbalance_factor: f64,
    ) -> Vec<Option<KeyRunIdOffsetBound>> {
        let mut bounds = Vec::with_capacity(num_partitions + 1);
        bounds.push(None);
        if num_partitions <= 1 {
            bounds.push(None);
            return bounds;
        }

        let total_entries: usize = indexes.iter().map(|index| index.len()).sum();
        for idx in 1..num_partitions {
            let target =
                partition_entry_target_at(num_partitions, imbalance_factor, idx, total_entries);
            let boundary = if total_entries == 0 {
                None
            } else {
                target.and_then(|value| expected_boundary_by_count(indexes, value))
            };
            bounds.push(boundary);
        }
        bounds.push(None);
        bounds
    }

    fn expected_bounds_by_size(
        indexes: &[MultiSparseIndexes<'_>],
        num_partitions: usize,
        imbalance_factor: f64,
    ) -> Vec<Option<KeyRunIdOffsetBound>> {
        let mut bounds = Vec::with_capacity(num_partitions + 1);
        bounds.push(None);
        if num_partitions <= 1 {
            bounds.push(None);
            return bounds;
        }

        let total_bytes: usize = indexes.iter().map(|index| index.total_bytes()).sum();
        for idx in 1..num_partitions {
            let target =
                partition_size_target_at(num_partitions, imbalance_factor, idx, total_bytes);
            let boundary = if total_bytes == 0 {
                None
            } else {
                target.and_then(|value| expected_boundary_by_size(indexes, value))
            };
            bounds.push(boundary);
        }
        bounds.push(None);
        bounds
    }

    fn assert_same_bound(
        got: &Option<KeyRunIdOffsetBound>,
        expected: &Option<KeyRunIdOffsetBound>,
    ) {
        match (got, expected) {
            (None, None) => {}
            (Some(g), Some(e)) => {
                assert_eq!(g.key, e.key);
                assert_eq!(g.run_id, e.run_id);
                assert_eq!(g.offset, e.offset);
            }
            _ => panic!("boundary mismatch: got={:?}, expected={:?}", got, expected),
        }
    }

    fn assert_monotonic_bounds(bounds: &[Option<KeyRunIdOffsetBound>]) {
        let mut prev: Option<&KeyRunIdOffsetBound> = None;
        for bound in bounds {
            if let Some(cur) = bound {
                if let Some(p) = prev {
                    let ord = cmp_key_run_offset(
                        (&p.key, p.run_id, p.offset),
                        (&cur.key, cur.run_id, cur.offset),
                    );
                    assert!(matches!(
                        ord,
                        std::cmp::Ordering::Less | std::cmp::Ordering::Equal
                    ));
                }
                prev = Some(cur);
            }
        }
    }

    #[test]
    fn test_partition_by_entry_count_two_runs() {
        let temp_dir = TempDir::new().unwrap();
        let run0 = build_run(
            &temp_dir,
            "run0.dat",
            &[
                b"a0", b"a1", b"a2", b"a3", b"a4", b"a5", b"a6", b"a7", b"a8", b"a9",
            ],
            1,
        );
        let run1 = build_run(
            &temp_dir,
            "run1.dat",
            &[
                b"b0", b"b1", b"b2", b"b3", b"b4", b"b5", b"b6", b"b7", b"b8", b"b9",
            ],
            1,
        );
        let run1_id = run1.run_id();

        let mergeable_runs = vec![MergeableRun::Single(run0), MergeableRun::Single(run1)];
        let indexes: Vec<_> = mergeable_runs
            .iter()
            .map(|run| run.sparse_indexes())
            .collect();

        let bounds = partition_by_key_range_and_entry_count(&indexes, 2, 1.0);
        let expected = expected_bounds_by_count(&indexes, 2, 1.0);
        assert_eq!(bounds.len(), 3);
        assert!(bounds[0].is_none());
        assert!(bounds[2].is_none());
        for (got, exp) in bounds.iter().zip(expected.iter()) {
            assert_same_bound(got, exp);
        }

        let mid = bounds[1].as_ref().expect("expected midpoint boundary");
        assert_eq!(mid.key, b"b0");
        assert_eq!(mid.run_id, run1_id);
        assert_eq!(mid.offset, 0);
    }

    #[test]
    fn test_partition_by_entry_count_two_runs_with_ovc() {
        let temp_dir = TempDir::new().unwrap();
        let run0 = build_ovc_run(
            &temp_dir,
            "ovc_run0.dat",
            &[
                b"a0", b"a1", b"a2", b"a3", b"a4", b"a5", b"a6", b"a7", b"a8", b"a9",
            ],
            1,
        );
        let run1 = build_ovc_run(
            &temp_dir,
            "ovc_run1.dat",
            &[
                b"b0", b"b1", b"b2", b"b3", b"b4", b"b5", b"b6", b"b7", b"b8", b"b9",
            ],
            1,
        );
        let run1_id = run1.run_id();

        let mergeable_runs = vec![
            MergeableRunWithOVC::Single(run0),
            MergeableRunWithOVC::Single(run1),
        ];
        let indexes: Vec<_> = mergeable_runs
            .iter()
            .map(|run| run.sparse_indexes())
            .collect();

        let bounds = partition_by_key_range_and_entry_count(&indexes, 2, 1.0);
        let expected = expected_bounds_by_count(&indexes, 2, 1.0);
        assert_eq!(bounds.len(), 3);
        assert!(bounds[0].is_none());
        assert!(bounds[2].is_none());
        for (got, exp) in bounds.iter().zip(expected.iter()) {
            assert_same_bound(got, exp);
        }

        let mid = bounds[1].as_ref().expect("expected midpoint boundary");
        assert_eq!(mid.key, b"b0");
        assert_eq!(mid.run_id, run1_id);
        assert_eq!(mid.offset, 0);
    }

    #[test]
    fn test_partition_by_entry_count_sparse_interval_10() {
        let temp_dir = TempDir::new().unwrap();
        let run0 = build_run_with_prefix_and_interval(&temp_dir, "run0_i10.dat", "a", 50, 1, 10);
        let run1 = build_run_with_prefix_and_interval(&temp_dir, "run1_i10.dat", "b", 50, 1, 10);
        let run1_id = run1.run_id();

        assert_eq!(run0.sparse_index().len(), 5);
        assert_eq!(run1.sparse_index().len(), 5);

        let mergeable_runs = vec![MergeableRun::Single(run0), MergeableRun::Single(run1)];
        let indexes: Vec<_> = mergeable_runs
            .iter()
            .map(|run| run.sparse_indexes())
            .collect();

        let bounds = partition_by_key_range_and_entry_count(&indexes, 2, 1.0);
        let expected = expected_bounds_by_count(&indexes, 2, 1.0);
        assert_eq!(bounds.len(), 3);
        for (got, exp) in bounds.iter().zip(expected.iter()) {
            assert_same_bound(got, exp);
        }

        let mid = bounds[1].as_ref().expect("expected midpoint boundary");
        assert_eq!(mid.key, b"b00000");
        assert_eq!(mid.run_id, run1_id);
        assert_eq!(mid.offset, 0);
    }

    #[test]
    fn test_partition_by_entry_count_sparse_interval_100() {
        let temp_dir = TempDir::new().unwrap();
        let run0 = build_run_with_prefix_and_interval(&temp_dir, "run0_i100.dat", "a", 50, 1, 100);
        let run1 = build_run_with_prefix_and_interval(&temp_dir, "run1_i100.dat", "b", 50, 1, 100);
        let run1_id = run1.run_id();

        assert_eq!(run0.sparse_index().len(), 1);
        assert_eq!(run1.sparse_index().len(), 1);

        let mergeable_runs = vec![MergeableRun::Single(run0), MergeableRun::Single(run1)];
        let indexes: Vec<_> = mergeable_runs
            .iter()
            .map(|run| run.sparse_indexes())
            .collect();

        let bounds = partition_by_key_range_and_entry_count(&indexes, 2, 1.0);
        let expected = expected_bounds_by_count(&indexes, 2, 1.0);
        assert_eq!(bounds.len(), 3);
        for (got, exp) in bounds.iter().zip(expected.iter()) {
            assert_same_bound(got, exp);
        }

        let mid = bounds[1].as_ref().expect("expected midpoint boundary");
        assert_eq!(mid.key, b"b00000");
        assert_eq!(mid.run_id, run1_id);
        assert_eq!(mid.offset, 0);
    }

    #[test]
    fn test_partition_by_size_two_runs() {
        let temp_dir = TempDir::new().unwrap();
        let run0 = build_run(
            &temp_dir,
            "run0_size.dat",
            &[b"a0", b"a1", b"a2", b"a3", b"a4"],
            1,
        );
        let run1 = build_run(
            &temp_dir,
            "run1_size.dat",
            &[b"b0", b"b1", b"b2", b"b3", b"b4"],
            100,
        );
        let run1_id = run1.run_id();

        let mergeable_runs = vec![MergeableRun::Single(run0), MergeableRun::Single(run1)];
        let indexes: Vec<_> = mergeable_runs
            .iter()
            .map(|run| run.sparse_indexes())
            .collect();

        let bounds = partition_by_key_range_and_size(&indexes, 2, 1.0);
        let expected = expected_bounds_by_size(&indexes, 2, 1.0);
        assert_eq!(bounds.len(), 3);
        assert!(bounds[0].is_none());
        assert!(bounds[2].is_none());
        for (got, exp) in bounds.iter().zip(expected.iter()) {
            assert_same_bound(got, exp);
        }

        let mid = bounds[1].as_ref().expect("expected midpoint boundary");
        assert_eq!(mid.key, b"b2");
        assert_eq!(mid.run_id, run1_id);
    }

    #[test]
    fn test_partition_by_size_two_runs_with_ovc() {
        let temp_dir = TempDir::new().unwrap();
        let run0 = build_ovc_run(
            &temp_dir,
            "ovc_run0_size.dat",
            &[b"a0", b"a1", b"a2", b"a3", b"a4"],
            1,
        );
        let run1 = build_ovc_run(
            &temp_dir,
            "ovc_run1_size.dat",
            &[b"b0", b"b1", b"b2", b"b3", b"b4"],
            100,
        );
        let run1_id = run1.run_id();

        let mergeable_runs = vec![
            MergeableRunWithOVC::Single(run0),
            MergeableRunWithOVC::Single(run1),
        ];
        let indexes: Vec<_> = mergeable_runs
            .iter()
            .map(|run| run.sparse_indexes())
            .collect();

        let bounds = partition_by_key_range_and_size(&indexes, 2, 1.0);
        let expected = expected_bounds_by_size(&indexes, 2, 1.0);
        assert_eq!(bounds.len(), 3);
        assert!(bounds[0].is_none());
        assert!(bounds[2].is_none());
        for (got, exp) in bounds.iter().zip(expected.iter()) {
            assert_same_bound(got, exp);
        }

        let mid = bounds[1].as_ref().expect("expected midpoint boundary");
        assert_eq!(mid.run_id, run1_id);
    }

    #[test]
    fn test_partition_by_entry_count_handles_duplicate_keys_with_run_id_order() {
        let temp_dir = TempDir::new().unwrap();
        let run0 = build_run(&temp_dir, "dup_run0.dat", &[b"k", b"k", b"k", b"k"], 4);
        let run1 = build_run(&temp_dir, "dup_run1.dat", &[b"k", b"k", b"k", b"k"], 4);
        let run1_id = run1.run_id();

        let mergeable_runs = vec![MergeableRun::Single(run0), MergeableRun::Single(run1)];
        let indexes: Vec<_> = mergeable_runs
            .iter()
            .map(|run| run.sparse_indexes())
            .collect();
        let bounds = partition_by_key_range_and_entry_count(&indexes, 2, 1.0);
        let expected = expected_bounds_by_count(&indexes, 2, 1.0);

        for (got, exp) in bounds.iter().zip(expected.iter()) {
            assert_same_bound(got, exp);
        }
        let mid = bounds[1].as_ref().unwrap();
        println!(
            "Mid boundary: key={:?} run_id={}, offset={}",
            mid.key, mid.run_id, mid.offset
        );
        assert_eq!(mid.key, b"k");
        assert_eq!(mid.run_id, run1_id);
        assert_eq!(mid.offset, 0);
    }

    #[test]
    fn test_partition_by_entry_count_range_partitioned_matches_bruteforce() {
        let temp_dir = TempDir::new().unwrap();
        let run0 = build_run(
            &temp_dir,
            "rp_run0.dat",
            &[b"a0", b"a1", b"a2", b"a3", b"a4", b"a5"],
            2,
        );
        let run1 = build_run(
            &temp_dir,
            "rp_run1.dat",
            &[b"b0", b"b0", b"b1", b"b2", b"b3"],
            5,
        );
        let run2 = build_run(
            &temp_dir,
            "rp_run2.dat",
            &[b"c0", b"c1", b"c2", b"d0", b"d1", b"e0"],
            7,
        );

        let mergeable_runs = vec![
            MergeableRun::RangePartitioned(vec![run0, run1]),
            MergeableRun::Single(run2),
        ];
        let indexes: Vec<_> = mergeable_runs
            .iter()
            .map(|run| run.sparse_indexes())
            .collect();

        let bounds = partition_by_key_range_and_entry_count(&indexes, 4, 1.0);
        let expected = expected_bounds_by_count(&indexes, 4, 1.0);
        assert_eq!(bounds.len(), 5);
        for (got, exp) in bounds.iter().zip(expected.iter()) {
            println!(
                "Got boundary: key={:?} run_id={} offset={}",
                got.as_ref()
                    .map_or("None", |b| std::str::from_utf8(&b.key).unwrap()),
                got.as_ref().map_or(0, |b| b.run_id),
                got.as_ref().map_or(0, |b| b.offset)
            );
            assert_same_bound(got, exp);
        }
        assert_monotonic_bounds(&bounds);
    }

    #[test]
    fn test_partition_by_size_range_partitioned_with_imbalance_matches_bruteforce() {
        let temp_dir = TempDir::new().unwrap();
        let run0 = build_run(
            &temp_dir,
            "size_rp_run0.dat",
            &[b"a0", b"a1", b"a2", b"a3", b"a4", b"a5"],
            1,
        );
        let run1 = build_run(
            &temp_dir,
            "size_rp_run1.dat",
            &[b"b0", b"b1", b"b2", b"b3", b"b4", b"b5"],
            40,
        );
        let run2 = build_run(
            &temp_dir,
            "size_rp_run2.dat",
            &[b"c0", b"c1", b"c2", b"c3", b"c4", b"c5"],
            200,
        );

        let mergeable_runs = vec![
            MergeableRun::RangePartitioned(vec![run0, run1]),
            MergeableRun::Single(run2),
        ];
        let indexes: Vec<_> = mergeable_runs
            .iter()
            .map(|run| run.sparse_indexes())
            .collect();

        let bounds = partition_by_key_range_and_size(&indexes, 5, 1.8);
        let expected = expected_bounds_by_size(&indexes, 5, 1.8);
        assert_eq!(bounds.len(), 6);
        for (got, exp) in bounds.iter().zip(expected.iter()) {
            println!(
                "Got boundary: key={:?} run_id={} offset={}",
                got.as_ref()
                    .map_or("None", |b| std::str::from_utf8(&b.key).unwrap()),
                got.as_ref().map_or(0, |b| b.run_id),
                got.as_ref().map_or(0, |b| b.offset)
            );
            assert_same_bound(got, exp);
        }
        assert_monotonic_bounds(&bounds);
    }

    #[test]
    fn test_partition_by_entry_count_extreme_case_uses_real_path() {
        let temp_dir = TempDir::new().unwrap();
        let run0 = build_run_with_prefix(&temp_dir, "count_big.dat", "a", 10_000, 1);
        let run0_id = run0.run_id();
        let run1 = build_run(&temp_dir, "count_small.dat", &[b"z0", b"z1"], 1);

        let mergeable_runs = vec![MergeableRun::Single(run0), MergeableRun::Single(run1)];
        let indexes: Vec<_> = mergeable_runs
            .iter()
            .map(|run| run.sparse_indexes())
            .collect();

        let bounds = partition_by_key_range_and_entry_count(&indexes, 2, 1.0);
        assert_eq!(bounds.len(), 3);
        let mid = bounds[1].as_ref().expect("expected midpoint boundary");
        assert_eq!(mid.run_id, run0_id);
    }

    #[test]
    fn test_partition_by_size_extreme_case_uses_real_path() {
        let temp_dir = TempDir::new().unwrap();
        let run0 = build_run_with_prefix(&temp_dir, "size_small.dat", "a", 10_000, 1);
        let run1 = build_run(&temp_dir, "size_big.dat", &[b"z0", b"z1"], 200_000);
        let run1_id = run1.run_id();

        let mergeable_runs = vec![MergeableRun::Single(run0), MergeableRun::Single(run1)];
        let indexes: Vec<_> = mergeable_runs
            .iter()
            .map(|run| run.sparse_indexes())
            .collect();

        let bounds = partition_by_key_range_and_size(&indexes, 2, 1.0);
        assert_eq!(bounds.len(), 3);
        let mid = bounds[1].as_ref().expect("expected midpoint boundary");
        assert_eq!(mid.run_id, run1_id);
    }

    #[test]
    fn test_partition_target_ratio_respects_imbalance_factor() {
        assert!(partition_target_ratio_at(4, 2.0, 0).is_none());
        assert!(partition_target_ratio_at(4, 2.0, 4).is_none());

        let r1 = partition_target_ratio_at(4, 2.0, 1).unwrap();
        let r2 = partition_target_ratio_at(4, 2.0, 2).unwrap();
        let r3 = partition_target_ratio_at(4, 2.0, 3).unwrap();
        assert!(r1 <= r2 && r2 <= r3);

        let first = r1;
        let second = r2 - r1;
        let third = r3 - r2;
        let fourth = 1.0 - r3;
        assert!(first > second);
        assert!((second - third).abs() < 1e-12);
        assert!((third - fourth).abs() < 1e-12);
    }
}
