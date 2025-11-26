use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::file::SharedFd;
use crate::diskio::io_stats::IoStatsTracker;
use crate::rand::small_thread_rng;
use crate::replacement_selection::run_replacement_selection;
use crate::sketch::kll::KLL;
use crate::sort::run_sink::RunSink;
use crate::sort::sort_buffer::SortBuffer;
use crate::{
    MergeStats, RunGenerationStats, RunInfo, SortInput, SortOutput, SortStats, TempDirInfo,
};
use rand::Rng;

const ENTRY_METADATA_SIZE: usize = std::mem::size_of::<u32>() * 2;

pub type Scanner = Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>;

pub struct RunGenerationThreadResult<R> {
    pub runs: Vec<R>,
    pub sketch: KLL<Vec<u8>>,
    pub load_ms: u128,
    pub sort_ms: u128,
    pub store_ms: u128,
}

pub trait RunSummary {
    fn total_entries(&self) -> usize;
    fn total_bytes(&self) -> usize;
}

pub fn execute_run_generation<R, F>(
    sort_input: Box<dyn SortInput>,
    num_threads: usize,
    sketch_size: usize,
    dir: impl AsRef<Path>,
    worker: F,
) -> Result<(Vec<R>, KLL<Vec<u8>>, RunGenerationStats), String>
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
            KLL::new(sketch_size),
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
    let mut sketch = KLL::new(sketch_size);
    let mut total_load_ms: u128 = 0;
    let mut total_sort_ms: u128 = 0;
    let mut total_store_ms: u128 = 0;
    let mut threads_count: u128 = 0;

    for handle in handles {
        let result = handle
            .join()
            .map_err(|_| "Run generation worker panicked".to_string())?;
        output_runs.extend(result.runs);
        sketch.merge(&result.sketch);
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
        sketch,
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
        run_indexing_interval: usize,
        sketch_size: usize,
        sketch_sampling_interval: usize,
    ) -> Self::Sink;

    fn dummy_run(&self) -> Self::MergeableRun;

    fn combine_runs(&self, runs: Vec<Self::MergeableRun>) -> Self::MergeableRun;

    fn merge_range(
        &self,
        runs: Arc<Vec<Self::MergeableRun>>,
        thread_id: usize,
        lower_bound: Vec<u8>,
        upper_bound: Vec<u8>,
        dir: &Path,
        io_tracker: Arc<IoStatsTracker>,
    ) -> Result<(Self::MergeableRun, u128), String>;

    fn into_output(&self, run: Self::MergeableRun, stats: SortStats) -> Box<dyn SortOutput>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunGenerationAlgorithm {
    ReplacementSelection,
    LoadSortStore,
}

impl std::fmt::Display for RunGenerationAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReplacementSelection => write!(f, "replacement-selection"),
            Self::LoadSortStore => write!(f, "load-sort-store"),
        }
    }
}

impl std::str::FromStr for RunGenerationAlgorithm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalized = s.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "replacement-selection"
            | "replacement_selection"
            | "replacementselection"
            | "replacement" => Ok(Self::ReplacementSelection),
            "load-sort-store" | "load_sort_store" | "loadsortstore" | "load" | "lss" => {
                Ok(Self::LoadSortStore)
            }
            other => Err(format!(
                "Invalid run generation algorithm '{other}'. Expected 'replacement-selection' or 'load-sort-store'."
            )),
        }
    }
}

pub struct SorterCore<H: SortHooks> {
    hooks: H,
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

impl<H: SortHooks> SorterCore<H> {
    pub fn with_hooks(
        hooks: H,
        run_gen_threads: usize,
        run_size: usize,
        merge_threads: usize,
        merge_fanin: usize,
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
            run_size,
            merge_threads,
            merge_fanin,
            sketch_size: 200,
            sketch_sampling_interval: 1000,
            run_indexing_interval: 1000,
            imbalance_factor: 1.0,
            temp_dir_info: Arc::new(TempDirInfo {
                path: temp_dir,
                should_delete: true,
            }),
            run_gen_algorithm: RunGenerationAlgorithm::ReplacementSelection,
        }
    }

    pub fn temp_dir(&self) -> &Path {
        self.temp_dir_info.as_ref().as_ref()
    }

    pub fn set_run_generation_algorithm(&mut self, algorithm: RunGenerationAlgorithm) {
        self.run_gen_algorithm = algorithm;
    }

    pub fn run_generation_algorithm(&self) -> RunGenerationAlgorithm {
        self.run_gen_algorithm
    }

    pub fn set_sketch_sampling_interval(&mut self, interval: usize) {
        self.sketch_sampling_interval = interval;
    }

    pub fn set_run_indexing_interval(&mut self, interval: usize) {
        self.run_indexing_interval = interval;
    }

    pub fn set_sketch_size(&mut self, size: usize) {
        self.sketch_size = size;
    }

    pub fn set_imbalance_factor(&mut self, factor: f64) {
        self.imbalance_factor = factor;
    }

    fn run_generation_internal(
        &self,
        sort_input: Box<dyn SortInput>,
    ) -> Result<(Vec<H::MergeableRun>, KLL<Vec<u8>>, RunGenerationStats), String> {
        run_generation_with_hooks(
            &self.hooks,
            sort_input,
            self.run_gen_threads,
            self.run_size,
            self.sketch_size,
            self.sketch_sampling_interval,
            self.run_indexing_interval,
            self.temp_dir_info.as_ref().as_ref(),
            self.run_gen_algorithm,
        )
    }

    fn multi_merge_internal(
        &self,
        runs: Vec<H::MergeableRun>,
        sketch: &KLL<Vec<u8>>,
    ) -> Result<(H::MergeableRun, Vec<MergeStats>), String> {
        multi_merge_with_hooks(
            &self.hooks,
            runs,
            self.merge_fanin,
            self.merge_threads,
            sketch,
            self.imbalance_factor,
            self.temp_dir_info.as_ref().as_ref(),
        )
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
    ) -> Result<(Vec<H::MergeableRun>, KLL<Vec<u8>>, RunGenerationStats), String>
    where
        H: Default,
    {
        let hooks = H::default();
        run_generation_with_hooks(
            &hooks,
            sort_input,
            num_threads,
            run_size,
            sketch_size,
            sketch_sampling_interval,
            run_indexing_interval,
            dir,
            algorithm,
        )
    }

    pub fn multi_merge(
        runs: Vec<H::MergeableRun>,
        fanin: usize,
        num_threads: usize,
        sketch: &KLL<Vec<u8>>,
        imbalance_factor: f64,
        dir: impl AsRef<Path>,
    ) -> Result<(H::MergeableRun, Vec<MergeStats>), String>
    where
        H: Default,
    {
        multi_merge_with_hooks(
            &H::default(),
            runs,
            fanin,
            num_threads,
            sketch,
            imbalance_factor,
            dir.as_ref(),
        )
    }
}

impl<H: SortHooks + Default> SorterCore<H> {
    pub fn with_defaults(
        run_gen_threads: usize,
        run_size: usize,
        merge_threads: usize,
        merge_fanin: usize,
        base_dir: impl AsRef<Path>,
    ) -> Self {
        Self::with_hooks(
            H::default(),
            run_gen_threads,
            run_size,
            merge_threads,
            merge_fanin,
            base_dir,
        )
    }
}

impl<H: SortHooks> crate::Sorter for SorterCore<H> {
    fn sort(&mut self, sort_input: Box<dyn SortInput>) -> Result<Box<dyn SortOutput>, String> {
        let (runs, sketch, run_gen_stats) = self.run_generation_internal(sort_input)?;
        let (merged_run, merge_stats) = self.multi_merge_internal(runs, &sketch)?;
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
    run_size: usize,
    sketch_size: usize,
    sketch_sampling_interval: usize,
    run_indexing_interval: usize,
    dir: impl AsRef<Path>,
    algorithm: RunGenerationAlgorithm,
) -> Result<(Vec<H::MergeableRun>, KLL<Vec<u8>>, RunGenerationStats), String> {
    let dir = dir.as_ref();
    let hooks = hooks.clone();
    let worker_hooks = hooks.clone();
    let worker = move |thread_id: usize,
                       mut scanner: Scanner,
                       io_tracker: Arc<IoStatsTracker>,
                       dir: PathBuf| {
        let run_path = dir.join(format!("intermediate_{}.dat", thread_id));
        let fd = Arc::new(
            SharedFd::new_from_path(&run_path, true)
                .expect("Failed to open run file with Direct I/O"),
        );
        let run_writer = AlignedWriter::from_fd_with_tracker(fd, Some((*io_tracker).clone()))
            .expect("Failed to create run writer");
        let mut sink = worker_hooks.create_sink(
            run_writer,
            run_indexing_interval,
            sketch_size,
            sketch_sampling_interval,
        );
        match algorithm {
            RunGenerationAlgorithm::ReplacementSelection => {
                let thread_start = Instant::now();
                let _ = run_replacement_selection(scanner, &mut sink, run_size);
                let (local_runs, sketch) = sink.finalize();
                let total_time_ms = thread_start.elapsed().as_millis();
                println!(
                    "Run gen thread {} ({:?}) {} ms, {} runs generated",
                    thread_id,
                    RunGenerationAlgorithm::ReplacementSelection,
                    total_time_ms,
                    local_runs.len()
                );
                RunGenerationThreadResult {
                    runs: local_runs,
                    sketch,
                    load_ms: 0,
                    sort_ms: total_time_ms,
                    store_ms: 0,
                }
            }
            RunGenerationAlgorithm::LoadSortStore => {
                let mut buffer = SortBuffer::new(run_size);
                let mut pending_record: Option<(Vec<u8>, Vec<u8>)> = None;
                let mut load_duration = Duration::default();
                let mut sort_duration = Duration::default();
                let mut store_duration = Duration::default();

                loop {
                    let load_start = Instant::now();
                    fill_sort_buffer(&mut buffer, scanner.as_mut(), &mut pending_record, run_size);
                    load_duration += load_start.elapsed();

                    if buffer.is_empty() {
                        break;
                    }

                    let sort_start = Instant::now();
                    let sorted_iter = buffer.sorted_iter();
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

                let (local_runs, sketch) = sink.finalize();
                let load_ms = load_duration.as_millis();
                let sort_ms = sort_duration.as_millis();
                let store_ms = store_duration.as_millis();
                println!(
                    "Run gen thread {} ({:?}) {} ms, {} runs generated",
                    thread_id,
                    RunGenerationAlgorithm::LoadSortStore,
                    load_ms + sort_ms + store_ms,
                    local_runs.len()
                );
                RunGenerationThreadResult {
                    runs: local_runs,
                    sketch,
                    load_ms,
                    sort_ms,
                    store_ms,
                }
            }
        }
    };

    execute_run_generation(sort_input, num_threads, sketch_size, dir, worker)
}

fn multi_merge_with_hooks<H: SortHooks>(
    hooks: &H,
    mut runs: Vec<H::MergeableRun>,
    fanin: usize,
    num_threads: usize,
    sketch: &KLL<Vec<u8>>,
    imbalance_factor: f64,
    dir: &Path,
) -> Result<(H::MergeableRun, Vec<MergeStats>), String> {
    let dir = dir.as_ref();
    let mut per_merge_stats = Vec::new();

    if runs.is_empty() {
        let merge_stats = MergeStats {
            output_runs: 0,
            merge_entry_num: vec![],
            time_ms: 0,
            io_stats: None,
            per_thread_times_ms: vec![],
        };
        return Ok((hooks.dummy_run(), vec![merge_stats]));
    }

    if runs.len() == 1 {
        let run = runs.into_iter().next().unwrap();
        let merge_stats = MergeStats {
            output_runs: 1,
            merge_entry_num: vec![run.total_entries() as u64],
            time_ms: 0,
            io_stats: None,
            per_thread_times_ms: vec![],
        };
        return Ok((run, vec![merge_stats]));
    }

    if fanin >= runs.len() {
        let (merged_run, stats) =
            merge_once_with_hooks(hooks, runs, num_threads, sketch, imbalance_factor, dir)?;
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
        let (merged_run, stats) =
            merge_once_with_hooks(hooks, batch, num_threads, sketch, imbalance_factor, dir)?;
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
    sketch: &KLL<Vec<u8>>,
    imbalance_factor: f64,
    dir: &Path,
) -> Result<(H::MergeableRun, MergeStats), String> {
    if output_runs.is_empty() {
        let merge_stats = MergeStats {
            output_runs: 0,
            merge_entry_num: vec![],
            time_ms: 0,
            io_stats: None,
            per_thread_times_ms: vec![],
        };
        return Ok((hooks.dummy_run(), merge_stats));
    }

    if output_runs.len() == 1 {
        let entry_count = output_runs[0].total_entries() as u64;
        let merge_stats = MergeStats {
            output_runs: 1,
            merge_entry_num: vec![entry_count],
            time_ms: 0,
            io_stats: None,
            per_thread_times_ms: vec![],
        };
        let run = output_runs.into_iter().next().unwrap();
        return Ok((run, merge_stats));
    }

    let merge_start = Instant::now();
    let merge_io_tracker = Arc::new(IoStatsTracker::new());
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

    let runs_arc = Arc::new(output_runs);
    let mut merge_handles = vec![];

    let k = merge_threads as f64;
    let r = if imbalance_factor <= 0.0 {
        1.0
    } else {
        imbalance_factor
    };
    let each = (1.0 - r * (1.0 / k)) / (k - 1.0);

    for thread_id in 0..merge_threads {
        let runs = Arc::clone(&runs_arc);
        let dir = dir.to_path_buf();
        let io_tracker = Arc::clone(&merge_io_tracker);
        let cdf = Arc::clone(&cdf);
        let hooks = hooks.clone();

        let handle = thread::spawn(move || {
            let tid = thread_id as f64;
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

            hooks.merge_range(runs, thread_id, lower_bound, upper_bound, &dir, io_tracker)
        });

        merge_handles.push(handle);
    }

    let mut output_runs = Vec::new();
    let mut per_thread_times_ms = Vec::new();

    for handle in merge_handles {
        let result = handle
            .join()
            .map_err(|_| "Merge worker panicked".to_string())?;
        let (run, thread_time) = result?;
        output_runs.push(run);
        per_thread_times_ms.push(thread_time);
    }

    let merge_time_ms = merge_start.elapsed().as_millis();
    let merge_io_stats = Some(merge_io_tracker.get_detailed_stats());

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

    Ok((hooks.combine_runs(output_runs), merge_stats))
}

fn append_record_to_buffer(
    buffer: &mut SortBuffer,
    key: Vec<u8>,
    value: Vec<u8>,
    run_size: usize,
) -> Result<(), (Vec<u8>, Vec<u8>)> {
    if buffer.has_space(&key, &value) {
        let appended = buffer.append(key, value);
        debug_assert!(appended, "SortBuffer reported space but append failed");
        Ok(())
    } else {
        if buffer.is_empty() {
            let entry_size = key.len() + value.len() + ENTRY_METADATA_SIZE;
            panic!(
                "Record of size {} bytes exceeds load-sort-store run size limit {} bytes",
                entry_size, run_size
            );
        }
        Err((key, value))
    }
}

fn fill_sort_buffer(
    buffer: &mut SortBuffer,
    scanner: &mut dyn Iterator<Item = (Vec<u8>, Vec<u8>)>,
    pending_record: &mut Option<(Vec<u8>, Vec<u8>)>,
    run_size: usize,
) {
    if let Some((key, value)) = pending_record.take() {
        append_record_to_buffer(buffer, key, value, run_size)
            .expect("pending record must fit inside an empty buffer");
    }

    while let Some((key, value)) = scanner.next() {
        match append_record_to_buffer(buffer, key, value, run_size) {
            Ok(()) => {}
            Err(record) => {
                *pending_record = Some(record);
                break;
            }
        }
    }
}
