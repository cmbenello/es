use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::diskio::io_stats::IoStatsTracker;
use crate::kll::Sketch;
use crate::{RunGenerationStats, RunInfo, SortInput};

pub type Scanner = Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>;

pub struct RunGenerationThreadResult<R> {
    pub runs: Vec<R>,
    pub sketch: Sketch<Vec<u8>>,
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
) -> Result<(Vec<R>, Sketch<Vec<u8>>, RunGenerationStats), String>
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
            Sketch::new(sketch_size),
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
    let mut sketch = Sketch::new(sketch_size);
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
