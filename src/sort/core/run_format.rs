use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::file::SharedFd;
use crate::diskio::io_stats::IoStatsTracker;
use crate::kll::Sketch;
use crate::sort::core::engine::RunSummary;
use crate::sort::core::engine::SortHooks;
use crate::sort::run_sink::RunSink;
use crate::{SortOutput, SortStats};

pub trait RunFormat: Clone + Send + Sync + 'static {
    type Run: RunSummary + Send + Sync + 'static;
    type Record: Send + 'static;
    type AppendState: Default + Send + 'static;

    fn new_run(writer: AlignedWriter, indexing_interval: usize) -> Result<Self::Run, String>;

    fn create_merge_run(writer: AlignedWriter) -> Result<Self::Run, String> {
        Self::new_run(writer, 1000)
    }

    fn finalize_run(run: &mut Self::Run) -> AlignedWriter;

    fn append_from_kv(state: &mut Self::AppendState, run: &mut Self::Run, key: &[u8], value: &[u8]);

    fn append_record(state: &mut Self::AppendState, run: &mut Self::Run, record: Self::Record) {
        let (key, value) = Self::record_into_kv(record);
        Self::append_from_kv(state, run, &key, &value);
    }

    fn scan_range(
        run: &Self::Run,
        lower_bound: &[u8],
        upper_bound: &[u8],
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = Self::Record> + Send>;

    fn merge_iterators(
        iterators: Vec<Box<dyn Iterator<Item = Self::Record> + Send>>,
    ) -> Box<dyn Iterator<Item = Self::Record> + Send>;

    fn start_key<'a>(run: &'a Self::Run) -> Option<&'a [u8]>;

    fn record_key<'a>(record: &'a Self::Record) -> &'a [u8];
    fn record_value<'a>(record: &'a Self::Record) -> &'a [u8];
    fn record_into_kv(record: Self::Record) -> (Vec<u8>, Vec<u8>);
}

pub enum MergeableRun<F: RunFormat> {
    Single(F::Run),
    RangePartitioned(Vec<F::Run>),
}

impl<F: RunFormat> MergeableRun<F> {
    pub fn scan_range_with_io_tracker(
        &self,
        lower_bound: &[u8],
        upper_bound: &[u8],
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = F::Record> + Send> {
        match self {
            MergeableRun::Single(run) => F::scan_range(run, lower_bound, upper_bound, io_tracker),
            MergeableRun::RangePartitioned(runs) => {
                let non_empty_runs: Vec<&F::Run> =
                    runs.iter().filter(|r| F::start_key(r).is_some()).collect();

                if non_empty_runs.is_empty() {
                    return Box::new(std::iter::empty());
                }

                let start_partition = if lower_bound.is_empty() {
                    0
                } else {
                    let mut ok = -1;
                    let mut ng = non_empty_runs.len() as isize;
                    while (ng - ok).abs() > 1 {
                        let mid = (ok + ng) / 2;
                        if F::start_key(non_empty_runs[mid as usize]).unwrap() < lower_bound {
                            ok = mid;
                        } else {
                            ng = mid;
                        }
                    }
                    if ok == -1 { 0 } else { ok as usize }
                };

                let end_partition = if upper_bound.is_empty() {
                    non_empty_runs.len()
                } else {
                    let mut ok = non_empty_runs.len() as isize;
                    let mut ng = -1;
                    while (ng - ok).abs() > 1 {
                        let mid = (ok + ng) / 2;
                        if upper_bound <= F::start_key(non_empty_runs[mid as usize]).unwrap() {
                            ok = mid;
                        } else {
                            ng = mid;
                        }
                    }
                    ok as usize
                };

                let mut iterators = Vec::with_capacity(end_partition - start_partition);
                for run in non_empty_runs[start_partition..end_partition].iter() {
                    iterators.push(F::scan_range(
                        run,
                        lower_bound,
                        upper_bound,
                        io_tracker.clone(),
                    ));
                }

                Box::new(RangePartitionedIterator::<F>::new(iterators))
            }
        }
    }

    pub fn total_entries(&self) -> usize {
        match self {
            MergeableRun::Single(run) => run.total_entries(),
            MergeableRun::RangePartitioned(runs) => {
                runs.iter().map(|run| run.total_entries()).sum()
            }
        }
    }

    pub fn total_bytes(&self) -> usize {
        match self {
            MergeableRun::Single(run) => run.total_bytes(),
            MergeableRun::RangePartitioned(runs) => runs.iter().map(|run| run.total_bytes()).sum(),
        }
    }

    pub fn into_runs(self) -> Vec<F::Run> {
        match self {
            MergeableRun::Single(run) => vec![run],
            MergeableRun::RangePartitioned(runs) => runs,
        }
    }
}

impl<F: RunFormat> RunSummary for MergeableRun<F> {
    fn total_entries(&self) -> usize {
        self.total_entries()
    }

    fn total_bytes(&self) -> usize {
        self.total_bytes()
    }
}

struct RangePartitionedIterator<F: RunFormat> {
    iterators: Vec<Box<dyn Iterator<Item = F::Record> + Send>>,
    current: usize,
}

impl<F: RunFormat> RangePartitionedIterator<F> {
    fn new(iterators: Vec<Box<dyn Iterator<Item = F::Record> + Send>>) -> Self {
        Self {
            iterators,
            current: 0,
        }
    }
}

impl<F: RunFormat> Iterator for RangePartitionedIterator<F> {
    type Item = F::Record;

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

pub struct RunWriterSink<F: RunFormat> {
    run_writer: Option<AlignedWriter>,
    run_indexing_interval: usize,
    sketch: Sketch<Vec<u8>>,
    sketch_sampling_interval: usize,
    records_seen: u64,
    current_run: Option<F::Run>,
    append_state: Option<F::AppendState>,
    runs: Vec<MergeableRun<F>>,
}

impl<F: RunFormat> RunWriterSink<F> {
    pub fn new(
        writer: AlignedWriter,
        run_indexing_interval: usize,
        sketch_size: usize,
        sketch_sampling_interval: usize,
    ) -> Self {
        Self {
            run_writer: Some(writer),
            run_indexing_interval,
            sketch: Sketch::new(sketch_size),
            sketch_sampling_interval,
            records_seen: 0,
            current_run: None,
            append_state: None,
            runs: Vec::new(),
        }
    }

    fn finalize_active_run(&mut self) {
        if let Some(mut run) = self.current_run.take() {
            let writer = F::finalize_run(&mut run);
            self.run_writer = Some(writer);
            self.runs.push(MergeableRun::Single(run));
        }
        self.append_state = None;
    }

    fn into_parts(mut self) -> (Vec<MergeableRun<F>>, Sketch<Vec<u8>>) {
        self.finalize_active_run();
        self.run_writer.take();
        (self.runs, self.sketch)
    }
}

impl<F: RunFormat> RunSink for RunWriterSink<F> {
    type MergeableRun = MergeableRun<F>;

    fn start_run(&mut self) {
        if self.current_run.is_some() {
            return;
        }
        let writer = self
            .run_writer
            .take()
            .expect("Run writer should be available when starting a run");
        let run = F::new_run(writer, self.run_indexing_interval).expect("Failed to create run");
        self.current_run = Some(run);
        self.append_state = Some(F::AppendState::default());
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

        if let Some(state) = &mut self.append_state {
            F::append_from_kv(state, run, key, value);
        } else {
            panic!("append state must exist while run is active");
        }
    }

    fn finish_run(&mut self) {
        self.finalize_active_run();
    }

    fn finalize(self) -> (Vec<Self::MergeableRun>, Sketch<Vec<u8>>) {
        self.into_parts()
    }
}

pub struct RunsOutput<F: RunFormat> {
    pub run: MergeableRun<F>,
    pub stats: SortStats,
}

impl<F: RunFormat> SortOutput for RunsOutput<F> {
    fn iter(&self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>> {
        Box::new(
            self.run
                .scan_range_with_io_tracker(&[], &[], None)
                .map(F::record_into_kv),
        )
    }

    fn stats(&self) -> SortStats {
        self.stats.clone()
    }
}

#[derive(Clone, Copy, Default)]
pub struct FormatSortHooks<F: RunFormat>(std::marker::PhantomData<F>);

impl<F: RunFormat> SortHooks for FormatSortHooks<F> {
    type MergeableRun = MergeableRun<F>;
    type Sink = RunWriterSink<F>;

    fn create_sink(
        &self,
        writer: AlignedWriter,
        run_indexing_interval: usize,
        sketch_size: usize,
        sketch_sampling_interval: usize,
    ) -> Self::Sink {
        RunWriterSink::new(
            writer,
            run_indexing_interval,
            sketch_size,
            sketch_sampling_interval,
        )
    }

    fn dummy_run(&self) -> Self::MergeableRun {
        MergeableRun::RangePartitioned(vec![])
    }

    fn combine_runs(&self, runs: Vec<Self::MergeableRun>) -> Self::MergeableRun {
        let mut combined = Vec::new();
        for run in runs {
            combined.extend(run.into_runs());
        }
        MergeableRun::RangePartitioned(combined)
    }

    fn merge_range(
        &self,
        runs: Arc<Vec<Self::MergeableRun>>,
        thread_id: usize,
        lower_bound: Vec<u8>,
        upper_bound: Vec<u8>,
        dir: &Path,
        io_tracker: Arc<IoStatsTracker>,
    ) -> Result<(Self::MergeableRun, u128), String> {
        let thread_start = Instant::now();
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

        let merged_iter = F::merge_iterators(iterators);

        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let ts = format!("{}{:09}", now.as_secs(), now.subsec_nanos());
        let run_path = dir.join(format!("merge_output_{}_{}.dat", thread_id, ts));
        let fd = Arc::new(
            SharedFd::new_from_path(&run_path)
                .map_err(|e| format!("Failed to open merge output file: {}", e))?,
        );
        let writer = AlignedWriter::from_fd_with_tracker(fd, Some((*io_tracker).clone()))
            .map_err(|e| format!("Failed to create run writer: {}", e))?;
        let mut output_run = F::create_merge_run(writer)
            .map_err(|e| format!("Failed to create merge output run: {}", e))?;
        let mut append_state = F::AppendState::default();

        for record in merged_iter {
            F::append_record(&mut append_state, &mut output_run, record);
        }

        let writer = F::finalize_run(&mut output_run);
        drop(writer);

        let thread_time_ms = thread_start.elapsed().as_millis();
        Ok((MergeableRun::Single(output_run), thread_time_ms))
    }

    fn into_output(&self, run: Self::MergeableRun, stats: SortStats) -> Box<dyn SortOutput> {
        Box::new(RunsOutput::<F> { run, stats })
    }
}
