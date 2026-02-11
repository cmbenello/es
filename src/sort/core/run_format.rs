use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::file::SharedFd;
use crate::diskio::io_stats::IoStatsTracker;
use crate::ovc::offset_value_coding_32::OVCU32;
use crate::sort::core::engine::SortHooks;
use crate::sort::core::engine::{RunSummary, Scanner};
use crate::sort::run_sink::RunSink;
use crate::{SortOutput, SortStats};

// Global run id generator
pub(crate) static RUN_ID_COUNTER: AtomicU32 = AtomicU32::new(0);

pub trait RunFormat: Clone + Send + Sync + 'static {
    type Run: RunSummary + Send + Sync + 'static;
    type Record: Send + 'static;
    type AppendState: Default + Send + 'static;

    fn new_run(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
    ) -> Result<Self::Run, String>;

    fn create_merge_run(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
    ) -> Result<Self::Run, String> {
        Self::new_run(writer, indexing_interval)
    }

    fn create_merge_run_with_id(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
        run_id: u32,
    ) -> Result<Self::Run, String>;

    fn finalize_run(run: &mut Self::Run) -> AlignedWriter;

    fn append_from_kv(state: &mut Self::AppendState, run: &mut Self::Run, key: &[u8], value: &[u8]);

    fn append_record(state: &mut Self::AppendState, run: &mut Self::Run, record: Self::Record) {
        let (key, value) = Self::record_into_kv(record);
        Self::append_from_kv(state, run, &key, &value);
    }

    /// Append a record that carries an OVC delta; default ignores the delta.
    fn append_with_ovc(
        state: &mut Self::AppendState,
        run: &mut Self::Run,
        ovc: OVCU32,
        key: &[u8],
        value: &[u8],
    ) {
        let _ = ovc;
        Self::append_from_kv(state, run, key, value);
    }

    fn scan_range(
        run: &Self::Run,
        lower_inc: Option<(&[u8], u32, usize)>, // KeyRunIdOffsetBound: (key, run_id, offset). Set run_id and offset to 0 if not applicable.
        upper_exc: Option<(&[u8], u32, usize)>, // KeyRunIdOffsetBound: (key, run_id, offset). Set run_id and offset to 0 if not applicable.
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = Self::Record> + Send>;

    fn merge_iterators(
        iterators: Vec<Box<dyn Iterator<Item = Self::Record> + Send>>,
    ) -> Box<dyn Iterator<Item = Self::Record> + Send>;

    fn run_id(run: &Self::Run) -> u32;

    fn sparse_index<'a>(run: &'a Self::Run) -> &'a [IndexEntry];

    fn start_key<'a>(run: &'a Self::Run) -> Option<(&'a [u8], u32, usize)>;

    fn record_key<'a>(record: &'a Self::Record) -> &'a [u8];
    fn record_value<'a>(record: &'a Self::Record) -> &'a [u8];
    fn record_into_kv(record: Self::Record) -> (Vec<u8>, Vec<u8>);

    /// Run replacement selection for this format
    fn run_replacement_selection<S: RunSink<MergeableRun = MergeableRun<Self>>>(
        scanner: Scanner,
        sink: &mut S,
        run_size: usize,
    ) -> crate::replacement_selection::ReplacementSelectionStats;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexingInterval {
    Records {
        stride: usize,
        budget_bytes: Option<usize>,
    },
    Bytes {
        stride: usize,
        budget_bytes: Option<usize>,
    },
}

impl IndexingInterval {
    pub fn records(interval: usize) -> Self {
        IndexingInterval::Records {
            stride: interval.max(1),
            budget_bytes: None,
        }
    }

    pub fn records_with_budget(interval: usize, budget_bytes: usize) -> Self {
        IndexingInterval::Records {
            stride: interval.max(1),
            budget_bytes: Some(budget_bytes.max(1)),
        }
    }

    pub fn bytes(stride: usize) -> Self {
        IndexingInterval::Bytes {
            stride: stride.max(1),
            budget_bytes: None,
        }
    }

    pub fn bytes_with_budget(stride: usize, budget_bytes: usize) -> Self {
        IndexingInterval::Bytes {
            stride: stride.max(1),
            budget_bytes: Some(budget_bytes.max(1)),
        }
    }

    pub fn value(self) -> usize {
        match self {
            IndexingInterval::Records { stride, .. } => stride,
            IndexingInterval::Bytes { stride, .. } => stride,
        }
    }

    pub fn budget_bytes(self) -> Option<usize> {
        match self {
            IndexingInterval::Records { budget_bytes, .. }
            | IndexingInterval::Bytes { budget_bytes, .. } => budget_bytes,
        }
    }

    pub fn with_value(self, interval: usize) -> Self {
        match self {
            IndexingInterval::Records { budget_bytes, .. } => IndexingInterval::Records {
                stride: interval.max(1),
                budget_bytes,
            },
            IndexingInterval::Bytes { budget_bytes, .. } => IndexingInterval::Bytes {
                stride: interval.max(1),
                budget_bytes,
            },
        }
    }
}

impl std::fmt::Display for IndexingInterval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexingInterval::Records { stride, .. } => write!(f, "{} records", stride),
            IndexingInterval::Bytes { stride, .. } => write!(f, "{} bytes", stride),
        }
    }
}
/// Sparse index entry inside a single run.
///
/// `file_offset` is relative to the start of that run's byte range.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    pub key: Vec<u8>,
    pub file_offset: usize,
}

#[derive(Clone, Copy)]
pub struct SparseIndexRef<'a> {
    pub run_id: u32,
    pub entry: &'a IndexEntry,
}

impl<'a> SparseIndexRef<'a> {
    pub fn key(&self) -> &'a [u8] {
        self.entry.key.as_slice()
    }

    pub fn file_offset(&self) -> usize {
        self.entry.file_offset
    }

    pub fn as_key_run_offset(&self) -> (&'a [u8], u32, usize) {
        (
            self.entry.key.as_slice(),
            self.run_id,
            self.entry.file_offset,
        )
    }
}

struct SparseIndexSegment<'a> {
    run_id: u32,
    entries: &'a [IndexEntry],
    total_bytes: usize,
    base_bytes: usize,
}

pub struct MultiSparseIndexes<'a> {
    segments: Vec<SparseIndexSegment<'a>>,
    segment_ends: Vec<usize>,
    total_len: usize,
    total_bytes: usize,
}

impl<'a> MultiSparseIndexes<'a> {
    /// Creates a logical concatenation view of per-run sparse indexes.
    ///
    /// No `IndexEntry` is copied. We only store segment metadata:
    ///
    /// ```text
    /// input segments
    ///   seg0 (run_id=10): [ (a,0), (d,128), (g,256) ]
    ///   seg1 (run_id=11): [ (h,0), (k,160) ]
    ///
    /// logical global array
    ///   idx: 0       1        2        3       4
    ///        (a,10,0)(d,10,128)(g,10,256)(h,11,0)(k,11,160)
    ///
    /// segment_ends = [3, 5]
    /// total_len    = 5
    /// ```
    ///
    /// `global_offset_at(i)` maps a logical index to cumulative bytes across
    /// segments using `base_bytes + entry.file_offset`.
    fn from_segments(mut segments: Vec<SparseIndexSegment<'a>>) -> Self {
        segments.retain(|segment| !segment.entries.is_empty());
        let mut segment_ends = Vec::with_capacity(segments.len());
        let mut total_len = 0;
        let mut total_bytes = 0;
        for segment in &mut segments {
            segment.base_bytes = total_bytes;
            total_len += segment.entries.len();
            segment_ends.push(total_len);
            total_bytes += segment.total_bytes;
        }

        Self {
            segments,
            segment_ends,
            total_len,
            total_bytes,
        }
    }

    pub fn len(&self) -> usize {
        self.total_len
    }

    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    fn locate(&self, index: usize) -> Option<(usize, usize)> {
        if index >= self.total_len {
            return None;
        }

        let needle = index + 1;
        let segment_index = match self.segment_ends.binary_search(&needle) {
            Ok(pos) | Err(pos) => pos,
        };
        let segment_start = if segment_index == 0 {
            0
        } else {
            self.segment_ends[segment_index - 1]
        };
        let entry_index = index - segment_start;
        Some((segment_index, entry_index))
    }

    pub fn get(&self, index: usize) -> Option<SparseIndexRef<'a>> {
        let (segment_index, entry_index) = self.locate(index)?;
        let segment = &self.segments[segment_index];
        Some(SparseIndexRef {
            run_id: segment.run_id,
            entry: &segment.entries[entry_index],
        })
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn global_offset_at(&self, index: usize) -> Option<usize> {
        let (segment_index, entry_index) = self.locate(index)?;
        let segment = &self.segments[segment_index];
        Some(segment.base_bytes + segment.entries[entry_index].file_offset)
    }

    pub fn iter(&'a self) -> impl Iterator<Item = SparseIndexRef<'a>> + 'a {
        self.segments.iter().flat_map(|segment| {
            let run_id = segment.run_id;
            segment
                .entries
                .iter()
                .map(move |entry| SparseIndexRef { run_id, entry })
        })
    }

    pub fn binary_search_by<F>(&self, mut cmp: F) -> Result<usize, usize>
    where
        F: FnMut(SparseIndexRef<'a>) -> std::cmp::Ordering,
    {
        let mut left = 0;
        let mut right = self.total_len;
        while left < right {
            let mid = left + (right - left) / 2;
            let entry = self.get(mid).expect("index is within bounds");
            match cmp(entry) {
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Err(left)
    }

    pub fn lower_bound_range(
        &self,
        key: &[u8],
        run_id: u32,
        offset: usize,
        mut lo: usize,
        mut hi: usize,
    ) -> usize {
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let entry = self.get(mid).expect("index is within bounds");
            if matches!(
                cmp_key_run_offset(entry.as_key_run_offset(), (key, run_id, offset)),
                std::cmp::Ordering::Less
            ) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    pub fn upper_bound_range(
        &self,
        key: &[u8],
        run_id: u32,
        offset: usize,
        mut lo: usize,
        mut hi: usize,
    ) -> usize {
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let entry = self.get(mid).expect("index is within bounds");
            if matches!(
                cmp_key_run_offset(entry.as_key_run_offset(), (key, run_id, offset)),
                std::cmp::Ordering::Less | std::cmp::Ordering::Equal
            ) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    pub fn upper_bound(&self, key: &[u8], run_id: u32, offset: usize) -> usize {
        self.upper_bound_range(key, run_id, offset, 0, self.total_len)
    }

    pub fn lower_bound(&self, key: &[u8], run_id: u32, offset: usize) -> usize {
        match self.binary_search_by(|entry| {
            cmp_key_run_offset(entry.as_key_run_offset(), (key, run_id, offset))
        }) {
            Ok(pos) | Err(pos) => pos,
        }
    }
}

pub enum MergeableRun<F: RunFormat> {
    Single(F::Run),
    RangePartitioned(Vec<F::Run>),
    StatsOnly { entries: usize, bytes: usize },
}

impl<F: RunFormat> MergeableRun<F> {
    pub fn scan_range_with_io_tracker(
        &self,
        lower_inc: Option<(&[u8], u32, usize)>, // (key, run_id, offset)
        upper_exc: Option<(&[u8], u32, usize)>, // (key, run_id, offset)
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = F::Record> + Send> {
        match self {
            MergeableRun::Single(run) => F::scan_range(run, lower_inc, upper_exc, io_tracker),
            MergeableRun::RangePartitioned(runs) => {
                let non_empty_runs: Vec<&F::Run> =
                    runs.iter().filter(|r| F::start_key(r).is_some()).collect();

                if non_empty_runs.is_empty() {
                    return Box::new(std::iter::empty());
                }

                let start_partition = if lower_inc.is_none() {
                    0
                } else {
                    let mut ok = -1;
                    let mut ng = non_empty_runs.len() as isize;
                    while (ng - ok).abs() > 1 {
                        let mid = (ok + ng) / 2;
                        if cmp_key_run_offset(
                            F::start_key(non_empty_runs[mid as usize]).unwrap(),
                            lower_inc.unwrap(),
                        ) == std::cmp::Ordering::Less
                        {
                            ok = mid;
                        } else {
                            ng = mid;
                        }
                    }
                    if ok == -1 { 0 } else { ok as usize }
                };

                let end_partition = if upper_exc.is_none() {
                    non_empty_runs.len()
                } else {
                    let mut ok = non_empty_runs.len() as isize;
                    let mut ng = -1;
                    while (ng - ok).abs() > 1 {
                        let mid = (ok + ng) / 2;
                        if matches!(
                            cmp_key_run_offset(
                                upper_exc.unwrap(),
                                F::start_key(non_empty_runs[mid as usize]).unwrap(),
                            ),
                            std::cmp::Ordering::Less | std::cmp::Ordering::Equal
                        ) {
                            ok = mid;
                        } else {
                            ng = mid;
                        }
                    }
                    ok as usize
                };

                let mut iterators = Vec::with_capacity(end_partition - start_partition);
                for run in non_empty_runs[start_partition..end_partition].iter() {
                    iterators.push(F::scan_range(run, lower_inc, upper_exc, io_tracker.clone()));
                }

                Box::new(RangePartitionedIterator::<F>::new(iterators))
            }
            MergeableRun::StatsOnly { .. } => Box::new(std::iter::empty()),
        }
    }

    pub fn total_entries(&self) -> usize {
        match self {
            MergeableRun::Single(run) => run.total_entries(),
            MergeableRun::RangePartitioned(runs) => {
                runs.iter().map(|run| run.total_entries()).sum()
            }
            MergeableRun::StatsOnly { entries, .. } => *entries,
        }
    }

    pub fn sparse_indexes(&self) -> MultiSparseIndexes<'_> {
        match self {
            MergeableRun::Single(run) => {
                MultiSparseIndexes::from_segments(vec![SparseIndexSegment {
                    run_id: F::run_id(run),
                    entries: F::sparse_index(run),
                    total_bytes: run.total_bytes(),
                    base_bytes: 0,
                }])
            }
            MergeableRun::RangePartitioned(runs) => {
                let segments = runs
                    .iter()
                    .map(|run| SparseIndexSegment {
                        run_id: F::run_id(run),
                        entries: F::sparse_index(run),
                        total_bytes: run.total_bytes(),
                        base_bytes: 0,
                    })
                    .collect();
                MultiSparseIndexes::from_segments(segments)
            }
            MergeableRun::StatsOnly { .. } => MultiSparseIndexes::from_segments(Vec::new()),
        }
    }

    pub fn total_bytes(&self) -> usize {
        match self {
            MergeableRun::Single(run) => run.total_bytes(),
            MergeableRun::RangePartitioned(runs) => runs.iter().map(|run| run.total_bytes()).sum(),
            MergeableRun::StatsOnly { bytes, .. } => *bytes,
        }
    }

    pub fn into_runs(self) -> Vec<F::Run> {
        match self {
            MergeableRun::Single(run) => vec![run],
            MergeableRun::RangePartitioned(runs) => runs,
            MergeableRun::StatsOnly { .. } => vec![],
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
    run_indexing_interval: IndexingInterval,
    current_run: Option<F::Run>,
    append_state: Option<F::AppendState>,
    runs: Vec<MergeableRun<F>>,
}

impl<F: RunFormat> RunWriterSink<F> {
    pub fn new(writer: AlignedWriter, run_indexing_interval: IndexingInterval) -> Self {
        Self {
            run_writer: Some(writer),
            run_indexing_interval,
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

    fn into_parts(mut self) -> Vec<MergeableRun<F>> {
        self.finalize_active_run();
        self.run_writer.take();
        self.runs
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

        if let Some(state) = &mut self.append_state {
            F::append_from_kv(state, run, key, value);
        } else {
            panic!("append state must exist while run is active");
        }
    }

    fn push_record_with_ovc(&mut self, ovc: OVCU32, key: &[u8], value: &[u8]) {
        let run = self
            .current_run
            .as_mut()
            .expect("run must be started before pushing records");

        if let Some(state) = &mut self.append_state {
            F::append_with_ovc(state, run, ovc, key, value);
        } else {
            panic!("append state must exist while run is active");
        }
    }

    fn finish_run(&mut self) {
        self.finalize_active_run();
    }

    fn finalize(self) -> Vec<Self::MergeableRun> {
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
                .scan_range_with_io_tracker(None, None, None)
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
        run_indexing_interval: IndexingInterval,
    ) -> Self::Sink {
        RunWriterSink::new(writer, run_indexing_interval)
    }

    fn dummy_run(&self) -> Self::MergeableRun {
        MergeableRun::RangePartitioned(vec![])
    }

    fn combine_runs(&self, runs: Vec<Self::MergeableRun>) -> Self::MergeableRun {
        let mut combined = Vec::new();
        let mut stats_entries = 0usize;
        let mut stats_bytes = 0usize;
        let mut has_stats = false;
        for run in runs {
            match run {
                MergeableRun::StatsOnly { entries, bytes } => {
                    has_stats = true;
                    stats_entries = stats_entries.saturating_add(entries);
                    stats_bytes = stats_bytes.saturating_add(bytes);
                }
                other => combined.extend(other.into_runs()),
            }
        }
        if has_stats {
            debug_assert!(
                combined.is_empty(),
                "stats-only runs should not be mixed with materialized runs"
            );
            return MergeableRun::StatsOnly {
                entries: stats_entries,
                bytes: stats_bytes,
            };
        }
        MergeableRun::RangePartitioned(combined)
    }

    fn merge_range(
        &self,
        runs: Arc<Vec<Self::MergeableRun>>,
        thread_id: usize,
        run_id: u32,
        run_indexing_interval: IndexingInterval,
        lower_bound: Option<(&[u8], u32, usize)>,
        upper_bound: Option<(&[u8], u32, usize)>,
        dir: &Path,
        io_tracker: Arc<IoStatsTracker>,
        discard_output: bool,
    ) -> Result<(Self::MergeableRun, u128), String> {
        let thread_start = Instant::now();
        let iterators: Vec<_> = runs
            .iter()
            .map(|run| {
                run.scan_range_with_io_tracker(
                    lower_bound,
                    upper_bound,
                    Some((*io_tracker).clone()),
                )
            })
            .collect();

        let merged_iter = F::merge_iterators(iterators);

        if discard_output {
            // Discard mode: iterate through all records but don't write
            let mut entries = 0usize;
            for record in merged_iter {
                entries = entries.saturating_add(1);
                black_box(record);
            }

            let thread_time_ms = thread_start.elapsed().as_millis();
            Ok((
                MergeableRun::StatsOnly { entries, bytes: 0 },
                thread_time_ms,
            ))
        } else {
            // Normal mode: write to file
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let ts = format!("{}{:09}", now.as_secs(), now.subsec_nanos());
            let run_path = dir.join(format!("merge_output_{}_{}.dat", thread_id, ts));
            let fd = Arc::new(
                SharedFd::new_from_path(&run_path, true)
                    .map_err(|e| format!("Failed to open merge output file: {}", e))?,
            );
            let writer = AlignedWriter::from_fd_with_tracker(fd, Some((*io_tracker).clone()))
                .map_err(|e| format!("Failed to create run writer: {}", e))?;
            let mut output_run = F::create_merge_run_with_id(writer, run_indexing_interval, run_id)
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
    }

    fn sparse_indexes<'a>(&self, run: &'a Self::MergeableRun) -> MultiSparseIndexes<'a> {
        run.sparse_indexes()
    }

    fn into_output(&self, run: Self::MergeableRun, stats: SortStats) -> Box<dyn SortOutput> {
        Box::new(RunsOutput::<F> { run, stats })
    }

    fn run_replacement_selection(
        &self,
        scanner: Scanner,
        sink: &mut Self::Sink,
        run_size: usize,
    ) -> crate::replacement_selection::ReplacementSelectionStats {
        F::run_replacement_selection(scanner, sink, run_size)
    }
}

#[derive(Debug, Clone)]
pub struct KeyRunIdOffsetBound {
    pub key: Vec<u8>,
    pub run_id: u32,
    pub offset: usize,
}

impl KeyRunIdOffsetBound {
    pub fn from_components((key, run_id, offset): (&[u8], u32, usize)) -> Self {
        Self {
            key: key.to_vec(),
            run_id,
            offset,
        }
    }

    #[inline]
    pub fn lt(&self, key: &[u8], run_id: u32, offset: usize) -> bool {
        cmp_key_run_offset((key, run_id, offset), (&self.key, self.run_id, self.offset))
            == std::cmp::Ordering::Less
    }

    #[inline]
    pub fn ge(&self, key: &[u8], run_id: u32, offset: usize) -> bool {
        matches!(
            cmp_key_run_offset((key, run_id, offset), (&self.key, self.run_id, self.offset)),
            std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
        )
    }
}

impl From<&[u8]> for KeyRunIdOffsetBound {
    fn from(key: &[u8]) -> Self {
        Self {
            key: key.to_vec(),
            run_id: 0,
            offset: 0,
        }
    }
}

pub fn cmp_key_run_offset(
    (key_a, run_id_a, offset_a): (&[u8], u32, usize),
    (key_b, run_id_b, offset_b): (&[u8], u32, usize),
) -> std::cmp::Ordering {
    match key_a.cmp(key_b) {
        std::cmp::Ordering::Equal => match run_id_a.cmp(&run_id_b) {
            std::cmp::Ordering::Equal => offset_a.cmp(&offset_b),
            ord => ord,
        },
        ord => ord,
    }
}
