use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::io_stats::IoStatsTracker;
use crate::sort::core::engine::SorterCore;
use crate::sort::core::run_format::{
    FormatSortHooks, IndexEntry, IndexingInterval, MergeableRun as GenericMergeableRun, RunFormat,
    RunsOutput as GenericRunsOutput,
};
use crate::sort::plain::merge::MergeIterator;
use crate::sort::plain::run::Run;

pub type PlainSortHooks = FormatSortHooks<PlainRunFormat>;
pub type ExternalSorter = SorterCore<PlainSortHooks>;
pub type MergeableRun = GenericMergeableRun<PlainRunFormat>;
pub type RunsOutput = GenericRunsOutput<PlainRunFormat>;

#[derive(Clone, Copy, Default)]
pub struct PlainRunFormat;
const DEFAULT_RUN_INDEXING_INTERVAL: usize = 100;

impl RunFormat for PlainRunFormat {
    type Run = Run;
    type Record = (Vec<u8>, Vec<u8>);
    type AppendState = ();

    fn new_run(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
    ) -> Result<Self::Run, String> {
        Run::from_writer_with_indexing_interval(writer, indexing_interval)
    }

    fn create_merge_run_with_id(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
        run_id: u32,
    ) -> Result<Self::Run, String> {
        Run::from_writer_with_indexing_interval_and_id(writer, indexing_interval, run_id)
    }

    fn finalize_run(run: &mut Self::Run) -> AlignedWriter {
        run.finalize_write()
    }

    fn append_from_kv(
        _state: &mut Self::AppendState,
        run: &mut Self::Run,
        key: &[u8],
        value: &[u8],
    ) {
        run.append(key, value);
    }

    fn scan_range(
        run: &Self::Run,
        lower_inc: Option<(&[u8], u32, usize)>,
        upper_exc: Option<(&[u8], u32, usize)>,
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = Self::Record> + Send> {
        run.scan_range_with_io_tracker(lower_inc, upper_exc, io_tracker)
    }

    fn merge_iterators(
        iterators: Vec<Box<dyn Iterator<Item = Self::Record> + Send>>,
    ) -> Box<dyn Iterator<Item = Self::Record> + Send> {
        Box::new(MergeIterator::new(iterators))
    }

    fn run_id(run: &Self::Run) -> u32 {
        run.run_id()
    }

    fn sparse_index<'a>(run: &'a Self::Run) -> &'a [IndexEntry] {
        run.sparse_index()
    }

    fn start_key<'a>(run: &'a Self::Run) -> Option<(&'a [u8], u32, usize)> {
        run.start_key().map(|key| (key, run.run_id(), 0))
    }

    fn record_key<'a>(record: &'a Self::Record) -> &'a [u8] {
        &record.0
    }

    fn record_value<'a>(record: &'a Self::Record) -> &'a [u8] {
        &record.1
    }

    fn record_into_kv(record: Self::Record) -> (Vec<u8>, Vec<u8>) {
        record
    }

    fn run_replacement_selection<
        S: crate::sort::run_sink::RunSink<
                MergeableRun = crate::sort::core::run_format::MergeableRun<Self>,
            >,
    >(
        scanner: crate::sort::core::engine::Scanner,
        sink: &mut S,
        run_gen_mem: usize,
    ) -> crate::replacement_selection::ReplacementSelectionStats {
        crate::replacement_selection::run_replacement_selection_mm(
            scanner,
            sink,
            run_gen_mem.saturating_mul(95) / 100,
        )
    }

    fn set_sparse_index_bootstrap(
        run: &mut Self::Run,
        avg_key_bytes: f64,
        avg_record_bytes: f64,
        sample_count: usize,
    ) {
        run.set_sparse_index_bootstrap(avg_key_bytes, avg_record_bytes, sample_count);
    }

    fn sparse_index_bootstrap(run: &Self::Run) -> Option<(f64, f64, usize)> {
        run.sparse_index_bootstrap()
    }
}

impl SorterCore<PlainSortHooks> {
    pub fn new(
        run_gen_threads: usize,
        run_gen_mem: usize,
        merge_threads: usize,
        merge_fanin: usize,
        base_dir: impl AsRef<std::path::Path>,
    ) -> Self {
        Self::new_with_indexing_interval(
            run_gen_threads,
            run_gen_mem,
            merge_threads,
            merge_fanin,
            DEFAULT_RUN_INDEXING_INTERVAL,
            base_dir,
        )
    }

    pub fn new_with_indexing_interval(
        run_gen_threads: usize,
        run_gen_mem: usize,
        merge_threads: usize,
        merge_fanin: usize,
        run_indexing_interval: usize,
        base_dir: impl AsRef<std::path::Path>,
    ) -> Self {
        SorterCore::with_indexing_interval(
            run_gen_threads,
            run_gen_mem,
            merge_threads,
            merge_fanin,
            run_indexing_interval,
            base_dir,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InMemInput;
    use crate::diskio::aligned_writer::AlignedWriter;
    use crate::diskio::file::SharedFd;
    use crate::rand::small_thread_rng;
    use rand::seq::SliceRandom;
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn test_range_partitioned_scan_with_io_tracker() {
        let temp_dir = TempDir::new().unwrap();

        let path0 = temp_dir.path().join("partition0.dat");
        let fd0 = Arc::new(SharedFd::new_from_path(&path0, true).unwrap());
        let writer0 = AlignedWriter::from_fd(fd0.clone()).unwrap();
        let mut run0 = Run::from_writer(writer0).unwrap();
        for i in 0..100 {
            let key = format!("a{:02}", i).into_bytes();
            let value = format!("value_{}", i).into_bytes();
            run0.append(&key, &value);
        }
        run0.finalize_write();

        let path1 = temp_dir.path().join("partition1.dat");
        let fd1 = Arc::new(SharedFd::new_from_path(&path1, true).unwrap());
        let writer1 = AlignedWriter::from_fd(fd1.clone()).unwrap();
        let mut run1 = Run::from_writer(writer1).unwrap();
        for i in 0..100 {
            let key = format!("b{:02}", i).into_bytes();
            let value = format!("value_{}", i + 100).into_bytes();
            run1.append(&key, &value);
        }
        run1.finalize_write();

        let path2 = temp_dir.path().join("partition2.dat");
        let fd2 = Arc::new(SharedFd::new_from_path(&path2, true).unwrap());
        let writer2 = AlignedWriter::from_fd(fd2.clone()).unwrap();
        let mut run2 = Run::from_writer(writer2).unwrap();
        for i in 0..100 {
            let key = format!("c{:02}", i).into_bytes();
            let value = format!("value_{}", i + 200).into_bytes();
            run2.append(&key, &value);
        }
        run2.finalize_write();

        let mergeable_run = MergeableRun::RangePartitioned(vec![run0, run1, run2]);

        let io_tracker1 = IoStatsTracker::new();
        let lower: (&[u8], _, _) = (b"a10", 0, 0);
        let upper: (&[u8], _, _) = (b"a20", 0, 0);
        let iter1 = mergeable_run.scan_range_with_io_tracker(
            Some(lower),
            Some(upper),
            Some(io_tracker1.clone()),
        );
        let results1: Vec<_> = iter1.collect();
        assert_eq!(results1.len(), 10);
        assert_eq!(results1[0].0, b"a10");
        assert_eq!(results1[9].0, b"a19");
        let (read_ops1, read_bytes1) = io_tracker1.get_read_stats();
        assert!(read_ops1 > 0);
        assert!(read_bytes1 > 0);

        let io_tracker2 = IoStatsTracker::new();
        let lower2: (&[u8], _, _) = (b"a90", 0, 0);
        let upper2: (&[u8], _, _) = (b"c10", 0, 0);
        let iter2 = mergeable_run.scan_range_with_io_tracker(
            Some(lower2),
            Some(upper2),
            Some(io_tracker2.clone()),
        );
        let results2: Vec<_> = iter2.collect();
        assert_eq!(results2.len(), 120);
        assert_eq!(results2[0].0, b"a90");
        assert_eq!(results2[10].0, b"b00");
        assert_eq!(results2[119].0, b"c09");
        let (read_ops2, read_bytes2) = io_tracker2.get_read_stats();
        assert!(read_ops2 > 0);
        assert!(read_bytes2 > 0);

        let lower3: (&[u8], _, _) = (b"d00", 0, 0);
        let upper3: (&[u8], _, _) = (b"d99", 0, 0);
        let iter3 = mergeable_run.scan_range_with_io_tracker(Some(lower3), Some(upper3), None);
        assert!(iter3.collect::<Vec<_>>().is_empty());

        let io_tracker4 = IoStatsTracker::new();
        let iter4 = mergeable_run.scan_range_with_io_tracker(None, None, Some(io_tracker4.clone()));
        let results4: Vec<_> = iter4.collect();
        assert_eq!(results4.len(), 300);
        let (read_ops4, read_bytes4) = io_tracker4.get_read_stats();
        assert!(read_ops4 > 0);
        assert!(read_bytes4 > 0);
    }

    #[test]
    fn test_multi_level_merge_small_fanout() {
        let num_records = 100000;
        let num_threads_run_gen = 2;
        let run_gen_mem = 512 * 1024;
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

        let (runs, _run_gen_stats) = ExternalSorter::run_generation(
            Box::new(InMemInput { data }),
            num_threads_run_gen,
            run_gen_mem,
            run_indexing_interval,
            temp_dir.path(),
        )
        .unwrap();

        let (merged_run, per_merge_stats) = ExternalSorter::multi_merge(
            runs,
            fanin,
            num_threads,
            imbalance_factor,
            crate::sort::core::engine::PartitionType::default(),
            temp_dir.path(),
        )
        .unwrap();

        assert!(per_merge_stats.len() >= 1);
        let final_runs = merged_run.into_runs();
        assert_eq!(final_runs.len(), num_threads);
    }
}
