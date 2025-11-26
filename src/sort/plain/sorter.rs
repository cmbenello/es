use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::io_stats::IoStatsTracker;
use crate::sort::core::engine::SorterCore;
use crate::sort::core::run_format::{
    FormatSortHooks, MergeableRun as GenericMergeableRun, RunFormat,
    RunsOutput as GenericRunsOutput,
};
use crate::sort::plain::merge::MergeIterator;
use crate::sort::plain::run::RunImpl;

pub type PlainSortHooks = FormatSortHooks<PlainRunFormat>;
pub type ExternalSorter = SorterCore<PlainSortHooks>;
pub type MergeableRun = GenericMergeableRun<PlainRunFormat>;
pub type RunsOutput = GenericRunsOutput<PlainRunFormat>;

#[derive(Clone, Copy, Default)]
pub struct PlainRunFormat;

impl RunFormat for PlainRunFormat {
    type Run = RunImpl;
    type Record = (Vec<u8>, Vec<u8>);
    type AppendState = ();

    fn new_run(writer: AlignedWriter, indexing_interval: usize) -> Result<Self::Run, String> {
        RunImpl::from_writer_with_indexing_interval(writer, indexing_interval)
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
        lower_bound: &[u8],
        upper_bound: &[u8],
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = Self::Record> + Send> {
        run.scan_range_with_io_tracker(lower_bound, upper_bound, io_tracker)
    }

    fn merge_iterators(
        iterators: Vec<Box<dyn Iterator<Item = Self::Record> + Send>>,
    ) -> Box<dyn Iterator<Item = Self::Record> + Send> {
        Box::new(MergeIterator::new(iterators))
    }

    fn start_key<'a>(run: &'a Self::Run) -> Option<&'a [u8]> {
        run.start_key()
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
}

impl SorterCore<PlainSortHooks> {
    pub fn new(
        run_gen_threads: usize,
        run_size: usize,
        merge_threads: usize,
        merge_fanin: usize,
        base_dir: impl AsRef<std::path::Path>,
    ) -> Self {
        SorterCore::with_defaults(
            run_gen_threads,
            run_size,
            merge_threads,
            merge_fanin,
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
    use crate::sketch::SketchType;
    use crate::sort::engine::RunGenerationAlgorithm;
    use rand::seq::SliceRandom;
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn test_range_partitioned_scan_with_io_tracker() {
        let temp_dir = TempDir::new().unwrap();

        let path0 = temp_dir.path().join("partition0.dat");
        let fd0 = Arc::new(SharedFd::new_from_path(&path0, true).unwrap());
        let writer0 = AlignedWriter::from_fd(fd0.clone()).unwrap();
        let mut run0 = RunImpl::from_writer(writer0).unwrap();
        for i in 0..100 {
            let key = format!("a{:02}", i).into_bytes();
            let value = format!("value_{}", i).into_bytes();
            run0.append(&key, &value);
        }
        run0.finalize_write();

        let path1 = temp_dir.path().join("partition1.dat");
        let fd1 = Arc::new(SharedFd::new_from_path(&path1, true).unwrap());
        let writer1 = AlignedWriter::from_fd(fd1.clone()).unwrap();
        let mut run1 = RunImpl::from_writer(writer1).unwrap();
        for i in 0..100 {
            let key = format!("b{:02}", i).into_bytes();
            let value = format!("value_{}", i + 100).into_bytes();
            run1.append(&key, &value);
        }
        run1.finalize_write();

        let path2 = temp_dir.path().join("partition2.dat");
        let fd2 = Arc::new(SharedFd::new_from_path(&path2, true).unwrap());
        let writer2 = AlignedWriter::from_fd(fd2.clone()).unwrap();
        let mut run2 = RunImpl::from_writer(writer2).unwrap();
        for i in 0..100 {
            let key = format!("c{:02}", i).into_bytes();
            let value = format!("value_{}", i + 200).into_bytes();
            run2.append(&key, &value);
        }
        run2.finalize_write();

        let mergeable_run = MergeableRun::RangePartitioned(vec![run0, run1, run2]);

        let io_tracker1 = IoStatsTracker::new();
        let iter1 =
            mergeable_run.scan_range_with_io_tracker(b"a10", b"a20", Some(io_tracker1.clone()));
        let results1: Vec<_> = iter1.collect();
        assert_eq!(results1.len(), 10);
        assert_eq!(results1[0].0, b"a10");
        assert_eq!(results1[9].0, b"a19");
        let (read_ops1, read_bytes1) = io_tracker1.get_read_stats();
        assert!(read_ops1 > 0);
        assert!(read_bytes1 > 0);

        let io_tracker2 = IoStatsTracker::new();
        let iter2 =
            mergeable_run.scan_range_with_io_tracker(b"a90", b"c10", Some(io_tracker2.clone()));
        let results2: Vec<_> = iter2.collect();
        assert_eq!(results2.len(), 120);
        assert_eq!(results2[0].0, b"a90");
        assert_eq!(results2[10].0, b"b00");
        assert_eq!(results2[119].0, b"c09");
        let (read_ops2, read_bytes2) = io_tracker2.get_read_stats();
        assert!(read_ops2 > 0);
        assert!(read_bytes2 > 0);

        let iter3 = mergeable_run.scan_range_with_io_tracker(b"d00", b"d99", None);
        assert!(iter3.collect::<Vec<_>>().is_empty());

        let io_tracker4 = IoStatsTracker::new();
        let iter4 = mergeable_run.scan_range_with_io_tracker(&[], &[], Some(io_tracker4.clone()));
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
        let run_size = 512;
        let sketch_type = SketchType::Kll;
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

        let (runs, sketch, _run_gen_stats) = ExternalSorter::run_generation(
            Box::new(InMemInput { data }),
            num_threads_run_gen,
            run_size,
            sketch_type,
            sketch_size,
            sketch_sampling_interval,
            run_indexing_interval,
            temp_dir.path(),
            RunGenerationAlgorithm::ReplacementSelection,
        )
        .unwrap();

        let (merged_run, per_merge_stats) = ExternalSorter::multi_merge(
            runs,
            fanin,
            num_threads,
            &sketch,
            imbalance_factor,
            temp_dir.path(),
        )
        .unwrap();

        assert!(per_merge_stats.len() >= 1);
        let final_runs = merged_run.into_runs();
        assert_eq!(final_runs.len(), num_threads);
    }
}
