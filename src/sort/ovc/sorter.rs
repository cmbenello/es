use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::io_stats::IoStatsTracker;
use crate::ovc::offset_value_coding_u64::OVCU64;
use crate::replacement_selection::compute_ovc_delta;
use crate::sort::core::engine::SorterCore;
use crate::sort::core::run_format::{
    FormatSortHooks, MergeableRun as GenericMergeableRun, RunFormat,
    RunsOutput as GenericRunsOutput,
};
use crate::sort::ovc::merge::MergeWithOVC;
use crate::sort::ovc::run::RunWithOVC;

pub type OvcSortHooks = FormatSortHooks<OvcRunFormat>;
pub type ExternalSorterWithOVC = SorterCore<OvcSortHooks>;
pub type MergeableRunWithOVC = GenericMergeableRun<OvcRunFormat>;
pub type RunsOutputWithOVC = GenericRunsOutput<OvcRunFormat>;

#[derive(Clone, Copy, Default)]
pub struct OvcRunFormat;

#[derive(Default, Clone)]
pub struct OvcAppendState {
    prev_key: Option<Vec<u8>>,
}

impl RunFormat for OvcRunFormat {
    type Run = RunWithOVC;
    type Record = (OVCU64, Vec<u8>, Vec<u8>);
    type AppendState = OvcAppendState;

    fn new_run(writer: AlignedWriter, indexing_interval: usize) -> Result<Self::Run, String> {
        RunWithOVC::from_writer_with_indexing_interval(writer, indexing_interval)
    }

    fn finalize_run(run: &mut Self::Run) -> AlignedWriter {
        run.finalize_write()
    }

    fn append_from_kv(
        state: &mut Self::AppendState,
        run: &mut Self::Run,
        key: &[u8],
        value: &[u8],
    ) {
        let ovc = compute_ovc_delta(state.prev_key.as_deref(), key);
        state.prev_key = Some(key.to_vec());
        run.append(ovc, key, value);
    }

    fn append_record(state: &mut Self::AppendState, run: &mut Self::Run, record: Self::Record) {
        let (ovc, key, value) = record;
        run.append(ovc, &key, &value);
        state.prev_key = Some(key);
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
        match iterators.len() {
            0 => Box::new(std::iter::empty()),
            1 => iterators.into_iter().next().unwrap(),
            _ => Box::new(MergeWithOVC::new(iterators)),
        }
    }

    fn start_key<'a>(run: &'a Self::Run) -> Option<&'a [u8]> {
        run.start_key()
    }

    fn record_key<'a>(record: &'a Self::Record) -> &'a [u8] {
        &record.1
    }

    fn record_value<'a>(record: &'a Self::Record) -> &'a [u8] {
        &record.2
    }

    fn record_into_kv(record: Self::Record) -> (Vec<u8>, Vec<u8>) {
        (record.1, record.2)
    }
}

impl SorterCore<OvcSortHooks> {
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
    use crate::sort::core::run_format::RunWriterSink;
    use crate::sort::engine::RunGenerationAlgorithm;
    use crate::sort::ovc::run::RunWithOVC;
    use crate::sort::ovc::sort_buffer_with_ovc::SortBufferOVC;
    use crate::sort::run_sink::RunSink;
    use rand::seq::SliceRandom;
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
    }

    #[test]
    fn test_range_partitioned_scan_with_io_tracker() {
        let temp_dir = TempDir::new().unwrap();

        let path0 = temp_dir.path().join("partition0.dat");
        let fd0 = Arc::new(SharedFd::new_from_path(&path0, true).unwrap());
        let writer0 = AlignedWriter::from_fd(fd0.clone()).unwrap();
        let mut run0 = RunWithOVC::from_writer(writer0).unwrap();
        for i in 0..100 {
            let key = format!("a{:02}", i).into_bytes();
            let value = format!("value_{}", i).into_bytes();
            run0.append(OVCU64::initial_value(), &key, &value);
        }
        run0.finalize_write();

        let path1 = temp_dir.path().join("partition1.dat");
        let fd1 = Arc::new(SharedFd::new_from_path(&path1, true).unwrap());
        let writer1 = AlignedWriter::from_fd(fd1.clone()).unwrap();
        let mut run1 = RunWithOVC::from_writer(writer1).unwrap();
        for i in 0..100 {
            let key = format!("b{:02}", i).into_bytes();
            let value = format!("value_{}", i + 100).into_bytes();
            run1.append(OVCU64::initial_value(), &key, &value);
        }
        run1.finalize_write();

        let path2 = temp_dir.path().join("partition2.dat");
        let fd2 = Arc::new(SharedFd::new_from_path(&path2, true).unwrap());
        let writer2 = AlignedWriter::from_fd(fd2.clone()).unwrap();
        let mut run2 = RunWithOVC::from_writer(writer2).unwrap();
        for i in 0..100 {
            let key = format!("c{:02}", i).into_bytes();
            let value = format!("value_{}", i + 200).into_bytes();
            run2.append(OVCU64::initial_value(), &key, &value);
        }
        run2.finalize_write();

        let mergeable_run = MergeableRunWithOVC::RangePartitioned(vec![run0, run1, run2]);

        let iter = mergeable_run.scan_range_with_io_tracker(b"a90", b"c10", None);
        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 120);
        assert_eq!(results[0].1, b"a90");
        assert_eq!(results[119].1, b"c09");
    }

    #[test]
    fn test_multi_level_merge_small_fanout() {
        let num_records = 100000;
        let num_threads_run_gen = 2;
        let run_size = 512;
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

        assert!(per_merge_stats.len() >= 1);
        let final_runs = merged_run.into_runs();
        assert_eq!(final_runs.len(), num_threads);
    }

    #[test]
    fn test_sort_buffer_flush_with_ovc() {
        let mut buffer = SortBufferOVC::new(1024);
        for i in 0..100 {
            buffer.append(
                format!("{:04}", 99 - i).into_bytes(),
                i.to_string().into_bytes(),
            );
        }
        let temp_dir = tempdir().unwrap();
        let fd = Arc::new(SharedFd::new_from_path(&temp_dir.path().join("run.dat"), true).unwrap());
        let writer = AlignedWriter::from_fd(fd).unwrap();
        let mut sink = RunWriterSink::<OvcRunFormat>::new(writer, 1000, 128, 1);
        sink.start_run();
        for (_ovc, key, value) in buffer.sorted_iter() {
            sink.push_record(&key, &value);
        }
        sink.finish_run();
        let (runs, _sketch) = sink.finalize();
        assert_eq!(runs.len(), 1);
    }

    #[test]
    fn test_consumed_runs_deleted_after_merge() {
        let temp_dir = TempDir::new().unwrap();
        let mut data: Vec<_> = (0..1000)
            .map(|i| {
                (
                    format!("key_{:05}", i).into_bytes(),
                    format!("value_{}", i).into_bytes(),
                )
            })
            .collect();
        data.shuffle(&mut small_thread_rng());

        // Generate runs
        let (runs, sketch, _) = ExternalSorterWithOVC::run_generation(
            Box::new(InMemInput { data }),
            2,        // num_threads
            256 * 20, // run_size
            100,      // sketch_size
            1000,     // sketch_sampling_interval
            1000,     // run_indexing_interval
            temp_dir.path(),
            RunGenerationAlgorithm::ReplacementSelection,
        )
        .unwrap();

        // Capture file count before merge
        let files_before_merge: Vec<_> = std::fs::read_dir(temp_dir.path())
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        assert!(
            !files_before_merge.is_empty(),
            "Should have generated some runs"
        );

        // Perform the merge
        let (merged_run, per_merge_stats) = ExternalSorterWithOVC::multi_merge(
            runs,
            2, // fanin
            2, // num_threads
            &sketch,
            1.0, // imbalance_factor
            temp_dir.path(),
        )
        .unwrap();

        // Verify merge happened
        assert!(!per_merge_stats.is_empty());

        // Check that input run files were deleted
        for path in &files_before_merge {
            assert!(
                !path.exists(),
                "Input run file {:?} should be deleted after merge",
                path
            );
        }

        // Drop the merged run
        let final_runs = merged_run.into_runs();
        drop(final_runs);

        // Verify all files are cleaned up
        let remaining_files: Vec<std::fs::DirEntry> = std::fs::read_dir(temp_dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(
            remaining_files.is_empty(),
            "All files should be deleted, but found: {:?}",
            remaining_files
        );
    }
}
