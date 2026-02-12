use std::sync::Barrier;

use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::io_stats::IoStatsTracker;
use crate::ovc::offset_value_coding_32::OVCU32;
use crate::sort::core::engine::SorterCore;
use crate::sort::core::run_format::{
    FormatSortHooks, IndexingInterval, MergeableRun as GenericMergeableRun, RunFormat,
    RunsOutput as GenericRunsOutput, cmp_key_run_offset,
};
use crate::sort::core::sparse_index::{
    SPARSE_INDEX_PAGE_SIZE, SparseIndex, SparseIndexPage, SparseIndexPagePool,
};
use crate::sort::ovc::merge::{MergeSource, MergeWithOVC, ZeroCopyMergeWithOVC};
use crate::sort::ovc::run::RunWithOVC;

pub type OvcSortHooks = FormatSortHooks<OvcRunFormat>;
pub type ExternalSorterWithOVC = SorterCore<OvcSortHooks>;
pub type MergeableRunWithOVC = GenericMergeableRun<OvcRunFormat>;
pub type RunsOutputWithOVC = GenericRunsOutput<OvcRunFormat>;

#[derive(Clone, Copy, Default)]
pub struct OvcRunFormat;
const DEFAULT_RUN_INDEXING_INTERVAL: usize = 100;

#[derive(Default, Clone)]
pub struct OvcAppendState {
    prev_key: Option<Vec<u8>>,
}

impl RunFormat for OvcRunFormat {
    type Run = RunWithOVC;
    type Record = (OVCU32, Vec<u8>, Vec<u8>);
    type AppendState = OvcAppendState;

    fn new_run(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
    ) -> Result<Self::Run, String> {
        RunWithOVC::from_writer_with_indexing_interval(writer, indexing_interval)
    }

    fn create_merge_run_with_id(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
        run_id: u32,
    ) -> Result<Self::Run, String> {
        RunWithOVC::from_writer_with_indexing_interval_and_id(writer, indexing_interval, run_id)
    }

    fn finalize_run(run: &mut Self::Run) -> AlignedWriter {
        run.finalize_write()
    }

    fn append_from_kv(
        _state: &mut Self::AppendState,
        _run: &mut Self::Run,
        _key: &[u8],
        _value: &[u8],
    ) {
        panic!("append_from_kv is not supported for OVC runs");
    }

    fn append_with_ovc(
        state: &mut Self::AppendState,
        run: &mut Self::Run,
        ovc: OVCU32,
        key: &[u8],
        value: &[u8],
    ) {
        run.append(ovc, key, value);
        state.prev_key = Some(key.to_vec());
    }

    fn append_record(state: &mut Self::AppendState, run: &mut Self::Run, record: Self::Record) {
        let (ovc, key, value) = record;
        run.append(ovc, &key, &value);
        state.prev_key = Some(key);
    }

    fn scan_range(
        run: &Self::Run,
        lower_bound: Option<(&[u8], u32, usize)>,
        upper_bound: Option<(&[u8], u32, usize)>,
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

    fn run_id(run: &Self::Run) -> u32 {
        run.run_id()
    }

    fn sparse_index<'a>(run: &'a Self::Run) -> &'a SparseIndex {
        run.sparse_index()
    }

    fn start_key<'a>(run: &'a Self::Run) -> Option<(&'a [u8], u32, usize)> {
        run.start_key().map(|key| (key, run.run_id(), 0))
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

    fn run_replacement_selection<
        S: crate::sort::run_sink::RunSink<
                MergeableRun = crate::sort::core::run_format::MergeableRun<Self>,
            >,
    >(
        scanner: crate::sort::core::engine::Scanner,
        sink: &mut S,
        run_size: usize,
    ) -> crate::replacement_selection::ReplacementSelectionStats {
        crate::replacement_selection::run_replacement_selection_ovc_mm(
            scanner,
            sink,
            run_size.saturating_mul(95) / 100,
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

    fn set_sparse_index_readers(run: &Self::Run, count: usize) {
        run.set_sparse_index_readers(count);
    }

    fn release_sparse_index(run: &Self::Run) {
        run.release_sparse_index();
    }

    fn release_sparse_index_to_pool(run: &Self::Run, pool: &SparseIndexPagePool) {
        run.release_sparse_index_to_pool(pool);
    }

    fn seed_sparse_index_buffer(run: &mut Self::Run, pages: Vec<SparseIndexPage>) {
        run.sparse_index_mut().seed_buffer(pages);
    }

    fn sparse_index_budget_pages(run: &Self::Run) -> usize {
        run.sparse_index().budget_pages()
    }

    fn direct_merge(
        _runs: &[GenericMergeableRun<Self>],
        _output_run: &mut Self::Run,
        _lower_bound: Option<(&[u8], u32, usize)>,
        _upper_bound: Option<(&[u8], u32, usize)>,
        _io_tracker: Option<IoStatsTracker>,
    ) -> bool {
        // No longer used; create_merge_sources_and_execute is used instead.
        false
    }

    fn create_merge_sources_and_execute(
        runs: &[GenericMergeableRun<Self>],
        output_run: &mut Self::Run,
        lower_bound: Option<(&[u8], u32, usize)>,
        upper_bound: Option<(&[u8], u32, usize)>,
        io_tracker: Option<IoStatsTracker>,
        page_pool: &SparseIndexPagePool,
        barrier: &Barrier,
        thread_id: usize,
        merge_threads: usize,
    ) -> bool {
        // Step 1: Create merge sources (reads sparse index for seek)
        let mut sources = Vec::new();

        for run in runs {
            match run {
                GenericMergeableRun::Single(r) => {
                    if let Some(src) =
                        r.scan_range_as_source(lower_bound, upper_bound, io_tracker.clone())
                    {
                        sources.push(src);
                    }
                }
                GenericMergeableRun::RangePartitioned(sub_runs) => {
                    let non_empty: Vec<&RunWithOVC> = sub_runs
                        .iter()
                        .filter(|r| r.start_key().is_some())
                        .collect();

                    if non_empty.is_empty() {
                        continue;
                    }

                    let start_partition = if lower_bound.is_none() {
                        0
                    } else {
                        let lb = lower_bound.unwrap();
                        let mut ok: isize = -1;
                        let mut ng: isize = non_empty.len() as isize;
                        while (ng - ok).abs() > 1 {
                            let mid = (ok + ng) / 2;
                            let r = non_empty[mid as usize];
                            let sk = (r.start_key().unwrap(), r.run_id(), 0usize);
                            if cmp_key_run_offset(sk, lb) == std::cmp::Ordering::Less {
                                ok = mid;
                            } else {
                                ng = mid;
                            }
                        }
                        if ok == -1 { 0 } else { ok as usize }
                    };

                    let end_partition = if upper_bound.is_none() {
                        non_empty.len()
                    } else {
                        let ub = upper_bound.unwrap();
                        let mut ok: isize = non_empty.len() as isize;
                        let mut ng: isize = -1;
                        while (ng - ok).abs() > 1 {
                            let mid = (ok + ng) / 2;
                            let r = non_empty[mid as usize];
                            let sk = (r.start_key().unwrap(), r.run_id(), 0usize);
                            if matches!(
                                cmp_key_run_offset(ub, sk),
                                std::cmp::Ordering::Less | std::cmp::Ordering::Equal
                            ) {
                                ok = mid;
                            } else {
                                ng = mid;
                            }
                        }
                        ok as usize
                    };

                    for sub_run in &non_empty[start_partition..end_partition] {
                        if let Some(src) = sub_run.scan_range_as_source(
                            lower_bound,
                            upper_bound,
                            io_tracker.clone(),
                        ) {
                            sources.push(src);
                        }
                    }
                }
                GenericMergeableRun::StatsOnly { .. } => {}
            }
        }

        // Step 2: Release sparse index pages to pool
        for run in runs {
            run.release_sparse_index_to_pool(page_pool);
        }

        // Step 3: Barrier - wait for all threads to release
        barrier.wait();

        // Step 4: Redistribute pages from pool to output run
        let total = page_pool.len();
        let budget_cap = output_run.sparse_index_mut().budget_pages();
        if thread_id == 0 {
            let taken = total.min(merge_threads * budget_cap);
            let freed = total.saturating_sub(taken);
            println!(
                "Sparse index page pool: {} pages ({:.2} MB), recycling {} pages ({:.2} MB), freeing {} pages ({:.2} MB)",
                total,
                (total * SPARSE_INDEX_PAGE_SIZE) as f64 / (1024.0 * 1024.0),
                taken,
                (taken * SPARSE_INDEX_PAGE_SIZE) as f64 / (1024.0 * 1024.0),
                freed,
                (freed * SPARSE_INDEX_PAGE_SIZE) as f64 / (1024.0 * 1024.0),
            );
        }
        let per_thread = total / merge_threads;
        let extra = total % merge_threads;
        let fair_share = per_thread + if thread_id < extra { 1 } else { 0 };
        let my_count = fair_share.min(budget_cap);
        let my_pages = page_pool.take(my_count);
        output_run.sparse_index_mut().seed_buffer(my_pages);

        // Step 5: Execute merge
        match sources.len() {
            0 => {}
            1 => {
                // Single source: direct copy without tree
                let mut source = sources.into_iter().next().unwrap();
                let mut key = Vec::new();
                while let Some((ovc, value_len)) = source.next_key_into(&mut key) {
                    output_run.append_header_and_key(ovc, &key, value_len);
                    source.copy_value_to(output_run.writer(), value_len);
                }
            }
            _ => {
                ZeroCopyMergeWithOVC::new(sources).merge_into(output_run);
            }
        }

        true
    }
}

impl SorterCore<OvcSortHooks> {
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
    use crate::ovc::offset_value_coding_32::compute_ovc_delta;
    use crate::rand::small_thread_rng;
    use crate::sort::ovc::run::RunWithOVC;
    use rand::seq::SliceRandom;
    use std::sync::Arc;
    use tempfile::TempDir;

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
            run0.append(OVCU32::initial_value(), &key, &value);
        }
        run0.finalize_write();

        let path1 = temp_dir.path().join("partition1.dat");
        let fd1 = Arc::new(SharedFd::new_from_path(&path1, true).unwrap());
        let writer1 = AlignedWriter::from_fd(fd1.clone()).unwrap();
        let mut run1 = RunWithOVC::from_writer(writer1).unwrap();
        for i in 0..100 {
            let key = format!("b{:02}", i).into_bytes();
            let value = format!("value_{}", i + 100).into_bytes();
            run1.append(OVCU32::initial_value(), &key, &value);
        }
        run1.finalize_write();

        let path2 = temp_dir.path().join("partition2.dat");
        let fd2 = Arc::new(SharedFd::new_from_path(&path2, true).unwrap());
        let writer2 = AlignedWriter::from_fd(fd2.clone()).unwrap();
        let mut run2 = RunWithOVC::from_writer(writer2).unwrap();
        for i in 0..100 {
            let key = format!("c{:02}", i).into_bytes();
            let value = format!("value_{}", i + 200).into_bytes();
            run2.append(OVCU32::initial_value(), &key, &value);
        }
        run2.finalize_write();

        let mergeable_run = MergeableRunWithOVC::RangePartitioned(vec![run0, run1, run2]);

        let iter = mergeable_run.scan_range_with_io_tracker(
            Some((b"a90", 0, 0)),
            Some((b"c10", 0, 0)),
            None,
        );
        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 120);
        assert_eq!(results[0].1, b"a90");
        assert_eq!(results[119].1, b"c09");
    }

    #[test]
    fn test_multi_level_merge_small_fanout() {
        let num_records = 1000000;
        let num_threads_run_gen = 2;
        let run_size = 512 * 1024; // 512 KB
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

        let (runs, _run_gen_stats) = ExternalSorterWithOVC::run_generation(
            Box::new(InMemInput { data }),
            num_threads_run_gen,
            run_size,
            run_indexing_interval,
            temp_dir.path(),
        )
        .unwrap();

        let (merged_run, per_merge_stats) = ExternalSorterWithOVC::multi_merge(
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
        let (runs, _) = ExternalSorterWithOVC::run_generation(
            Box::new(InMemInput { data }),
            2,          // num_threads
            256 * 1024, // run_size
            1000,       // run_indexing_interval
            temp_dir.path(),
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
            2,   // fanin
            2,   // num_threads
            1.0, // imbalance_factor
            crate::sort::core::engine::PartitionType::default(),
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
