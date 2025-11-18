mod common;

use common::{ovc_test_dir, sorter_behavior};
use es::{ExternalSorterWithOVC, RunGenerationAlgorithm};

#[test]
fn test_basic_sort_with_ovc() {
    sorter_behavior::basic_sort(|| ExternalSorterWithOVC::new(2, 512, 2, 10000, ovc_test_dir()));
}

#[test]
fn test_load_sort_store_run_generation_with_ovc() {
    sorter_behavior::load_sort_store(|| {
        let mut sorter = ExternalSorterWithOVC::new(2, 4 * 1024, 2, 10000, ovc_test_dir());
        sorter.set_run_generation_algorithm(RunGenerationAlgorithm::LoadSortStore);
        sorter
    });
}

#[test]
fn test_external_sort_with_ovc() {
    sorter_behavior::small_buffer_external_sort(|| {
        ExternalSorterWithOVC::new(2, 512, 2, 10000, ovc_test_dir())
    });
}

#[test]
fn test_empty_input_with_ovc() {
    sorter_behavior::empty_input(|| ExternalSorterWithOVC::new(2, 512, 2, 10000, ovc_test_dir()));
}

#[test]
fn test_single_element_with_ovc() {
    sorter_behavior::single_element(|| {
        ExternalSorterWithOVC::new(2, 512, 2, 10000, ovc_test_dir())
    });
}

#[test]
fn test_duplicate_keys_with_ovc() {
    sorter_behavior::duplicate_keys(|| {
        ExternalSorterWithOVC::new(2, 512, 2, 10000, ovc_test_dir())
    });
}

#[test]
fn test_large_values_with_ovc() {
    sorter_behavior::large_values(|| ExternalSorterWithOVC::new(2, 512, 2, 10000, ovc_test_dir()));
}

#[test]
fn test_parallel_sorting_with_ovc() {
    sorter_behavior::large_dataset_with_threads(|threads| {
        ExternalSorterWithOVC::new(threads, 1024 * 1024, threads, 10000, ovc_test_dir())
    });
}

#[test]
fn test_sort_stats_with_ovc() {
    sorter_behavior::stats_populated(|| {
        ExternalSorterWithOVC::new(2, 512, 2, 10000, ovc_test_dir())
    });
}

#[test]
fn test_ovc_run_generation_stats_with_small_buffer() {
    sorter_behavior::run_generation_stats_small_buffer(|| {
        ExternalSorterWithOVC::new(1, 30, 1, 10000, ovc_test_dir())
    });
}

#[test]
fn test_binary_keys_with_ovc() {
    sorter_behavior::binary_keys(|| ExternalSorterWithOVC::new(2, 512, 2, 10000, ovc_test_dir()));
}

#[test]
fn test_variable_length_keys_with_ovc() {
    sorter_behavior::variable_length_keys(|| {
        ExternalSorterWithOVC::new(2, 512, 2, 10000, ovc_test_dir())
    });
}

#[test]
fn test_entry_too_large_panics_with_ovc() {
    sorter_behavior::entry_too_large_errors(|| {
        ExternalSorterWithOVC::new(1, 50, 1, 10000, ovc_test_dir())
    });
}

#[test]
fn test_concurrent_sorters_with_ovc() {
    sorter_behavior::concurrent_sorters(|| {
        ExternalSorterWithOVC::new(2, 512, 2, 10000, ovc_test_dir())
    });
}
