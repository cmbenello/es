mod common;

use common::{sorter_behavior, test_dir};
use es::{ExternalSorter, RunGenerationAlgorithm};

#[test]
fn test_basic_functionality() {
    sorter_behavior::basic_sort(|| ExternalSorter::new(1, 512, 1, 10000, test_dir()));
}

#[test]
fn test_quicksort_run_generation_path() {
    sorter_behavior::load_sort_store(|| {
        let mut sorter = ExternalSorter::new(2, 4 * 1024, 2, 10000, test_dir());
        sorter.set_run_generation_algorithm(RunGenerationAlgorithm::LoadSortStore);
        sorter
    });
}

#[test]
fn test_empty_input() {
    sorter_behavior::empty_input(|| ExternalSorter::new(2, 512, 2, 10000, test_dir()));
}

#[test]
fn test_single_element() {
    sorter_behavior::single_element(|| ExternalSorter::new(2, 512, 2, 10000, test_dir()));
}

#[test]
fn test_large_dataset_with_threading() {
    sorter_behavior::large_dataset_with_threads(|threads| {
        ExternalSorter::new(threads, 1024 * 1024, threads, 10000, test_dir())
    });
}

#[test]
fn test_small_buffer_forces_external_sort() {
    sorter_behavior::small_buffer_external_sort(|| {
        ExternalSorter::new(2, 512, 2, 10000, test_dir())
    });
}

#[test]
fn test_duplicate_keys_and_stability() {
    sorter_behavior::duplicate_keys(|| ExternalSorter::new(2, 512, 2, 10000, test_dir()));
}

#[test]
fn test_run_generation_stats_with_small_buffer() {
    sorter_behavior::run_generation_stats_small_buffer(|| {
        ExternalSorter::new(1, 30, 1, 10000, test_dir())
    });
}

#[test]
fn test_variable_sized_data() {
    sorter_behavior::variable_sized_data(|| {
        ExternalSorter::new(2, 1024 * 1024, 2, 10000, test_dir())
    });
}

#[test]
fn test_binary_data_with_null_bytes() {
    sorter_behavior::binary_data_with_nulls(|| {
        ExternalSorter::new(2, 1024 * 1024, 2, 10000, test_dir())
    });
}

#[test]
fn test_buffer_boundary_edge_cases() {
    sorter_behavior::buffer_boundary_cases(|| {
        ExternalSorter::new(1, 32 * 100, 1, 10000, test_dir())
    });
}

#[test]
fn test_entry_too_large_panics() {
    sorter_behavior::entry_too_large_errors(|| ExternalSorter::new(1, 50, 1, 10000, test_dir()));
}

#[test]
fn test_concurrent_sorters() {
    sorter_behavior::concurrent_sorters(|| ExternalSorter::new(2, 512, 2, 10000, test_dir()));
}

#[test]
fn test_large_values() {
    sorter_behavior::large_values(|| ExternalSorter::new(2, 512, 2, 10000, test_dir()));
}

#[test]
fn test_sort_stats() {
    sorter_behavior::stats_populated(|| ExternalSorter::new(2, 512, 2, 10000, test_dir()));
}

#[test]
fn test_binary_keys() {
    sorter_behavior::binary_keys(|| ExternalSorter::new(2, 512, 2, 10000, test_dir()));
}

#[test]
fn test_variable_length_keys() {
    sorter_behavior::variable_length_keys(|| ExternalSorter::new(2, 512, 2, 10000, test_dir()));
}
