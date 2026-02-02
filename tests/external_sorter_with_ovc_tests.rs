mod common;

use common::{ovc_test_dir, sorter_behavior};
use es::ovc::offset_value_coding::OVCKeyValue;
use es::ovc::tree_of_losers_ovc::LoserTreeOVC;
use es::replacement_selection::RecordSize;
use es::{ExternalSorterWithOVC, InMemInput, Sorter};

#[test]
fn test_basic_sort_with_ovc() {
    sorter_behavior::basic_sort(|| ExternalSorterWithOVC::new(2, 512, 2, 10000, ovc_test_dir()));
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
    sorter_behavior::large_values(|| ExternalSorterWithOVC::new(2, 1024, 2, 10000, ovc_test_dir()));
}

#[test]
fn test_parallel_sorting_with_ovc() {
    sorter_behavior::large_dataset_with_threads(|threads| {
        ExternalSorterWithOVC::new(threads, 10 * 1024, threads, 10000, ovc_test_dir())
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
        ExternalSorterWithOVC::new(1, 100, 1, 10000, ovc_test_dir())
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

#[test]
fn test_run_size_too_small_for_tree_overhead_with_ovc() {
    let record_size = OVCKeyValue::new(vec![b'k'], vec![b'v']).size();
    let overhead = LoserTreeOVC::<OVCKeyValue>::structure_overhead_bytes_for_capacity(1);
    let run_size = record_size + overhead - 1;

    let mut sorter = ExternalSorterWithOVC::new(1, run_size, 1, 10000, ovc_test_dir());
    let input = InMemInput {
        data: vec![(vec![b'k'], vec![b'v'])],
    };

    assert!(sorter.sort(Box::new(input)).is_err());
}
