mod common;

use common::{ovc_test_dir, sorter_behavior};
use es::{ExternalSorterWithOVC, InMemInput, Sorter};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_basic_sort_with_ovc() {
        sorter_behavior::basic_sort(|| {
            ExternalSorterWithOVC::new(2, 512 * 1024, 2, 10000, ovc_test_dir())
        });
    }

    #[test]
    fn test_external_sort_with_ovc() {
        sorter_behavior::small_buffer_external_sort(|| {
            ExternalSorterWithOVC::new(2, 128 * 1024, 2, 10000, ovc_test_dir())
        });
    }

    #[test]
    fn test_empty_input_with_ovc() {
        sorter_behavior::empty_input(|| {
            ExternalSorterWithOVC::new(2, 512 * 1024, 2, 10000, ovc_test_dir())
        });
    }

    #[test]
    fn test_single_element_with_ovc() {
        sorter_behavior::single_element(|| {
            ExternalSorterWithOVC::new(2, 512 * 1024, 2, 10000, ovc_test_dir())
        });
    }

    #[test]
    fn test_duplicate_keys_with_ovc() {
        sorter_behavior::duplicate_keys(|| {
            ExternalSorterWithOVC::new(2, 512 * 1024, 2, 10000, ovc_test_dir())
        });
    }

    #[test]
    fn test_pathological_key_distribution_with_ovc() {
        let mut sorter = ExternalSorterWithOVC::new(4, 1024 * 1024, 4, 10000, ovc_test_dir());
        sorter.set_partition_type(es::sort::engine::PartitionType::KeyOnly);

        let mut data = Vec::new();

        // Create a pathological distribution:
        // - Many entries with same prefix
        // - Keys that differ only in last character
        // - Keys designed to cause poor partitioning

        // Group 1: Same long prefix
        for _ in 0..1000 {
            let key = "aaaaaaaaaaaaaaaaaaaaaaaaaaaa";
            data.push((key.as_bytes().to_vec(), b"v1".to_vec()));
        }

        // Group 2: Sequential but reverse order
        for i in (0..1000).rev() {
            let key = format!("b{:04}", i);
            data.push((key.into_bytes(), b"v2".to_vec()));
        }

        // Group 3: All same key
        for i in 0..1000 {
            data.push((b"same_key".to_vec(), format!("v3_{}", i).into_bytes()));
        }

        let input = InMemInput { data };
        let output = sorter.sort(Box::new(input)).unwrap();

        let results: Vec<_> = output.iter().collect();
        assert_eq!(results.len(), 3000);

        // Verify sorted order
        for i in 1..results.len() {
            assert!(results[i - 1].0 <= results[i].0);
        }
    }

    #[test]
    fn test_large_values_with_ovc() {
        sorter_behavior::large_values(|| {
            ExternalSorterWithOVC::new(2, 1024 * 1024, 2, 10000, ovc_test_dir())
        });
    }

    #[test]
    fn test_parallel_sorting_with_ovc() {
        sorter_behavior::large_dataset_with_threads(|threads| {
            ExternalSorterWithOVC::new(threads, 128 * 1024, threads, 10000, ovc_test_dir())
        });
    }

    #[test]
    fn test_sort_stats_with_ovc() {
        sorter_behavior::stats_populated(|| {
            ExternalSorterWithOVC::new(2, 512 * 1024, 2, 10000, ovc_test_dir())
        });
    }

    #[test]
    fn test_ovc_run_generation_stats_with_small_buffer() {
        sorter_behavior::run_generation_stats_small_buffer(|| {
            ExternalSorterWithOVC::new(1, 128 * 1024, 1, 10000, ovc_test_dir())
        });
    }

    #[test]
    fn test_binary_keys_with_ovc() {
        sorter_behavior::binary_keys(|| {
            ExternalSorterWithOVC::new(2, 512 * 1024, 2, 10000, ovc_test_dir())
        });
    }

    #[test]
    fn test_variable_length_keys_with_ovc() {
        sorter_behavior::variable_length_keys(|| {
            ExternalSorterWithOVC::new(2, 512 * 1024, 2, 10000, ovc_test_dir())
        });
    }

    #[test]
    fn test_concurrent_sorters_with_ovc() {
        sorter_behavior::concurrent_sorters(|| {
            ExternalSorterWithOVC::new(2, 512 * 1024, 2, 10000, ovc_test_dir())
        });
    }
}
