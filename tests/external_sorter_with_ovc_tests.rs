use std::path::PathBuf;
use std::sync::Once;

static INIT: Once = Once::new();

fn test_dir() -> PathBuf {
    INIT.call_once(|| {
        let dir = PathBuf::from("./test_runs_ovc");
        std::fs::create_dir_all(&dir).expect("Failed to create test directory");
    });
    PathBuf::from("./test_runs_ovc")
}

use es::{ExternalSorterWithOVC, InMemInput, Sorter};

#[test]
fn test_basic_sort_with_ovc() {
    let mut sorter = ExternalSorterWithOVC::new(2, 512, 2, 10000, test_dir());

    let data = vec![
        (b"c".to_vec(), b"3".to_vec()),
        (b"a".to_vec(), b"1".to_vec()),
        (b"b".to_vec(), b"2".to_vec()),
    ];

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, b"a");
    assert_eq!(results[0].1, b"1");
    assert_eq!(results[1].0, b"b");
    assert_eq!(results[1].1, b"2");
    assert_eq!(results[2].0, b"c");
    assert_eq!(results[2].1, b"3");
}

#[test]
fn test_external_sort_with_ovc() {
    // Use small buffer to force external sorting with multiple runs
    let mut sorter = ExternalSorterWithOVC::new(2, 512, 2, 10000, test_dir());

    let mut data = Vec::new();
    for i in (0..100).rev() {
        let key = format!("key_{:03}", i);
        let value = format!("value_{}", i);
        data.push((key.into_bytes(), value.into_bytes()));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 100);

    // Verify sorting order
    for i in 0..100 {
        let expected_key = format!("key_{:03}", i);
        let expected_value = format!("value_{}", i);
        assert_eq!(results[i].0, expected_key.as_bytes());
        assert_eq!(results[i].1, expected_value.as_bytes());
    }
}

#[test]
fn test_empty_input_with_ovc() {
    let mut sorter = ExternalSorterWithOVC::new(2, 512, 2, 10000, test_dir());

    let data = vec![];
    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_single_element_with_ovc() {
    let mut sorter = ExternalSorterWithOVC::new(2, 512, 2, 10000, test_dir());

    let data = vec![(b"only".to_vec(), b"one".to_vec())];
    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, b"only");
    assert_eq!(results[0].1, b"one");
}

#[test]
fn test_duplicate_keys_with_ovc() {
    let mut sorter = ExternalSorterWithOVC::new(2, 512, 2, 10000, test_dir());

    let data = vec![
        (b"b".to_vec(), b"2".to_vec()),
        (b"a".to_vec(), b"1".to_vec()),
        (b"b".to_vec(), b"3".to_vec()),
        (b"a".to_vec(), b"4".to_vec()),
        (b"c".to_vec(), b"5".to_vec()),
    ];

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 5);

    // All 'a' keys should come first
    assert_eq!(results[0].0, b"a");
    assert_eq!(results[1].0, b"a");

    // Then 'b' keys
    assert_eq!(results[2].0, b"b");
    assert_eq!(results[3].0, b"b");

    // Finally 'c' key
    assert_eq!(results[4].0, b"c");
    assert_eq!(results[4].1, b"5");
}

#[test]
fn test_large_values_with_ovc() {
    let mut sorter = ExternalSorterWithOVC::new(2, 512, 2, 10000, test_dir());

    // Create large values to test buffer handling
    let large_value = vec![b'x'; 500];
    let data = vec![
        (b"c".to_vec(), large_value.clone()),
        (b"a".to_vec(), large_value.clone()),
        (b"b".to_vec(), large_value.clone()),
    ];

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, b"a");
    assert_eq!(results[1].0, b"b");
    assert_eq!(results[2].0, b"c");

    // Verify values are preserved
    assert_eq!(results[0].1.len(), 500);
    assert_eq!(results[0].1[0], b'x');
}

#[test]
fn test_parallel_sorting_with_ovc() {
    // Test with different thread configurations
    let mut sorter = ExternalSorterWithOVC::new(4, 1024, 4, 10000, test_dir());

    let mut data = Vec::new();
    for i in (0..1000).rev() {
        let key = format!("{:04}", i);
        let value = format!("val_{}", i);
        data.push((key.into_bytes(), value.into_bytes()));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 1000);

    // Verify correct ordering
    for i in 0..1000 {
        let expected_key = format!("{:04}", i);
        assert_eq!(results[i].0, expected_key.as_bytes());
    }
}

#[test]
fn test_sort_stats_with_ovc() {
    // Small buffer to force multiple runs
    let mut sorter = ExternalSorterWithOVC::new(2, 512, 2, 10000, test_dir());

    let mut data = Vec::new();
    for i in (0..5000).rev() {
        let key = format!("key_{:02}", i);
        let value = format!("value_{}", i);
        data.push((key.into_bytes(), value.into_bytes()));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    // Get stats and verify they're populated
    let stats = output.stats();

    // Should have created multiple runs with small buffer
    let num_runs = stats.run_gen_stats.num_runs;
    assert!(num_runs > 0);
    assert!(stats.run_gen_stats.time_ms > 0);

    // If there were multiple runs, there should be non-zero merge time recorded across merges
    if num_runs > 1 {
        let total_merge_time: u128 = stats.per_merge_stats.iter().map(|m| m.time_ms).sum();
        assert!(total_merge_time > 0);
    }
}

#[test]
fn test_binary_keys_with_ovc() {
    let mut sorter = ExternalSorterWithOVC::new(2, 512, 2, 10000, test_dir());

    // Test with binary (non-UTF8) keys
    let data = vec![
        (vec![255, 255, 255], b"max".to_vec()),
        (vec![0, 0, 0], b"min".to_vec()),
        (vec![127, 127, 127], b"mid".to_vec()),
        (vec![1, 2, 3], b"low".to_vec()),
        (vec![200, 201, 202], b"high".to_vec()),
    ];

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 5);

    // Verify binary ordering
    assert_eq!(results[0].0, vec![0, 0, 0]);
    assert_eq!(results[0].1, b"min");
    assert_eq!(results[1].0, vec![1, 2, 3]);
    assert_eq!(results[1].1, b"low");
    assert_eq!(results[2].0, vec![127, 127, 127]);
    assert_eq!(results[2].1, b"mid");
    assert_eq!(results[3].0, vec![200, 201, 202]);
    assert_eq!(results[3].1, b"high");
    assert_eq!(results[4].0, vec![255, 255, 255]);
    assert_eq!(results[4].1, b"max");
}

#[test]
fn test_variable_length_keys_with_ovc() {
    let mut sorter = ExternalSorterWithOVC::new(2, 512, 2, 10000, test_dir());

    let data = vec![
        (b"zzz".to_vec(), b"3_chars".to_vec()),
        (b"a".to_vec(), b"1_char".to_vec()),
        (b"bb".to_vec(), b"2_chars".to_vec()),
        (b"aaaa".to_vec(), b"4_chars".to_vec()),
        (b"aaa".to_vec(), b"3_chars_a".to_vec()),
    ];

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 5);

    // Verify lexicographic ordering
    assert_eq!(results[0].0, b"a");
    assert_eq!(results[0].1, b"1_char");
    assert_eq!(results[1].0, b"aaa");
    assert_eq!(results[1].1, b"3_chars_a");
    assert_eq!(results[2].0, b"aaaa");
    assert_eq!(results[2].1, b"4_chars");
    assert_eq!(results[3].0, b"bb");
    assert_eq!(results[3].1, b"2_chars");
    assert_eq!(results[4].0, b"zzz");
    assert_eq!(results[4].1, b"3_chars");
}
