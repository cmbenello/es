mod common;
use common::test_dir;

use es::{ExternalSorter, InMemInput, Sorter};
use std::collections::HashMap;
use std::thread;

#[test]
fn test_basic_functionality() {
    let mut sorter = ExternalSorter::new(1, 512, 1, 10000, test_dir());

    let data = vec![
        (b"z".to_vec(), b"26".to_vec()),
        (b"a".to_vec(), b"1".to_vec()),
        (b"m".to_vec(), b"13".to_vec()),
    ];

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0], (b"a".to_vec(), b"1".to_vec()));
    assert_eq!(results[1], (b"m".to_vec(), b"13".to_vec()));
    assert_eq!(results[2], (b"z".to_vec(), b"26".to_vec()));
}

#[test]
fn test_empty_input() {
    let mut sorter = ExternalSorter::new(2, 512, 2, 10000, test_dir());

    let input = InMemInput { data: vec![] };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_large_dataset_with_threading() {
    // Test both single and multi-threaded sorting with 100K entries
    for num_threads in [1, 4] {
        let mut sorter =
            ExternalSorter::new(num_threads, 1024 * 1024, num_threads, 10000, test_dir());

        let mut data = Vec::new();
        for i in 0..100_000 {
            // Mix up the order with prime number modulo
            let key = format!("{:08}", (i * 7919) % 100_000);
            let value = format!("value_{}", i);
            data.push((key.into_bytes(), value.into_bytes()));
        }

        let input = InMemInput { data };
        let output = sorter.sort(Box::new(input)).unwrap();

        let results: Vec<_> = output.iter().collect();
        assert_eq!(results.len(), 100_000);

        // Verify sorted order
        for i in 0..100_000 {
            assert_eq!(results[i].0, format!("{:08}", i).as_bytes());
        }
    }
}

#[test]
fn test_small_buffer_forces_external_sort() {
    // Use very small buffer to force multiple runs and test merge
    let mut sorter = ExternalSorter::new(2, 512, 2, 10000, test_dir());

    let mut data = Vec::new();
    for i in 0..5000 {
        let key = format!("{:05}", 5000 - i - 1); // Reverse order
        let value = vec![b'v'; 20];
        data.push((key.into_bytes(), value));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 5000);

    // Verify sorted order
    for i in 0..5000 {
        assert_eq!(results[i].0, format!("{:05}", i).as_bytes());
    }
}

#[test]
fn test_duplicate_keys_and_stability() {
    let mut sorter = ExternalSorter::new(2, 512, 2, 10000, test_dir());

    let mut data = Vec::new();
    // Create many duplicates: 100 keys, 10 values each
    for i in 0..100 {
        for j in 0..10 {
            let key = format!("key_{:03}", i);
            let value = format!("value_{}_{}", i, j);
            data.push((key.into_bytes(), value.into_bytes()));
        }
    }

    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 1000);

    // Verify all duplicates are preserved
    let mut counts = HashMap::new();
    for (key, _) in &results {
        *counts.entry(key.clone()).or_insert(0) += 1;
    }

    for (_key, count) in counts {
        assert_eq!(count, 10);
    }

    // Verify sorted order
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_variable_sized_data() {
    let mut sorter = ExternalSorter::new(2, 1024 * 1024, 2, 10000, test_dir());

    let mut data = Vec::new();
    for i in 0..1000 {
        // Variable key sizes
        let key = if i % 3 == 0 {
            format!("{:03}", i).into_bytes()
        } else if i % 3 == 1 {
            format!("longer_key_{:06}", i).into_bytes()
        } else {
            format!("very_long_key_with_more_data_{:09}", i).into_bytes()
        };

        // Variable value sizes
        let value = vec![b'x'; (i % 100) + 1];
        data.push((key, value));
    }

    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 1000);

    // Verify sorted order
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_binary_data_with_null_bytes() {
    let mut sorter = ExternalSorter::new(2, 1024 * 1024, 2, 10000, test_dir());

    let mut data = Vec::new();
    // Test with binary data including null bytes
    for i in 0..256 {
        let key = vec![i as u8, 0, 255 - i as u8];
        let value = vec![0, i as u8, 0, 255 - i as u8];
        data.push((key, value));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 256);

    // Verify sorted order
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }

    // Verify first and last entries
    assert_eq!(results[0].0, vec![0, 0, 255]);
    assert_eq!(results[255].0, vec![255, 0, 0]);
}

#[test]
fn test_buffer_boundary_edge_cases() {
    // Test when entries fill the buffer and when single entry is small relative to buffer
    let entry_size = 32;
    let max_memory = entry_size * 100;
    let mut sorter = ExternalSorter::new(1, max_memory, 1, 10000, test_dir());

    let mut data = Vec::new();
    for i in 0..100 {
        let key = format!("k{:02}", i).into_bytes();
        let value = b"v".to_vec();
        data.push((key, value));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 100);

    for i in 0..100 {
        assert_eq!(results[i].0, format!("k{:02}", i).as_bytes());
    }
}

#[test]
#[should_panic]
fn test_entry_too_large_panics() {
    let mut sorter = ExternalSorter::new(1, 50, 1, 10000, test_dir());

    // Create entries where key + value + overhead > buffer size
    let key = vec![b'k'; 100];
    let value = vec![b'v'; 100];

    let data = vec![(key, value)];
    let input = InMemInput { data };

    let _ = sorter.sort(Box::new(input));
}

#[test]
fn test_concurrent_sorters() {
    // Test multiple sorters running concurrently without interference
    let handles: Vec<_> = (0..4)
        .map(|id| {
            thread::spawn(move || {
                let mut sorter = ExternalSorter::new(2, 512, 2, 10000, test_dir());

                let mut data = Vec::new();
                for i in 0..1000 {
                    let key = format!("sorter_{}_key_{:04}", id, i);
                    let value = format!("value_{}", i);
                    data.push((key.into_bytes(), value.into_bytes()));
                }

                let input = InMemInput { data };
                let output = sorter.sort(Box::new(input)).unwrap();
                let results: Vec<_> = output.iter().collect();

                assert_eq!(results.len(), 1000);
                for i in 1..results.len() {
                    assert!(results[i - 1].0 <= results[i].0);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}
