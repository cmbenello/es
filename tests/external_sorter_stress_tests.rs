use es::{ExternalSorter, InMemInput, Sorter};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use std::path::PathBuf;
use std::sync::Once;

static INIT: Once = Once::new();

fn test_dir() -> PathBuf {
    INIT.call_once(|| {
        let dir = PathBuf::from("./test_runs");
        std::fs::create_dir_all(&dir).expect("Failed to create test directory");
    });
    PathBuf::from("./test_runs")
}

#[test]
#[ignore] // Run with: cargo test --ignored test_very_large_dataset
fn test_very_large_dataset() {
    // Test with 1M entries
    let mut sorter = ExternalSorter::new(8, 16 * 1024 * 1024, 8, 10000, test_dir()); // 16MB buffer

    let mut data = Vec::new();
    for i in 0..1_000_000 {
        let key = format!("{:08}", (i * 999983) % 1_000_000); // Prime to shuffle
        let value = format!("value_{}", i);
        data.push((key.into_bytes(), value.into_bytes()));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 1_000_000);

    // Verify a sample of entries
    for i in (0..1_000_000).step_by(10_000) {
        assert_eq!(results[i].0, format!("{:08}", i).as_bytes());
    }
}

#[test]
fn test_pathological_key_distribution() {
    let mut sorter = ExternalSorter::new(4, 1024 * 1024, 4, 10000, test_dir());

    let mut data = Vec::new();

    // Create a pathological distribution:
    // - Many entries with same prefix
    // - Keys that differ only in last character
    // - Keys designed to cause poor partitioning

    // Group 1: Same long prefix
    for i in 0..1000 {
        let key = format!("aaaaaaaaaaaaaaaaaaaaaaaaaaaa{:04}", i);
        data.push((key.into_bytes(), b"v1".to_vec()));
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
fn test_memory_pressure() {
    // Test behavior under memory pressure with large values
    let mut sorter = ExternalSorter::new(2, 15000, 2, 10000, test_dir()); // 15KB buffer

    let mut data = Vec::new();
    for i in 0..100 {
        let key = format!("{:03}", i);
        // Large values to stress memory
        let value = vec![b'v'; 10_000];
        data.push((key.into_bytes(), value));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 100);

    // Verify sorted and values preserved
    for i in 0..100 {
        assert_eq!(results[i].0, format!("{:03}", i).as_bytes());
        assert_eq!(results[i].1.len(), 10_000);
    }
}

#[test]
fn test_extreme_thread_counts() {
    // Test with various thread counts including edge cases
    for num_threads in [1, 16, 32, 64] {
        let mut sorter =
            ExternalSorter::new(num_threads, 1024 * 1024, num_threads, 10000, test_dir());

        let mut data = Vec::new();
        let entries = 100000;
        for i in 0..entries {
            let key = format!("{:08}", entries - i - 1); // Reverse order to ensure sorting
            data.push((key.into_bytes(), b"value".to_vec()));
        }

        let input = InMemInput { data };
        let output = sorter.sort(Box::new(input)).unwrap();

        let results: Vec<_> = output.iter().collect();
        assert_eq!(results.len(), entries);

        // Verify sorted
        for i in 0..entries {
            assert_eq!(results[i].0, format!("{:08}", i).as_bytes());
        }
    }
}

#[test]
fn test_max_memory_edge_cases() {
    // Test various buffer sizes including very small and exact multiples
    let max_memories = [1024, 4096, 8192];

    for &max_memory in &max_memories {
        let mut sorter = ExternalSorter::new(2, max_memory / 2, 2, 10000, test_dir());

        let mut data = Vec::new();
        for i in 0..500 {
            let key = format!("{:04}", i);
            let value = format!("val_{}", i);
            data.push((key.into_bytes(), value.into_bytes()));
        }

        let input = InMemInput { data };
        let output = sorter.sort(Box::new(input)).unwrap();

        let results: Vec<_> = output.iter().collect();
        assert_eq!(results.len(), 500);

        // Just verify it's sorted
        for i in 1..results.len() {
            assert!(results[i - 1].0 <= results[i].0);
        }
    }
}

#[test]
fn test_concurrent_massive_sorts() {
    // Multiple threads doing large sorts simultaneously
    let num_concurrent = 4;
    let entries_per_sort = 10_000;

    let handles: Vec<_> = (0..num_concurrent)
        .map(|thread_id| {
            thread::spawn(move || {
                let mut sorter = ExternalSorter::new(4, 1024 * 1024, 4, 10000, test_dir());

                let mut data = Vec::new();
                for i in 0..entries_per_sort {
                    // Create unique keys per thread
                    let key = format!("t{:02}_{:06}", thread_id, i);
                    let value = vec![b'x'; 100];
                    data.push((key.into_bytes(), value));
                }

                // Shuffle
                use rand::seq::SliceRandom;
                let mut rng = rand::rng();
                data.shuffle(&mut rng);

                let input = InMemInput { data };
                let output = sorter.sort(Box::new(input)).unwrap();

                let results: Vec<_> = output.iter().collect();
                assert_eq!(results.len(), entries_per_sort);

                // Verify sorted
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

#[test]
fn test_all_identical_keys() {
    // Test when all keys are identical
    let mut sorter = ExternalSorter::new(2, 512, 2, 10000, test_dir());

    let mut data = Vec::new();
    for i in 0..10_000 {
        data.push((b"same".to_vec(), i.to_string().into_bytes()));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 10_000);

    // All keys should be "same"
    for (key, _) in &results {
        assert_eq!(key, b"same");
    }
}

#[test]
fn test_alternating_small_large() {
    // Alternate between very small and very large entries
    let mut sorter = ExternalSorter::new(2, 2048, 2, 10000, test_dir());

    let mut data = Vec::new();
    for i in 0..1000 {
        if i % 2 == 0 {
            // Small entry
            let key = format!("{:04}", i);
            data.push((key.into_bytes(), b"s".to_vec()));
        } else {
            // Large entry
            let key = format!("{:04}", i);
            let value = vec![b'L'; 5000];
            data.push((key.into_bytes(), value));
        }
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 1000);

    // Verify sorted and sizes preserved
    for i in 0..1000 {
        assert_eq!(results[i].0, format!("{:04}", i).as_bytes());
        if i % 2 == 0 {
            assert_eq!(results[i].1.len(), 1);
        } else {
            assert_eq!(results[i].1.len(), 5000);
        }
    }
}

#[test]
fn test_random_binary_keys() {
    use rand::Rng;
    let mut rng = rand::rng();

    let mut sorter = ExternalSorter::new(4, 1024 * 1024, 4, 10000, test_dir());

    let mut data = Vec::new();
    for _ in 0..5000 {
        // Random binary keys of varying lengths
        let key_len = rng.random_range(1..50);
        let key: Vec<u8> = (0..key_len).map(|_| rng.random()).collect();

        let value_len = rng.random_range(1..100);
        let value: Vec<u8> = (0..value_len).map(|_| rng.random()).collect();

        data.push((key, value));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 5000);

    // Verify sorted
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_progressive_key_lengths() {
    // Keys that progressively get longer
    let mut sorter = ExternalSorter::new(2, 512, 2, 10000, test_dir());

    let mut data = Vec::new();
    for i in 0..100 {
        // Reduce to 100 to keep key sizes reasonable
        let key = "a".repeat(i + 1);
        let value = format!("value_{}", i);
        data.push((key.into_bytes(), value.into_bytes()));
    }

    // Shuffle
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 100);

    // Verify sorted by length (since all are 'a's)
    for i in 0..100 {
        assert_eq!(results[i].0.len(), i + 1);
    }
}

#[test]
fn test_interleaved_runs() {
    // Force creation of many runs with interleaved data
    let mut sorter = ExternalSorter::new(2, 512, 2, 10000, test_dir());

    let mut data = Vec::new();
    // Create data that will form many runs
    for round in 0..10 {
        for i in 0..100 {
            let key = format!("{:02}{:03}", i, round);
            data.push((key.into_bytes(), b"value".to_vec()));
        }
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 1000);

    // Verify sorted
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_stress_file_handles() {
    // Create many concurrent sorts to stress file handle limits
    let num_concurrent = 20;
    let counter = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_concurrent)
        .map(|_| {
            let counter = Arc::clone(&counter);
            thread::spawn(move || {
                let id = counter.fetch_add(1, Ordering::SeqCst);
                let mut sorter = ExternalSorter::new(4, 1024 * 1024, 4, 10000, test_dir());

                let mut data = Vec::new();
                for i in 0..1000 {
                    let key = format!("sort{:02}_key{:04}", id, i);
                    data.push((key.into_bytes(), b"val".to_vec()));
                }

                let input = InMemInput { data };
                let output = sorter.sort(Box::new(input)).unwrap();
                let results: Vec<_> = output.iter().collect();
                assert_eq!(results.len(), 1000);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

// Removed test_zero_threads_fallback as the implementation requires at least 1 thread

#[test]
fn test_max_value_sizes() {
    // Test with very large values
    let mut sorter = ExternalSorter::new(2, 10 * 1024 * 1024, 2, 10000, test_dir()); // 10MB buffer

    let mut data = Vec::new();
    for i in 0..10 {
        let key = format!("{:02}", i);
        // 1MB value
        let value = vec![b'v'; 1_000_000];
        data.push((key.into_bytes(), value));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 10);

    // Verify large values preserved
    for i in 0..10 {
        assert_eq!(results[i].0, format!("{:02}", i).as_bytes());
        assert_eq!(results[i].1.len(), 1_000_000);
    }
}
