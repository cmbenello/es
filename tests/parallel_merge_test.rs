use es::{ExternalSorter, InMemInput, Sorter};
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
fn test_parallel_merge_with_multiple_threads() {
    // Create a scenario that guarantees multiple runs and parallel merge
    let mut sorter = ExternalSorter::new(4, 512, 4, 10000, test_dir());

    // Generate enough data to create many runs
    let mut data = Vec::new();
    for i in 0..10_000 {
        // Create keys that will distribute well across partitions
        let key = format!("{:08}", i);
        let value = format!("value_{}", i);
        data.push((key.into_bytes(), value.into_bytes()));
    }

    // Shuffle to ensure runs contain mixed data
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    println!(
        "Sorting {} entries with 4 threads and 512 byte buffer",
        data.len()
    );

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 10_000);

    // Verify sorted order
    for i in 1..results.len() {
        assert!(
            results[i - 1].0 <= results[i].0,
            "Results not sorted at position {}",
            i
        );
    }

    // Verify all entries are present
    for i in 0..10_000 {
        let expected_key = format!("{:08}", i);
        assert_eq!(results[i].0, expected_key.as_bytes());
    }
}

#[test]
fn test_parallel_merge_boundary_conditions() {
    // Test with exact partition boundaries
    let mut sorter = ExternalSorter::new(3, 1024, 3, 10000, test_dir());

    let mut data = Vec::new();

    // Create data that will partition nicely into 3 groups
    // Group 1: keys starting with 'a'
    for i in 0..1000 {
        let key = format!("a{:06}", i);
        data.push((key.into_bytes(), b"group1".to_vec()));
    }

    // Group 2: keys starting with 'b'
    for i in 0..1000 {
        let key = format!("b{:06}", i);
        data.push((key.into_bytes(), b"group2".to_vec()));
    }

    // Group 3: keys starting with 'c'
    for i in 0..1000 {
        let key = format!("c{:06}", i);
        data.push((key.into_bytes(), b"group3".to_vec()));
    }

    // Shuffle within groups to create multiple runs
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 3000);

    // Verify sorted order
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }

    // Verify grouping
    assert!(results[999].0 < results[1000].0); // Last 'a' < first 'b'
    assert!(results[1999].0 < results[2000].0); // Last 'b' < first 'c'
}

#[test]
fn test_parallel_merge_performance() {
    use std::time::Instant;

    // Compare single-threaded vs multi-threaded merge
    let sizes = vec![5000, 10000];

    for size in sizes {
        // Single thread
        let mut sorter1 = ExternalSorter::new(1, 2048, 1, 10000, test_dir());
        let mut data1 = Vec::new();
        for i in 0..size {
            let key = format!("{:08}", (i * 7919) % size);
            data1.push((key.into_bytes(), b"value".to_vec()));
        }

        let start1 = Instant::now();
        let _ = sorter1.sort(Box::new(InMemInput { data: data1 })).unwrap();
        let time1 = start1.elapsed();

        // Multiple threads
        let mut sorter4 = ExternalSorter::new(4, 2048, 4, 10000, test_dir());
        let mut data4 = Vec::new();
        for i in 0..size {
            let key = format!("{:08}", (i * 7919) % size);
            data4.push((key.into_bytes(), b"value".to_vec()));
        }

        let start4 = Instant::now();
        let _ = sorter4.sort(Box::new(InMemInput { data: data4 })).unwrap();
        let time4 = start4.elapsed();

        println!(
            "Size {}: 1 thread = {:?}, 4 threads = {:?}",
            size, time1, time4
        );

        // Multi-threaded should not be significantly slower
        // (might not be faster due to overhead on small datasets)
        // assert!(time4.as_millis() < time1.as_millis() * 2,
        //     "Multi-threaded sort should not be much slower than single-threaded");
    }
}
