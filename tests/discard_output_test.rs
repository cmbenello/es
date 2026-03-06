use es::{ExternalSorter, InMemInput, Sorter};
use tempfile::TempDir;

#[test]
fn test_discard_final_output() {
    let temp_dir = TempDir::new().unwrap();

    // Create test data
    let mut data = Vec::new();
    for i in (0..1000).rev() {
        let key = format!("key_{:05}", i).into_bytes();
        let value = format!("value_{}", i).into_bytes();
        data.push((key, value));
    }

    // Test with discard_final_output = true
    let mut sorter = ExternalSorter::new(2, 512 * 1024, 2, 10, temp_dir.path());
    sorter.set_discard_final_output(true);

    let input = InMemInput { data: data.clone() };
    let output = sorter.sort(Box::new(input)).unwrap();

    // Output should be empty (discarded)
    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 0, "Output should be empty when discarded");

    // Stats should still be populated
    let stats = output.stats();
    assert!(
        stats.run_gen_stats.num_runs > 0,
        "Run generation should have occurred"
    );
    assert!(
        stats.per_merge_stats.len() > 0,
        "Merge should have occurred"
    );

    println!("✓ Discard mode test passed");
    println!("Stats: {}", stats);
}

#[test]
fn test_normal_output_still_works() {
    let temp_dir = TempDir::new().unwrap();

    // Create test data
    let mut data = Vec::new();
    for i in (0..100).rev() {
        let key = format!("key_{:05}", i).into_bytes();
        let value = format!("value_{}", i).into_bytes();
        data.push((key, value));
    }

    // Test with discard_final_output = false (default)
    let mut sorter = ExternalSorter::new(2, 512 * 1024, 2, 10, temp_dir.path());

    let input = InMemInput { data: data.clone() };
    let output = sorter.sort(Box::new(input)).unwrap();

    // Output should have all records
    let results: Vec<_> = output.iter().collect();
    assert_eq!(
        results.len(),
        100,
        "Output should have all records in normal mode"
    );

    // Verify sorted
    for i in 0..100 {
        let expected_key = format!("key_{:05}", i).into_bytes();
        assert_eq!(results[i].0, expected_key, "Records should be sorted");
    }

    println!("✓ Normal mode test passed");
}
