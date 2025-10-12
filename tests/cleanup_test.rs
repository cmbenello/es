use es::{ExternalSorter, InMemInput, Sorter};
use std::fs;
use std::path::PathBuf;

fn test_dir() -> PathBuf {
    PathBuf::from("./test_cleanup")
}

#[test]
fn test_external_sorter_cleanup() {
    // Create test directory
    let base_dir = test_dir();
    fs::create_dir_all(&base_dir).unwrap();

    // Count directories before
    let before = count_subdirs(&base_dir);

    // Run a sort that creates temp files
    {
        let mut sorter = ExternalSorter::new(2, 512, 2, 10000, &base_dir);

        let mut data = Vec::new();
        for i in 0..1000 {
            let key = format!("{:04}", i);
            let value = vec![b'v'; 100]; // Large enough to create multiple runs
            data.push((key.into_bytes(), value));
        }

        let input = InMemInput { data };
        let output = sorter.sort(Box::new(input)).unwrap();

        // Consume output to ensure everything is processed
        let results: Vec<_> = output.iter().collect();
        assert_eq!(results.len(), 1000);

        // While sorter is alive, temp directory should exist
        let during = count_subdirs(&base_dir);
        assert!(
            during > before,
            "Temp directory should be created during sort"
        );
    } // sorter dropped here

    // After sorter is dropped, temp directory should be cleaned up
    let after = count_subdirs(&base_dir);
    assert_eq!(
        after, before,
        "Temp directory should be cleaned up after drop"
    );

    // Clean up test directory
    let _ = fs::remove_dir_all(&base_dir);
}

fn count_subdirs(dir: &PathBuf) -> usize {
    match fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .count(),
        Err(_) => 0,
    }
}
