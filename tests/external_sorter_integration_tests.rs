mod common;
use common::test_dir;

use es::{ExternalSorter, InMemInput, Sorter};
use std::fs;

#[test]
fn test_real_world_log_sorting() {
    // Simulate sorting real-world log entries with timestamps, levels, and components
    let mut sorter = ExternalSorter::new(4, 1024 * 1024, 4, 100000, test_dir());

    let log_levels = ["DEBUG", "INFO", "WARN", "ERROR"];
    let components = ["auth", "api", "db", "cache", "web"];

    let mut data = Vec::new();
    use rand::Rng;
    let mut rng = rand::rng();

    for i in 0..50_000 {
        let timestamp = format!(
            "2024-01-15T{:02}:{:02}:{:02}.{:03}Z",
            i / 3600 % 24,
            i / 60 % 60,
            i % 60,
            i % 1000
        );

        let level = log_levels[rng.random_range(0..log_levels.len())];
        let component = components[rng.random_range(0..components.len())];
        let message = format!("[{}] [{}] Processing request {}", level, component, i);

        // Use timestamp as key for chronological sorting
        data.push((timestamp.into_bytes(), message.into_bytes()));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 50_000);

    // Verify chronological order
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_structured_data_sorting() {
    // Test sorting various structured data formats: JSON, URLs, composite keys
    let mut sorter = ExternalSorter::new(4, 512, 4, 10000, test_dir());

    let mut data = Vec::new();

    // Add JSON-like keys
    for i in 0..100 {
        let key = format!(
            r#"{{"user_id":{},"timestamp":{},"action":"click"}}"#,
            i % 20,
            1000000 + i
        );
        data.push((key.into_bytes(), b"json_data".to_vec()));
    }

    // Add URLs
    let urls = [
        "https://api.example.com/v1",
        "https://blog.example.com/post",
        "http://example.com",
    ];
    for url in &urls {
        for i in 0..50 {
            data.push((format!("{}/{}", url, i).into_bytes(), b"url_data".to_vec()));
        }
    }

    // Add composite keys (timestamp + user_id)
    for hour in 0..10 {
        for user_id in 0..20 {
            let key = format!("2024-01-01T{:02}:00:00_user_{:04}", hour, user_id);
            data.push((key.into_bytes(), b"composite_data".to_vec()));
        }
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();

    // Verify lexicographic ordering across all data types
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_ascii_and_unicode_ordering() {
    // Test ASCII ordering (numbers < uppercase < lowercase) and Unicode handling
    let mut sorter = ExternalSorter::new(2, 512, 2, 10000, test_dir());

    let test_data = vec![
        "Apple",
        "apple",
        "APPLE",
        "123",
        "ABC",
        "abc",
        "Abc",
        "zebra",
        "ñoño",
        "中文",
        "日本語",
        "🦀",
        "Åpple",
        "école",
        "Москва",
    ];

    let mut data = Vec::new();
    for s in &test_data {
        data.push((s.as_bytes().to_vec(), b"value".to_vec()));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();

    // Verify byte-order sorting
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }

    let sorted_keys: Vec<String> = results
        .iter()
        .map(|(k, _)| String::from_utf8_lossy(k).to_string())
        .collect();

    // Verify ASCII ordering: numbers < uppercase < lowercase
    assert!(sorted_keys[0].starts_with(|c: char| c.is_numeric()));
    let apple_pos = sorted_keys.iter().position(|s| s == "Apple").unwrap();
    let apple_lower_pos = sorted_keys.iter().position(|s| s == "apple").unwrap();
    assert!(apple_pos < apple_lower_pos);
}

#[test]
fn test_cleanup_on_drop() {
    // Verify that temporary directories are created during sorting
    let temp_dir_pattern = "external_sort_";
    let temp_dir = std::env::temp_dir();

    let before_count = fs::read_dir(&temp_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(temp_dir_pattern))
        .count();

    {
        let mut sorter = ExternalSorter::new(2, 256, 2, 10000, &temp_dir);
        let mut data = Vec::new();
        for i in 0..1000 {
            data.push((format!("{:04}", i).into_bytes(), b"value".to_vec()));
        }
        let input = InMemInput { data };
        let _output = sorter.sort(Box::new(input)).unwrap();
    }

    let after_count = fs::read_dir(&temp_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(temp_dir_pattern))
        .count();

    assert!(after_count >= before_count);
}
