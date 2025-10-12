use es::{ExternalSorter, GenSortInputDirect, SortInput, Sorter};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;

/// Helper to create a test GenSort file with the given records
fn create_test_gensort_file(path: &Path, records: &[(Vec<u8>, Vec<u8>)]) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    for (key, payload) in records {
        assert_eq!(key.len(), 10, "Key must be exactly 10 bytes");
        assert_eq!(payload.len(), 90, "Payload must be exactly 90 bytes");
        file.write_all(key)?;
        file.write_all(payload)?;
    }

    file.sync_all()?;
    Ok(())
}

#[test]
fn test_gensort_basic_read() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.gensort");

    // Create test data
    let records = vec![
        (b"key0000001".to_vec(), vec![1u8; 90]),
        (b"key0000002".to_vec(), vec![2u8; 90]),
        (b"key0000003".to_vec(), vec![3u8; 90]),
    ];

    create_test_gensort_file(&test_file, &records).unwrap();

    // Test reading
    let input = GenSortInputDirect::new(&test_file).unwrap();
    assert_eq!(input.len(), 3);
    assert!(!input.is_empty());

    // Create single scanner and read all records
    let scanners = input.create_parallel_scanners(1, None);
    assert_eq!(scanners.len(), 1);

    let scanner = scanners.into_iter().next().unwrap();
    let mut read_records = Vec::new();

    for record in scanner {
        read_records.push(record);
    }

    assert_eq!(read_records.len(), 3);
    assert_eq!(read_records[0].0, b"key0000001");
    assert_eq!(read_records[0].1, vec![1u8; 90]);
    assert_eq!(read_records[1].0, b"key0000002");
    assert_eq!(read_records[2].0, b"key0000003");
}

#[test]
fn test_gensort_parallel_scanners() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test_parallel.gensort");

    // Create test data with more records
    let mut records = Vec::new();
    for i in 0..100 {
        let key = format!("key{:07}", i).into_bytes();
        let payload = vec![(i % 256) as u8; 90];
        records.push((key, payload));
    }

    create_test_gensort_file(&test_file, &records).unwrap();

    // Test with multiple scanners
    let input = GenSortInputDirect::new(&test_file).unwrap();
    assert_eq!(input.len(), 100);

    let scanners = input.create_parallel_scanners(4, None);
    assert_eq!(scanners.len(), 4);

    // Collect all records from all scanners
    let mut all_records = Vec::new();
    for scanner in scanners {
        for record in scanner {
            all_records.push(record);
        }
    }

    assert_eq!(all_records.len(), 100);

    // Verify all records are present (order may vary due to parallel scanning)
    all_records.sort_by(|a, b| a.0.cmp(&b.0));
    for i in 0..100 {
        let expected_key = format!("key{:07}", i).into_bytes();
        assert_eq!(all_records[i].0, expected_key);
        assert_eq!(all_records[i].1[0], (i % 256) as u8);
    }
}

#[test]
fn test_gensort_invalid_file_size() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("invalid.gensort");

    // Create a file with invalid size (not multiple of 100)
    let mut file = File::create(&test_file).unwrap();
    file.write_all(&[0u8; 150]).unwrap(); // 1.5 records
    file.sync_all().unwrap();
    drop(file);

    // Should fail to open
    let result = GenSortInputDirect::new(&test_file);
    assert!(result.is_err());
}

#[test]
fn test_gensort_empty_file() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("empty.gensort");

    // Create empty file
    File::create(&test_file).unwrap();

    // Should succeed with 0 records
    let input = GenSortInputDirect::new(&test_file).unwrap();
    assert_eq!(input.len(), 0);
    assert!(input.is_empty());

    let scanners = input.create_parallel_scanners(1, None);
    assert_eq!(scanners.len(), 0);
}

#[test]
fn test_gensort_with_actual_file() {
    // Test with the actual test_gen_sort.dat file if it exists
    let test_file = Path::new("test_gen_sort.dat");
    if !test_file.exists() {
        eprintln!("Skipping test with actual file - test_gen_sort.dat not found");
        return;
    }

    let input = GenSortInputDirect::new(test_file).unwrap();
    println!("test_gen_sort.dat contains {} records", input.len());
    assert_eq!(input.len(), 100); // File should have exactly 100 records

    // Read all records and verify they're valid
    let scanners = input.create_parallel_scanners(1, None);
    let mut count = 0;

    for scanner in scanners {
        for (key, payload) in scanner {
            assert_eq!(key.len(), 10);
            assert_eq!(payload.len(), 90);
            count += 1;
        }
    }

    assert_eq!(count, 100);
}

#[test]
fn test_gensort_sort_integration() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("unsorted.gensort");

    // Create unsorted test data
    let mut records = Vec::new();
    let keys = [
        b"key0000005".to_vec(),
        b"key0000001".to_vec(),
        b"key0000003".to_vec(),
        b"key0000002".to_vec(),
        b"key0000004".to_vec(),
    ];

    for (i, key) in keys.iter().enumerate() {
        records.push((key.clone(), vec![i as u8; 90]));
    }

    create_test_gensort_file(&test_file, &records).unwrap();

    // Sort using ExternalSorter
    let input = GenSortInputDirect::new(&test_file).unwrap();
    let mut sorter = ExternalSorter::new(2, 512, 2, 10000, temp_dir.path());
    let output = sorter.sort(Box::new(input)).unwrap();

    // Verify sorted output
    let mut sorted_keys = Vec::new();
    for (key, _payload) in output.iter() {
        sorted_keys.push(key);
    }

    assert_eq!(sorted_keys.len(), 5);
    assert_eq!(sorted_keys[0], b"key0000001");
    assert_eq!(sorted_keys[1], b"key0000002");
    assert_eq!(sorted_keys[2], b"key0000003");
    assert_eq!(sorted_keys[3], b"key0000004");
    assert_eq!(sorted_keys[4], b"key0000005");
}

#[test]
fn test_gensort_large_sort() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("large.gensort");

    // Create larger dataset with random order
    let mut records = Vec::new();
    let mut keys: Vec<usize> = (0..1000).collect();

    // Shuffle using simple method
    for i in 0..keys.len() {
        let j = (i * 7 + 11) % keys.len();
        keys.swap(i, j);
    }

    for &key_num in &keys {
        let key = format!("key{:07}", key_num).into_bytes();
        let payload = vec![(key_num % 256) as u8; 90];
        records.push((key, payload));
    }

    create_test_gensort_file(&test_file, &records).unwrap();

    // Sort with multiple threads
    let input = GenSortInputDirect::new(&test_file).unwrap();
    let mut sorter = ExternalSorter::new(4, 1024 * 1024, 4, 10000, temp_dir.path());
    let output = sorter.sort(Box::new(input)).unwrap();

    // Verify sorted output
    let mut prev_key: Option<Vec<u8>> = None;
    let mut count = 0;

    for (key, payload) in output.iter() {
        if let Some(ref prev) = prev_key {
            assert!(
                key >= *prev,
                "Sort order violation: {:?} < {:?}",
                std::str::from_utf8(&key).unwrap(),
                std::str::from_utf8(prev).unwrap()
            );
        }

        // Extract key number and verify payload
        let key_str = std::str::from_utf8(&key).unwrap();
        let key_num: usize = key_str[3..].parse().unwrap();
        assert_eq!(payload[0], (key_num % 256) as u8);

        prev_key = Some(key);
        count += 1;
    }

    assert_eq!(count, 1000);

    // Print statistics
    let stats = output.stats();
    println!("Sort statistics:");
    println!("{}", stats);
}
