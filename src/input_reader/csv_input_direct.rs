//! CSV Input implementation using Direct I/O with BufReader

use crate::diskio::aligned_reader::AlignedReader;
use crate::diskio::file::SharedFd;
use crate::diskio::file::file_size_fd;
use crate::{IoStatsTracker, SortInput, order_preserving_encoding::*};
use arrow::datatypes::{DataType, Schema};
use chrono::Datelike;
use std::io::Seek;
use std::path::Path;
use std::sync::Arc;

// Convert CSV string values to bytes based on schema
fn convert_csv_value_to_bytes(value: &str, data_type: &DataType) -> Result<Vec<u8>, String> {
    match data_type {
        DataType::Utf8 | DataType::LargeUtf8 => Ok(value.as_bytes().to_vec()),
        DataType::Int32 => {
            let parsed = value
                .parse::<i32>()
                .map_err(|e| format!("Failed to parse '{}' as Int32: {}", value, e))?;
            Ok(i32_to_order_preserving_bytes(parsed).to_vec())
        }
        DataType::Int64 => {
            let parsed = value
                .parse::<i64>()
                .map_err(|e| format!("Failed to parse '{}' as Int64: {}", value, e))?;
            Ok(i64_to_order_preserving_bytes(parsed).to_vec())
        }
        DataType::Float32 => {
            let parsed = value
                .parse::<f32>()
                .map_err(|e| format!("Failed to parse '{}' as Float32: {}", value, e))?;
            Ok(f32_to_order_preserving_bytes(parsed).to_vec())
        }
        DataType::Float64 => {
            let parsed = value
                .parse::<f64>()
                .map_err(|e| format!("Failed to parse '{}' as Float64: {}", value, e))?;
            Ok(f64_to_order_preserving_bytes(parsed).to_vec())
        }
        DataType::Date32 => {
            // yyyy-MM-dd format
            let parsed = chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d")
                .map_err(|e| format!("Failed to parse '{}' as Date32: {}", value, e))?;
            Ok(i32_to_order_preserving_bytes(parsed.num_days_from_ce()).to_vec())
        }
        DataType::Binary | DataType::LargeBinary => {
            // For binary, we might expect hex-encoded strings
            // For now, just use the raw bytes
            Ok(value.as_bytes().to_vec())
        }
        _ => Err(format!("Unsupported data type for CSV: {:?}", data_type)),
    }
}

/// Configuration for CSV parsing with direct I/O
#[derive(Clone)]
pub struct CsvDirectConfig {
    /// Delimiter character (default: ',')
    pub delimiter: u8,
    /// Column indices to use as key (0-based)
    pub key_columns: Vec<usize>,
    /// Column indices to use as value (0-based)
    pub value_columns: Vec<usize>,
    /// Whether the CSV has headers
    pub has_headers: bool,
    /// Schema defining column types (required)
    pub schema: Arc<Schema>,
}

impl CsvDirectConfig {
    /// Create a new config with schema
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            delimiter: b',',
            key_columns: vec![0],
            value_columns: vec![1],
            has_headers: true,
            schema,
        }
    }
}

/// CSV input reader using Direct I/O
pub struct CsvInputDirect {
    fd: Arc<SharedFd>,
    config: CsvDirectConfig,
    file_size: u64,
}

impl CsvInputDirect {
    /// Create a new CSV input reader using Direct I/O with GlobalFileManager
    pub fn new(path: impl AsRef<Path>, config: CsvDirectConfig) -> Result<Self, String> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(format!("File does not exist: {:?}", path));
        }

        if config.key_columns.is_empty() {
            return Err("At least one key column must be specified".to_string());
        }

        if config.value_columns.is_empty() {
            return Err("At least one value column must be specified".to_string());
        }

        let fd = Arc::new(SharedFd::new_from_path(path, false).map_err(|e| {
            format!(
                "Failed to open file with Direct I/O: {}: {}",
                path.display(),
                e
            )
        })?);

        let file_size = file_size_fd(fd.as_raw_fd())
            .map_err(|e| format!("Failed to get file size: {}: {}", path.display(), e))?;

        if file_size == 0 {
            return Err("CSV file is empty".to_string());
        }

        Ok(Self {
            fd,
            config,
            file_size,
        })
    }
}

impl SortInput for CsvInputDirect {
    fn create_parallel_scanners(
        &self,
        num_scanners: usize,
        io_tracker: Option<IoStatsTracker>,
    ) -> Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>> {
        if num_scanners == 0 || self.file_size == 0 {
            return vec![];
        }

        let bytes_per_scanner = self.file_size.div_ceil(num_scanners as u64);
        let mut scanners = Vec::new();

        for i in 0..num_scanners {
            let start_byte = i as u64 * bytes_per_scanner;
            let end_byte = if i == num_scanners - 1 {
                u64::MAX // Last partition reads to EOF
            } else {
                ((i + 1) as u64 * bytes_per_scanner).min(self.file_size)
            };

            if start_byte >= self.file_size {
                break;
            }

            let scanner = CsvPartitionDirect {
                fd: self.fd.clone(),
                config: self.config.clone(),
                start_byte,
                end_byte,
                skip_first_partial: i > 0,
                initialized: false,
                reader: None,
                line_buffer: String::new(),
                io_stats: io_tracker.clone(),
            };

            scanners.push(Box::new(scanner) as Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>);
        }

        scanners
    }
}

// ReaderType enum removed - now only using ManagedAlignedReader

/// Partition iterator for CSV files using Direct I/O
struct CsvPartitionDirect {
    fd: Arc<SharedFd>,
    config: CsvDirectConfig,
    start_byte: u64,
    end_byte: u64,
    skip_first_partial: bool,
    initialized: bool,
    reader: Option<AlignedReader>,
    line_buffer: String,
    io_stats: Option<IoStatsTracker>,
}

impl Iterator for CsvPartitionDirect {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        // println!("[DEBUG] CsvPartitionDirect::next() called");

        // Initialize reader on first call
        if !self.initialized {
            // println!("[DEBUG] Initializing CSV partition");
            if let Err(e) = self.initialize() {
                eprintln!("Failed to initialize CSV partition: {}", e);
                return None;
            }
        }

        // Loop instead of recursion to avoid stack overflow
        loop {
            // Read next line
            self.line_buffer.clear();
            // println!("[DEBUG] Reading line from CSV");

            let reader = self.reader.as_mut()?;
            let bytes_read = match reader.read_line(&mut self.line_buffer) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("Failed to read line: {}", e);
                    return None;
                }
            };

            if bytes_read == 0 {
                // println!("[DEBUG] EOF reached");
                return None;
            }

            // println!("[DEBUG] Read {} bytes, line length: {}", bytes_read, self.line_buffer.len());

            // Check if we should stop reading
            if self.end_byte < u64::MAX {
                let reader = self.reader.as_mut()?;
                match reader.stream_position() {
                    Ok(pos) => {
                        // Calculate where this line started
                        let line_start = pos - self.line_buffer.len() as u64;

                        // If this line started after our boundary, stop
                        // This ensures lines starting exactly at the boundary are read by this partition
                        if line_start > self.end_byte {
                            return None;
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to get stream position: {}", e);
                        return None;
                    }
                }
            }

            // Parse CSV line
            // println!("[DEBUG] Parsing CSV line");
            match self.parse_line(&self.line_buffer) {
                Some((key, value)) => {
                    // println!("[DEBUG] Successfully parsed line, key len: {}, value len: {}", key.len(), value.len());
                    return Some((key, value));
                }
                None => {
                    // println!("[DEBUG] Failed to parse line, continuing to next line");
                    // Continue loop instead of recursive call
                    continue;
                }
            }
        }
    }
}

// No need for Drop implementation since we're tracking stats in real-time

impl CsvPartitionDirect {
    fn initialize(&mut self) -> Result<(), String> {
        self.initialized = true;

        // Always use ManagedAlignedReader with the required file_manager
        // Use the new constructor that handles start_byte properly
        let mut managed_reader = AlignedReader::from_fd_with_start_position(
            self.fd.clone(),
            self.start_byte,
            self.io_stats.clone(),
        )
        .map_err(|e| e.to_string())?;

        // Skip to next complete line if not the first partition
        if self.skip_first_partial && self.start_byte > 0 {
            managed_reader
                .skip_to_newline()
                .map_err(|e| format!("Failed to skip to newline: {}", e))?;
        }

        // Skip headers if this is the first partition
        if self.config.has_headers && self.start_byte == 0 {
            managed_reader
                .skip_to_newline()
                .map_err(|e| format!("Failed to skip header line: {}", e))?;
        }

        self.reader = Some(managed_reader);

        Ok(())
    }

    fn parse_line(&self, line: &str) -> Option<(Vec<u8>, Vec<u8>)> {
        // println!("[DEBUG] parse_line called with line length: {}", line.len());
        let line = line.trim_end_matches('\n').trim_end_matches('\r');
        if line.is_empty() {
            // println!("[DEBUG] Line is empty after trimming");
            return None;
        }

        // println!("[DEBUG] Calling parse_csv_line");
        let fields = parse_csv_line(line, self.config.delimiter);
        // println!("[DEBUG] parse_csv_line returned {} fields", fields.len());

        let schema_fields = self.config.schema.fields();

        // Build key from key columns
        let mut key = Vec::new();
        for (i, &col_idx) in self.config.key_columns.iter().enumerate() {
            if i > 0 {
                key.push(0); // Null separator
            }

            let field_value = fields.get(col_idx)?;
            let field_type = schema_fields.get(col_idx)?.data_type();

            match convert_csv_value_to_bytes(field_value, field_type) {
                Ok(bytes) => key.extend(bytes),
                Err(e) => {
                    eprintln!("Failed to convert key column {}: {}", col_idx, e);
                    return None;
                }
            }
        }

        // Build value from value columns
        let mut value = Vec::new();
        for (i, &col_idx) in self.config.value_columns.iter().enumerate() {
            if i > 0 {
                value.push(0); // Null separator
            }

            let field_value = fields.get(col_idx)?;
            let field_type = schema_fields.get(col_idx)?.data_type();

            match convert_csv_value_to_bytes(field_value, field_type) {
                Ok(bytes) => value.extend(bytes),
                Err(e) => {
                    eprintln!("Failed to convert value column {}: {}", col_idx, e);
                    return None;
                }
            }
        }

        Some((key, value))
    }
}

/// Parse a CSV line handling quoted fields
fn parse_csv_line(line: &str, delimiter: u8) -> Vec<String> {
    // println!("[DEBUG] parse_csv_line called with line length: {}", line.len());
    // if line.len() > 10000 {
    //     println!("[DEBUG] WARNING: Very long line detected: {} chars", line.len());
    // }
    let mut fields = Vec::new();
    let mut current_field = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes {
                    // Check if this is an escaped quote
                    if chars.peek() == Some(&'"') {
                        current_field.push('"');
                        chars.next(); // consume the second quote
                    } else {
                        in_quotes = false;
                    }
                } else {
                    in_quotes = true;
                }
            }
            c if c == delimiter as char && !in_quotes => {
                fields.push(current_field.clone());
                current_field.clear();
            }
            c => {
                current_field.push(c);
            }
        }
    }

    // Add the last field
    fields.push(current_field);

    // println!("[DEBUG] parse_csv_line completed with {} fields", fields.len());
    // if fields.len() > 100 {
    //     println!("[DEBUG] WARNING: Large number of fields: {}", fields.len());
    // }

    fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sorter;
    use crate::sort::sorter::ExternalSorter;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::Path;
    use std::sync::Arc;

    fn create_csv_file(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut file = File::create(path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn test_csv_byte_partitioning() {
        let dir = std::env::temp_dir().join("csv_byte_partition_tests");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("large_csv.csv");

        // Create a CSV with known byte boundaries
        let mut content = String::from("id,value\n");
        for i in 0..20 {
            // Each data line is exactly 10 bytes: "XX,val_XX\n"
            content.push_str(&format!("{:02},val_{:02}\n", i, i));
        }
        create_csv_file(&path, &content);

        // File should be 9 (header) + 20*10 = 209 bytes
        let file_size = fs::metadata(&path).unwrap().len();
        assert_eq!(file_size, 209);

        // Test with 3 partitions - should split at ~70 bytes each
        // Create schema for test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let config = CsvDirectConfig::new(schema);
        let csv_input = CsvInputDirect::new(&path, config).unwrap();
        let partitions = csv_input.create_parallel_scanners(3, None);
        assert_eq!(partitions.len(), 3);

        // Collect all entries from all partitions
        let mut all_entries = Vec::new();
        for (i, partition) in partitions.into_iter().enumerate() {
            let entries: Vec<_> = partition.collect();
            println!("Partition {} has {} entries", i, entries.len());
            all_entries.extend(entries);
        }

        // Should have all 20 entries
        assert_eq!(all_entries.len(), 20);

        // Verify all IDs are present
        for i in 0..20 {
            let expected_key = format!("{:02}", i);
            assert!(
                all_entries
                    .iter()
                    .any(|(k, _)| k == expected_key.as_bytes()),
                "Missing key: {}",
                expected_key
            );
        }

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_csv_partial_line_handling() {
        let dir = std::env::temp_dir().join("csv_partial_line_tests");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("partial_lines.csv");

        // Create a CSV where partition boundaries will fall in the middle of lines
        let content = "name,description\n\
                       Alice,Short description\n\
                       Bob,This is a much longer description that will span across partition boundaries\n\
                       Charlie,Another short one\n\
                       David,Yet another long description to test partial line handling across multiple partitions\n\
                       Eve,Final entry\n";
        create_csv_file(&path, content);

        // Create schema for test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("description", DataType::Utf8, false),
        ]));
        let config = CsvDirectConfig::new(schema);
        let csv_input = CsvInputDirect::new(&path, config).unwrap();
        let partitions = csv_input.create_parallel_scanners(4, None);

        // Collect all entries
        let mut all_entries = Vec::new();
        for partition in partitions {
            let entries: Vec<_> = partition.collect();
            all_entries.extend(entries);
        }

        // Should have exactly 5 entries
        assert_eq!(all_entries.len(), 5);

        // Verify all names are present
        let names = vec!["Alice", "Bob", "Charlie", "David", "Eve"];
        for name in names {
            assert!(
                all_entries.iter().any(|(k, _)| k == name.as_bytes()),
                "Missing name: {}",
                name
            );
        }

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_csv_with_external_sorter() {
        let dir = std::env::temp_dir().join("csv_sorter_tests");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sort_test.csv");

        // Create a CSV with data in reverse order
        let mut content = String::from("key,value\n");
        for i in (0..50).rev() {
            content.push_str(&format!("key_{:03},value_{}\n", i, i));
        }
        create_csv_file(&path, &content);

        // Create schema for test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let config = CsvDirectConfig::new(schema);
        let csv_input = CsvInputDirect::new(&path, config).unwrap();

        // Sort using external sorter with multiple threads
        let mut sorter = ExternalSorter::new(4, 256, 4, 10000, dir.as_path());
        let output = sorter.sort(Box::new(csv_input)).unwrap();

        let results: Vec<_> = output.iter().collect();
        assert_eq!(results.len(), 50);

        // Verify sorted order
        for i in 0..50 {
            let expected_key = format!("key_{:03}", i);
            assert_eq!(results[i].0, expected_key.as_bytes());
        }

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_csv_empty_fields() {
        let dir = std::env::temp_dir().join("csv_empty_fields_tests");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_fields.csv");

        // CSV with some empty fields
        let content = "key,value\n\
                       ,empty_key\n\
                       normal,\n\
                       key1,value1\n\
                       ,\n";
        create_csv_file(&path, content);

        // Create schema for test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let config = CsvDirectConfig::new(schema);
        let csv_input = CsvInputDirect::new(&path, config).unwrap();
        let partitions = csv_input.create_parallel_scanners(1, None);

        let entries: Vec<_> = partitions.into_iter().flatten().collect();
        assert_eq!(entries.len(), 4);

        // Check empty field handling
        assert_eq!(entries[0], (b"".to_vec(), b"empty_key".to_vec()));
        assert_eq!(entries[1], (b"normal".to_vec(), b"".to_vec()));
        assert_eq!(entries[2], (b"key1".to_vec(), b"value1".to_vec()));
        assert_eq!(entries[3], (b"".to_vec(), b"".to_vec()));

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_csv_custom_delimiter() {
        let dir = std::env::temp_dir().join("csv_delimiter_tests");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pipe_delimited.csv");

        // Create a pipe-delimited file
        let content = "name|age|city\n\
                       Alice|30|New York\n\
                       Bob|25|San Francisco\n\
                       Charlie|35|Boston\n";
        create_csv_file(&path, content);

        // Create schema for pipe-delimited data
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Utf8, false),
            Field::new("city", DataType::Utf8, false),
        ]));
        let mut config = CsvDirectConfig::new(schema);
        config.delimiter = b'|';
        config.key_columns = vec![0]; // name
        config.value_columns = vec![2]; // city
        config.has_headers = true;

        let csv_input = CsvInputDirect::new(&path, config).unwrap();
        let partitions = csv_input.create_parallel_scanners(2, None);

        let mut all_entries = Vec::new();
        for partition in partitions {
            all_entries.extend(partition.collect::<Vec<_>>());
        }

        assert_eq!(all_entries.len(), 3);
        assert_eq!(all_entries[0], (b"Alice".to_vec(), b"New York".to_vec()));
        assert_eq!(all_entries[1], (b"Bob".to_vec(), b"San Francisco".to_vec()));
        assert_eq!(all_entries[2], (b"Charlie".to_vec(), b"Boston".to_vec()));

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_direct_io_alignment() {
        // Test that ManagedAlignedReader handles non-aligned start positions correctly
        let dir = std::env::temp_dir().join("direct_io_alignment_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("alignment_test.csv");

        // Create content that will force non-aligned reads
        let mut content = String::from("header1,header2\n");
        for i in 0..100 {
            content.push_str(&format!("row{},value{}\n", i, i));
        }
        create_csv_file(&path, &content);

        // Test with multiple partitions to ensure alignment handling works
        // Create schema for test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("header1", DataType::Utf8, false),
            Field::new("header2", DataType::Utf8, false),
        ]));
        let config = CsvDirectConfig::new(schema);
        let csv_input = CsvInputDirect::new(&path, config).unwrap();
        let partitions = csv_input.create_parallel_scanners(5, None);

        let mut total_count = 0;
        for partition in partitions {
            total_count += partition.count();
        }

        // Should have 100 data rows (header is skipped)
        assert_eq!(total_count, 100);

        fs::remove_file(&path).unwrap();
    }
}
