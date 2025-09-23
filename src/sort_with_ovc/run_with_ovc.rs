use crate::diskio::aligned_reader::AlignedReader;
use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::constants::{PAGE_SIZE, align_down};
use crate::diskio::file::SharedFd;
use crate::diskio::io_stats::IoStatsTracker;
use crate::ovc::offset_value_coding_u64::OVCU64;
use std::io::{Read, Write};
use std::sync::Arc;

// Sparse index entry
#[derive(Debug, Clone)]
pub struct IndexEntry {
    pub key: Vec<u8>,
    pub file_offset: usize,
    pub entry_number: usize,
}

// File-based run implementation with direct I/O
pub struct RunWithOVC {
    fd: Arc<SharedFd>,
    writer: Option<AlignedWriter>,
    total_entries: usize,
    start_bytes: usize,
    total_bytes: usize,
    sparse_index: Vec<IndexEntry>,
    sampling_interval: usize,
    entries_seen: usize,
}

impl RunWithOVC {
    pub fn from_writer(writer: AlignedWriter) -> Result<Self, String> {
        Self::from_writer_with_sampling_interval(writer, 1000)
    }

    pub fn from_writer_with_sampling_interval(
        writer: AlignedWriter,
        sampling_interval: usize,
    ) -> Result<Self, String> {
        // Get current position in the file
        let start_bytes = writer.position() as usize;
        let fd = writer.get_fd();

        Ok(Self {
            fd,
            writer: Some(writer),
            total_entries: 0,
            start_bytes,
            total_bytes: 0,
            sparse_index: Vec::new(),
            sampling_interval,
            entries_seen: 0,
        })
    }

    pub fn finalize_write(&mut self) -> AlignedWriter {
        // Sort sparse index by file offset before finalizing
        self.sparse_index
            .sort_unstable_by_key(|entry| entry.file_offset);
        self.writer.take().unwrap()
    }

    pub fn byte_range(&self) -> (usize, usize) {
        (self.start_bytes, self.start_bytes + self.total_bytes)
    }

    pub fn total_entries(&self) -> usize {
        self.total_entries
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    fn find_start_position(&self, lower_bound: &[u8]) -> Option<usize> {
        if self.sparse_index.is_empty() || lower_bound.is_empty() {
            return None;
        }

        // Binary search to find the last entry with key < lower_bound
        let mut left = 0;
        let mut right = self.sparse_index.len();
        let mut best_pos = None;

        while left < right {
            let mid = left + (right - left) / 2;
            if &self.sparse_index[mid].key[..] < lower_bound {
                best_pos = Some(self.sparse_index[mid].file_offset);
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        best_pos
    }
}

impl RunWithOVC {
    pub fn append(&mut self, ovc: OVCU64, key: Vec<u8>, value: Vec<u8>) {
        let writer = self
            .writer
            .as_mut()
            .expect("RunWithOVC is not initialized with a writer");

        // Use sampling interval for sparse index
        if self.entries_seen % self.sampling_interval == 0 {
            let index_entry = IndexEntry {
                key: key.clone(),
                file_offset: self.total_bytes,
                entry_number: self.total_entries,
            };
            self.sparse_index.push(index_entry);
        }
        self.entries_seen += 1;

        let key_len = key.len() as u32;
        let value_len = value.len() as u32;

        // Write entry directly to DirectFileWriter (it handles buffering)
        writer.write_all(&ovc.to_le_bytes()).unwrap();
        writer.write_all(&key_len.to_le_bytes()).unwrap();
        writer.write_all(&value_len.to_le_bytes()).unwrap();
        writer.write_all(&key).unwrap();
        writer.write_all(&value).unwrap();

        self.total_bytes += 8 + 8 + key.len() + value.len();
        self.total_entries += 1;
    }

    pub fn scan_range(
        &self,
        lower_inc: &[u8],
        upper_exc: &[u8],
    ) -> Box<dyn Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)> + Send> {
        self.scan_range_with_io_tracker(lower_inc, upper_exc, None)
    }

    pub fn scan_range_with_io_tracker(
        &self,
        lower_inc: &[u8],
        upper_exc: &[u8],
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)> + Send> {
        // Handle empty runs (no entries written)
        if self.total_entries == 0 {
            return Box::new(std::iter::empty());
        }

        // Open for reading with direct I/O, optionally with tracker
        let mut reader = if let Some(tracker) = io_tracker {
            AlignedReader::from_fd_with_tracer(self.fd.clone(), Some(tracker)).unwrap()
        } else {
            AlignedReader::from_fd(self.fd.clone()).unwrap()
        };

        // Use sparse index to seek to a good starting position
        // Since runs contain sorted data, we can safely skip ahead
        let mut start_offset = self.start_bytes;
        if let Some(offset) = self.find_start_position(lower_inc) {
            // Use the offset from sparse index if we have a lower bound
            // offset is relative to start_bytes
            start_offset = self.start_bytes + offset;
        }

        // Seek to the start position if needed
        if start_offset > 0 {
            // Align to page boundary for direct I/O
            let aligned_offset = align_down(start_offset as u64, PAGE_SIZE as u64) as usize;
            let skip_bytes = start_offset - aligned_offset;

            // Seek to aligned position
            if aligned_offset > 0 {
                reader.seek(aligned_offset as u64).unwrap();
            }

            // We'll need to skip the first few bytes after seeking
            return Box::new(RunIteratorWithOVC {
                reader,
                lower_bound: lower_inc.to_vec(),
                upper_bound: upper_exc.to_vec(),
                bytes_read: aligned_offset,
                total_bytes: self.total_bytes,
                skip_bytes,
                actual_start: self.start_bytes,
            }) as Box<dyn Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)> + Send>;
        }

        Box::new(RunIteratorWithOVC {
            reader,
            lower_bound: lower_inc.to_vec(),
            upper_bound: upper_exc.to_vec(),
            bytes_read: self.start_bytes, // Start from the beginning of this run
            total_bytes: self.total_bytes,
            skip_bytes: 0,
            actual_start: self.start_bytes,
        }) as Box<dyn Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)> + Send>
    }
}

struct RunIteratorWithOVC {
    reader: AlignedReader,
    lower_bound: Vec<u8>,
    upper_bound: Vec<u8>,
    bytes_read: usize,
    total_bytes: usize,
    skip_bytes: usize,   // Bytes to skip after seeking to aligned position
    actual_start: usize, // Where this run actually starts in the file
}

impl Iterator for RunIteratorWithOVC {
    type Item = (OVCU64, Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        use std::io::ErrorKind;

        // First, skip bytes if we sought to an aligned position
        if self.skip_bytes > 0 {
            let mut skip_buf = vec![0u8; self.skip_bytes];
            self.reader
                .read_exact(&mut skip_buf)
                .expect("Failed to skip bytes after seek");
            self.bytes_read += self.skip_bytes;
            self.skip_bytes = 0;
        }

        loop {
            // Check if we've read all the actual data for this run
            if self.total_bytes > 0 && self.bytes_read - self.actual_start >= self.total_bytes {
                return None;
            }

            // Read OVC
            let mut ovc_bytes = [0u8; 8];
            match self.reader.read_exact(&mut ovc_bytes) {
                Ok(_) => {}
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                    // Legitimate EOF - we've reached the end of data
                    return None;
                }
                Err(e) => {
                    panic!("Failed to read OVC: {}", e);
                }
            }
            let ovc = OVCU64::from_le_bytes(ovc_bytes);

            // Read key length
            let mut key_len_bytes = [0u8; 4];
            self.reader
                .read_exact(&mut key_len_bytes)
                .expect("Failed to read key length");
            let key_len = u32::from_le_bytes(key_len_bytes) as usize;

            // Read value length
            let mut value_len_bytes = [0u8; 4];
            self.reader
                .read_exact(&mut value_len_bytes)
                .expect("Failed to read value length");
            let value_len = u32::from_le_bytes(value_len_bytes) as usize;

            // Read key
            let mut key = vec![0u8; key_len];
            self.reader
                .read_exact(&mut key)
                .expect("Failed to read key");

            // Read value
            let mut value = vec![0u8; value_len];
            self.reader
                .read_exact(&mut value)
                .expect("Failed to read value");

            // Update bytes read
            let entry_size = 8 + 8 + key.len() + value.len();
            self.bytes_read += entry_size;

            // Check if this entry belongs to our run
            if self.bytes_read - entry_size < self.actual_start {
                continue; // This entry is before our run
            }

            // Check if key is in range [lower_inc, upper_exc)
            if !self.lower_bound.is_empty() && key < self.lower_bound {
                continue;
            }
            if !self.upper_bound.is_empty() && key >= self.upper_bound {
                // Since data is sorted in runs, we can stop here
                return None;
            }

            return Some((ovc, key, value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskio::aligned_writer::AlignedWriter;
    use crate::diskio::file::SharedFd;
    use std::fs;
    use std::path::PathBuf;

    fn test_dir() -> PathBuf {
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let dir = std::env::temp_dir().join(format!("run_with_ovc_test_{}", time));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn create_test_file(name: &str) -> Arc<SharedFd> {
        let path = test_dir().join(name);
        Arc::new(SharedFd::new_from_path(&path).expect("Failed to create test file"))
    }

    #[test]
    fn test_create_run_from_writer() {
        let fd = create_test_file("test_create_run.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let run = RunWithOVC::from_writer(writer).unwrap();

        assert_eq!(run.total_entries, 0);
        assert_eq!(run.total_bytes, 0);
        assert!(run.sparse_index.is_empty());
    }

    #[test]
    fn test_append_and_finalize() {
        let fd = create_test_file("test_append.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();

        // Append some data
        run.append(
            OVCU64::initial_value(),
            b"key1".to_vec(),
            b"value1".to_vec(),
        );
        run.append(
            OVCU64::initial_value(),
            b"key2".to_vec(),
            b"value2".to_vec(),
        );
        run.append(
            OVCU64::initial_value(),
            b"key3".to_vec(),
            b"value3".to_vec(),
        );

        assert_eq!(run.total_entries, 3);
        assert!(run.total_bytes > 0);

        // Finalize should return the writer
        let _writer = run.finalize_write();
        assert!(run.writer.is_none());
    }

    #[test]
    fn test_scan_range_full() {
        let fd = create_test_file("test_scan_full.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();

        // Write sorted data
        run.append(OVCU64::initial_value(), b"a".to_vec(), b"1".to_vec());
        run.append(OVCU64::initial_value(), b"b".to_vec(), b"2".to_vec());
        run.append(OVCU64::initial_value(), b"c".to_vec(), b"3".to_vec());

        let mut writer = run.finalize_write();
        writer.flush().unwrap();

        // Read all data back
        let results: Vec<_> = run.scan_range(&[], &[]).collect();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].1, b"a");
        assert_eq!(results[0].2, b"1");
        assert_eq!(results[1].1, b"b");
        assert_eq!(results[1].2, b"2");
        assert_eq!(results[2].1, b"c");
        assert_eq!(results[2].2, b"3");
    }

    #[test]
    fn test_scan_range_with_bounds() {
        let fd = create_test_file("test_scan_bounds.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();

        // Write sorted data
        run.append(OVCU64::initial_value(), b"a".to_vec(), b"1".to_vec());
        run.append(OVCU64::initial_value(), b"b".to_vec(), b"2".to_vec());
        run.append(OVCU64::initial_value(), b"c".to_vec(), b"3".to_vec());
        run.append(OVCU64::initial_value(), b"d".to_vec(), b"4".to_vec());
        run.append(OVCU64::initial_value(), b"e".to_vec(), b"5".to_vec());

        let mut writer = run.finalize_write();
        writer.flush().unwrap();

        // Scan with bounds [b, d)
        let results: Vec<_> = run.scan_range(b"b", b"d").collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1, b"b");
        assert_eq!(results[1].1, b"c");
    }

    #[test]
    fn test_empty_run() {
        let fd = create_test_file("test_empty_run.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();
        let _writer = run.finalize_write();

        // Scanning empty run should return empty iterator
        let results: Vec<_> = run.scan_range(&[], &[]).collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_sparse_index_creation() {
        let fd = create_test_file("test_sparse_index.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();
        run.sampling_interval = 5; // Small sampling interval for testing

        // Add more entries than reservoir size
        for i in 0..20 {
            let key = format!("key_{:02}", i);
            run.append(OVCU64::initial_value(), key.into_bytes(), b"value".to_vec());
        }

        let writer = run.finalize_write();
        drop(writer); // Close writer to flush data

        // Sparse index should have entries based on sampling interval
        // With interval 5 and 20 entries (0-19), we sample at: 0, 5, 10, 15
        let expected_entries = (0..20).filter(|&i| i % run.sampling_interval == 0).count();
        assert_eq!(run.sparse_index.len(), expected_entries);

        // Sparse index should be sorted by file offset
        for i in 1..run.sparse_index.len() {
            assert!(run.sparse_index[i].file_offset >= run.sparse_index[i - 1].file_offset);
        }
    }

    #[test]
    fn test_find_start_position() {
        let fd = create_test_file("test_find_start.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();

        // Manually create sparse index for testing
        run.sparse_index = vec![
            IndexEntry {
                key: b"a".to_vec(),
                file_offset: 0,
                entry_number: 0,
            },
            IndexEntry {
                key: b"c".to_vec(),
                file_offset: 100,
                entry_number: 10,
            },
            IndexEntry {
                key: b"e".to_vec(),
                file_offset: 200,
                entry_number: 20,
            },
        ];

        // Test finding start position
        assert_eq!(run.find_start_position(b"b"), Some(0));
        assert_eq!(run.find_start_position(b"d"), Some(100));
        assert_eq!(run.find_start_position(b"f"), Some(200));
        assert_eq!(run.find_start_position(b"a"), None);
    }

    #[test]
    fn test_large_values() {
        let fd = create_test_file("test_large_values.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();

        // Create large values
        let large_value = vec![b'x'; 1000];

        run.append(
            OVCU64::initial_value(),
            b"key1".to_vec(),
            large_value.clone(),
        );
        run.append(
            OVCU64::initial_value(),
            b"key2".to_vec(),
            large_value.clone(),
        );

        let mut writer = run.finalize_write();
        writer.flush().unwrap();

        // Read back and verify
        let results: Vec<_> = run.scan_range(&[], &[]).collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].2.len(), 1000);
        assert_eq!(results[1].2.len(), 1000);
    }

    #[test]
    fn test_byte_range() {
        let fd = create_test_file("test_byte_range.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let start_pos = writer.position() as usize;
        let mut run = RunWithOVC::from_writer(writer).unwrap();

        run.append(OVCU64::initial_value(), b"key".to_vec(), b"value".to_vec());
        run.append(
            OVCU64::initial_value(),
            b"key2".to_vec(),
            b"value2".to_vec(),
        );

        let (start, end) = run.byte_range();
        assert_eq!(start, start_pos);
        assert_eq!(end, start_pos + run.total_bytes);
    }

    #[test]
    fn test_ovc_values_preserved() {
        let fd = create_test_file("test_ovc_preserved.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();

        // Write with specific OVC values
        run.append(
            OVCU64::normal_value(&[10], 0),
            b"key1".to_vec(),
            b"val1".to_vec(),
        );
        run.append(
            OVCU64::normal_value(&[20], 0),
            b"key2".to_vec(),
            b"val2".to_vec(),
        );
        run.append(
            OVCU64::normal_value(&[30], 0),
            b"key3".to_vec(),
            b"val3".to_vec(),
        );

        let writer = run.finalize_write();
        drop(writer); // Close writer to flush data

        // Read back and verify OVC values
        let results: Vec<_> = run.scan_range(&[], &[]).collect();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, OVCU64::normal_value(&[10], 0));
        assert_eq!(results[1].0, OVCU64::normal_value(&[20], 0));
        assert_eq!(results[2].0, OVCU64::normal_value(&[30], 0));
    }
}
