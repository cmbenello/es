use crate::diskio::aligned_reader::AlignedReader;
use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::constants::align_down;
use crate::diskio::file::SharedFd;
use crate::diskio::io_stats::IoStatsTracker;
use crate::ovc::offset_value_coding::{OVCFlag, OVCU64};
use crate::sort::core::engine::RunSummary;
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
    indexing_interval: usize,
}

impl RunWithOVC {
    pub fn from_writer(writer: AlignedWriter) -> Result<Self, String> {
        Self::from_writer_with_indexing_interval(writer, 1000)
    }

    pub fn from_writer_with_indexing_interval(
        writer: AlignedWriter,
        indexing_interval: usize,
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
            indexing_interval,
        })
    }

    pub fn finalize_write(&mut self) -> AlignedWriter {
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

    pub fn start_key(&self) -> Option<&[u8]> {
        self.sparse_index.first().map(|entry| entry.key.as_slice())
    }

    fn find_start_position(&self, lower_bound: &[u8]) -> Option<(usize, Vec<u8>)> {
        if self.sparse_index.is_empty() {
            return None;
        }
        let mut best_entry = &self.sparse_index[0];
        if lower_bound.is_empty() {
            return Some((best_entry.file_offset, best_entry.key.clone()));
        }

        // Binary search to find the last entry with key < lower_bound
        let mut left = 0;
        let mut right = self.sparse_index.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if &self.sparse_index[mid].key[..] < lower_bound {
                best_entry = &self.sparse_index[mid];
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        Some((best_entry.file_offset, best_entry.key.clone()))
    }
}

impl RunWithOVC {
    pub fn append(&mut self, ovc: OVCU64, key: &[u8], value: &[u8]) {
        let writer = self
            .writer
            .as_mut()
            .expect("RunWithOVC is not initialized with a writer");

        // Use sampling interval for sparse index
        if self.total_entries % self.indexing_interval == 0 {
            let index_entry = IndexEntry {
                key: key.to_vec(),
                file_offset: self.total_bytes,
                entry_number: self.total_entries,
            };
            self.sparse_index.push(index_entry);
        }

        let value_len = value.len() as u32;

        // Optimize for duplicate keys: skip key_len field since it's always 0
        if ovc.flag() == OVCFlag::DuplicateValue {
            writer.write_all(&ovc.to_le_bytes()).unwrap();
            writer.write_all(&value_len.to_le_bytes()).unwrap();
            writer.write_all(&value).unwrap();
            self.total_bytes += 8 + 4 + value.len();
            self.total_entries += 1;
            return;
        }

        // Write entry directly to DirectFileWriter (it handles buffering)
        let offset = match ovc.flag() {
            OVCFlag::EarlyFence | OVCFlag::LateFence => {
                panic!("EarlyFence and LateFence OVC flags are not supported in RunWithOVC");
            }
            OVCFlag::InitialValue => 0,
            OVCFlag::NormalValue => ovc.offset(),
            OVCFlag::DuplicateValue => unreachable!(),
        };
        let truncated_key = &key[offset..];
        let truncated_key_len = truncated_key.len() as u32;
        writer.write_all(&ovc.to_le_bytes()).unwrap();
        writer.write_all(&truncated_key_len.to_le_bytes()).unwrap();
        writer.write_all(&value_len.to_le_bytes()).unwrap();
        writer.write_all(&truncated_key).unwrap();
        writer.write_all(&value).unwrap();

        self.total_bytes += 8 + 8 + truncated_key.len() + value.len();
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

        // Use sparse index to seek to a good starting position.
        // find_start_position will return None iff total_entries == 0
        // That case is handled above.
        // If there is at least one entry, the first entry in sparse index is always at offset 0.
        let (offset, start_key) = self.find_start_position(lower_inc).unwrap();
        let start_offset = self.start_bytes + offset;

        // Seek to the start position if needed
        if start_offset > 0 {
            // Align to page boundary for direct I/O
            let aligned_offset = align_down(start_offset as u64, 4096) as usize;
            let skip_bytes = start_offset - aligned_offset;

            // Seek to aligned position
            if aligned_offset > 0 {
                reader.seek(aligned_offset as u64).unwrap();
            }

            // We'll need to skip the first few bytes after seeking
            return Box::new(RunIteratorWithOVC {
                reader,
                prev_key: start_key,
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
            prev_key: start_key,
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
    prev_key: Vec<u8>,
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

            // Optimize for duplicate keys: skip reading key_len, just use prev_key
            if ovc.flag() == OVCFlag::DuplicateValue {
                // Read value length
                let mut value_len_bytes = [0u8; 4];
                self.reader
                    .read_exact(&mut value_len_bytes)
                    .expect("Failed to read value length");
                let value_len = u32::from_le_bytes(value_len_bytes) as usize;

                // Read value
                let mut value = vec![0u8; value_len];
                self.reader
                    .read_exact(&mut value)
                    .expect("Failed to read value");

                // Update bytes read (no key_len field for duplicates)
                self.bytes_read += 8 + 4 + value_len;

                // Use previous key
                let key = self.prev_key.clone();

                // Check if key is in range [lower_inc, upper_exc)
                if !self.lower_bound.is_empty() && key < self.lower_bound {
                    continue;
                }
                if !self.upper_bound.is_empty() && key >= self.upper_bound {
                    return None;
                }

                return Some((ovc, key, value));
            }

            // Read key length
            let mut key_len_bytes = [0u8; 4];
            self.reader
                .read_exact(&mut key_len_bytes)
                .expect("Failed to read key length");
            let truncated_key_len = u32::from_le_bytes(key_len_bytes) as usize;

            // Read value length
            let mut value_len_bytes = [0u8; 4];
            self.reader
                .read_exact(&mut value_len_bytes)
                .expect("Failed to read value length");
            let value_len = u32::from_le_bytes(value_len_bytes) as usize;

            let offset = match ovc.flag() {
                OVCFlag::EarlyFence | OVCFlag::LateFence => {
                    panic!("EarlyFence and LateFence OVC flags are not supported in RunWithOVC");
                }
                OVCFlag::InitialValue => 0,
                OVCFlag::NormalValue => ovc.offset(),
                OVCFlag::DuplicateValue => unreachable!(),
            };
            let mut key = vec![0u8; offset + truncated_key_len];
            key[..offset].copy_from_slice(&self.prev_key[..offset]);

            // Read key
            self.reader
                .read_exact(&mut key[offset..])
                .expect("Failed to read key");
            self.prev_key.resize(key.len(), 0);
            self.prev_key.copy_from_slice(&key);

            // Read value
            let mut value = vec![0u8; value_len];
            self.reader
                .read_exact(&mut value)
                .expect("Failed to read value");

            // Update bytes read
            self.bytes_read += 8 + 8 + truncated_key_len + value_len;

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

impl RunSummary for RunWithOVC {
    fn total_entries(&self) -> usize {
        self.total_entries
    }

    fn total_bytes(&self) -> usize {
        self.total_bytes
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
        Arc::new(SharedFd::new_from_path(&path, true).expect("Failed to create test file"))
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
        run.append(OVCU64::initial_value(), b"key1", b"value1");
        run.append(OVCU64::initial_value(), b"key2", b"value2");
        run.append(OVCU64::initial_value(), b"key3", b"value3");

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
        run.append(OVCU64::initial_value(), b"a", b"1");
        run.append(OVCU64::initial_value(), b"b", b"2");
        run.append(OVCU64::initial_value(), b"c", b"3");

        let mut writer = run.finalize_write();
        writer.flush().unwrap();

        // Read all data back
        let results: Vec<_> = run.scan_range(&[], &[]).collect();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, OVCU64::initial_value());
        assert_eq!(results[0].1, b"a");
        assert_eq!(results[0].2, b"1");
        assert_eq!(results[1].0, OVCU64::initial_value());
        assert_eq!(results[1].1, b"b");
        assert_eq!(results[1].2, b"2");
        assert_eq!(results[2].0, OVCU64::initial_value());
        assert_eq!(results[2].1, b"c");
        assert_eq!(results[2].2, b"3");
    }

    #[test]
    fn test_scan_range_with_bounds() {
        let fd = create_test_file("test_scan_bounds.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();

        // Write sorted data
        run.append(OVCU64::initial_value(), b"a", b"1");
        run.append(OVCU64::initial_value(), b"b", b"2");
        run.append(OVCU64::initial_value(), b"c", b"3");
        run.append(OVCU64::initial_value(), b"d", b"4");
        run.append(OVCU64::initial_value(), b"e", b"5");

        let mut writer = run.finalize_write();
        writer.flush().unwrap();

        // Scan with bounds [b, d)
        let results: Vec<_> = run.scan_range(b"b", b"d").collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, OVCU64::initial_value());
        assert_eq!(results[0].1, b"b");
        assert_eq!(results[0].2, b"2");
        assert_eq!(results[1].0, OVCU64::initial_value());
        assert_eq!(results[1].1, b"c");
        assert_eq!(results[1].2, b"3");
    }

    #[test]
    fn test_empty_run() {
        let fd = create_test_file("test_empty_run.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();
        let mut writer = run.finalize_write();
        writer.flush().unwrap();

        // Scanning empty run should return empty iterator
        let results: Vec<_> = run.scan_range(&[], &[]).collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_sparse_index_creation() {
        let fd = create_test_file("test_sparse_index.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();
        run.indexing_interval = 5; // Small sampling interval for testing

        // Add more entries than reservoir size
        for i in 0..20 {
            let key = format!("key_{:02}", i);
            run.append(OVCU64::initial_value(), &key.into_bytes(), b"value");
        }

        let writer = run.finalize_write();
        drop(writer); // Close writer to flush data

        // Sparse index should have entries based on sampling interval
        // With interval 5 and 20 entries (0-19), we sample at: 0, 5, 10, 15
        let expected_entries = (0..20).filter(|&i| i % run.indexing_interval == 0).count();
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
                key: b"b".to_vec(),
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
        assert_eq!(run.find_start_position(b"a"), Some((0, b"b".to_vec()))); // The first entry will be always returned
        assert_eq!(run.find_start_position(b"b"), Some((0, b"b".to_vec())));
        assert_eq!(run.find_start_position(b"d"), Some((100, b"c".to_vec())));
        assert_eq!(run.find_start_position(b"f"), Some((200, b"e".to_vec())));
    }

    #[test]
    fn test_large_values() {
        let fd = create_test_file("test_large_values.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();

        // Create large values
        let large_value = vec![b'x'; 1000];

        run.append(OVCU64::initial_value(), b"key1", &large_value);
        run.append(OVCU64::initial_value(), b"key2", &large_value);

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

        run.append(OVCU64::initial_value(), b"key", b"value");
        run.append(OVCU64::initial_value(), b"key2", b"value2");

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
        run.append(OVCU64::normal_value(&[10], 0), b"key1", b"val1");
        run.append(OVCU64::normal_value(&[20], 0), b"key2", b"val2");
        run.append(OVCU64::normal_value(&[30], 0), b"key3", b"val3");

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
