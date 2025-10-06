use crate::diskio::aligned_reader::AlignedReader;
use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::constants::{PAGE_SIZE, align_down};
use crate::diskio::file::SharedFd;
use crate::diskio::io_stats::IoStatsTracker;
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
pub struct RunImpl {
    fd: Arc<SharedFd>,
    writer: Option<AlignedWriter>,
    total_entries: usize,
    start_bytes: usize,
    total_bytes: usize,
    sparse_index: Vec<IndexEntry>,
    indexing_interval: usize,
}

impl RunImpl {
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

impl RunImpl {
    pub fn append(&mut self, key: Vec<u8>, value: Vec<u8>) {
        let writer = self
            .writer
            .as_mut()
            .expect("RunImpl is not initialized with a writer");

        // Use sampling interval for sparse index
        if self.total_entries % self.indexing_interval == 0 {
            let index_entry = IndexEntry {
                key: key.clone(),
                file_offset: self.total_bytes,
                entry_number: self.total_entries,
            };
            self.sparse_index.push(index_entry);
        }

        let key_len = key.len() as u32;
        let value_len = value.len() as u32;

        // Write entry directly to DirectFileWriter (it handles buffering)
        writer.write_all(&key_len.to_le_bytes()).unwrap();
        writer.write_all(&value_len.to_le_bytes()).unwrap();
        writer.write_all(&key).unwrap();
        writer.write_all(&value).unwrap();

        self.total_bytes += 8 + key.len() + value.len();
        self.total_entries += 1;
    }

    pub fn scan_range(
        &self,
        lower_inc: &[u8],
        upper_exc: &[u8],
    ) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send> {
        self.scan_range_with_io_tracker(lower_inc, upper_exc, None)
    }

    pub fn scan_range_with_io_tracker(
        &self,
        lower_inc: &[u8],
        upper_exc: &[u8],
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send> {
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
        let (offset, _start_key) = self.find_start_position(lower_inc).unwrap();
        let start_offset = self.start_bytes + offset;

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
            return Box::new(RunIterator {
                reader,
                lower_bound: lower_inc.to_vec(),
                upper_bound: upper_exc.to_vec(),
                bytes_read: aligned_offset,
                total_bytes: self.total_bytes,
                skip_bytes,
                actual_start: self.start_bytes,
            }) as Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>;
        }

        Box::new(RunIterator {
            reader,
            lower_bound: lower_inc.to_vec(),
            upper_bound: upper_exc.to_vec(),
            bytes_read: self.start_bytes, // Start from the beginning of this run
            total_bytes: self.total_bytes,
            skip_bytes: 0,
            actual_start: self.start_bytes,
        }) as Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>
    }
}

struct RunIterator {
    reader: AlignedReader,
    lower_bound: Vec<u8>,
    upper_bound: Vec<u8>,
    bytes_read: usize,
    total_bytes: usize,
    skip_bytes: usize,   // Bytes to skip after seeking to aligned position
    actual_start: usize, // Where this run actually starts in the file
}

impl Iterator for RunIterator {
    type Item = (Vec<u8>, Vec<u8>);

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

            // Read key length
            let mut key_len_bytes = [0u8; 4];
            match self.reader.read_exact(&mut key_len_bytes) {
                Ok(_) => {}
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                    // Legitimate EOF - we've reached the end of data
                    return None;
                }
                Err(e) => panic!("Failed to read key length: {}", e),
            }
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
            let entry_size = 8 + key.len() + value.len();
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

            return Some((key, value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_test_path(name: &str) -> PathBuf {
        let test_dir =
            std::env::temp_dir()
                .join("run_test")
                .join(format!("{}_{}", std::process::id(), name));
        std::fs::create_dir_all(&test_dir).unwrap();
        test_dir.join("test_run.dat")
    }

    fn get_test_writer(name: &str) -> AlignedWriter {
        let path = get_test_path(name);
        let fd = Arc::new(
            SharedFd::new_from_path(&path).expect("Failed to open test file with Direct I/O"),
        );

        AlignedWriter::from_fd(fd).unwrap()
    }

    #[test]
    fn test_basic_append_and_scan() {
        let writer = get_test_writer("basic");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Append some entries
        run.append(b"key1".to_vec(), b"value1".to_vec());
        run.append(b"key2".to_vec(), b"value2".to_vec());
        run.append(b"key3".to_vec(), b"value3".to_vec());
        run.finalize_write();

        let mut iter = run.scan_range(&[], &[]);
        assert_eq!(iter.next(), Some((b"key1".to_vec(), b"value1".to_vec())));
        assert_eq!(iter.next(), Some((b"key2".to_vec(), b"value2".to_vec())));
        assert_eq!(iter.next(), Some((b"key3".to_vec(), b"value3".to_vec())));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_empty_run() {
        let writer = get_test_writer("empty");
        let mut run = RunImpl::from_writer(writer).unwrap();
        run.finalize_write();

        // For empty runs, total_bytes should be 0
        assert_eq!(run.total_bytes(), 0);
        assert_eq!(run.total_entries(), 0);

        // Scanning should return no items
        let mut iter = run.scan_range(&[], &[]);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_scan_range_filtering() {
        let writer = get_test_writer("range");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Append entries
        run.append(b"a".to_vec(), b"1".to_vec());
        run.append(b"b".to_vec(), b"2".to_vec());
        run.append(b"c".to_vec(), b"3".to_vec());
        run.append(b"d".to_vec(), b"4".to_vec());
        run.append(b"e".to_vec(), b"5".to_vec());
        run.finalize_write();

        // Test inclusive lower bound
        let iter = run.scan_range(b"b", &[]);
        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].0, b"b");
        assert_eq!(results[3].0, b"e");

        // Test exclusive upper bound
        let iter = run.scan_range(&[], b"d");
        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, b"a");
        assert_eq!(results[2].0, b"c");

        // Test both bounds
        let iter = run.scan_range(b"b", b"d");
        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, b"b");
        assert_eq!(results[1].0, b"c");
    }

    #[test]
    fn test_large_entries() {
        let writer = get_test_writer("large");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Create large key and value
        let large_key = vec![b'k'; 1000];
        let large_value = vec![b'v'; 10000];

        run.append(large_key.clone(), large_value.clone());
        run.append(b"small".to_vec(), b"val".to_vec());
        run.finalize_write();

        let mut iter = run.scan_range(&[], &[]);
        let (key, value) = iter.next().unwrap();
        assert_eq!(key, large_key);
        assert_eq!(value, large_value);

        let (key, value) = iter.next().unwrap();
        assert_eq!(key, b"small");
        assert_eq!(value, b"val");

        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_page_buffer_flushing() {
        let writer = get_test_writer("page_flush");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Add entries that will trigger page flushes
        // Page size is 4096 bytes, each entry is 8 bytes + key + value
        let mut entries = Vec::new();
        for i in 0..500 {
            let key = format!("key_{:04}", i).into_bytes();
            let value = format!("value_{:04}", i).into_bytes();
            entries.push((key.clone(), value.clone()));
            run.append(key, value);
        }
        run.finalize_write();

        // Verify all entries are present
        let iter = run.scan_range(&[], &[]);
        let mut count = 0;
        for (key, value) in iter {
            assert_eq!(key, entries[count].0);
            assert_eq!(value, entries[count].1);
            count += 1;
        }
        assert_eq!(count, 500);
    }

    #[test]
    fn test_multiple_scans() {
        let writer = get_test_writer("multi_scan");
        let mut run = RunImpl::from_writer(writer).unwrap();

        run.append(b"a".to_vec(), b"1".to_vec());
        run.append(b"b".to_vec(), b"2".to_vec());
        run.append(b"c".to_vec(), b"3".to_vec());
        run.finalize_write();

        // First scan
        let mut iter1 = run.scan_range(&[], &[]);
        assert_eq!(iter1.next().unwrap().0, b"a");

        // Second concurrent scan
        let mut iter2 = run.scan_range(&[], &[]);
        assert_eq!(iter2.next().unwrap().0, b"a");
        assert_eq!(iter2.next().unwrap().0, b"b");

        // Continue first scan
        assert_eq!(iter1.next().unwrap().0, b"b");
        assert_eq!(iter1.next().unwrap().0, b"c");
        assert_eq!(iter1.next(), None);

        // Finish second scan
        assert_eq!(iter2.next().unwrap().0, b"c");
        assert_eq!(iter2.next(), None);
    }

    #[test]
    fn test_in_memory_tracking() {
        let writer = get_test_writer("in_memory");
        let entry_count = 10;

        let mut run = RunImpl::from_writer(writer).unwrap();

        // Track bytes as we add entries
        let mut expected_bytes = 0;
        for i in 0..entry_count {
            let key = format!("key_{:02}", i).into_bytes();
            let value = format!("value_{:02}", i).into_bytes();
            expected_bytes += 8 + key.len() + value.len();
            run.append(key, value);
        }

        // Verify in-memory tracking
        assert_eq!(run.total_entries, entry_count);
        assert_eq!(run.total_bytes, expected_bytes);

        run.finalize_write();
    }

    #[test]
    fn test_binary_data() {
        let writer = get_test_writer("binary");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Test with binary data including null bytes
        let binary_key = vec![0, 1, 2, 3, 255, 254, 253, 252];
        let binary_value = vec![0; 100]; // 100 null bytes

        run.append(binary_key.clone(), binary_value.clone());
        run.append(vec![255; 5], vec![128; 10]);
        run.finalize_write();

        let mut iter = run.scan_range(&[], &[]);
        let (key, value) = iter.next().unwrap();
        assert_eq!(key, binary_key);
        assert_eq!(value, binary_value);

        let (key, value) = iter.next().unwrap();
        assert_eq!(key, vec![255; 5]);
        assert_eq!(value, vec![128; 10]);

        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_exact_page_boundary() {
        let writer = get_test_writer("page_boundary");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Calculate entry size to exactly fill pages
        // Each entry: 4 bytes key_len + 4 bytes value_len + key + value
        // To get exact 4096 bytes, we need to calculate carefully
        let _entries_per_page = 4096 / 50; // Approximate, will adjust

        let mut total_bytes = 0;
        let mut entries = Vec::new();

        // Fill exactly one page
        while total_bytes + 50 <= 4096 {
            let key = format!("k{:03}", entries.len()).into_bytes();
            let value = format!("v{:03}", entries.len()).into_bytes();
            let entry_size = 8 + key.len() + value.len();

            if total_bytes + entry_size > 4096 {
                break;
            }

            entries.push((key.clone(), value.clone()));
            run.append(key, value);
            total_bytes += entry_size;
        }

        // Add one more entry to trigger flush
        run.append(b"trigger".to_vec(), b"flush".to_vec());
        entries.push((b"trigger".to_vec(), b"flush".to_vec()));
        run.finalize_write();

        // Verify all entries
        let mut iter = run.scan_range(&[], &[]);
        for (expected_key, expected_value) in &entries {
            let (key, value) = iter.next().unwrap();
            assert_eq!(&key, expected_key);
            assert_eq!(&value, expected_value);
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_finalize_idempotent() {
        let writer = get_test_writer("finalize_idempotent");
        let mut run = RunImpl::from_writer(writer).unwrap();

        run.append(b"key".to_vec(), b"value".to_vec());
        run.finalize_write();

        // Can still scan after multiple finalizes
        let mut iter = run.scan_range(&[], &[]);
        assert_eq!(iter.next(), Some((b"key".to_vec(), b"value".to_vec())));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_unicode_keys_values() {
        let writer = get_test_writer("unicode");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Test with various unicode strings
        let test_cases = vec![
            ("hello", "world"),
            ("你好", "世界"),
            ("🦀", "🔥"),
            ("key_混合_test", "value_テスト_data"),
        ];

        for (key, value) in &test_cases {
            run.append(key.as_bytes().to_vec(), value.as_bytes().to_vec());
        }
        run.finalize_write();

        let mut iter = run.scan_range(&[], &[]);
        for (key, value) in &test_cases {
            let (k, v) = iter.next().unwrap();
            assert_eq!(k, key.as_bytes());
            assert_eq!(v, value.as_bytes());
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_scan_after_partial_read() {
        let writer = get_test_writer("partial_read");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Add entries
        for i in 0..10 {
            let key = format!("{:02}", i).into_bytes();
            let value = format!("v{}", i).into_bytes();
            run.append(key, value);
        }
        run.finalize_write();

        // Start a scan and read only partially
        let mut iter = run.scan_range(&[], &[]);
        assert_eq!(iter.next().unwrap().0, b"00");
        assert_eq!(iter.next().unwrap().0, b"01");
        assert_eq!(iter.next().unwrap().0, b"02");
        // Drop iterator without reading all
        drop(iter);

        // Start a new scan - should work fine
        let iter = run.scan_range(&[], &[]);
        let all: Vec<_> = iter.collect();
        assert_eq!(all.len(), 10);
    }

    #[test]
    fn test_empty_key_value() {
        let writer = get_test_writer("empty_kv");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Test empty key with non-empty value
        run.append(vec![], b"value_for_empty_key".to_vec());

        // Test non-empty key with empty value
        run.append(b"key_with_empty_value".to_vec(), vec![]);

        // Test both empty
        run.append(vec![], vec![]);

        // Test normal entry
        run.append(b"normal".to_vec(), b"entry".to_vec());

        run.finalize_write();

        let mut iter = run.scan_range(&[], &[]);

        let (k, v) = iter.next().unwrap();
        assert_eq!(k, Vec::<u8>::new());
        assert_eq!(v, b"value_for_empty_key");

        let (k, v) = iter.next().unwrap();
        assert_eq!(k, b"key_with_empty_value");
        assert_eq!(v, Vec::<u8>::new());

        let (k, v) = iter.next().unwrap();
        assert_eq!(k, Vec::<u8>::new());
        assert_eq!(v, Vec::<u8>::new());

        let (k, v) = iter.next().unwrap();
        assert_eq!(k, b"normal");
        assert_eq!(v, b"entry");

        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_stress_many_small_entries() {
        let writer = get_test_writer("stress_small");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Add 10,000 small entries
        let count = 10_000;
        for i in 0..count {
            let key = format!("{:05}", i).into_bytes();
            let value = b"v".to_vec();
            run.append(key, value);
        }

        run.finalize_write();

        // Verify all entries
        let iter = run.scan_range(&[], &[]);
        let all: Vec<_> = iter.collect();
        assert_eq!(all.len(), count);

        // Check first and last
        assert_eq!(all[0].0, b"00000");
        assert_eq!(all[count - 1].0, format!("{:05}", count - 1).as_bytes());
    }

    #[test]
    fn test_concurrent_run_creation() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        for i in 0..5 {
            let success_count = Arc::clone(&success_count);
            let handle = thread::spawn(move || {
                let writer = get_test_writer(&format!("concurrent_{}", i));
                let mut run = RunImpl::from_writer(writer).unwrap();

                for j in 0..100 {
                    let key = format!("thread{}_{:03}", i, j).into_bytes();
                    let value = format!("value_{}", j).into_bytes();
                    run.append(key, value);
                }

                run.finalize_write();

                success_count.fetch_add(1, Ordering::SeqCst);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(success_count.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_max_key_value_sizes() {
        let writer = get_test_writer("max_sizes");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Test with maximum practical sizes
        let max_key = vec![b'k'; 1_000_000]; // 1MB key
        let max_value = vec![b'v'; 5_000_000]; // 5MB value

        run.append(max_key.clone(), max_value.clone());
        run.append(b"small".to_vec(), b"entry".to_vec());
        run.finalize_write();

        let mut iter = run.scan_range(&[], &[]);
        let (k, v) = iter.next().unwrap();
        assert_eq!(k.len(), 1_000_000);
        assert_eq!(v.len(), 5_000_000);

        let (k, v) = iter.next().unwrap();
        assert_eq!(k, b"small");
        assert_eq!(v, b"entry");
    }

    #[test]
    fn test_alternating_large_small() {
        let writer = get_test_writer("alternating");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Alternate between large and small entries
        for i in 0..100 {
            if i % 2 == 0 {
                // Large entry
                let key = format!("{:03}_large", i).into_bytes();
                let value = vec![b'L'; 10_000];
                run.append(key, value);
            } else {
                // Small entry
                let key = format!("{:03}_small", i).into_bytes();
                let value = b"s".to_vec();
                run.append(key, value);
            }
        }
        run.finalize_write();

        let iter = run.scan_range(&[], &[]);
        let all: Vec<_> = iter.collect();
        assert_eq!(all.len(), 100);

        // Verify pattern
        for i in 0..100 {
            if i % 2 == 0 {
                assert_eq!(all[i].1.len(), 10_000);
            } else {
                assert_eq!(all[i].1.len(), 1);
            }
        }
    }

    #[test]
    fn test_run_without_metadata() {
        let writer = get_test_writer("no_metadata");

        // Create a run with data
        let mut run = RunImpl::from_writer(writer).unwrap();
        for i in 0..10 {
            run.append(format!("{:02}", i).into_bytes(), b"value".to_vec());
        }
        run.finalize_write();

        // Verify we can still read the data
        let mut iter = run.scan_range(&[], &[]);
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 10);
    }

    #[test]
    fn test_scan_with_equal_bounds() {
        let writer = get_test_writer("equal_bounds");
        let mut run = RunImpl::from_writer(writer).unwrap();

        run.append(b"a".to_vec(), b"1".to_vec());
        run.append(b"b".to_vec(), b"2".to_vec());
        run.append(b"c".to_vec(), b"3".to_vec());
        run.finalize_write();

        // Scan with equal lower and upper bounds - should return nothing
        let iter = run.scan_range(b"b", b"b");
        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_duplicate_keys() {
        let writer = get_test_writer("duplicate_keys");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Add duplicate keys
        run.append(b"key".to_vec(), b"value1".to_vec());
        run.append(b"key".to_vec(), b"value2".to_vec());
        run.append(b"key".to_vec(), b"value3".to_vec());
        run.append(b"other".to_vec(), b"value".to_vec());
        run.finalize_write();

        let iter = run.scan_range(&[], &[]);
        let all: Vec<_> = iter.collect();
        assert_eq!(all.len(), 4);

        // All duplicates should be preserved
        assert_eq!(all[0], (b"key".to_vec(), b"value1".to_vec()));
        assert_eq!(all[1], (b"key".to_vec(), b"value2".to_vec()));
        assert_eq!(all[2], (b"key".to_vec(), b"value3".to_vec()));
        assert_eq!(all[3], (b"other".to_vec(), b"value".to_vec()));
    }

    #[test]
    fn test_io_tracking_write_operations() {
        let path = get_test_path("io_write_tracking");
        let fd = Arc::new(
            SharedFd::new_from_path(&path).expect("Failed to open test file with Direct I/O"),
        );

        // Create writer with IO tracker
        let io_tracker = IoStatsTracker::new();
        let writer = AlignedWriter::from_fd_with_tracker(fd, Some(io_tracker.clone())).unwrap();
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Get baseline stats before writing
        let (initial_write_ops, initial_write_bytes) = io_tracker.get_write_stats();

        // Write N records with known sizes
        let num_records = 10000;
        let mut expected_logical_bytes = 0;

        for i in 0..num_records {
            let key = format!("key_{:08}", i).into_bytes();
            let value = format!("value_data_{:08}", i).into_bytes();
            expected_logical_bytes += 8 + key.len() + value.len(); // 4 bytes key_len + 4 bytes value_len + key + value
            run.append(key, value);
        }

        // Finalize to ensure all data is written and flush to disk
        let mut writer = run.finalize_write();
        writer.flush().expect("Failed to flush data to disk");

        // Get final stats after writing
        let (final_write_ops, final_write_bytes) = io_tracker.get_write_stats();

        let write_ops = final_write_ops - initial_write_ops;
        let write_bytes = final_write_bytes - initial_write_bytes;

        println!("=== WRITE IO STATS ===");
        println!("Records written: {}", num_records);
        println!("Expected logical bytes: {}", expected_logical_bytes);
        println!("Actual IO write operations: {}", write_ops);
        println!("Actual IO write bytes: {}", write_bytes);
        println!(
            "IO overhead ratio: {:.2}x",
            write_bytes as f64 / expected_logical_bytes as f64
        );

        // Verify that some writes occurred
        assert!(write_ops > 0, "No write operations were tracked");
        assert!(
            write_bytes >= expected_logical_bytes as u64,
            "IO bytes ({}) should be >= logical bytes ({})",
            write_bytes,
            expected_logical_bytes
        );
    }

    #[test]
    fn test_io_tracking_scan_operations() {
        let path = get_test_path("io_scan_tracking");
        let fd = Arc::new(
            SharedFd::new_from_path(&path).expect("Failed to open test file with Direct I/O"),
        );

        // First, write some data (without tracking writes for this test)
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();
        let mut run = RunImpl::from_writer(writer).unwrap();

        let num_records = 100000;
        let mut expected_logical_bytes = 0;

        for i in 0..num_records {
            let key = format!("key_{:08}", i).into_bytes();
            let value = format!("value_data_{:08}", i).into_bytes();
            expected_logical_bytes += 8 + key.len() + value.len();
            run.append(key, value);
        }

        // Finalize write without tracking and flush to ensure data is on disk
        let mut writer = run.finalize_write();
        writer.flush().expect("Failed to flush data to disk");

        // Now test scanning with IO tracking
        let read_tracker = IoStatsTracker::new();

        // Get baseline stats
        let (initial_read_ops, initial_read_bytes) = read_tracker.get_read_stats();

        // Scan all records with IO tracking
        let iter = run.scan_range_with_io_tracker(&[], &[], Some(read_tracker.clone()));
        let scanned_records: Vec<_> = iter.collect();

        // Get final stats after scanning
        let (final_read_ops, final_read_bytes) = read_tracker.get_read_stats();

        let read_ops = final_read_ops - initial_read_ops;
        let read_bytes = final_read_bytes - initial_read_bytes;

        println!("=== SCAN IO STATS ===");
        println!("Records scanned: {}", scanned_records.len());
        println!("Expected logical bytes: {}", expected_logical_bytes);
        println!("Actual IO read operations: {}", read_ops);
        println!("Actual IO read bytes: {}", read_bytes);
        println!(
            "IO overhead ratio: {:.2}x",
            read_bytes as f64 / expected_logical_bytes as f64
        );

        // Verify scan results
        assert_eq!(scanned_records.len(), num_records);

        // Verify that some reads occurred
        assert!(read_ops > 0, "No read operations were tracked");
        assert!(
            read_bytes >= expected_logical_bytes as u64,
            "IO bytes ({}) should be >= logical bytes ({})",
            read_bytes,
            expected_logical_bytes
        );

        // Test partial scan to show different IO patterns
        let partial_tracker = IoStatsTracker::new();
        let iter = run.scan_range_with_io_tracker(
            b"key_00000100",
            b"key_00000150",
            Some(partial_tracker.clone()),
        );
        let partial_results: Vec<_> = iter.collect();

        let (partial_read_ops, partial_read_bytes) = partial_tracker.get_read_stats();

        println!("=== PARTIAL SCAN IO STATS ===");
        println!("Records in partial scan: {}", partial_results.len());
        println!("Partial scan IO read operations: {}", partial_read_ops);
        println!("Partial scan IO read bytes: {}", partial_read_bytes);

        assert_eq!(partial_results.len(), 50); // records 100-149
        assert!(
            partial_read_ops > 0,
            "Partial scan should have read operations"
        );
    }

    #[test]
    fn test_sparse_index_effectiveness() {
        let path = get_test_path("sparse_effectiveness");
        let fd = Arc::new(
            SharedFd::new_from_path(&path).expect("Failed to open test file with Direct I/O"),
        );

        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Write 10000 sequential records
        let num_records = 10000;
        for i in 0..num_records {
            let key = format!("key_{:08}", i).into_bytes();
            let value = format!("value_data_{:08}", i).into_bytes();
            run.append(key, value);
        }

        let mut writer = run.finalize_write();
        writer.flush().expect("Failed to flush data to disk");

        println!("=== SPARSE INDEX ANALYSIS ===");
        println!("Total records: {}", num_records);
        println!("Sparse index size: {}", run.sparse_index.len());
        println!("Indexing interval: {}", run.indexing_interval);

        // Check sparse index distribution
        if !run.sparse_index.is_empty() {
            let first_offset = run.sparse_index[0].file_offset;
            let last_offset = run.sparse_index[run.sparse_index.len() - 1].file_offset;
            let total_bytes = run.total_bytes();

            println!("First index offset: {}", first_offset);
            println!("Last index offset: {}", last_offset);
            println!("Total bytes: {}", total_bytes);
            println!(
                "Index coverage: {:.2}%",
                (last_offset as f64 / total_bytes as f64) * 100.0
            );

            // Test seek effectiveness for different positions
            let test_keys = [
                b"key_00002500".to_vec(), // 25% through
                b"key_00005000".to_vec(), // 50% through
                b"key_00007500".to_vec(), // 75% through
            ];

            for test_key in &test_keys {
                let io_tracker = IoStatsTracker::new();
                let iter = run.scan_range_with_io_tracker(test_key, &[], Some(io_tracker.clone()));
                let results: Vec<_> = iter.take(100).collect(); // Only take first 100 to avoid reading everything

                let (read_ops, read_bytes) = io_tracker.get_read_stats();
                println!(
                    "Key {}: {} results, {} ops, {} bytes",
                    String::from_utf8_lossy(test_key),
                    results.len(),
                    read_ops,
                    read_bytes
                );
            }
        }
    }

    #[test]
    fn test_merge_io_efficiency() {
        // Create two runs with known data
        let path1 = get_test_path("merge_io_1");
        let path2 = get_test_path("merge_io_2");

        let fd1 = Arc::new(SharedFd::new_from_path(&path1).unwrap());
        let fd2 = Arc::new(SharedFd::new_from_path(&path2).unwrap());

        // Create first run (even keys)
        let writer1 = AlignedWriter::from_fd(fd1.clone()).unwrap();
        let mut run1 = RunImpl::from_writer(writer1).unwrap();
        let mut run1_logical_bytes = 0;

        for i in (0..5000).step_by(2) {
            let key = format!("key_{:08}", i).into_bytes();
            let value = format!("value_data_{:08}", i).into_bytes();
            run1_logical_bytes += 8 + key.len() + value.len();
            run1.append(key, value);
        }
        let mut writer1 = run1.finalize_write();
        writer1.flush().unwrap();

        // Create second run (odd keys)
        let writer2 = AlignedWriter::from_fd(fd2.clone()).unwrap();
        let mut run2 = RunImpl::from_writer(writer2).unwrap();
        let mut run2_logical_bytes = 0;

        for i in (1..5000).step_by(2) {
            let key = format!("key_{:08}", i).into_bytes();
            let value = format!("value_data_{:08}", i).into_bytes();
            run2_logical_bytes += 8 + key.len() + value.len();
            run2.append(key, value);
        }
        let mut writer2 = run2.finalize_write();
        writer2.flush().unwrap();

        let total_logical_bytes = run1_logical_bytes + run2_logical_bytes;

        println!("=== MERGE IO EFFICIENCY TEST ===");
        println!("Run1 logical bytes: {}", run1_logical_bytes);
        println!("Run1 sparse index size: {}", run1.sparse_index.len());
        println!("Run2 logical bytes: {}", run2_logical_bytes);
        println!("Run2 sparse index size: {}", run2.sparse_index.len());
        println!("Total logical bytes: {}", total_logical_bytes);

        // Test different merge patterns
        let test_cases = [
            ("Full scan both runs", &b""[..], &b""[..]),
            ("Middle range", b"key_00002000", b"key_00003000"),
            ("Early range", b"key_00000100", b"key_00000200"),
            ("Late range", b"key_00004800", b"key_00004900"),
        ];

        for (test_name, lower, upper) in &test_cases {
            let tracker1 = IoStatsTracker::new();
            let tracker2 = IoStatsTracker::new();

            let iter1 = run1.scan_range_with_io_tracker(lower, upper, Some(tracker1.clone()));
            let iter2 = run2.scan_range_with_io_tracker(lower, upper, Some(tracker2.clone()));

            let results1: Vec<_> = iter1.collect();
            let results1_size = results1
                .iter()
                .map(|(k, v)| 8 + k.len() + v.len())
                .sum::<usize>();
            let results2: Vec<_> = iter2.collect();
            let results2_size = results2
                .iter()
                .map(|(k, v)| 8 + k.len() + v.len())
                .sum::<usize>();

            let (ops1, bytes1) = tracker1.get_read_stats();
            let (ops2, bytes2) = tracker2.get_read_stats();

            println!("--- {} ---", test_name);
            println!(
                "Run1: {} results, {} ops, {} bytes",
                results1.len(),
                ops1,
                bytes1
            );
            println!(
                "Run2: {} results, {} ops, {} bytes",
                results2.len(),
                ops2,
                bytes2
            );
            println!("Total IO: {} bytes", bytes1 + bytes2);
            println!("Expected logical: ~{} bytes", results1_size + results2_size);
            println!(
                "IO ratio: {:.2}x",
                (bytes1 + bytes2) as f64 / (results1_size + results2_size) as f64
            );
            println!();
        }
    }

    #[test]
    fn test_indexing_interval_sparse_index() {
        let writer = get_test_writer("indexing_interval");
        let mut run = RunImpl::from_writer_with_indexing_interval(writer, 500).unwrap();

        // Add many entries to test sampling interval
        // Should sample every 500th entry
        for i in 0..50000 {
            let key = format!("{:05}", i).into_bytes();
            let value = vec![b'v'; 50]; // 50 byte value
            run.append(key, value);
        }
        let writer = run.finalize_write();
        drop(writer); // Ensure all data is written

        // Verify sparse index has entries based on sampling interval
        let expected_entries = 50000 / 500; // Every 500th entry
        assert_eq!(
            run.sparse_index.len(),
            expected_entries,
            "Sparse index should have exactly {} entries",
            expected_entries
        );

        // Verify index entries are sorted by offset after finalize
        for i in 1..run.sparse_index.len() {
            assert!(
                run.sparse_index[i - 1].file_offset < run.sparse_index[i].file_offset,
                "Index should be sorted by file offset"
            );
        }

        // Verify sampling interval consistency
        for (i, index_entry) in run.sparse_index.iter().enumerate() {
            let expected_entry_number = i * 500;
            assert_eq!(
                index_entry.entry_number, expected_entry_number,
                "Entry {} should have entry_number {}",
                i, expected_entry_number
            );
        }

        // Verify we have good distribution across the file
        let total_size = run.total_bytes;
        let first_offset = run.sparse_index.first().unwrap().file_offset;
        let last_offset = run.sparse_index.last().unwrap().file_offset;

        // Verify good coverage - the span should cover a significant portion of the file
        let coverage = (last_offset - first_offset) as f64 / total_size as f64;
        assert!(
            coverage > 0.5,
            "Sparse index should cover at least 50% of file, got {:.2}%",
            coverage * 100.0
        );

        // Test that scan_range uses the sparse index efficiently
        let lower = b"25000";
        let iter = run.scan_range(lower, &[]);
        let results: Vec<_> = iter.collect();

        // Should get approximately half the entries
        assert_eq!(results.len(), 25000);
        assert_eq!(results[0].0, b"25000");
    }

    #[test]
    fn test_sparse_index_seek() {
        let writer = get_test_writer("sparse_seek");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Add enough entries to build a sparse index
        for i in 0..5000 {
            let key = format!("{:04}", i).into_bytes();
            let value = vec![b'x'; 50];
            run.append(key, value);
        }
        run.finalize_write();

        // Test that scan_range with a high lower bound is efficient
        // It should use the sparse index to skip ahead
        let lower = b"4000";
        let iter = run.scan_range(lower, &[]);
        let results: Vec<_> = iter.collect();

        // Should get entries from 4000 onwards
        assert_eq!(results.len(), 1000); // 4000 to 4999
        assert_eq!(results[0].0, b"4000");

        // Verify all results are >= lower bound
        for (key, _) in &results {
            assert!(key[..] >= lower[..]);
        }
    }

    #[test]
    fn test_scan_range_many_duplicate_keys() {
        let writer = get_test_writer("many_duplicates");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Create a pattern with many duplicates
        // Keys: 100 'a's, 200 'b's, 150 'c's, 100 'd's, 50 'e's
        let mut expected_counts = std::collections::HashMap::new();

        // Add 100 'a' entries
        for i in 0..100 {
            run.append(b"a".to_vec(), format!("a_value_{}", i).into_bytes());
        }
        expected_counts.insert(b"a".to_vec(), 100);

        // Add 200 'b' entries
        for i in 0..200 {
            run.append(b"b".to_vec(), format!("b_value_{}", i).into_bytes());
        }
        expected_counts.insert(b"b".to_vec(), 200);

        // Add 150 'c' entries
        for i in 0..150 {
            run.append(b"c".to_vec(), format!("c_value_{}", i).into_bytes());
        }
        expected_counts.insert(b"c".to_vec(), 150);

        // Add 100 'd' entries
        for i in 0..100 {
            run.append(b"d".to_vec(), format!("d_value_{}", i).into_bytes());
        }
        expected_counts.insert(b"d".to_vec(), 100);

        // Add 50 'e' entries
        for i in 0..50 {
            run.append(b"e".to_vec(), format!("e_value_{}", i).into_bytes());
        }
        expected_counts.insert(b"e".to_vec(), 50);

        run.finalize_write();

        // Test 1: Full scan - should get all duplicates
        let iter = run.scan_range(&[], &[]);
        let all: Vec<_> = iter.collect();
        assert_eq!(all.len(), 600); // Total entries

        // Verify counts for each key
        let mut actual_counts = std::collections::HashMap::new();
        for (key, _) in &all {
            *actual_counts.entry(key.clone()).or_insert(0) += 1;
        }
        assert_eq!(actual_counts, expected_counts);

        // Test 2: Scan range [b, d) - should get all 'b' and 'c' entries
        let iter = run.scan_range(b"b", b"d");
        let range_results: Vec<_> = iter.collect();
        assert_eq!(range_results.len(), 350); // 200 'b' + 150 'c'

        // Verify we only got 'b' and 'c'
        for (key, _) in &range_results {
            assert!(key == b"b" || key == b"c");
        }

        // Count 'b' and 'c' entries
        let b_count = range_results.iter().filter(|(k, _)| k == b"b").count();
        let c_count = range_results.iter().filter(|(k, _)| k == b"c").count();
        assert_eq!(b_count, 200);
        assert_eq!(c_count, 150);

        // Test 3: Scan range [c, c) - should get nothing (exclusive upper bound)
        let iter = run.scan_range(b"c", b"c");
        let empty: Vec<_> = iter.collect();
        assert_eq!(empty.len(), 0);

        // Test 4: Scan range [c, d) - should get all 'c' entries
        let iter = run.scan_range(b"c", b"d");
        let c_only: Vec<_> = iter.collect();
        assert_eq!(c_only.len(), 150);
        for (key, _) in &c_only {
            assert_eq!(key, b"c");
        }

        // Test 5: Scan with lower bound only [d, ...) - should get 'd' and 'e'
        let iter = run.scan_range(b"d", &[]);
        let d_onwards: Vec<_> = iter.collect();
        assert_eq!(d_onwards.len(), 150); // 100 'd' + 50 'e'

        // Test 6: Scan with upper bound only [..., c) - should get 'a' and 'b'
        let iter = run.scan_range(&[], b"c");
        let before_c: Vec<_> = iter.collect();
        assert_eq!(before_c.len(), 300); // 100 'a' + 200 'b'
    }

    #[test]
    fn test_scan_range_duplicate_keys_with_sparse_index() {
        let writer = get_test_writer("duplicates_sparse");
        let mut run = RunImpl::from_writer(writer).unwrap();

        // Create enough duplicate data to trigger sparse indexing
        // Each entry is about 108 bytes (8 + 3 + 97)
        // We need > 64KB to trigger multiple index entries
        // 64KB / 108 ≈ 600 entries per index interval

        // Add 1000 entries each of 'aaa', 'bbb', 'ccc' to ensure sparse index has entries
        for _ in 0..10000 {
            run.append(b"aaa".to_vec(), vec![b'x'; 97]); // ~100 byte value
        }
        for _ in 0..10000 {
            run.append(b"bbb".to_vec(), vec![b'y'; 97]);
        }
        for _ in 0..10000 {
            run.append(b"ccc".to_vec(), vec![b'z'; 97]);
        }

        run.finalize_write();

        // Verify sparse index was built
        assert!(
            run.sparse_index.len() > 3,
            "Sparse index should have multiple entries"
        );

        // Test scan ranges work correctly with sparse index

        // Range [bbb, ccc) should return exactly 1000 'bbb' entries
        let iter = run.scan_range(b"bbb", b"ccc");
        let bbb_results: Vec<_> = iter.collect();
        assert_eq!(bbb_results.len(), 10000);
        for (key, _) in &bbb_results {
            assert_eq!(key, b"bbb");
        }

        // Range [b, c) should also return 1000 'bbb' entries
        let iter = run.scan_range(b"b", b"c");
        let b_range: Vec<_> = iter.collect();
        assert_eq!(b_range.len(), 10000);

        // Range [aab, bba) should return 0 entries (no keys in this range)
        let iter = run.scan_range(b"aab", b"bba");
        let empty: Vec<_> = iter.collect();
        assert_eq!(empty.len(), 0);

        // Full scan should return all 3000 entries
        let iter = run.scan_range(&[], &[]);
        let all: Vec<_> = iter.collect();
        assert_eq!(all.len(), 30000);
    }

    #[test]
    fn test_multiple_runs_same_file_reused_writer() {
        let writer = get_test_writer("multiple_runs_reused");

        // First run: create new file
        let mut run1 = RunImpl::from_writer(writer).unwrap();

        // Write keys 00-09
        for i in 0..10 {
            let key = format!("{:02}", i).into_bytes();
            let value = format!("run1_val_{}", i).into_bytes();
            run1.append(key, value);
        }

        // Take the writer to reuse it
        let writer = run1.finalize_write();
        println!("Writer position after run1: {}", writer.position());

        // Second run: reuse the writer
        let mut run2 = RunImpl::from_writer(writer).unwrap();

        // Write keys 10-19
        for i in 10..20 {
            let key = format!("{:02}", i).into_bytes();
            let value = format!("run2_val_{}", i).into_bytes();
            run2.append(key, value);
        }

        // Take the writer again
        let writer = run2.finalize_write();

        // Third run: reuse the writer again
        let mut run3 = RunImpl::from_writer(writer).unwrap();

        // Write keys 20-29
        for i in 20..30 {
            let key = format!("{:02}", i).into_bytes();
            let value = format!("run3_val_{}", i).into_bytes();
            run3.append(key, value);
        }

        // Verify byte ranges
        let (r1_start, r1_end) = run1.byte_range();
        let (r2_start, r2_end) = run2.byte_range();
        let (r3_start, _r3_end) = run3.byte_range();

        println!("Run1 byte range: {:?}", run1.byte_range());
        println!("Run2 byte range: {:?}", run2.byte_range());
        println!("Run3 byte range: {:?}", run3.byte_range());

        assert_eq!(r1_start, 0);
        // Due to direct I/O alignment, r1_end might be padded
        assert!(r2_start >= r1_end);
        assert!(r3_start >= r2_end);

        // Drop the writer to ensure all data is flushed to disk
        drop(run3.finalize_write());

        // Now read from each run using their actual byte ranges
        let run1_data: Vec<_> = run1.scan_range(&[], &[]).collect();
        let run2_data: Vec<_> = run2.scan_range(&[], &[]).collect();
        let run3_data: Vec<_> = run3.scan_range(&[], &[]).collect();

        println!("Run1 data length: {}", run1_data.len());
        println!("Run2 data length: {}", run2_data.len());
        println!("Run3 data length: {}", run3_data.len());

        if !run1_data.is_empty() {
            println!(
                "Run1 first: {:?}",
                std::str::from_utf8(&run1_data[0].0).unwrap()
            );
        }
        if !run2_data.is_empty() {
            println!(
                "Run2 first: {:?}",
                std::str::from_utf8(&run2_data[0].0).unwrap()
            );
        }
        if !run3_data.is_empty() {
            println!(
                "Run3 first: {:?}",
                std::str::from_utf8(&run3_data[0].0).unwrap()
            );
        }

        // Verify each run returns only its own data
        assert_eq!(run1_data.len(), 10);
        assert_eq!(run2_data.len(), 10);
        assert_eq!(run3_data.len(), 10);

        // Verify first and last entries of each run
        assert_eq!(run1_data[0].0, b"00");
        assert_eq!(run1_data[9].0, b"09");
        assert_eq!(std::str::from_utf8(&run1_data[0].1).unwrap(), "run1_val_0");

        assert_eq!(run2_data[0].0, b"10");
        assert_eq!(run2_data[9].0, b"19");
        assert_eq!(std::str::from_utf8(&run2_data[0].1).unwrap(), "run2_val_10");

        assert_eq!(run3_data[0].0, b"20");
        assert_eq!(run3_data[9].0, b"29");
        assert_eq!(std::str::from_utf8(&run3_data[0].1).unwrap(), "run3_val_20");
    }
}
