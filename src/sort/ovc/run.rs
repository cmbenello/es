use crate::diskio::aligned_reader::AlignedReader;
use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::constants::align_down;
use crate::diskio::file::SharedFd;
use crate::diskio::io_stats::IoStatsTracker;
use crate::ovc::offset_value_coding_32::OVCU32;
use crate::ovc::offset_value_coding_64::OVCFlag;
use crate::sort::core::engine::RunSummary;
use crate::sort::core::run_format::{
    IndexEntry, IndexingInterval, KeyRunIdOffsetBound, RUN_ID_COUNTER, cmp_key_run_offset,
};
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::Ordering;

// File-based run implementation with direct I/O
pub struct RunWithOVC {
    run_id: u32,
    fd: Arc<SharedFd>,
    writer: Option<AlignedWriter>,
    total_entries: usize,
    start_bytes: usize,
    total_bytes: usize,
    sparse_index: Vec<IndexEntry>,
    index_bytes: usize,
    indexing_interval: IndexingInterval,
    next_index_at_entry: usize,
    next_index_at_bytes: usize,
}

impl RunWithOVC {
    #[cfg(test)]
    pub(crate) fn from_writer(writer: AlignedWriter) -> Result<Self, String> {
        Self::from_writer_with_indexing_interval(writer, IndexingInterval::records(1000))
    }

    pub fn from_writer_with_indexing_interval(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
    ) -> Result<Self, String> {
        let run_id = RUN_ID_COUNTER.fetch_add(1, Ordering::AcqRel);
        Self::from_writer_with_indexing_interval_and_id(writer, indexing_interval, run_id)
    }

    pub fn from_writer_with_indexing_interval_and_id(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
        run_id: u32,
    ) -> Result<Self, String> {
        // Get current position in the file
        let start_bytes = writer.position() as usize;
        let fd = writer.get_fd();

        Ok(Self {
            run_id,
            fd,
            writer: Some(writer),
            total_entries: 0,
            start_bytes,
            total_bytes: 0,
            sparse_index: Vec::new(),
            index_bytes: 0,
            indexing_interval,
            next_index_at_entry: 0,
            next_index_at_bytes: 0,
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

    pub fn run_id(&self) -> u32 {
        self.run_id
    }

    pub fn start_key(&self) -> Option<&[u8]> {
        self.sparse_index.first().map(|entry| entry.key.as_slice())
    }

    pub fn sparse_index(&self) -> &Vec<IndexEntry> {
        &self.sparse_index
    }

    fn find_start_position(
        &self,
        lower_inc: Option<&KeyRunIdOffsetBound>,
    ) -> Option<(usize, Vec<u8>)> {
        if self.sparse_index.is_empty() {
            return None;
        }
        let mut best_entry = &self.sparse_index[0];
        if lower_inc.is_none() {
            return Some((best_entry.file_offset, best_entry.key.clone()));
        }

        // Binary search to find the last entry with key < lower_inc
        let mut left = 0;
        let mut right = self.sparse_index.len();
        let lower_inc = lower_inc.unwrap();

        while left < right {
            let mid = left + (right - left) / 2;
            let mid_entry = &self.sparse_index[mid];
            if cmp_key_run_offset(
                (&mid_entry.key, self.run_id, mid_entry.file_offset),
                (&lower_inc.key, lower_inc.run_id, lower_inc.offset),
            ) == std::cmp::Ordering::Less
            {
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
    fn should_sample_index(&self) -> bool {
        self.sparse_index.is_empty()
            || match self.indexing_interval {
                IndexingInterval::Records { .. } => self.total_entries >= self.next_index_at_entry,
                IndexingInterval::Bytes { .. } => self.total_bytes >= self.next_index_at_bytes,
            }
    }

    fn record_sparse_index(&mut self, key: &[u8]) {
        self.sparse_index.push(IndexEntry {
            key: key.to_vec(),
            file_offset: self.total_bytes,
        });
        let entry_bytes = std::mem::size_of::<IndexEntry>().saturating_add(key.len());
        self.index_bytes = self.index_bytes.saturating_add(entry_bytes);
        let interval = self.indexing_interval;
        match interval {
            IndexingInterval::Records {
                stride,
                budget_bytes,
            } => self.update_record_interval(stride, budget_bytes),
            IndexingInterval::Bytes {
                stride,
                budget_bytes,
            } => self.update_byte_interval(stride, budget_bytes),
        }
    }

    fn avg_index_entry_bytes(&self) -> usize {
        let entries = self.sparse_index.len().max(1);
        self.index_bytes.saturating_add(entries.saturating_sub(1)) / entries
    }

    fn update_record_interval(&mut self, stride: usize, budget_bytes: Option<usize>) {
        let records_seen = self.total_entries.saturating_add(1);
        match budget_bytes {
            Some(budget) => {
                if self.index_bytes >= budget {
                    self.indexing_interval = IndexingInterval::Records {
                        stride: usize::MAX,
                        budget_bytes: Some(budget),
                    };
                    self.next_index_at_entry = usize::MAX;
                } else {
                    let avg_entry_bytes = self.avg_index_entry_bytes();
                    let max_entries = (budget / avg_entry_bytes).max(1);
                    let target_stride =
                        records_seen.saturating_add(max_entries.saturating_sub(1)) / max_entries;
                    let new_stride = stride.max(target_stride.max(1));
                    self.indexing_interval = IndexingInterval::Records {
                        stride: new_stride,
                        budget_bytes: Some(budget),
                    };
                    self.next_index_at_entry = self.total_entries.saturating_add(new_stride);
                }
            }
            None => {
                self.next_index_at_entry = self.total_entries.saturating_add(stride);
            }
        }
    }

    fn update_byte_interval(&mut self, stride: usize, budget_bytes: Option<usize>) {
        match budget_bytes {
            Some(budget) => {
                if self.index_bytes >= budget {
                    self.indexing_interval = IndexingInterval::Bytes {
                        stride: usize::MAX,
                        budget_bytes: Some(budget),
                    };
                    self.next_index_at_bytes = usize::MAX;
                } else {
                    let avg_entry_bytes = self.avg_index_entry_bytes();
                    let target_stride = avg_entry_bytes
                        .saturating_mul(self.total_bytes)
                        .saturating_add(budget.saturating_sub(1))
                        / budget;
                    let new_stride = stride.max(target_stride.max(1));
                    self.indexing_interval = IndexingInterval::Bytes {
                        stride: new_stride,
                        budget_bytes: Some(budget),
                    };
                    self.next_index_at_bytes = self.total_bytes.saturating_add(new_stride);
                }
            }
            None => {
                self.next_index_at_bytes = self.total_bytes.saturating_add(stride);
            }
        }
    }

    pub fn append(&mut self, ovc: OVCU32, key: &[u8], value: &[u8]) {
        if self.writer.is_none() {
            panic!("RunWithOVC is not initialized with a writer");
        }

        // Use sampling interval for sparse index
        if self.should_sample_index() {
            self.record_sparse_index(key);
        }

        let writer = self.writer.as_mut().unwrap();

        let value_len = value.len() as u32;

        // Optimize for duplicate keys: skip key_len field since it's always 0
        if ovc.flag() == OVCFlag::DuplicateValue {
            writer.write_all(&ovc.to_le_bytes()).unwrap();
            writer.write_all(&value_len.to_le_bytes()).unwrap();
            writer.write_all(&value).unwrap();
            self.total_bytes += 4 + 4 + value.len();
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

        self.total_bytes += 4 + 8 + truncated_key.len() + value.len();
        self.total_entries += 1;
    }

    pub fn scan_range(
        &self,
        lower_inc: Option<(&[u8], u32, usize)>,
        upper_exc: Option<(&[u8], u32, usize)>,
    ) -> Box<dyn Iterator<Item = (OVCU32, Vec<u8>, Vec<u8>)> + Send> {
        self.scan_range_with_io_tracker(lower_inc, upper_exc, None)
    }

    pub fn scan_range_with_io_tracker(
        &self,
        lower_inc: Option<(&[u8], u32, usize)>,
        upper_exc: Option<(&[u8], u32, usize)>,
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = (OVCU32, Vec<u8>, Vec<u8>)> + Send> {
        let lower_inc = lower_inc.map(KeyRunIdOffsetBound::from_components);
        let upper_exc = upper_exc.map(KeyRunIdOffsetBound::from_components);
        self.scan_range_with_parsed_bounds(lower_inc, upper_exc, io_tracker)
    }

    fn scan_range_with_parsed_bounds(
        &self,
        lower_inc: Option<KeyRunIdOffsetBound>,
        upper_exc: Option<KeyRunIdOffsetBound>,
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = (OVCU32, Vec<u8>, Vec<u8>)> + Send> {
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
        let (offset, start_key) = self.find_start_position(lower_inc.as_ref()).unwrap();
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
                run_id: self.run_id,
                prev_key: start_key,
                lower_inc,
                upper_exc,
                file_pos: aligned_offset,
                run_len_bytes: self.total_bytes,
                align_skip_bytes: skip_bytes,
                run_start: self.start_bytes,
                done: false,
            }) as Box<dyn Iterator<Item = (OVCU32, Vec<u8>, Vec<u8>)> + Send>;
        }

        Box::new(RunIteratorWithOVC {
            reader,
            run_id: self.run_id,
            prev_key: start_key,
            lower_inc,
            upper_exc,
            file_pos: self.start_bytes, // Start from the beginning of this run
            run_len_bytes: self.total_bytes,
            align_skip_bytes: 0,
            run_start: self.start_bytes,
            done: false,
        }) as Box<dyn Iterator<Item = (OVCU32, Vec<u8>, Vec<u8>)> + Send>
    }
}

struct RunIteratorWithOVC {
    reader: AlignedReader,
    run_id: u32,
    prev_key: Vec<u8>,
    lower_inc: Option<KeyRunIdOffsetBound>,
    upper_exc: Option<KeyRunIdOffsetBound>,
    /// Absolute file offset where the reader currently is.
    file_pos: usize,
    /// Total bytes in this run (length from `run_start`).
    run_len_bytes: usize,
    /// Bytes to skip after seeking to aligned position (absolute).
    align_skip_bytes: usize,
    /// Absolute file offset where this run begins.
    run_start: usize,
    /// Once we return None, remain exhausted.
    done: bool,
}

impl Iterator for RunIteratorWithOVC {
    type Item = (OVCU32, Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        use std::io::ErrorKind;

        if self.done {
            return None;
        }

        // First, skip bytes if we sought to an aligned position
        if self.align_skip_bytes > 0 {
            let mut skip_buf = vec![0u8; self.align_skip_bytes];
            self.reader
                .read_exact(&mut skip_buf)
                .expect("Failed to skip bytes after seek");
            self.file_pos += self.align_skip_bytes;
            self.align_skip_bytes = 0;
        }

        loop {
            // Check if we've read all the actual data for this run
            if self.run_len_bytes > 0 && self.file_pos - self.run_start >= self.run_len_bytes {
                self.done = true;
                return None;
            }

            // Read OVC
            let mut ovc_bytes = [0u8; 4];
            match self.reader.read_exact(&mut ovc_bytes) {
                Ok(_) => {}
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                    // Legitimate EOF - we've reached the end of data
                    self.done = true;
                    return None;
                }
                Err(e) => {
                    panic!("Failed to read OVC: {}", e);
                }
            }
            let ovc = OVCU32::from_le_bytes(ovc_bytes);
            let cur_offset = self.file_pos - self.run_start;

            // Optimize for duplicate keys: skip reading key_len, just use prev_key
            if ovc.flag() == OVCFlag::DuplicateValue {
                // Read value length
                let mut value_len_bytes = [0u8; 4];
                self.reader
                    .read_exact(&mut value_len_bytes)
                    .expect("Failed to read value length");
                let value_len = u32::from_le_bytes(value_len_bytes) as usize;

                let key_ref = &self.prev_key;

                // Check if key is in range [lower_inc, upper_exc)
                if let Some(lb) = &self.lower_inc {
                    if lb.lt(key_ref, self.run_id, cur_offset) {
                        let new_pos = self.reader.position().saturating_add(value_len as u64);
                        self.reader
                            .seek(new_pos)
                            .expect("Failed to skip value bytes");
                        // Update bytes read (no key_len field for duplicates)
                        self.file_pos += 4 + 4 + value_len;
                        continue;
                    }
                }
                if let Some(ub) = &self.upper_exc {
                    if ub.ge(key_ref, self.run_id, cur_offset) {
                        self.done = true;
                        return None;
                    }
                }

                // Read value
                let mut value = vec![0u8; value_len];
                self.reader
                    .read_exact(&mut value)
                    .expect("Failed to read value");

                // Update bytes read (no key_len field for duplicates)
                self.file_pos += 4 + 4 + value_len;

                let key = key_ref.clone();
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

            // Check if key is in range [lower_inc, upper_exc)
            if let Some(lb) = &self.lower_inc {
                if lb.lt(&key, self.run_id, cur_offset) {
                    let new_pos = self.reader.position().saturating_add(value_len as u64);
                    self.reader
                        .seek(new_pos)
                        .expect("Failed to skip value bytes");
                    // Update bytes read
                    self.file_pos += 4 + 8 + truncated_key_len + value_len;
                    continue;
                }
            }
            if let Some(ub) = &self.upper_exc {
                if ub.ge(&key, self.run_id, cur_offset) {
                    // Since data is sorted in runs, we can stop here
                    self.done = true;
                    return None;
                }
            }

            // Read value
            let mut value = vec![0u8; value_len];
            self.reader
                .read_exact(&mut value)
                .expect("Failed to read value");

            // Update bytes read
            self.file_pos += 4 + 8 + truncated_key_len + value_len;

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
        run.append(OVCU32::initial_value(), b"key1", b"value1");
        run.append(OVCU32::initial_value(), b"key2", b"value2");
        run.append(OVCU32::initial_value(), b"key3", b"value3");

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
        run.append(OVCU32::initial_value(), b"a", b"1");
        run.append(OVCU32::initial_value(), b"b", b"2");
        run.append(OVCU32::initial_value(), b"c", b"3");

        let mut writer = run.finalize_write();
        writer.flush().unwrap();

        // Read all data back
        let results: Vec<_> = run.scan_range(None, None).collect();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, OVCU32::initial_value());
        assert_eq!(results[0].1, b"a");
        assert_eq!(results[0].2, b"1");
        assert_eq!(results[1].0, OVCU32::initial_value());
        assert_eq!(results[1].1, b"b");
        assert_eq!(results[1].2, b"2");
        assert_eq!(results[2].0, OVCU32::initial_value());
        assert_eq!(results[2].1, b"c");
        assert_eq!(results[2].2, b"3");
    }

    #[test]
    fn test_scan_range_with_bounds() {
        let fd = create_test_file("test_scan_bounds.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();

        // Write sorted data
        run.append(OVCU32::initial_value(), b"a", b"1");
        run.append(OVCU32::initial_value(), b"b", b"2");
        run.append(OVCU32::initial_value(), b"c", b"3");
        run.append(OVCU32::initial_value(), b"d", b"4");
        run.append(OVCU32::initial_value(), b"e", b"5");

        let mut writer = run.finalize_write();
        writer.flush().unwrap();

        // Scan with bounds [b, d)
        let lower = Some((b"b".as_ref(), 0, 0));
        let upper = Some((b"d".as_ref(), 0, 0));
        let results: Vec<_> = run.scan_range(lower, upper).collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, OVCU32::initial_value());
        assert_eq!(results[0].1, b"b");
        assert_eq!(results[0].2, b"2");
        assert_eq!(results[1].0, OVCU32::initial_value());
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
        let results: Vec<_> = run.scan_range(None, None).collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_sparse_index_creation() {
        let fd = create_test_file("test_sparse_index.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();
        let indexing_interval = 5;
        run.indexing_interval = IndexingInterval::records(indexing_interval);

        // Add more entries than reservoir size
        for i in 0..20 {
            let key = format!("key_{:02}", i);
            run.append(OVCU32::initial_value(), &key.into_bytes(), b"value");
        }

        let writer = run.finalize_write();
        drop(writer); // Close writer to flush data

        // Sparse index should have entries based on sampling interval
        // With interval 5 and 20 entries (0-19), we sample at: 0, 5, 10, 15
        let expected_entries = (0..20).filter(|&i| i % indexing_interval == 0).count();
        assert_eq!(run.sparse_index.len(), expected_entries);

        let first_entry = run.sparse_index.first().unwrap();
        assert_eq!(first_entry.key, b"key_00");

        for (i, entry) in run.sparse_index.iter().enumerate() {
            let expected_index = i * indexing_interval;
            let expected_key = format!("key_{:02}", expected_index);
            assert_eq!(entry.key, expected_key.as_bytes());
        }

        // Sparse index should be sorted by file offset
        for i in 1..run.sparse_index.len() {
            assert!(run.sparse_index[i].file_offset >= run.sparse_index[i - 1].file_offset);
        }
    }

    #[test]
    fn test_record_budget_adaptive_sampling() {
        let fd = create_test_file("test_record_budget_adaptive.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();
        let stride = 7;
        let key_len = 4;
        let index_entry_bytes = std::mem::size_of::<IndexEntry>() + key_len;
        let budget = index_entry_bytes * 3;
        let mut run = RunWithOVC::from_writer_with_indexing_interval(
            writer,
            IndexingInterval::records_with_budget(stride, budget),
        )
        .unwrap();

        for i in 0..50 {
            let key = format!("{:04}", i).into_bytes();
            run.append(OVCU32::initial_value(), &key, b"v");
        }

        let writer = run.finalize_write();
        drop(writer);

        assert_eq!(run.sparse_index.len(), 3);
        assert_eq!(run.sparse_index.first().unwrap().file_offset, 0);
        let expected_indices = [0, stride, stride * 2];
        for (entry, expected_index) in run.sparse_index.iter().zip(expected_indices) {
            let expected_key = format!("{:04}", expected_index);
            assert_eq!(entry.key, expected_key.as_bytes());
        }
        let index_bytes = run.sparse_index.len() * index_entry_bytes;
        assert_eq!(index_bytes, budget);
    }

    #[test]
    fn test_byte_budget_adaptive_sampling() {
        let fd = create_test_file("test_byte_budget_adaptive.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();
        let stride = 64;
        let key_len = 3;
        let index_entry_bytes = std::mem::size_of::<IndexEntry>() + key_len;
        let budget = index_entry_bytes * 2;
        let mut run = RunWithOVC::from_writer_with_indexing_interval(
            writer,
            IndexingInterval::bytes_with_budget(stride, budget),
        )
        .unwrap();

        for i in 0..50 {
            let key = format!("{:03}", i).into_bytes();
            run.append(OVCU32::initial_value(), &key, b"v");
        }

        let writer = run.finalize_write();
        drop(writer);

        assert_eq!(run.sparse_index.len(), 2);
        let offsets: Vec<usize> = run
            .sparse_index
            .iter()
            .map(|entry| entry.file_offset)
            .collect();
        assert_eq!(offsets[0], 0);
        assert!(offsets[1] >= stride);
        let index_bytes = run.sparse_index.len() * index_entry_bytes;
        assert_eq!(index_bytes, budget);
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
            },
            IndexEntry {
                key: b"c".to_vec(),
                file_offset: 100,
            },
            IndexEntry {
                key: b"e".to_vec(),
                file_offset: 200,
            },
        ];

        let b_a = KeyRunIdOffsetBound::from_components((b"a", 0, 0));
        let b_b = KeyRunIdOffsetBound::from_components((b"b", 0, 0));
        let b_d = KeyRunIdOffsetBound::from_components((b"d", 0, 0));
        let b_f = KeyRunIdOffsetBound::from_components((b"f", 0, 0));

        // Test finding start position
        assert_eq!(
            run.find_start_position(Some(&b_a)),
            Some((0, b"b".to_vec()))
        ); // The first entry will be always returned
        assert_eq!(
            run.find_start_position(Some(&b_b)),
            Some((0, b"b".to_vec()))
        );
        assert_eq!(
            run.find_start_position(Some(&b_d)),
            Some((100, b"c".to_vec()))
        );
        assert_eq!(
            run.find_start_position(Some(&b_f)),
            Some((200, b"e".to_vec()))
        );
    }

    #[test]
    fn test_large_values() {
        let fd = create_test_file("test_large_values.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();

        // Create large values
        let large_value = vec![b'x'; 1000];

        run.append(OVCU32::initial_value(), b"key1", &large_value);
        run.append(OVCU32::initial_value(), b"key2", &large_value);

        let mut writer = run.finalize_write();
        writer.flush().unwrap();

        // Read back and verify
        let results: Vec<_> = run.scan_range(None, None).collect();

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

        run.append(OVCU32::initial_value(), b"key", b"value");
        run.append(OVCU32::initial_value(), b"key2", b"value2");

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
        run.append(OVCU32::normal_value(&[10], 0), b"key1", b"val1");
        run.append(OVCU32::normal_value(&[20], 0), b"key2", b"val2");
        run.append(OVCU32::normal_value(&[30], 0), b"key3", b"val3");

        let writer = run.finalize_write();
        drop(writer); // Close writer to flush data

        // Read back and verify OVC values
        let results: Vec<_> = run.scan_range(None, None).collect();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, OVCU32::normal_value(&[10], 0));
        assert_eq!(results[1].0, OVCU32::normal_value(&[20], 0));
        assert_eq!(results[2].0, OVCU32::normal_value(&[30], 0));
    }
}
