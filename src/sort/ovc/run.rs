use crate::diskio::aligned_reader::AlignedReader;
use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::constants::align_down;
use crate::diskio::file::SharedFd;
use crate::diskio::io_stats::IoStatsTracker;
use crate::ovc::offset_value_coding_32::OVCU32;
use crate::ovc::offset_value_coding_64::OVCFlag;
use crate::sort::core::engine::RunSummary;
use crate::sort::core::run_format::{
    IndexingInterval, KeyRunIdOffsetBound, RUN_ID_COUNTER, cmp_key_run_offset,
};
use crate::sort::core::sparse_index::{SparseIndex, SparseIndexPagePool};
use std::cell::UnsafeCell;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

// File-based run implementation with direct I/O
pub struct RunWithOVC {
    run_id: u32,
    fd: Arc<SharedFd>,
    writer: Option<AlignedWriter>,
    total_entries: usize,
    start_bytes: usize,
    total_bytes: usize,
    sparse_index: UnsafeCell<SparseIndex>,
    sparse_index_refcount: AtomicUsize,
}

// SAFETY: RunWithOVC is shared across threads via Arc during merge.
// The sparse_index UnsafeCell is safe because:
// - During write phase: single-threaded, only the owner mutates
// - During merge: all threads read concurrently (no mutation),
//   then each calls release_sparse_index(); the last thread (atomic
//   decrement to 0) clears the index with no concurrent readers remaining.
// - RunIteratorWithOVC does NOT hold a reference to the sparse index.
unsafe impl Sync for RunWithOVC {}

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
            sparse_index: UnsafeCell::new(SparseIndex::new(indexing_interval)),
            sparse_index_refcount: AtomicUsize::new(0),
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
        // SAFETY: No concurrent mutation during reads.
        let index = unsafe { &*self.sparse_index.get() };
        index.first_key()
    }

    pub fn sparse_index(&self) -> &SparseIndex {
        // SAFETY: No concurrent mutation during reads.
        unsafe { &*self.sparse_index.get() }
    }

    /// Set the number of threads that will read the sparse index.
    /// Must be called before spawning threads.
    pub fn set_sparse_index_readers(&self, count: usize) {
        self.sparse_index_refcount.store(count, Ordering::Release);
    }

    /// Decrement the sparse index refcount. The thread that decrements to 0
    /// clears the sparse index, freeing the memory.
    pub fn release_sparse_index(&self) {
        let prev = self.sparse_index_refcount.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            // Last reader: safe to clear since no other thread will access it.
            // SAFETY: All concurrent readers have finished before decrementing.
            let index = unsafe { &mut *self.sparse_index.get() };
            index.clear();
        }
    }

    /// Decrement the sparse index refcount. The last thread drains pages to the pool.
    pub fn release_sparse_index_to_pool(&self, pool: &SparseIndexPagePool) {
        let prev = self.sparse_index_refcount.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            let index = unsafe { &mut *self.sparse_index.get() };
            pool.return_pages(index.take_buffer());
        }
    }

    /// Get mutable access to the sparse index (for seeding buffer).
    /// SAFETY: Caller must ensure no concurrent access.
    pub fn sparse_index_mut(&self) -> &mut SparseIndex {
        unsafe { &mut *self.sparse_index.get() }
    }

    fn find_start_position(
        &self,
        lower_inc: Option<&KeyRunIdOffsetBound>,
    ) -> Option<(usize, Vec<u8>)> {
        // SAFETY: No concurrent mutation during reads.
        let sparse_index = unsafe { &*self.sparse_index.get() };
        if sparse_index.is_empty() {
            return None;
        }
        let mut best_idx: usize = 0;
        if lower_inc.is_none() {
            return Some((sparse_index.file_offset(0), sparse_index.key(0).to_vec()));
        }

        // Binary search to find the last entry with key < lower_inc
        let mut left = 0;
        let mut right = sparse_index.len();
        let lower_inc = lower_inc.unwrap();

        while left < right {
            let mid = left + (right - left) / 2;
            if cmp_key_run_offset(
                (
                    sparse_index.key(mid),
                    self.run_id,
                    sparse_index.file_offset(mid),
                ),
                (&lower_inc.key, lower_inc.run_id, lower_inc.offset),
            ) == std::cmp::Ordering::Less
            {
                best_idx = mid;
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        Some((
            sparse_index.file_offset(best_idx),
            sparse_index.key(best_idx).to_vec(),
        ))
    }
}

impl RunWithOVC {
    fn should_sample_index(&self) -> bool {
        // SAFETY: Called only during single-threaded write phase.
        let sparse_index = unsafe { &*self.sparse_index.get() };
        sparse_index.should_sample(self.total_entries, self.total_bytes)
    }

    pub fn set_sparse_index_bootstrap(
        &mut self,
        avg_key_bytes: f64,
        avg_record_bytes: f64,
        sample_count: usize,
    ) {
        let sparse_index = self.sparse_index.get_mut();
        sparse_index.set_bootstrap(avg_key_bytes, avg_record_bytes, sample_count);
    }

    pub fn sparse_index_bootstrap(&self) -> Option<(f64, f64, usize)> {
        let sparse_index = unsafe { &*self.sparse_index.get() };
        sparse_index.bootstrap()
    }

    fn observe_record(&mut self, key_len: usize, record_bytes: usize) {
        let sparse_index = self.sparse_index.get_mut();
        sparse_index.observe_record(key_len, record_bytes);
    }

    fn record_sparse_index(
        &mut self,
        key: &[u8],
        sampled_record_index: usize,
        sampled_file_offset: usize,
        _record_bytes: usize,
    ) {
        // SAFETY: Called only during single-threaded write phase.
        let sparse_index = self.sparse_index.get_mut();
        sparse_index.record_sample(
            key,
            sampled_file_offset,
            sampled_record_index,
            sampled_file_offset,
        );
    }

    pub fn append(&mut self, ovc: OVCU32, key: &[u8], value: &[u8]) {
        if self.writer.is_none() {
            panic!("RunWithOVC is not initialized with a writer");
        }

        // Use sampling interval for sparse index
        let should_sample = self.should_sample_index();
        let record_index = self.total_entries;
        let record_offset = self.total_bytes;

        let writer = self.writer.as_mut().unwrap();

        let value_len = value.len() as u32;

        // Optimize for duplicate keys: skip key_len field since it's always 0
        if ovc.flag() == OVCFlag::DuplicateValue {
            writer.write_all(&ovc.to_le_bytes()).unwrap();
            writer.write_all(&value_len.to_le_bytes()).unwrap();
            writer.write_all(&value).unwrap();
            let record_bytes = 4 + 4 + value.len();
            self.total_bytes += record_bytes;
            self.total_entries += 1;
            self.observe_record(key.len(), record_bytes);
            if should_sample {
                self.record_sparse_index(key, record_index, record_offset, record_bytes);
            }
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

        let record_bytes = 4 + 8 + truncated_key.len() + value.len();
        self.total_bytes += record_bytes;
        self.total_entries += 1;
        self.observe_record(key.len(), record_bytes);
        if should_sample {
            self.record_sparse_index(key, record_index, record_offset, record_bytes);
        }
    }

    /// Write only the header (OVC + key_len + value_len) and key to the output.
    /// The caller is responsible for writing the value bytes separately via `writer()`.
    pub fn append_header_and_key(&mut self, ovc: OVCU32, key: &[u8], value_len: usize) {
        if self.writer.is_none() {
            panic!("RunWithOVC is not initialized with a writer");
        }

        let should_sample = self.should_sample_index();
        let record_index = self.total_entries;
        let record_offset = self.total_bytes;

        let writer = self.writer.as_mut().unwrap();
        let value_len_u32 = value_len as u32;

        if ovc.flag() == OVCFlag::DuplicateValue {
            writer.write_all(&ovc.to_le_bytes()).unwrap();
            writer.write_all(&value_len_u32.to_le_bytes()).unwrap();
            // No key bytes for duplicates; value written by caller
            let record_bytes = 4 + 4 + value_len;
            self.total_bytes += record_bytes;
            self.total_entries += 1;
            self.observe_record(key.len(), record_bytes);
            if should_sample {
                self.record_sparse_index(key, record_index, record_offset, record_bytes);
            }
            return;
        }

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
        writer.write_all(&value_len_u32.to_le_bytes()).unwrap();
        writer.write_all(truncated_key).unwrap();
        // Value written by caller via writer()

        let record_bytes = 4 + 8 + truncated_key.len() + value_len;
        self.total_bytes += record_bytes;
        self.total_entries += 1;
        self.observe_record(key.len(), record_bytes);
        if should_sample {
            self.record_sparse_index(key, record_index, record_offset, record_bytes);
        }
    }

    /// Get a mutable reference to the underlying writer.
    /// Used for zero-copy value transfer from source reader.
    pub fn writer(&mut self) -> &mut AlignedWriter {
        self.writer.as_mut().expect("RunWithOVC has no writer")
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

    /// Create a MergeSource for zero-copy merge.
    /// Returns None for empty runs.
    pub fn scan_range_as_source(
        &self,
        lower_inc: Option<(&[u8], u32, usize)>,
        upper_exc: Option<(&[u8], u32, usize)>,
        io_tracker: Option<IoStatsTracker>,
    ) -> Option<RunIteratorWithOVC> {
        if self.total_entries == 0 {
            return None;
        }

        let lower_inc_bound = lower_inc.map(KeyRunIdOffsetBound::from_components);
        let upper_exc_bound = upper_exc.map(KeyRunIdOffsetBound::from_components);

        let mut reader = if let Some(tracker) = io_tracker {
            AlignedReader::from_fd_with_tracer(self.fd.clone(), Some(tracker)).unwrap()
        } else {
            AlignedReader::from_fd(self.fd.clone()).unwrap()
        };

        let (offset, start_key) = self.find_start_position(lower_inc_bound.as_ref()).unwrap();
        let start_offset = self.start_bytes + offset;

        if start_offset > 0 {
            let aligned_offset = align_down(start_offset as u64, 4096) as usize;
            let skip_bytes = start_offset - aligned_offset;
            if aligned_offset > 0 {
                reader.seek(aligned_offset as u64).unwrap();
            }
            Some(RunIteratorWithOVC {
                reader,
                run_id: self.run_id,
                prev_key: start_key,
                lower_inc: lower_inc_bound,
                upper_exc: upper_exc_bound,
                file_pos: aligned_offset,
                run_len_bytes: self.total_bytes,
                align_skip_bytes: skip_bytes,
                run_start: self.start_bytes,
                done: false,
            })
        } else {
            Some(RunIteratorWithOVC {
                reader,
                run_id: self.run_id,
                prev_key: start_key,
                lower_inc: lower_inc_bound,
                upper_exc: upper_exc_bound,
                file_pos: self.start_bytes,
                run_len_bytes: self.total_bytes,
                align_skip_bytes: 0,
                run_start: self.start_bytes,
                done: false,
            })
        }
    }
}

pub struct RunIteratorWithOVC {
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

use crate::sort::ovc::merge::MergeSource;

impl MergeSource for RunIteratorWithOVC {
    fn next_key_into(&mut self, key: &mut Vec<u8>) -> Option<(OVCU32, usize)> {
        use std::io::ErrorKind;

        if self.done {
            return None;
        }

        // Skip alignment bytes on first call
        if self.align_skip_bytes > 0 {
            let mut skip_buf = vec![0u8; self.align_skip_bytes];
            self.reader
                .read_exact(&mut skip_buf)
                .expect("Failed to skip bytes after seek");
            self.file_pos += self.align_skip_bytes;
            self.align_skip_bytes = 0;
        }

        loop {
            // Check if we've read all data for this run
            if self.run_len_bytes > 0 && self.file_pos - self.run_start >= self.run_len_bytes {
                self.done = true;
                return None;
            }

            // Read OVC
            let mut ovc_bytes = [0u8; 4];
            match self.reader.read_exact(&mut ovc_bytes) {
                Ok(_) => {}
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                    self.done = true;
                    return None;
                }
                Err(e) => panic!("Failed to read OVC: {}", e),
            }
            let ovc = OVCU32::from_le_bytes(ovc_bytes);
            let cur_offset = self.file_pos - self.run_start;

            // Duplicate key: no key_len field
            if ovc.flag() == OVCFlag::DuplicateValue {
                let mut value_len_bytes = [0u8; 4];
                self.reader
                    .read_exact(&mut value_len_bytes)
                    .expect("Failed to read value length");
                let value_len = u32::from_le_bytes(value_len_bytes) as usize;

                let key_ref = &self.prev_key;

                // Range check
                if let Some(lb) = &self.lower_inc {
                    if lb.lt(key_ref, self.run_id, cur_offset) {
                        // Skip value and continue
                        let new_pos = self.reader.position().saturating_add(value_len as u64);
                        self.reader.seek(new_pos).expect("Failed to skip value");
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

                // Copy prev_key into the provided buffer
                key.resize(self.prev_key.len(), 0);
                key.copy_from_slice(&self.prev_key);

                // Header bytes consumed: ovc(4) + value_len(4) = 8
                // Value bytes NOT consumed yet (reader is at value start)
                // We track the header bytes; value bytes tracked in copy_value_to/skip_value
                self.file_pos += 4 + 4;

                return Some((ovc, value_len));
            }

            // Normal/initial key: read key_len + value_len + key
            let mut key_len_bytes = [0u8; 4];
            self.reader
                .read_exact(&mut key_len_bytes)
                .expect("Failed to read key length");
            let truncated_key_len = u32::from_le_bytes(key_len_bytes) as usize;

            let mut value_len_bytes = [0u8; 4];
            self.reader
                .read_exact(&mut value_len_bytes)
                .expect("Failed to read value length");
            let value_len = u32::from_le_bytes(value_len_bytes) as usize;

            let offset = match ovc.flag() {
                OVCFlag::EarlyFence | OVCFlag::LateFence => {
                    panic!("EarlyFence and LateFence OVC flags are not supported");
                }
                OVCFlag::InitialValue => 0,
                OVCFlag::NormalValue => ovc.offset(),
                OVCFlag::DuplicateValue => unreachable!(),
            };

            // Reconstruct full key into the provided buffer
            key.resize(offset + truncated_key_len, 0);
            key[..offset].copy_from_slice(&self.prev_key[..offset]);

            self.reader
                .read_exact(&mut key[offset..])
                .expect("Failed to read key");

            // Update prev_key
            self.prev_key.resize(key.len(), 0);
            self.prev_key.copy_from_slice(key);

            // Range check
            if let Some(lb) = &self.lower_inc {
                if lb.lt(key, self.run_id, cur_offset) {
                    // Skip value and continue
                    let new_pos = self.reader.position().saturating_add(value_len as u64);
                    self.reader.seek(new_pos).expect("Failed to skip value");
                    self.file_pos += 4 + 8 + truncated_key_len + value_len;
                    continue;
                }
            }
            if let Some(ub) = &self.upper_exc {
                if ub.ge(key, self.run_id, cur_offset) {
                    self.done = true;
                    return None;
                }
            }

            // Header bytes consumed: ovc(4) + key_len(4) + value_len(4) + key
            // Value bytes NOT consumed yet
            self.file_pos += 4 + 8 + truncated_key_len;

            return Some((ovc, value_len));
        }
    }

    fn copy_value_to(&mut self, writer: &mut dyn std::io::Write, len: usize) {
        use std::io::BufRead;
        if len == 0 {
            return;
        }
        // Zero-copy transfer: pass slices from reader's internal 64KB buffer
        // directly to writer, avoiding any intermediate heap/stack buffer.
        let mut remaining = len;
        while remaining > 0 {
            let buf = self
                .reader
                .fill_buf()
                .expect("Failed to fill reader buffer");
            let available = buf.len().min(remaining);
            writer
                .write_all(&buf[..available])
                .expect("Failed to write value bytes");
            self.reader.consume(available);
            remaining -= available;
        }
        self.file_pos += len;
    }

    fn skip_value(&mut self, len: usize) {
        if len == 0 {
            return;
        }
        let new_pos = self.reader.position().saturating_add(len as u64);
        self.reader.seek(new_pos).expect("Failed to skip value");
        self.file_pos += len;
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
        assert!(run.sparse_index().is_empty());
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

        let indexing_interval = 5;
        let mut run = RunWithOVC::from_writer_with_indexing_interval(
            writer,
            IndexingInterval::records(indexing_interval),
        )
        .unwrap();

        // Add more entries than reservoir size
        for i in 0..20 {
            let key = format!("key_{:02}", i);
            run.append(OVCU32::initial_value(), &key.into_bytes(), b"value");
        }

        let writer = run.finalize_write();
        drop(writer); // Close writer to flush data

        // Sparse index should have entries based on sampling interval
        // With interval 5 and 20 entries (0-19), we sample at: 0, 5, 10, 15
        let si = run.sparse_index();
        let expected_entries = (0..20).filter(|&i| i % indexing_interval == 0).count();
        assert_eq!(si.len(), expected_entries);

        assert_eq!(si.key(0), b"key_00");

        for i in 0..si.len() {
            let expected_index = i * indexing_interval;
            let expected_key = format!("key_{:02}", expected_index);
            assert_eq!(si.key(i), expected_key.as_bytes());
        }

        // Sparse index should be sorted by file offset
        for i in 1..si.len() {
            assert!(si.file_offset(i) >= si.file_offset(i - 1));
        }
    }

    #[test]
    fn test_record_budget_ignored_fixed_stride() {
        let fd = create_test_file("test_record_budget_adaptive.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();
        let stride = 7;
        let mut run = RunWithOVC::from_writer_with_indexing_interval(
            writer,
            IndexingInterval::records_with_budget(stride, 1),
        )
        .unwrap();

        let total_entries = 50;
        for i in 0..total_entries {
            let key = format!("{:04}", i).into_bytes();
            run.append(OVCU32::initial_value(), &key, b"v");
        }

        let writer = run.finalize_write();
        drop(writer);

        let si = run.sparse_index();
        let expected_entries = (total_entries + stride - 1) / stride;
        assert_eq!(si.len(), expected_entries);
        assert_eq!(si.file_offset(0), 0);
        for i in 0..si.len() {
            let expected_index = i * stride;
            let expected_key = format!("{:04}", expected_index);
            assert_eq!(si.key(i), expected_key.as_bytes());
        }
    }

    #[test]
    fn test_byte_budget_ignored_fixed_stride() {
        let fd = create_test_file("test_byte_budget_adaptive.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();
        let stride = 64;
        let mut run = RunWithOVC::from_writer_with_indexing_interval(
            writer,
            IndexingInterval::bytes_with_budget(stride, 1),
        )
        .unwrap();

        for i in 0..50 {
            let key = format!("{:03}", i).into_bytes();
            run.append(OVCU32::initial_value(), &key, b"v");
        }

        let writer = run.finalize_write();
        drop(writer);

        let si = run.sparse_index();
        let offsets: Vec<usize> = (0..si.len()).map(|i| si.file_offset(i)).collect();
        assert_eq!(offsets[0], 0);
        assert!(offsets.len() > 2);
        for pair in offsets.windows(2) {
            assert!(pair[1] >= pair[0] + stride);
        }
    }

    #[test]
    fn test_find_start_position() {
        let fd = create_test_file("test_find_start.dat");
        let writer = AlignedWriter::from_fd(fd.clone()).unwrap();

        let mut run = RunWithOVC::from_writer(writer).unwrap();

        // Manually create sparse index for testing
        let si = run.sparse_index.get_mut();
        si.push(b"b", 0);
        si.push(b"c", 100);
        si.push(b"e", 200);

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
