use std::sync::Mutex;

use crate::sort::core::run_format::IndexingInterval;

/// 4KB page size for sparse index storage.
///
/// Small pages minimise per-run waste: a run with a few hundred entries
/// only needs a handful of 4KB pages instead of one 64KB page that is
/// mostly empty.
pub const SPARSE_INDEX_PAGE_SIZE: usize = 4 * 1024;

/// Bit width for byte offsets within a page (12 bits for 4KB pages).
///
/// Handle encoding: `(page_idx << 12) | byte_offset` in a u32.
/// - **page_idx** (upper 20 bits): up to 1,048,576 pages = 4 GB of page storage
/// - **byte_offset** (lower 12 bits): up to 4,096 bytes (covers full 4KB page)
const OFFSET_BITS: u32 = 12;
const OFFSET_MASK: u32 = (1u32 << OFFSET_BITS) - 1;

/// Entry header size: file_offset (8B) + key_len (2B) = 10 bytes.
const ENTRY_HEADER_SIZE: usize = 8 + 2;

/// A 4KB page storing serialized sparse index entries.
///
/// Each entry is laid out as:
/// ```text
/// [file_offset: u64 (8B)] [key_len: u16 (2B)] [key: [u8; key_len]]
/// ```
/// Entries never span page boundaries.
pub struct SparseIndexPage {
    data: Box<[u8; SPARSE_INDEX_PAGE_SIZE]>,
    used: usize,
}

impl SparseIndexPage {
    pub fn new() -> Self {
        Self {
            data: Box::new([0u8; SPARSE_INDEX_PAGE_SIZE]),
            used: 0,
        }
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        SPARSE_INDEX_PAGE_SIZE - self.used
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }
}

/// Pool of recyclable sparse index pages.
///
/// Thread-safe via `Mutex`. Multiple merge threads may concurrently call
/// `return_pages` (before barrier) and `take`/`len` (after barrier).
pub struct SparseIndexPagePool {
    pages: Mutex<Vec<SparseIndexPage>>,
}

impl SparseIndexPagePool {
    pub fn new() -> Self {
        Self {
            pages: Mutex::new(Vec::new()),
        }
    }

    /// Return pages to the pool (resets their `used` counter).
    pub fn return_pages(&self, pages: Vec<SparseIndexPage>) {
        let mut pool = self.pages.lock().unwrap();
        for mut page in pages {
            page.reset();
            pool.push(page);
        }
    }

    /// Take up to `count` pages from the pool.
    pub fn take(&self, count: usize) -> Vec<SparseIndexPage> {
        let mut pool = self.pages.lock().unwrap();
        let len = pool.len();
        let take = count.min(len);
        pool.split_off(len - take)
    }

    /// Number of pages currently in the pool.
    pub fn len(&self) -> usize {
        self.pages.lock().unwrap().len()
    }
}

/// Encode a page index and byte offset into a u32 handle.
#[inline]
fn encode_handle(page_idx: usize, byte_offset: usize) -> u32 {
    debug_assert!(byte_offset < SPARSE_INDEX_PAGE_SIZE);
    debug_assert!(page_idx < (1 << 20));
    ((page_idx as u32) << OFFSET_BITS) | (byte_offset as u32)
}

/// Decode a u32 handle into (page_idx, byte_offset).
#[inline]
fn decode_handle(handle: u32) -> (usize, usize) {
    let page_idx = (handle >> OFFSET_BITS) as usize;
    let byte_offset = (handle & OFFSET_MASK) as usize;
    (page_idx, byte_offset)
}

/// Page-backed sparse index.
///
/// Key data and file offsets are serialized into recyclable 4KB pages.
/// A separate `pos` vector stores packed (page_idx, byte_offset) handles
/// for O(1) indexed access.
pub struct SparseIndex {
    // Storage
    pos: Vec<u32>,
    buffer: Vec<SparseIndexPage>,
    spare_pages: Vec<SparseIndexPage>,

    // Sampling control
    indexing_interval: IndexingInterval,
    next_index_at_entry: usize,
    next_index_at_bytes: usize,
    thread_index_budget_bytes: Option<usize>,
    estimated_total_data_bytes: Option<usize>,
    bootstrap_avg_key_bytes: f64,
    bootstrap_avg_record_bytes: f64,
    bootstrap_sample_count: usize,
    observed_key_bytes_total: usize,
    observed_record_bytes_total: usize,
    observed_record_count: usize,
    warned_dynamic_stride_fallback: bool,
}

impl SparseIndex {
    pub fn new(indexing_interval: IndexingInterval) -> Self {
        Self {
            pos: Vec::new(),
            buffer: Vec::new(),
            spare_pages: Vec::new(),
            indexing_interval,
            next_index_at_entry: 0,
            next_index_at_bytes: 0,
            thread_index_budget_bytes: indexing_interval.budget_bytes(),
            estimated_total_data_bytes: indexing_interval.estimated_total_data_bytes(),
            bootstrap_avg_key_bytes: 0.0,
            bootstrap_avg_record_bytes: 0.0,
            bootstrap_sample_count: 0,
            observed_key_bytes_total: 0,
            observed_record_bytes_total: 0,
            observed_record_count: 0,
            warned_dynamic_stride_fallback: false,
        }
    }

    // ── Storage access ──────────────────────────────────────────────

    #[inline]
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// Get the key bytes for entry at `index`.
    #[inline]
    pub fn key(&self, index: usize) -> &[u8] {
        let (page_idx, offset) = decode_handle(self.pos[index]);
        let page = &self.buffer[page_idx];
        let key_len = u16::from_le_bytes([page.data[offset + 8], page.data[offset + 9]]) as usize;
        &page.data[offset + ENTRY_HEADER_SIZE..offset + ENTRY_HEADER_SIZE + key_len]
    }

    /// Get the file offset for entry at `index`.
    #[inline]
    pub fn file_offset(&self, index: usize) -> usize {
        let (page_idx, offset) = decode_handle(self.pos[index]);
        let page = &self.buffer[page_idx];
        u64::from_le_bytes(page.data[offset..offset + 8].try_into().unwrap()) as usize
    }

    /// Get the key of the first entry, if any.
    pub fn first_key(&self) -> Option<&[u8]> {
        if self.pos.is_empty() {
            None
        } else {
            Some(self.key(0))
        }
    }

    /// Push a new entry (key + file_offset) into the index.
    ///
    /// Serializes into the current page. If there's not enough room,
    /// allocates a new page. Returns false (and does nothing) if the key
    /// is too large to fit in a single page.
    pub fn push(&mut self, key: &[u8], file_offset: usize) -> bool {
        let entry_size = ENTRY_HEADER_SIZE + key.len();
        if entry_size > SPARSE_INDEX_PAGE_SIZE {
            return false;
        }

        // Ensure we have a page with enough space.
        if self.buffer.is_empty() || self.buffer.last().unwrap().remaining() < entry_size {
            if let Some(mut page) = self.spare_pages.pop() {
                page.reset();
                self.buffer.push(page);
            } else {
                self.buffer.push(SparseIndexPage::new());
            }
        }

        let page_idx = self.buffer.len() - 1;
        let page = &mut self.buffer[page_idx];
        let offset = page.used;

        // Write file_offset as u64 LE
        page.data[offset..offset + 8].copy_from_slice(&(file_offset as u64).to_le_bytes());
        // Write key_len as u16 LE
        page.data[offset + 8..offset + 10].copy_from_slice(&(key.len() as u16).to_le_bytes());
        // Write key bytes
        page.data[offset + ENTRY_HEADER_SIZE..offset + ENTRY_HEADER_SIZE + key.len()]
            .copy_from_slice(key);

        page.used = offset + entry_size;
        self.pos.push(encode_handle(page_idx, offset));
        true
    }

    // ── Buffer management ───────────────────────────────────────────

    /// Drain all pages out of this index, clearing entry positions.
    /// Returns the owned pages (caller can return them to a pool).
    pub fn take_buffer(&mut self) -> Vec<SparseIndexPage> {
        self.pos.clear();
        let mut pages = std::mem::take(&mut self.buffer);
        pages.append(&mut self.spare_pages);
        pages
    }

    /// Seed this index with recycled pages.  Pages are stored in a spare
    /// pool and consumed on demand by `push` instead of heap-allocating.
    pub fn seed_buffer(&mut self, pages: Vec<SparseIndexPage>) {
        self.spare_pages.extend(pages);
    }

    /// Clear all entries and drop pages.
    pub fn clear(&mut self) {
        self.pos.clear();
        self.buffer.clear();
        self.spare_pages.clear();
    }

    /// Number of pages currently held (active + spare).
    pub fn page_count(&self) -> usize {
        self.buffer.len() + self.spare_pages.len()
    }

    /// Maximum pages this index should hold based on its budget.
    /// Returns `usize::MAX` if no budget is set (unlimited).
    pub fn budget_pages(&self) -> usize {
        match self.thread_index_budget_bytes {
            Some(b) if b > 0 => b.div_ceil(SPARSE_INDEX_PAGE_SIZE).max(1),
            _ => usize::MAX,
        }
    }

    // ── Sampling control ────────────────────────────────────────────

    pub fn indexing_interval(&self) -> IndexingInterval {
        self.indexing_interval
    }

    /// Returns true if a new sample should be recorded at the given run position.
    pub fn should_sample(&self, total_entries: usize, total_bytes: usize) -> bool {
        self.pos.is_empty()
            || match self.indexing_interval {
                IndexingInterval::Records { .. } => total_entries >= self.next_index_at_entry,
                IndexingInterval::Bytes { .. } => total_bytes >= self.next_index_at_bytes,
            }
    }

    /// Record a sparse-index sample and advance the next-sample point.
    /// If the key is too large for a page, it is truncated to fit.
    /// (The truncated key is only used for seek lower-bounding, so a
    /// prefix is always a safe under-approximation.)
    pub fn record_sample(
        &mut self,
        key: &[u8],
        file_offset: usize,
        sampled_record_index: usize,
        sampled_file_offset: usize,
    ) {
        if !self.push(key, file_offset) {
            // Key too large: truncate to the maximum that fits in a page.
            let max_key_len = SPARSE_INDEX_PAGE_SIZE - ENTRY_HEADER_SIZE;
            self.push(&key[..max_key_len], file_offset);
        }
        self.update_next_sample_point(sampled_record_index, sampled_file_offset);
    }

    /// Track per-record statistics for bootstrap estimation.
    pub fn observe_record(&mut self, key_len: usize, record_bytes: usize) {
        self.observed_key_bytes_total = self.observed_key_bytes_total.saturating_add(key_len);
        self.observed_record_bytes_total = self
            .observed_record_bytes_total
            .saturating_add(record_bytes);
        self.observed_record_count = self.observed_record_count.saturating_add(1);
    }

    pub fn set_bootstrap(
        &mut self,
        avg_key_bytes: f64,
        avg_record_bytes: f64,
        sample_count: usize,
    ) {
        if sample_count == 0 {
            return;
        }
        self.bootstrap_avg_key_bytes = avg_key_bytes.max(1.0);
        self.bootstrap_avg_record_bytes = avg_record_bytes.max(1.0);
        self.bootstrap_sample_count = sample_count;
    }

    pub fn bootstrap(&self) -> Option<(f64, f64, usize)> {
        let total_count = self
            .bootstrap_sample_count
            .saturating_add(self.observed_record_count);
        if total_count == 0 {
            return None;
        }
        let total_weight = total_count as f64;
        let key_sum = self.bootstrap_avg_key_bytes * (self.bootstrap_sample_count as f64)
            + self.observed_key_bytes_total as f64;
        let record_sum = self.bootstrap_avg_record_bytes * (self.bootstrap_sample_count as f64)
            + self.observed_record_bytes_total as f64;
        Some((
            (key_sum / total_weight).max(1.0),
            (record_sum / total_weight).max(1.0),
            total_count,
        ))
    }

    /// Compute the average bytes per index entry (for dynamic stride calculation).
    fn avg_index_entry_bytes(&self) -> f64 {
        let (avg_key_bytes, _, _) = self.bootstrap().unwrap_or((1.0, 1.0, 0));
        // Overhead per entry in pages: 8 (file_offset) + 2 (key_len) + 4 (pos handle in vec)
        (14.0 + avg_key_bytes).max(1.0)
    }

    pub fn dynamic_stride(&self) -> Option<usize> {
        let budget_bytes = self.thread_index_budget_bytes?;
        let estimated_total_data_bytes = self.estimated_total_data_bytes?;
        if budget_bytes == 0 || estimated_total_data_bytes == 0 {
            return None;
        }

        let (_avg_key_bytes, avg_record_bytes, _) = self.bootstrap().unwrap_or((1.0, 1.0, 0));
        let avg_index_entry_bytes = self.avg_index_entry_bytes();
        let target_samples = ((budget_bytes as f64) / avg_index_entry_bytes).max(1.0);

        let stride = match self.indexing_interval {
            IndexingInterval::Records { .. } => {
                let estimated_records = ((estimated_total_data_bytes as f64) / avg_record_bytes)
                    .ceil()
                    .max(1.0);
                (estimated_records / target_samples).ceil() as usize
            }
            IndexingInterval::Bytes { .. } => {
                ((estimated_total_data_bytes as f64) / target_samples).ceil() as usize
            }
        };

        Some(stride.max(1))
    }

    fn dynamic_stride_unavailable_reason(&self) -> &'static str {
        if self.thread_index_budget_bytes.is_none() {
            return "thread index budget is unavailable";
        }
        if self.estimated_total_data_bytes.is_none() {
            return "estimated total data bytes is unavailable";
        }
        if self.thread_index_budget_bytes == Some(0) {
            return "thread index budget is zero";
        }
        if self.estimated_total_data_bytes == Some(0) {
            return "estimated total data bytes is zero";
        }
        "dynamic stride is unavailable"
    }

    fn warn_dynamic_stride_fallback_once(&mut self, run_id: u32, fallback_stride: usize) {
        if self.warned_dynamic_stride_fallback {
            return;
        }
        println!(
            "WARNING: sparse index dynamic stride fallback is in use (run_id={}, reason={}, fallback_stride={})",
            run_id,
            self.dynamic_stride_unavailable_reason(),
            fallback_stride
        );
        self.warned_dynamic_stride_fallback = true;
    }

    pub fn update_next_sample_point(
        &mut self,
        sampled_record_index: usize,
        sampled_file_offset: usize,
    ) {
        self.update_next_sample_point_for_run(0, sampled_record_index, sampled_file_offset);
    }

    pub fn update_next_sample_point_for_run(
        &mut self,
        run_id: u32,
        sampled_record_index: usize,
        sampled_file_offset: usize,
    ) {
        let stride = match self.dynamic_stride() {
            Some(stride) => stride,
            None => {
                let fallback_stride = self.indexing_interval.value().max(1);
                self.warn_dynamic_stride_fallback_once(run_id, fallback_stride);
                fallback_stride
            }
        };
        match self.indexing_interval {
            IndexingInterval::Records {
                budget_bytes,
                estimated_total_data_bytes,
                ..
            } => {
                self.indexing_interval = IndexingInterval::Records {
                    stride,
                    budget_bytes,
                    estimated_total_data_bytes,
                };
                self.next_index_at_entry = sampled_record_index.saturating_add(stride);
            }
            IndexingInterval::Bytes {
                budget_bytes,
                estimated_total_data_bytes,
                ..
            } => {
                self.indexing_interval = IndexingInterval::Bytes {
                    stride,
                    budget_bytes,
                    estimated_total_data_bytes,
                };
                self.next_index_at_bytes = sampled_file_offset.saturating_add(stride);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_index_push_and_access() {
        let mut idx = SparseIndex::new(IndexingInterval::records(100));
        idx.push(b"hello", 0);
        idx.push(b"world", 128);
        idx.push(b"foo", 256);

        assert_eq!(idx.len(), 3);
        assert_eq!(idx.key(0), b"hello");
        assert_eq!(idx.file_offset(0), 0);
        assert_eq!(idx.key(1), b"world");
        assert_eq!(idx.file_offset(1), 128);
        assert_eq!(idx.key(2), b"foo");
        assert_eq!(idx.file_offset(2), 256);
    }

    #[test]
    fn test_sparse_index_first_key() {
        let mut idx = SparseIndex::new(IndexingInterval::records(100));
        assert_eq!(idx.first_key(), None);
        idx.push(b"abc", 0);
        assert_eq!(idx.first_key(), Some(b"abc".as_slice()));
    }

    #[test]
    fn test_sparse_index_page_overflow() {
        let mut idx = SparseIndex::new(IndexingInterval::records(1));
        // Push many entries to overflow a single page
        let key = vec![b'x'; 100];
        let entries_per_page = SPARSE_INDEX_PAGE_SIZE / (ENTRY_HEADER_SIZE + 100);
        for i in 0..entries_per_page + 5 {
            idx.push(&key, i * 100);
        }
        assert!(idx.page_count() >= 2);
        // Verify last entry
        let last = idx.len() - 1;
        assert_eq!(idx.key(last), key.as_slice());
        assert_eq!(idx.file_offset(last), last * 100);
    }

    #[test]
    fn test_take_and_seed_buffer() {
        let mut idx = SparseIndex::new(IndexingInterval::records(100));
        idx.push(b"a", 0);
        idx.push(b"b", 100);
        assert_eq!(idx.page_count(), 1);

        let pages = idx.take_buffer();
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.page_count(), 0);
        assert_eq!(pages.len(), 1);

        let mut idx2 = SparseIndex::new(IndexingInterval::records(100));
        idx2.seed_buffer(pages);
        // Pages are seeded but reset; no entries yet
        assert_eq!(idx2.len(), 0);
        assert_eq!(idx2.page_count(), 1);
        // Can push into the seeded page
        idx2.push(b"c", 200);
        assert_eq!(idx2.len(), 1);
        assert_eq!(idx2.key(0), b"c");
    }

    #[test]
    fn test_page_pool() {
        let pool = SparseIndexPagePool::new();
        assert_eq!(pool.len(), 0);

        let mut idx = SparseIndex::new(IndexingInterval::records(100));
        idx.push(b"test", 42);
        let pages = idx.take_buffer();
        pool.return_pages(pages);
        assert_eq!(pool.len(), 1);

        let taken = pool.take(1);
        assert_eq!(taken.len(), 1);
        assert_eq!(pool.len(), 0);
        // Returned pages have their `used` reset
        assert_eq!(taken[0].used, 0);
    }

    #[test]
    fn test_should_sample() {
        let idx = SparseIndex::new(IndexingInterval::records(10));
        // Empty index always samples
        assert!(idx.should_sample(0, 0));
    }

    #[test]
    fn test_handle_encoding() {
        let h = encode_handle(3, 1024);
        let (pi, bo) = decode_handle(h);
        assert_eq!(pi, 3);
        assert_eq!(bo, 1024);

        let h = encode_handle(0, 0);
        let (pi, bo) = decode_handle(h);
        assert_eq!(pi, 0);
        assert_eq!(bo, 0);

        // Max values: 20-bit page_idx, 12-bit byte_offset
        let h = encode_handle((1 << 20) - 1, (1 << 12) - 1);
        let (pi, bo) = decode_handle(h);
        assert_eq!(pi, (1 << 20) - 1);
        assert_eq!(bo, (1 << 12) - 1);
    }
}
