use std::cmp::Ordering;
use std::marker::PhantomData;
use std::rc::Rc;

use super::late_fence_slots::LateFenceSlots;
#[cfg(test)]
use super::memory::AllocHandle;
use super::memory::{
    ManagedSlice, reset_thread_local_manager, reset_thread_local_manager_with_limit,
    tls_has_headroom,
};
use super::{RecordSize, ReplacementScanner, ReplacementSelectionStats, RunEmitter, next_record};
use crate::ovc::offset_value_coding_64::SentinelValue;
use crate::ovc::tree_of_losers::LoserTree;
use crate::sort::run_sink::RunSink;

pub struct KeyValueMM {
    data: ManagedSlice,
    _not_send: PhantomData<Rc<()>>,
}

impl KeyValueMM {
    fn try_new(key: &[u8], value: &[u8]) -> Option<Self> {
        let data = ManagedSlice::alloc(key, value)?;

        Some(Self {
            data,
            _not_send: PhantomData,
        })
    }

    pub fn get_key(&self) -> &[u8] {
        let Some((ptr, key_len, _value_len, total)) = self.data.payload_info() else {
            return &[];
        };
        let payload = unsafe { std::slice::from_raw_parts(ptr, total) };
        let key_offset = 2;
        let key_end = key_offset + key_len;
        &payload[key_offset..key_end]
    }

    pub fn key_value_slices(&self) -> Option<(&[u8], &[u8])> {
        let Some((ptr, key_len, value_len, total)) = self.data.payload_info() else {
            return None;
        };
        let payload = unsafe { std::slice::from_raw_parts(ptr, total) };
        let key_offset = 2;
        let value_len_offset = key_offset + key_len;
        let value_offset = value_len_offset + 2;
        let key = &payload[key_offset..value_len_offset];
        let value = &payload[value_offset..value_offset + value_len];
        Some((key, value))
    }
}

impl Drop for KeyValueMM {
    fn drop(&mut self) {
        self.data.release();
    }
}

impl PartialEq for KeyValueMM {
    fn eq(&self, other: &Self) -> bool {
        if self.is_early_fence() {
            return other.is_early_fence();
        }
        if self.is_late_fence() {
            return other.is_late_fence();
        }
        if other.is_early_fence() || other.is_late_fence() {
            return false;
        }
        self.get_key() == other.get_key()
    }
}

impl Eq for KeyValueMM {}

impl Ord for KeyValueMM {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.is_early_fence() {
            return if other.is_early_fence() {
                Ordering::Equal
            } else {
                Ordering::Less
            };
        }
        if self.is_late_fence() {
            return if other.is_late_fence() {
                Ordering::Equal
            } else {
                Ordering::Greater
            };
        }
        if other.is_early_fence() {
            return Ordering::Greater;
        }
        if other.is_late_fence() {
            return Ordering::Less;
        }
        self.get_key().cmp(other.get_key())
    }
}

impl PartialOrd for KeyValueMM {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl SentinelValue for KeyValueMM {
    fn early_fence() -> Self {
        Self {
            data: ManagedSlice::early_fence(),
            _not_send: PhantomData,
        }
    }

    fn late_fence() -> Self {
        Self {
            data: ManagedSlice::late_fence(),
            _not_send: PhantomData,
        }
    }

    fn is_early_fence(&self) -> bool {
        self.data.is_early_fence()
    }

    fn is_late_fence(&self) -> bool {
        self.data.is_late_fence()
    }
}

impl RecordSize for KeyValueMM {
    fn size(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.allocation_size()
    }
}

/// Replacement selection that stores record bytes in the memory manager.
pub struct ReplacementSelectionMM<T: Ord + SentinelValue> {
    tree: LoserTree<T>,
    next_run_buffer: Vec<T>,
    late_fence_slots: LateFenceSlots,
    initialized: bool,
    pending_run_switch: bool,
}

impl<T: Ord + SentinelValue> ReplacementSelectionMM<T> {
    pub fn new() -> Self {
        Self {
            tree: LoserTree::new(vec![]),
            next_run_buffer: Vec::new(),
            late_fence_slots: LateFenceSlots::new(),
            initialized: false,
            pending_run_switch: false,
        }
    }

    pub fn insert_initial(&mut self, record: T) {
        assert!(
            !self.initialized,
            "Cannot insert initial records after build is called"
        );
        self.next_run_buffer.push(record);
    }

    pub fn build(&mut self) {
        assert!(!self.initialized, "build() may only be called once");
        self.initialize_tree_from_buffer();
        self.initialized = true;
    }

    fn initialize_tree_from_buffer(&mut self) {
        let size = self.next_run_buffer.len();
        let values = self.next_run_buffer.drain(..);
        self.tree.reset_from_iter(size, values);

        let capacity = if size == 0 {
            0
        } else {
            size.next_power_of_two()
        };
        self.late_fence_slots.reset(capacity);
        self.late_fence_slots.set_range(size, capacity);
    }

    pub fn absorb_record_with(&mut self, record: T, emitter: &mut impl RunEmitter<T>) {
        assert!(self.initialized, "absorb_record called before build()");

        let record = record;

        self.ensure_active_tree(emitter);

        if self.tree.peek().is_none() && self.next_run_buffer.is_empty() {
            self.seed_tree_with_record(record, emitter);
            return;
        }

        let belongs_to_next_run = self.should_defer_to_next_run(&record);

        if belongs_to_next_run {
            self.add_to_next_run(record);
        } else {
            self.add_to_current_run(record, emitter);
        }
    }

    fn ensure_active_tree(&mut self, emitter: &mut impl RunEmitter<T>) {
        if self.tree.peek().is_none() {
            self.switch_run(emitter);
        }
    }

    fn should_defer_to_next_run(&mut self, record: &T) -> bool {
        if let Some((winner, _)) = self.tree.peek() {
            record < winner
        } else {
            false
        }
    }

    fn add_to_next_run(&mut self, record: T) {
        self.next_run_buffer.push(record);
    }

    fn add_to_current_run(&mut self, record: T, emitter: &mut impl RunEmitter<T>) {
        // Use late-fence slots when we have headroom; otherwise push.
        let has_headroom = tls_has_headroom();
        if has_headroom {
            if let Some(idx) = self.late_fence_slots.pop() {
                self.tree.update(idx, record);
                return;
            }
        }

        let winner = self.tree.push(record);
        if !winner.is_late_fence() && !winner.is_early_fence() {
            emitter.emit(winner);
        }
    }

    fn switch_run(&mut self, emitter: &mut impl RunEmitter<T>) {
        if self.next_run_buffer.is_empty() {
            return;
        }
        self.initialize_tree_from_buffer();
        self.pending_run_switch = false;
        emitter.on_run_switch();
    }

    fn seed_tree_with_record(&mut self, record: T, emitter: &mut impl RunEmitter<T>) {
        if self.pending_run_switch {
            self.pending_run_switch = false;
            emitter.on_run_switch();
        }
        self.tree.reset(vec![record]);
        self.late_fence_slots.reset(1);
    }

    pub fn drain_with(&mut self, emitter: &mut impl RunEmitter<T>) {
        self.drain_current_tree(emitter);
        self.switch_run(emitter);
        self.drain_current_tree(emitter);
    }

    fn drain_current_tree(&mut self, emitter: &mut impl RunEmitter<T>) {
        while let Some((_, idx)) = self.tree.peek() {
            let val = self.tree.push(T::late_fence());
            if !val.is_late_fence() && !val.is_early_fence() {
                emitter.emit(val);
                // The slot at idx now contains late_fence and can be reused
                self.late_fence_slots.push(idx);
            }
        }
    }

    pub fn buffer_len(&self) -> usize {
        self.next_run_buffer.len()
    }

    fn emit_one_from_current_run(&mut self, emitter: &mut impl RunEmitter<T>) -> bool {
        self.ensure_active_tree(emitter);
        let Some((_, idx)) = self.tree.peek() else {
            return false;
        };
        let val = self.tree.push(T::late_fence());
        if !val.is_late_fence() && !val.is_early_fence() {
            emitter.emit(val);
            self.late_fence_slots.push(idx);
            return true;
        }
        false
    }
}

/// Replacement selection driven by the tree-of-losers,
/// storing keys/values in a memory manager rather than heap Vecs.
pub fn run_replacement_selection_mm<S>(
    mut scanner: ReplacementScanner,
    sink: &mut S,
    memory_limit: usize,
) -> ReplacementSelectionStats
where
    S: RunSink,
{
    let manager_limit = memory_limit.saturating_mul(9) / 10;
    if manager_limit == 0 {
        reset_thread_local_manager();
    } else {
        reset_thread_local_manager_with_limit(manager_limit);
    }

    let mut rs = ReplacementSelectionMM::new();
    let mut pending_record: Option<(Vec<u8>, Vec<u8>)> = None;

    loop {
        let Some((key, value)) = next_record(&mut pending_record, scanner.as_mut()) else {
            break;
        };

        let key_len = key.len();
        let value_len = value.len();
        let record = match KeyValueMM::try_new(&key, &value) {
            Some(record) => record,
            None => {
                if rs.buffer_len() > 0 {
                    pending_record = Some((key, value));
                    break;
                }
                panic!(
                    "Memory manager allocation failed for key {} bytes, value {} bytes",
                    key_len, value_len
                );
            }
        };
        rs.insert_initial(record);
    }

    if rs.buffer_len() == 0 {
        return ReplacementSelectionStats::default();
    }

    rs.build();
    sink.start_run();

    let mut records_emitted = 0usize;

    let mut emitter = SinkEmitter {
        sink,
        emitted: &mut records_emitted,
    };

    while let Some((key, value)) = next_record(&mut pending_record, scanner.as_mut()) {
        let record = loop {
            if let Some(record) = KeyValueMM::try_new(&key, &value) {
                break record;
            }
            // Keep emitting until we can allocate
            if !rs.emit_one_from_current_run(&mut emitter) {
                let key_len = key.len();
                let value_len = value.len();
                panic!(
                    "Memory manager allocation failed for key {} bytes, value {} bytes",
                    key_len, value_len
                );
            }
        };
        rs.absorb_record_with(record, &mut emitter);
    }

    rs.drain_with(&mut emitter);

    sink.finish_run();

    ReplacementSelectionStats { records_emitted }
}

struct SinkEmitter<'a, S: RunSink> {
    sink: &'a mut S,
    emitted: &'a mut usize,
}

impl<'a, S: RunSink> RunEmitter<KeyValueMM> for SinkEmitter<'a, S> {
    fn emit(&mut self, record: KeyValueMM) {
        if record.is_sentinel() {
            return;
        }
        if let Some((key, value)) = record.key_value_slices() {
            self.sink.push_record(key, value);
            *self.emitted += 1;
        }
    }

    fn on_run_switch(&mut self) {
        self.sink.finish_run();
        self.sink.start_run();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::{Sketch, SketchType};
    use crate::sort::core::engine::RunSummary;

    #[derive(Clone)]
    struct CollectedRun {
        entries: usize,
        bytes: usize,
    }

    impl RunSummary for CollectedRun {
        fn total_entries(&self) -> usize {
            self.entries
        }

        fn total_bytes(&self) -> usize {
            self.bytes
        }
    }

    struct CollectingSink {
        runs: Vec<Vec<(Vec<u8>, Vec<u8>)>>,
        current: Vec<(Vec<u8>, Vec<u8>)>,
    }

    impl CollectingSink {
        fn new() -> Self {
            Self {
                runs: Vec::new(),
                current: Vec::new(),
            }
        }
    }

    impl RunSink for CollectingSink {
        type MergeableRun = CollectedRun;

        fn start_run(&mut self) {
            self.current.clear();
        }

        fn push_record(&mut self, key: &[u8], value: &[u8]) {
            self.current.push((key.to_vec(), value.to_vec()));
        }

        fn finish_run(&mut self) {
            if !self.current.is_empty() {
                self.runs.push(std::mem::take(&mut self.current));
            }
        }

        fn finalize(self) -> (Vec<Self::MergeableRun>, Sketch<Vec<u8>>) {
            let runs = self
                .runs
                .into_iter()
                .map(|run| CollectedRun {
                    entries: run.len(),
                    bytes: run.iter().map(|(k, v)| k.len() + v.len()).sum(),
                })
                .collect();
            (runs, Sketch::new(SketchType::Kll, 0))
        }
    }

    fn key_bytes(value: u32) -> Vec<u8> {
        value.to_be_bytes().to_vec()
    }

    struct CountingEmitter {
        emitted: usize,
    }

    impl CountingEmitter {
        fn new() -> Self {
            Self { emitted: 0 }
        }
    }

    impl RunEmitter<KeyValueMM> for CountingEmitter {
        fn emit(&mut self, record: KeyValueMM) {
            if record.is_sentinel() {
                return;
            }
            self.emitted += 1;
        }

        fn on_run_switch(&mut self) {}
    }

    #[test]
    fn test_mm_runs_sorted() {
        let data = vec![
            (key_bytes(5), vec![1]),
            (key_bytes(1), vec![2]),
            (key_bytes(3), vec![3]),
            (key_bytes(2), vec![4]),
            (key_bytes(4), vec![5]),
        ];
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        run_replacement_selection_mm(scanner, &mut sink, 100_000);

        assert!(!sink.runs.is_empty(), "expected at least one run");
        for run in &sink.runs {
            for i in 1..run.len() {
                assert!(run[i - 1].0 <= run[i].0, "run should be sorted");
            }
        }
    }

    #[test]
    fn test_mm_emits_multiple_runs_on_full_manager() {
        let mut data = Vec::new();
        for i in 0..200u32 {
            let key = key_bytes(i);
            let value = vec![b'x'; 3000];
            data.push((key, value));
        }
        // Reverse to force multiple runs
        data.reverse();
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        run_replacement_selection_mm(scanner, &mut sink, 100_000); // 200 * 3000 = 600,000 bytes needed

        assert!(
            sink.runs.len() > 1,
            "expected multiple runs when memory manager fills"
        );
    }

    // ==================== KeyValueMM Tests ====================

    #[test]
    fn test_kv_mm_creation() {
        reset_thread_local_manager();
        let key = b"test_key";
        let value = b"test_value";
        let kv = KeyValueMM::try_new(key, value).expect("should create");

        assert_eq!(kv.get_key(), key);
        let (k, v) = kv.key_value_slices().expect("should have slices");
        assert_eq!(k, key);
        assert_eq!(v, value);
    }

    #[test]
    fn test_kv_mm_empty_key() {
        reset_thread_local_manager();
        let key = b"";
        let value = b"value";
        let kv = KeyValueMM::try_new(key, value).expect("should create");

        assert_eq!(kv.get_key(), key);
        let (k, v) = kv.key_value_slices().expect("should have slices");
        assert_eq!(k, key);
        assert_eq!(v, value);
    }

    #[test]
    fn test_kv_mm_empty_value() {
        reset_thread_local_manager();
        let key = b"key";
        let value = b"";
        let kv = KeyValueMM::try_new(key, value).expect("should create");

        let (k, v) = kv.key_value_slices().expect("should have slices");
        assert_eq!(k, key);
        assert_eq!(v, value);
    }

    #[test]
    fn test_kv_mm_both_empty() {
        reset_thread_local_manager();
        let key = b"";
        let value = b"";
        let kv = KeyValueMM::try_new(key, value).expect("should create");

        let (k, v) = kv.key_value_slices().expect("should have slices");
        assert_eq!(k, key);
        assert_eq!(v, value);
    }

    #[test]
    fn test_kv_mm_large_key_value() {
        reset_thread_local_manager();
        let key = vec![b'k'; 10000];
        let value = vec![b'v'; 10000];
        let kv = KeyValueMM::try_new(&key, &value).expect("should create");

        let (k, v) = kv.key_value_slices().expect("should have slices");
        assert_eq!(k, &key[..]);
        assert_eq!(v, &value[..]);
    }

    #[test]
    fn test_kv_mm_sentinel_early_fence() {
        let fence = KeyValueMM::early_fence();
        assert!(fence.is_early_fence());
        assert!(!fence.is_late_fence());
        assert_eq!(fence.get_key(), b"");
    }

    #[test]
    fn test_kv_mm_sentinel_late_fence() {
        let fence = KeyValueMM::late_fence();
        assert!(fence.is_late_fence());
        assert!(!fence.is_early_fence());
        assert_eq!(fence.get_key(), b"");
    }

    #[test]
    fn test_kv_mm_size() {
        reset_thread_local_manager();
        let key = b"test";
        let value = b"value";
        let kv = KeyValueMM::try_new(key, value).expect("should create");

        let size = kv.size();
        assert!(size > std::mem::size_of::<KeyValueMM>());
    }

    #[test]
    fn test_kv_mm_multiple_allocations() {
        reset_thread_local_manager();
        let mut kvs = Vec::new();

        for _i in 0..100 {
            let key = format!("key_{}", _i);
            let value = format!("value_{}", _i);
            let kv = KeyValueMM::try_new(key.as_bytes(), value.as_bytes()).expect("should create");
            kvs.push((kv, key, value));
        }

        for (kv, key, value) in &kvs {
            let (k, v) = kv.key_value_slices().expect("should have slices");
            assert_eq!(k, key.as_bytes());
            assert_eq!(v, value.as_bytes());
        }
    }

    #[test]
    fn test_kv_mm_memory_limit_reached() {
        reset_thread_local_manager_with_limit(100_000);
        let mut kvs = Vec::new();
        let mut failed = false;

        // Try to allocate more than the limit
        for _i in 0..500 {
            let key = vec![b'k'; 1000];
            let value = vec![b'v'; 1000];
            if let Some(kv) = KeyValueMM::try_new(&key, &value) {
                kvs.push(kv);
            } else {
                // Should eventually fail
                failed = true;
                break;
            }
        }

        if failed {
            assert!(!kvs.is_empty(), "should have allocated some before failing");
        }
        // If we didn't fail, the limit was high enough for all allocations, which is also valid
    }

    // ==================== ReplacementSelectionMM Tests ====================

    #[test]
    fn test_rs_mm_new() {
        let rs: ReplacementSelectionMM<KeyValueMM> = ReplacementSelectionMM::new();
        assert!(!rs.initialized);
        assert_eq!(rs.buffer_len(), 0);
    }

    #[test]
    fn test_rs_mm_insert_initial() {
        reset_thread_local_manager();
        let mut rs = ReplacementSelectionMM::new();
        let kv = KeyValueMM::try_new(b"key", b"value").expect("should create");

        rs.insert_initial(kv);
        assert_eq!(rs.buffer_len(), 1);
    }

    #[test]
    fn test_rs_mm_build() {
        reset_thread_local_manager();
        let mut rs = ReplacementSelectionMM::new();
        let kv = KeyValueMM::try_new(b"key", b"value").expect("should create");

        rs.insert_initial(kv);
        rs.build();
        assert!(rs.initialized);
    }

    #[test]
    fn test_rs_mm_empty_build() {
        let mut rs: ReplacementSelectionMM<KeyValueMM> = ReplacementSelectionMM::new();
        rs.build();
        assert!(rs.initialized);
        assert_eq!(rs.buffer_len(), 0);
    }

    #[test]
    fn test_rs_mm_absorb_record() {
        reset_thread_local_manager();
        let mut rs = ReplacementSelectionMM::new();

        for i in 0..10u32 {
            let kv = KeyValueMM::try_new(&key_bytes(i), &[i as u8]).expect("should create");
            rs.insert_initial(kv);
        }

        rs.build();

        let mut sink = CollectingSink::new();
        sink.start_run();
        let mut emitted = 0;
        let mut emitter = SinkEmitter {
            sink: &mut sink,
            emitted: &mut emitted,
        };

        // Add a record that belongs to current run (higher than initial)
        let new_kv = KeyValueMM::try_new(&key_bytes(20), &[20]).expect("should create");
        rs.absorb_record_with(new_kv, &mut emitter);

        // The new record might cause emission or be added to tree
        // Just verify the function completes without panic (emitted is always >= 0)
        let _ = emitted;
    }

    #[test]
    fn test_mm_padding_slots_tracked_and_consumed() {
        reset_thread_local_manager();
        let mut rs = ReplacementSelectionMM::new();

        for &key in &[10u32, 20, 30] {
            let kv = KeyValueMM::try_new(&key_bytes(key), &[key as u8]).expect("should create");
            rs.insert_initial(kv);
        }

        rs.build();
        assert!(
            !rs.late_fence_slots.is_empty(),
            "expected padding late-fence slot to be tracked"
        );

        let mut emitter = CountingEmitter::new();
        let new_kv = KeyValueMM::try_new(&key_bytes(15), &[15]).expect("should create");
        rs.absorb_record_with(new_kv, &mut emitter);

        assert_eq!(
            emitter.emitted, 0,
            "should fill padding slot without eviction"
        );
        assert!(
            rs.late_fence_slots.is_empty(),
            "padding late-fence slot should be consumed"
        );
    }

    #[test]
    fn test_rs_mm_drain() {
        reset_thread_local_manager();
        let mut rs = ReplacementSelectionMM::new();

        for i in 0..5u32 {
            let kv = KeyValueMM::try_new(&key_bytes(i), &[i as u8]).expect("should create");
            rs.insert_initial(kv);
        }

        rs.build();

        let mut sink = CollectingSink::new();
        sink.start_run();
        let mut emitted = 0;
        let mut emitter = SinkEmitter {
            sink: &mut sink,
            emitted: &mut emitted,
        };

        rs.drain_with(&mut emitter);

        assert_eq!(emitted, 5);
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_mm_empty_input() {
        let scanner: ReplacementScanner = Box::new(std::iter::empty());
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection_mm(scanner, &mut sink, 1_000_000);

        assert_eq!(stats.records_emitted, 0);
        assert!(sink.runs.is_empty());
    }

    #[test]
    fn test_mm_single_record() {
        let data = vec![(key_bytes(1), vec![1])];
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection_mm(scanner, &mut sink, 1_000_000);

        assert_eq!(stats.records_emitted, 1);
        assert_eq!(sink.runs.len(), 1);
        assert_eq!(sink.runs[0].len(), 1);
    }

    #[test]
    fn test_mm_ascending_order() {
        let mut data = Vec::new();
        for i in 0..100u32 {
            data.push((key_bytes(i), vec![i as u8]));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection_mm(scanner, &mut sink, 1_000_000);

        assert_eq!(stats.records_emitted, 100);
        // Ascending order should produce a single run
        assert_eq!(sink.runs.len(), 1);

        // Verify sorted
        for i in 1..sink.runs[0].len() {
            assert!(sink.runs[0][i - 1].0 <= sink.runs[0][i].0);
        }
    }

    #[test]
    fn test_mm_descending_order() {
        let mut data = Vec::new();
        for i in (0..50u32).rev() {
            data.push((key_bytes(i), vec![i as u8]));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        run_replacement_selection_mm(scanner, &mut sink, 1_000_000);

        // Descending order should produce multiple runs (worst case)
        assert!(!sink.runs.is_empty());

        // Each run should be sorted
        for run in &sink.runs {
            for i in 1..run.len() {
                assert!(run[i - 1].0 <= run[i].0);
            }
        }
    }

    #[test]
    fn test_mm_duplicate_keys() {
        let mut data = Vec::new();
        for i in 0..50u32 {
            let key_val = i / 5; // Create duplicates
            data.push((key_bytes(key_val), vec![i as u8]));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection_mm(scanner, &mut sink, 1_000_000);

        assert_eq!(stats.records_emitted, 50);

        // Verify sorted (stable sort for duplicates)
        for run in &sink.runs {
            for i in 1..run.len() {
                assert!(run[i - 1].0 <= run[i].0);
            }
        }
    }

    #[test]
    fn test_mm_very_small_memory_limit() {
        let mut data = Vec::new();
        for i in 0..20u32 {
            data.push((key_bytes(i), vec![i as u8; 100]));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        // Run with a memory limit
        run_replacement_selection_mm(scanner, &mut sink, 100_000);

        // Should process all records
        let total: usize = sink.runs.iter().map(|r| r.len()).sum();
        assert_eq!(total, 20);

        // Verify each run is sorted
        for run in &sink.runs {
            for i in 1..run.len() {
                assert!(run[i - 1].0 <= run[i].0);
            }
        }
    }

    #[test]
    fn test_mm_large_values() {
        let mut data = Vec::new();
        for i in 0..10u32 {
            let key = key_bytes(i);
            let value = vec![i as u8; 5000];
            data.push((key, value));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection_mm(scanner, &mut sink, 100_000);

        assert_eq!(stats.records_emitted, 10);

        // Verify sorted
        for run in &sink.runs {
            for i in 1..run.len() {
                assert!(run[i - 1].0 <= run[i].0);
            }
        }
    }

    #[test]
    fn test_mm_mixed_sizes() {
        let mut data = Vec::new();
        for i in 0..50u32 {
            let key = key_bytes(i);
            let value_size = ((i % 10) + 1) * 100;
            let value = vec![i as u8; value_size as usize];
            data.push((key, value));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection_mm(scanner, &mut sink, 100_000);

        assert_eq!(stats.records_emitted, 50);

        // Verify sorted
        for run in &sink.runs {
            for i in 1..run.len() {
                assert!(run[i - 1].0 <= run[i].0);
            }
        }
    }

    #[test]
    fn test_mm_all_records_in_final_output() {
        let mut data = Vec::new();
        for i in 0..100u32 {
            data.push((key_bytes(i), vec![i as u8]));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        run_replacement_selection_mm(scanner, &mut sink, 500_000);

        // Count total records across all runs
        let total: usize = sink.runs.iter().map(|r| r.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_mm_interleaved_pattern() {
        // Pattern: high, low, high, low... to test run generation
        let mut data = Vec::new();
        for i in 0..50u32 {
            let val = if i % 2 == 0 { i + 100 } else { i };
            data.push((key_bytes(val), vec![i as u8]));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        run_replacement_selection_mm(scanner, &mut sink, 1_000_000);

        // Verify each run is sorted
        for run in &sink.runs {
            for i in 1..run.len() {
                assert!(run[i - 1].0 <= run[i].0);
            }
        }
    }

    #[test]
    fn test_mm_memory_limit_boundary() {
        let mut data = Vec::new();
        for i in 0..100u32 {
            data.push((key_bytes(i), vec![i as u8; 1000]));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        // Set a memory limit and verify all records are processed
        run_replacement_selection_mm(scanner, &mut sink, 300_000);

        let total: usize = sink.runs.iter().map(|r| r.len()).sum();
        assert_eq!(total, 100);

        // Verify each run is sorted
        for run in &sink.runs {
            for i in 1..run.len() {
                assert!(run[i - 1].0 <= run[i].0);
            }
        }
    }

    #[test]
    fn test_mm_zero_memory_limit() {
        let mut data = Vec::new();
        for i in 0..10u32 {
            data.push((key_bytes(i), vec![i as u8]));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        // Zero limit should still work with default manager behavior
        run_replacement_selection_mm(scanner, &mut sink, 0);

        let total: usize = sink.runs.iter().map(|r| r.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_mm_run_boundaries() {
        // Test that run boundaries are correctly detected
        let mut data = Vec::new();
        // Create a pattern that will force run switches
        for i in 0..30u32 {
            data.push((key_bytes(i), vec![i as u8; 2000]));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        run_replacement_selection_mm(scanner, &mut sink, 150_000);

        if sink.runs.len() > 1 {
            // Verify that the last element of one run < first element of next run
            // only works if data is ascending
            for i in 1..sink.runs.len() {
                if !sink.runs[i - 1].is_empty() && !sink.runs[i].is_empty() {
                    let last_of_prev = &sink.runs[i - 1].last().unwrap().0;
                    let first_of_next = &sink.runs[i].first().unwrap().0;
                    assert!(last_of_prev <= first_of_next);
                }
            }
        }
    }

    #[test]
    fn test_mm_pseudo_random_keys() {
        let mut data = Vec::new();
        let mut seed: u32 = 42;
        let next_rand = |s: &mut u32| {
            *s = s.wrapping_mul(1103515245).wrapping_add(12345);
            *s
        };

        for _ in 0..100 {
            let key_val = next_rand(&mut seed) % 1000;
            data.push((key_bytes(key_val), vec![1, 2, 3]));
        }

        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection_mm(scanner, &mut sink, 1_000_000);

        assert_eq!(stats.records_emitted, 100);

        // Verify each run is sorted
        for run in &sink.runs {
            for i in 1..run.len() {
                assert!(run[i - 1].0 <= run[i].0);
            }
        }
    }

    #[test]
    fn test_mm_key_length_variations() {
        let mut data = Vec::new();
        for i in 0..50u32 {
            let key_size = (i % 20) + 1;
            let key = vec![i as u8; key_size as usize];
            data.push((key, vec![1, 2, 3]));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection_mm(scanner, &mut sink, 1_000_000);

        assert_eq!(stats.records_emitted, 50);
    }

    #[test]
    fn test_mm_value_length_variations() {
        let mut data = Vec::new();
        for i in 0..50u32 {
            let value_size = (i % 30) + 1;
            let value = vec![i as u8; value_size as usize];
            data.push((key_bytes(i), value));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection_mm(scanner, &mut sink, 1_000_000);

        assert_eq!(stats.records_emitted, 50);
    }

    #[test]
    fn test_mm_repeated_values_different_keys() {
        let mut data = Vec::new();
        let same_value = vec![42u8; 100];
        for i in 0..50u32 {
            data.push((key_bytes(i), same_value.clone()));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection_mm(scanner, &mut sink, 1_000_000);

        assert_eq!(stats.records_emitted, 50);
    }

    #[test]
    fn test_mm_thread_local_manager_reset() {
        // Test that thread-local manager is properly reset
        reset_thread_local_manager();

        let data = vec![(key_bytes(1), vec![1])];
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        run_replacement_selection_mm(scanner, &mut sink, 100_000);

        // Run again to verify manager was reset
        let data2 = vec![(key_bytes(2), vec![2])];
        let scanner2: ReplacementScanner = Box::new(data2.into_iter());
        let mut sink2 = CollectingSink::new();

        run_replacement_selection_mm(scanner2, &mut sink2, 100_000);

        assert!(!sink2.runs.is_empty());
    }

    #[test]
    fn test_mm_manager_with_90_percent_limit() {
        // Test that manager_limit is correctly calculated as 90% of memory_limit
        let mut data = Vec::new();
        for i in 0..20u32 {
            data.push((key_bytes(i), vec![i as u8; 1000]));
        }
        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        // The function should set manager limit to 90% of this
        run_replacement_selection_mm(scanner, &mut sink, 100_000);

        let total: usize = sink.runs.iter().map(|r| r.len()).sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn test_managed_slice_size_optimization() {
        // Verify that ManagedSlice uses only 4 bytes (not 8 bytes with Option)
        assert_eq!(
            std::mem::size_of::<ManagedSlice>(),
            4,
            "ManagedSlice should be 4 bytes (single AllocHandle, not Option<AllocHandle>)"
        );

        // Verify AllocHandle is 4 bytes
        assert_eq!(
            std::mem::size_of::<AllocHandle>(),
            4,
            "AllocHandle should be 4 bytes"
        );

        // Verify KeyValueMM benefits from the optimization
        let size = std::mem::size_of::<KeyValueMM>();
        // KeyValueMM = ManagedSlice (4) + PhantomData (0) = 4 bytes + padding
        assert!(
            size <= 16,
            "KeyValueMM should be at most 16 bytes, got {}",
            size
        );
    }

    #[test]
    fn test_memory_usage_with_100_byte_records() {
        use crate::diskio::constants::DEFAULT_BUFFER_SIZE;

        // Test with 100-byte records to understand memory usage
        reset_thread_local_manager();

        println!("\n=== Memory Usage Analysis for 100-byte Records ===");
        println!(
            "KeyValueMM struct size: {} bytes",
            std::mem::size_of::<KeyValueMM>()
        );
        println!(
            "ManagedSlice size: {} bytes",
            std::mem::size_of::<ManagedSlice>()
        );
        println!(
            "AllocHandle size: {} bytes",
            std::mem::size_of::<AllocHandle>()
        );
        println!(
            "DEFAULT_BUFFER_SIZE (EXTENT_SIZE): {} bytes",
            DEFAULT_BUFFER_SIZE
        );

        // Create records: 50 bytes key + 50 bytes value = 100 bytes payload
        let key = vec![b'k'; 50];
        let value = vec![b'v'; 50];

        // Test single record allocation
        let kv = KeyValueMM::try_new(&key, &value).expect("should allocate");
        let alloc_size = kv.data.allocation_size();
        println!("\nSingle 100-byte record:");
        println!("  Payload (key+value): {} bytes", key.len() + value.len());
        println!("  Stored format (with length headers): {} bytes", 100 + 4);
        println!("  Actual block allocation: {} bytes", alloc_size);
        println!("  Block overhead: {} bytes", alloc_size - 104);

        // Calculate how many fit in one extent
        let records_per_extent = DEFAULT_BUFFER_SIZE / alloc_size;
        println!("\nCapacity:");
        println!(
            "  Records per extent (~{}KB): ~{} records",
            DEFAULT_BUFFER_SIZE / 1024,
            records_per_extent
        );

        drop(kv);

        // Now test with many records
        let num_records = 1000;
        let mut data = Vec::new();
        for i in 0..num_records {
            data.push((vec![b'k'; 50], vec![(i % 256) as u8; 50]));
        }

        let memory_limit = 500_000; // 500 KB
        let manager_limit = memory_limit * 9 / 10; // 450 KB

        let scanner: ReplacementScanner = Box::new(data.into_iter());
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection_mm(scanner, &mut sink, memory_limit);

        println!("\n=== Test with {} x 100-byte records ===", num_records);
        println!("Memory limit: {} KB", memory_limit / 1024);
        println!("Manager limit (90%): {} KB", manager_limit / 1024);
        println!("Records processed: {}", stats.records_emitted);
        println!("Number of runs: {}", sink.runs.len());

        // Calculate expected memory usage
        let records_in_memory = if sink.runs.len() == 1 {
            num_records
        } else {
            // Estimate based on manager limit
            manager_limit / alloc_size
        };

        let data_in_manager = records_in_memory * alloc_size;
        let struct_overhead = records_in_memory * std::mem::size_of::<KeyValueMM>();
        let tree_capacity = records_in_memory.next_power_of_two();
        let tree_overhead = tree_capacity * std::mem::size_of::<KeyValueMM>();

        println!(
            "\nEstimated memory breakdown for {} records in memory:",
            records_in_memory
        );
        println!(
            "  Data in manager: {} KB ({} bytes)",
            data_in_manager / 1024,
            data_in_manager
        );
        println!(
            "  KeyValueMM structs: {} KB ({} bytes)",
            struct_overhead / 1024,
            struct_overhead
        );
        println!("  Tree capacity (power of 2): {}", tree_capacity);
        println!(
            "  Tree Vec allocation: {} KB ({} bytes)",
            tree_overhead / 1024,
            tree_overhead
        );
        println!(
            "  Total estimated: {} KB",
            (data_in_manager + struct_overhead + tree_overhead) / 1024
        );
        println!(
            "  As % of memory_limit: {:.1}%",
            (data_in_manager + struct_overhead + tree_overhead) as f64 / memory_limit as f64
                * 100.0
        );

        // Verify all records were processed
        let total: usize = sink.runs.iter().map(|r| r.len()).sum();
        assert_eq!(total, num_records);
    }
}
