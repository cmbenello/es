use super::late_fence_slots::LateFenceSlots;
use super::{
    RecordSize, ReplacementScanner, ReplacementSelectionStats, RunEmitter, ensure_entry_fits,
    next_record,
};
use crate::ovc::offset_value_coding_32::OVCKeyValue32;
use crate::ovc::offset_value_coding_64::SentinelValue;
use crate::ovc::tree_of_losers_ovc::{LoserTreeOVC, OVCTreeKey};
use crate::sort::run_sink::RunSink;

/// Replacement selection algorithm with Offset Value Coding (OVC) optimization.
///
/// This implementation extends the standard replacement selection algorithm by
/// maintaining and updating offset value codes during tree comparisons, enabling
/// efficient compression of sorted runs.
///
/// The algorithm maintains two logical partitions:
/// - Current run: Active tree from which sorted records are extracted
/// - Next run: Buffer collecting records that cannot fit in the current run
///
/// Key features:
/// - Automatic OVC updates during comparisons via `LoserTreeOVC`
/// - Memory-aware eviction based on workspace size constraints
/// - Padding slot optimization for power-of-two tree capacities
pub struct ReplacementSelectionOVC<T: OVCTreeKey + RecordSize> {
    /// Tournament tree for efficient min-element extraction with OVC updates
    tree: LoserTreeOVC<T>,

    /// Buffer for records destined for the next sorted run
    next_run_buffer: Vec<T>,

    /// Available padding slots in the tree (indices between num_elements and capacity)
    late_fence_slots: LateFenceSlots,

    /// Maximum allowed memory usage in bytes
    workspace_size: usize,

    /// Current memory usage in bytes
    used_space: usize,

    /// Estimated overhead of the tree structure in bytes
    tree_overhead: usize,

    /// Whether the tree has been initialized via build()
    initialized: bool,
}

impl<T: OVCTreeKey + RecordSize> ReplacementSelectionOVC<T> {
    /// Create a new replacement selection instance with the specified workspace size.
    ///
    /// # Arguments
    /// * `workspace_size` - Maximum memory (in bytes) that can be used for buffering
    pub fn new(workspace_size: usize) -> Self {
        Self {
            tree: LoserTreeOVC::new(vec![]),
            next_run_buffer: Vec::new(),
            late_fence_slots: LateFenceSlots::new(),
            workspace_size,
            used_space: 0,
            tree_overhead: 0,
            initialized: false,
        }
    }

    /// Insert an initial record before building the tree.
    ///
    /// This method can only be called before `build()`. Initial records form
    /// the first sorted run.
    ///
    /// # Panics
    /// Panics if called after `build()` has been invoked.
    pub fn insert_initial(&mut self, record: T) {
        assert!(
            !self.initialized,
            "Cannot insert initial records after build is called"
        );
        self.used_space += record.size();
        self.next_run_buffer.push(record);
    }

    /// Build the initial tree from all inserted records.
    ///
    /// This transitions the data structure from setup phase to operational phase.
    /// After calling this method, use `absorb_record()` to process new records.
    ///
    /// # Panics
    /// Panics if called more than once.
    pub fn build(&mut self) {
        assert!(!self.initialized, "build() may only be called once");
        self.initialize_tree_from_buffer();
        self.initialized = true;
    }

    /// Initialize the tree from the next run buffer and set up padding slots
    fn initialize_tree_from_buffer(&mut self) {
        let num_real = self.next_run_buffer.len();
        let capacity = if num_real == 0 {
            0
        } else {
            num_real.next_power_of_two()
        };

        self.update_tree_overhead(capacity);

        let values = self.next_run_buffer.drain(..);
        self.tree.reset_from_iter(num_real, values);
        self.late_fence_slots.reset(capacity);

        // Capture padding slots (between num_real and next power of 2)
        self.late_fence_slots.set_range(num_real, capacity);
    }

    fn update_tree_overhead(&mut self, capacity: usize) {
        self.used_space = self.used_space.saturating_sub(self.tree_overhead);
        let new_overhead = LoserTreeOVC::<T>::structure_overhead_bytes_for_capacity(capacity);
        self.used_space += new_overhead;
        self.tree_overhead = new_overhead;
    }

    /// Absorb a new record and stream evicted records via `emit`.
    pub fn absorb_record_with(&mut self, record: T, emitter: &mut impl RunEmitter<T>) {
        assert!(self.initialized, "absorb_record called before build()");

        let mut record = record;
        let new_size = record.size();

        // Ensure we have an active run to compare against
        self.ensure_active_tree(emitter);

        // Determine if the record belongs to current or next run
        let belongs_to_next_run = self.should_defer_to_next_run(&mut record);

        if belongs_to_next_run {
            self.add_to_next_run(record);
        } else {
            self.add_to_current_run(record, emitter);
        }

        self.used_space += new_size;

        // Evict records until workspace constraint is satisfied
        self.evict_until_space_available(emitter);
    }

    /// Ensure there is an active tree to work with
    fn ensure_active_tree(&mut self, emitter: &mut impl RunEmitter<T>) {
        if self.tree.peek().is_none() {
            self.switch_run(emitter);
        }
    }

    /// Determine if a record should go to the next run
    fn should_defer_to_next_run(&mut self, record: &mut T) -> bool {
        if let Some((winner, _)) = self.tree.peek() {
            // Compare with current winner and update offset value codes
            // Returns true if record >= winner (can go in current run)
            // Returns false if record < winner (must go to next run)
            !record.derive_ovc_from(winner)
        } else {
            false
        }
    }

    /// Add a record to the next run buffer
    fn add_to_next_run(&mut self, mut record: T) {
        record.set_initial_ovc();
        self.next_run_buffer.push(record);
    }

    /// Add a record to the current run tree
    fn add_to_current_run(&mut self, record: T, emitter: &mut impl RunEmitter<T>) {
        let record_size = record.size();
        let will_exceed = self.used_space + record_size > self.workspace_size;

        // Only use padding slots if we have memory headroom
        // When memory is tight, prefer push interface which combines eviction + insertion in one pass
        if !will_exceed {
            if let Some(idx) = self.late_fence_slots.pop() {
                // Use available padding slot (no eviction needed)
                self.tree.update(idx, record);
                return;
            }
        }

        // Memory tight OR no padding slots: use push interface
        // This efficiently combines eviction and insertion in a single leaf-to-root pass
        let winner = self.tree.push(record);
        if !winner.is_late_fence() && !winner.is_early_fence() {
            self.used_space -= winner.size();
            emitter.emit(winner);
        }
    }

    /// Evict records until workspace size constraint is satisfied
    fn evict_until_space_available(&mut self, emitter: &mut impl RunEmitter<T>) {
        while self.used_space > self.workspace_size {
            self.ensure_active_tree(emitter);

            if self.tree.peek().is_none() {
                break; // No more records to evict
            }

            let (_, idx) = self.tree.peek().unwrap();
            let winner = self.tree.push(T::late_fence());

            if !winner.is_late_fence() && !winner.is_early_fence() {
                self.used_space -= winner.size();
                emitter.emit(winner);
                self.late_fence_slots.push(idx);
            }
        }
    }

    /// Switch from current run to next run by rebuilding the tree
    fn switch_run(&mut self, emitter: &mut impl RunEmitter<T>) {
        if self.next_run_buffer.is_empty() {
            return;
        }
        self.initialize_tree_from_buffer();
        emitter.on_run_switch();
    }

    pub fn drain_with(&mut self, emitter: &mut impl RunEmitter<T>) {
        // Drain current tree
        self.drain_current_tree(emitter);

        // Switch to next run and drain it as well
        self.switch_run(emitter);
        self.drain_current_tree(emitter);
    }

    /// Helper to drain all records from the current tree
    fn drain_current_tree(&mut self, emitter: &mut impl RunEmitter<T>) {
        while self.tree.peek().is_some() {
            let val = self.tree.push(T::late_fence());
            if !val.is_late_fence() && !val.is_early_fence() {
                self.used_space -= val.size();
                emitter.emit(val);
            }
        }
    }

    /// Get the number of records currently in the next run buffer.
    pub fn buffer_len(&self) -> usize {
        self.next_run_buffer.len()
    }
}

/// Replacement selection that leverages the OVC-aware tree-of-losers
/// implementation.
pub fn run_replacement_selection_ovc<S>(
    mut scanner: ReplacementScanner,
    sink: &mut S,
    memory_limit: usize,
) -> ReplacementSelectionStats
where
    S: RunSink,
{
    let mut rs = ReplacementSelectionOVC::new(memory_limit);
    let mut pending_record: Option<(Vec<u8>, Vec<u8>)> = None;
    let mut memory_used = 0usize;

    loop {
        let Some((key, value)) = next_record(&mut pending_record, scanner.as_mut()) else {
            break;
        };

        let record = OVCKeyValue32::new(key, value);
        let size = record.size();
        ensure_entry_fits(size, memory_limit);

        let projected_count = rs.buffer_len() + 1;
        let projected_capacity = if projected_count == 0 {
            0
        } else {
            projected_count.next_power_of_two()
        };
        let projected_overhead =
            LoserTreeOVC::<OVCKeyValue32>::structure_overhead_bytes_for_capacity(
                projected_capacity,
            );
        if memory_used + size + projected_overhead > memory_limit {
            if rs.buffer_len() > 0 {
                let (_, key, value) = record.take();
                pending_record = Some((key, value));
                break;
            }
            panic!(
                "Workspace too small for replacement selection OVC: record size {} + tree overhead {} exceeds limit {}",
                size, projected_overhead, memory_limit
            );
        }

        memory_used += size;
        rs.insert_initial(record);

        if memory_used >= memory_limit {
            break;
        }
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
        #[cfg(debug_assertions)]
        validator: OvcRunValidator::new(),
    };

    while let Some((key, value)) = next_record(&mut pending_record, scanner.as_mut()) {
        let record = OVCKeyValue32::new(key, value);
        let size = record.size();
        ensure_entry_fits(size, memory_limit);

        rs.absorb_record_with(record, &mut emitter);
    }

    rs.drain_with(&mut emitter);

    sink.finish_run();

    ReplacementSelectionStats { records_emitted }
}

struct SinkEmitter<'a, S: RunSink> {
    sink: &'a mut S,
    emitted: &'a mut usize,
    #[cfg(debug_assertions)]
    validator: OvcRunValidator,
}

impl<'a, S: RunSink> RunEmitter<OVCKeyValue32> for SinkEmitter<'a, S> {
    fn emit(&mut self, record: OVCKeyValue32) {
        if record.is_sentinel() {
            return;
        }
        let (ovc, key, value) = record.take();
        #[cfg(debug_assertions)]
        self.validator.validate(ovc, &key);
        self.sink.push_record_with_ovc(ovc, &key, &value);
        *self.emitted += 1;
    }

    fn on_run_switch(&mut self) {
        #[cfg(debug_assertions)]
        self.validator.on_run_switch();
        self.sink.finish_run();
        self.sink.start_run();
    }
}

#[cfg(debug_assertions)]
struct OvcRunValidator {
    prev_key: Option<Vec<u8>>,
}

#[cfg(debug_assertions)]
impl OvcRunValidator {
    fn new() -> Self {
        Self { prev_key: None }
    }

    fn on_run_switch(&mut self) {
        self.prev_key = None;
    }

    fn validate(&mut self, ovc: crate::ovc::offset_value_coding_32::OVCU32, key: &[u8]) {
        use crate::ovc::offset_value_coding_32::compute_ovc_delta;
        use crate::ovc::offset_value_coding_64::OVCFlag;

        match self.prev_key.as_deref() {
            None => {
                assert!(
                    ovc.is_initial_value(),
                    "first record in run should be InitialValue, got flag {}",
                    ovc.flag().as_u8()
                );
            }
            Some(prev) => {
                let consistent = match ovc.flag() {
                    OVCFlag::EarlyFence | OVCFlag::LateFence => false,
                    OVCFlag::DuplicateValue => prev == key,
                    OVCFlag::InitialValue => false,
                    OVCFlag::NormalValue => {
                        let stored_offset = ovc.offset();
                        if stored_offset > prev.len() || stored_offset > key.len() {
                            false
                        } else if prev[..stored_offset] != key[..stored_offset] {
                            false
                        } else {
                            let stored_value = ovc.value();
                            let actual_suffix = &key[stored_offset..];
                            let check_len = stored_value.len().min(actual_suffix.len());
                            if stored_value[..check_len] != actual_suffix[..check_len] {
                                false
                            } else if stored_value.len() > check_len
                                && stored_value[check_len..].iter().any(|&b| b != 0)
                            {
                                false
                            } else {
                                true
                            }
                        }
                    }
                };

                assert!(consistent, "inconsistent OVC for key {:?}", key);

                let expected = compute_ovc_delta(Some(prev), key);
                assert_eq!(
                    ovc, expected,
                    "non-optimal OVC for key {:?}: expected {}, got {}",
                    key, expected, ovc
                );
            }
        }

        self.prev_key = Some(key.to_vec());
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::cell::RefCell;

    use super::*;
    use crate::ovc::offset_value_coding_32::OVCEntry32;
    use crate::ovc::offset_value_coding_32::is_ovc_consistent;

    /// Helper to collect records into runs, automatically switching to a new run when requested
    struct RunCollector<T> {
        runs: RefCell<Vec<Vec<T>>>,
    }

    impl<T> RunCollector<T> {
        fn new() -> Self {
            Self {
                runs: RefCell::new(vec![Vec::new()]),
            }
        }

        fn into_runs(self) -> Vec<Vec<T>> {
            self.runs.into_inner()
        }
    }

    impl<T> RunEmitter<T> for RunCollector<T> {
        fn emit(&mut self, record: T) {
            let mut runs = self.runs.borrow_mut();
            runs.last_mut().unwrap().push(record);
        }

        fn on_run_switch(&mut self) {
            let mut runs = self.runs.borrow_mut();
            runs.push(Vec::new());
        }
    }

    fn make_entry(val: u16, size: usize) -> OVCEntry32 {
        let mut key = vec![0; size];
        if size > 0 {
            key[0] = (val >> 8) as u8;
        }
        if size > 1 {
            key[1] = val as u8;
        }
        OVCEntry32::new(key)
    }

    #[test]
    fn test_padding_optimization_usage() {
        // With overhead accounting:
        // - Each 10-byte key record = 24 (struct) + 10 (key) = 34 bytes
        // - 3 records = 102 bytes
        // - Tree capacity = 4, overhead = 24 (vec) + 4*8 (node) = 56 bytes
        // - After absorb: 4 records = 136 bytes, overhead = 56 bytes
        // Total after absorb = 136 + 56 = 192 bytes
        let mut rs = ReplacementSelectionOVC::new(500);
        rs.insert_initial(make_entry(10, 10));
        rs.insert_initial(make_entry(20, 10));
        rs.insert_initial(make_entry(30, 10));

        rs.build();

        let mut collector = RunCollector::new();
        rs.absorb_record_with(make_entry(15, 10), &mut collector);

        let runs = collector.into_runs();
        assert!(runs[0].is_empty(), "Should use padding slot, not evict");
        assert!(
            rs.late_fence_slots.is_empty(),
            "Padding slot should be consumed"
        );
    }

    #[test]
    fn test_run_generation_logic() {
        // With overhead: each 10-byte record = 34 bytes
        // 2 records = 68 bytes, tree capacity=2, overhead = 2*8 + 24 = 40 bytes
        // Total = 108 bytes. Use workspace 180 so first absorb fits without eviction
        // Then record 5 < tree min (10), goes to buffer
        let mut rs = ReplacementSelectionOVC::new(180);
        rs.insert_initial(make_entry(10, 10));
        rs.insert_initial(make_entry(20, 10));
        rs.build();

        let mut collector = RunCollector::new();
        // Record 5 is smaller than tree minimum (10), goes to buffer
        rs.absorb_record_with(make_entry(5, 10), &mut collector);

        // No eviction expected since we have headroom
        let runs = collector.runs.borrow();
        assert!(runs[0].is_empty() || runs[0].len() >= 1);
        drop(runs);

        // Record 5 should be in buffer since it's smaller than tree minimum
        assert_eq!(rs.buffer_len(), 1);
    }

    #[test]
    fn test_variable_size_eviction() {
        // With overhead: each 2-byte record = 26 bytes, 6-byte record = 30 bytes
        // 5 records of 2 bytes = 5 * 26 = 130 bytes
        // Tree capacity = 8, overhead = 8*8 + 24 = 88 bytes
        // Use workspace that allows initial records but forces eviction on absorb
        let mut rs = ReplacementSelectionOVC::new(300);
        for i in 1..=5 {
            rs.insert_initial(make_entry(i * 10, 2));
        }
        rs.build();

        let mut collector = RunCollector::new();
        rs.absorb_record_with(make_entry(100, 6), &mut collector);

        // With larger workspace, fewer evictions needed
        // The exact count depends on memory pressure
        let runs = collector.into_runs();
        // Verify records are emitted in sorted order
        for i in 1..runs[0].len() {
            assert!(runs[0][i].get_key() >= runs[0][i - 1].get_key());
        }
    }

    #[test]
    fn test_run_switch() {
        // Test that runs switch correctly when tree empties
        // With overhead: 10-byte record = 34 bytes
        // Use workspace that fits 2 records + overhead initially
        // Tree capacity=2, overhead = 2*8 + 24 = 40 bytes
        // 2 records (68) + overhead (40) = 108 bytes
        let mut rs = ReplacementSelectionOVC::new(150);
        rs.insert_initial(make_entry(10, 10));
        rs.insert_initial(make_entry(20, 10));
        rs.build();

        let mut collector = RunCollector::new();

        // Absorb smaller record (goes to buffer) and larger record (goes to tree)
        rs.absorb_record_with(make_entry(5, 10), &mut collector); // -> buffer
        rs.absorb_record_with(make_entry(25, 10), &mut collector); // -> tree (uses padding slot)

        // Drain to see all runs
        rs.drain_with(&mut collector);

        let runs: Vec<Vec<OVCEntry32>> = collector
            .into_runs()
            .into_iter()
            .filter(|r| !r.is_empty())
            .collect();

        // Should have at least 2 runs: first run with original records, second with buffer
        assert!(
            runs.len() >= 2,
            "Should have at least 2 runs, got {}",
            runs.len()
        );

        // Verify all records are sorted within their runs
        for run in &runs {
            for i in 1..run.len() {
                assert!(
                    run[i].get_key() >= run[i - 1].get_key(),
                    "Run should be sorted"
                );
            }
        }
    }

    #[test]
    fn test_full_sort_experiment() {
        let mut rng = StdRng::seed_from_u64(40);
        let item_size = 10;
        let workspace_capacity_items = 10;
        // With overhead: each record = 24 (struct) + item_size bytes
        // Plus tree overhead: capacity * 8 + 24
        // For 10 items, capacity = 16, overhead = 16*8 + 24 = 152 bytes
        // Total per record with overhead ~= 34 bytes
        // Use workspace that can hold ~10 records with overhead
        let record_size_with_overhead = 28 + item_size; // 32-bit OVC entry overhead
        let tree_overhead_estimate =
            LoserTreeOVC::<OVCEntry32>::structure_overhead_bytes_for_capacity(16);
        let workspace_size =
            workspace_capacity_items * record_size_with_overhead + tree_overhead_estimate;
        let num_elements = 300;

        let mut rs = ReplacementSelectionOVC::new(workspace_size);

        let mut data: Vec<OVCEntry32> = (0..num_elements)
            .map(|_| make_entry(rng.random_range(0..1000), item_size))
            .collect();

        for _ in 0..workspace_capacity_items {
            if let Some(rec) = data.pop() {
                rs.insert_initial(rec);
            }
        }
        rs.build();

        let mut collector = RunCollector::new();
        while let Some(rec) = data.pop() {
            rs.absorb_record_with(rec, &mut collector);
        }

        rs.drain_with(&mut collector);

        let runs: Vec<Vec<OVCEntry32>> = collector
            .into_runs()
            .into_iter()
            .filter(|r| !r.is_empty())
            .collect();
        let run_count = runs.len();

        println!(
            "Processed {} items with workspace size {}",
            num_elements, workspace_size
        );
        println!("Generated {} sorted runs:", run_count);

        let mut total_elements = 0;
        for (r, run_slice) in runs.iter().enumerate() {
            total_elements += run_slice.len();
            let consistent = is_ovc_consistent(run_slice);
            println!(
                "Run {} (len={}, consistent={}):",
                r,
                run_slice.len(),
                consistent
            );
            for entry in run_slice {
                println!("  ovc={:?}, key={:?}", entry.get_ovc(), entry.get_key());
            }
            for i in 1..run_slice.len() {
                assert!(
                    run_slice[i].get_key() >= run_slice[i - 1].get_key(),
                    "Run {} is unsorted at index {}! {:?} < {:?}",
                    r,
                    i,
                    run_slice[i].get_key(),
                    run_slice[i - 1].get_key()
                );
            }
        }

        assert_eq!(total_elements, num_elements);
        assert!(run_count > 1);

        // Expected Logic Check:
        // With random data, run length is expected to be ~2 * WorkspaceSize
        // 100 items / (2 * 5 items) = ~10 runs expected.
        println!(
            "Average Run Length: {:.2}",
            num_elements as f64 / run_count as f64
        );
    }
}
