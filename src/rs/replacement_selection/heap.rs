use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::{
    ReplacementScanner, ReplacementSelectionSink, ReplacementSelectionStats, ensure_entry_fits,
    next_record,
};

const ENTRY_METADATA_SIZE: usize = std::mem::size_of::<u32>() * 2;

/// Primary replacement-selection entry point (binary heap strategy).
pub fn run_replacement_selection<S>(
    scanner: ReplacementScanner,
    sink: &mut S,
    memory_limit: usize,
) -> ReplacementSelectionStats
where
    S: ReplacementSelectionSink,
{
    BinaryHeapReplacementSelection::new(scanner, memory_limit).run(sink)
}

struct BinaryHeapReplacementSelection {
    scanner: ReplacementScanner,
    memory_limit: usize,
}

impl BinaryHeapReplacementSelection {
    fn new(scanner: ReplacementScanner, memory_limit: usize) -> Self {
        Self {
            scanner,
            memory_limit,
        }
    }

    fn run<S>(mut self, sink: &mut S) -> ReplacementSelectionStats
    where
        S: ReplacementSelectionSink,
    {
        let mut current_heap = BinaryHeap::<HeapRecord>::new();
        let mut future_heap = BinaryHeap::<HeapRecord>::new();
        let mut pending_record: Option<(Vec<u8>, Vec<u8>)> = None;
        let mut records_emitted = 0usize;
        let mut memory_used = 0usize;

        self.fill_heap(
            &mut current_heap,
            &mut memory_used,
            &mut pending_record,
        );

        if current_heap.is_empty() {
            return ReplacementSelectionStats::default();
        }

        sink.start_run();

        loop {
            if current_heap.is_empty() {
                if future_heap.is_empty() {
                    self.fill_heap(
                        &mut current_heap,
                        &mut memory_used,
                        &mut pending_record,
                    );
                    if current_heap.is_empty() {
                        break;
                    }
                    continue;
                } else {
                    sink.finish_run();
                    std::mem::swap(&mut current_heap, &mut future_heap);
                    sink.start_run();
                }
            }

            let record = current_heap.pop().expect("heap shouldn't be empty");
            memory_used = memory_used.saturating_sub(record.size);

            sink.push_record(&record.key, &record.value);
            records_emitted += 1;

            self.try_insert_next(
                &mut current_heap,
                &mut future_heap,
                &mut memory_used,
                &mut pending_record,
                &record.key,
            );
        }

        sink.finish_run();

        #[cfg(test)]
        {
            // no-op to silence warnings during tests
        }

        ReplacementSelectionStats {
            records_emitted,
        }
    }

    fn fill_heap(
        &mut self,
        heap: &mut BinaryHeap<HeapRecord>,
        memory_used: &mut usize,
        pending_record: &mut Option<(Vec<u8>, Vec<u8>)>,
    ) {
        loop {
            let maybe_record = next_record(pending_record, self.scanner.as_mut());
            let Some((key, value)) = maybe_record else {
                break;
            };

            let size = key.len() + value.len() + ENTRY_METADATA_SIZE;
            ensure_entry_fits(size, self.memory_limit);

            if *memory_used + size > self.memory_limit && !heap.is_empty() {
                *pending_record = Some((key, value));
                break;
            }

            *memory_used += size;
            heap.push(HeapRecord::with_size(key, value, size));

            if *memory_used >= self.memory_limit {
                break;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_insert_next(
        &mut self,
        current_heap: &mut BinaryHeap<HeapRecord>,
        future_heap: &mut BinaryHeap<HeapRecord>,
        memory_used: &mut usize,
        pending_record: &mut Option<(Vec<u8>, Vec<u8>)>,
        last_output_key: &[u8],
    ) {
        let maybe_record = next_record(pending_record, self.scanner.as_mut());
        let Some((key, value)) = maybe_record else {
            return;
        };

        let size = key.len() + value.len() + ENTRY_METADATA_SIZE;
        ensure_entry_fits(size, self.memory_limit);

        if *memory_used + size > self.memory_limit {
            *pending_record = Some((key, value));
            return;
        }

        *memory_used += size;
        let target_heap = if key.as_slice() < last_output_key {
            future_heap
        } else {
            current_heap
        };

        target_heap.push(HeapRecord::with_size(key, value, size));
    }
}

#[derive(Debug, Eq)]
struct HeapRecord {
    key: Vec<u8>,
    value: Vec<u8>,
    size: usize,
}

impl HeapRecord {
    fn with_size(key: Vec<u8>, value: Vec<u8>, size: usize) -> Self {
        Self { key, value, size }
    }
}

impl PartialEq for HeapRecord {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.value == other.value
    }
}

impl Ord for HeapRecord {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed ordering for min-heap behavior
        other.key.cmp(&self.key)
    }
}

impl PartialOrd for HeapRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CollectingSink {
        runs: Vec<Vec<(Vec<u8>, Vec<u8>)>>,
        current_run: Vec<(Vec<u8>, Vec<u8>)>,
    }

    impl CollectingSink {
        fn new() -> Self {
            Self {
                runs: Vec::new(),
                current_run: Vec::new(),
            }
        }
    }

    impl ReplacementSelectionSink for CollectingSink {
        fn start_run(&mut self) {
            self.current_run.clear();
        }

        fn push_record(&mut self, key: &[u8], value: &[u8]) {
            self.current_run.push((key.to_vec(), value.to_vec()));
        }

        fn finish_run(&mut self) {
            if !self.current_run.is_empty() {
                self.runs.push(std::mem::take(&mut self.current_run));
            }
        }
    }

    fn create_scanner(data: Vec<(Vec<u8>, Vec<u8>)>) -> ReplacementScanner {
        Box::new(data.into_iter())
    }

    #[test]
    fn test_empty_input() {
        let scanner = create_scanner(vec![]);
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection(scanner, &mut sink, 1024);
        assert_eq!(stats.records_emitted, 0);
        assert!(sink.runs.is_empty());
    }

    #[test]
    fn test_single_run_sorted_input() {
        let scanner = create_scanner(vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"b".to_vec(), b"2".to_vec()),
            (b"c".to_vec(), b"3".to_vec()),
        ]);
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection(scanner, &mut sink, 128);
        assert_eq!(stats.records_emitted, 3);
        assert_eq!(sink.runs.len(), 1);
        assert_eq!(sink.runs[0][0].0, b"a");
        assert_eq!(sink.runs[0][2].0, b"c");
    }

    #[test]
    fn test_multiple_runs_for_descending_input() {
        let scanner = create_scanner(vec![
            (b"c".to_vec(), b"3".to_vec()),
            (b"b".to_vec(), b"2".to_vec()),
            (b"a".to_vec(), b"1".to_vec()),
        ]);
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection(scanner, &mut sink, 16);
        assert_eq!(stats.records_emitted, 3);
        assert!(sink.runs.len() >= 2);
        assert_eq!(sink.runs.iter().map(|r| r.len()).sum::<usize>(), 3);
    }

    #[test]
    fn test_run_lengths_with_small_buffer_sorted_input() {
        let data: Vec<_> = (0u8..10).map(|k| (vec![k], vec![k])).collect();
        let scanner = create_scanner(data);
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection(scanner, &mut sink, 30);
        assert_eq!(stats.records_emitted, 10);
        assert_eq!(sink.runs.len(), 1);
        assert_eq!(sink.runs[0].len(), 10);
    }

    #[test]
    fn test_run_lengths_with_small_buffer_descending_input() {
        let data: Vec<_> = (0u8..10).rev().map(|k| (vec![k], vec![k])).collect();
        let scanner = create_scanner(data);
        let mut sink = CollectingSink::new();

        let stats = run_replacement_selection(scanner, &mut sink, 30);
        assert_eq!(stats.records_emitted, 10);
        let run_lengths: Vec<_> = sink.runs.iter().map(|r| r.len()).collect();
        assert_eq!(run_lengths, vec![3, 3, 3, 1]);
    }
}
