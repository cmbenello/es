// Bitset-backed free-slot tracker with a summary bitmap.
//
// ASCII view (capacity = 20 slots, 1st level = 64-bit words shown as 8 bits):
//
// slots (indices):  0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 | 16 17 18 19
// level[0] words:  [0 0 0 1 0 1 0 0]  [1 0 0 0 0 0 0 0]   [0 1 1 0 ....]
//                         ^   ^        ^                     ^ ^
//                         |   |        |                     | |
//                      free  free    free                 free free
//
// level[1] (summary of level[0] words):
//   one bit per word in level[0]; bit=1 means that word has at least one free slot.
//   e.g. [1 1 1 0 0 0 0 0] -> words 0,1,2 are non-empty
//
// pop():
// 1) read top summary word, find first set bit (trailing_zeros) -> picks a word index
// 2) descend until level[0] -> picks a bit within that word
// 3) clear the bit; if a word becomes empty, clear its parent summary bit
//
// This is O(1) because the number of levels is log_64(capacity).
pub(crate) struct LateFenceSlots {
    capacity: usize,
    levels: Vec<Vec<u64>>,
}

impl LateFenceSlots {
    pub(crate) fn new() -> Self {
        Self {
            capacity: 0,
            levels: Vec::new(),
        }
    }

    // Prepares bitsets for a new capacity without dropping existing allocations
    // unless the level structure must change (resize may still grow as needed).
    pub(crate) fn reset(&mut self, capacity: usize) {
        self.capacity = capacity;
        let desired = Self::level_lengths(capacity);
        if desired.is_empty() {
            for level in &mut self.levels {
                level.fill(0);
            }
            return;
        }

        if self.levels.len() > desired.len() {
            self.levels.truncate(desired.len());
        } else if self.levels.len() < desired.len() {
            self.levels.resize_with(desired.len(), Vec::new);
        }

        for (level, &len) in self.levels.iter_mut().zip(desired.iter()) {
            level.resize(len, 0);
            level.fill(0);
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.levels.last().map_or(true, |top| top[0] == 0)
    }

    pub(crate) fn push(&mut self, idx: usize) {
        if self.capacity == 0 {
            return;
        }
        debug_assert!(idx < self.capacity);

        let word_idx = idx >> 6;
        let bit = idx & 63;
        let word = &mut self.levels[0][word_idx];
        let before = *word;
        *word |= 1u64 << bit;
        if before != 0 {
            return;
        }
        self.propagate_set(word_idx);
    }

    pub(crate) fn pop(&mut self) -> Option<usize> {
        if self.is_empty() {
            return None;
        }

        let mut word_idx = 0usize;
        // Walk summary levels from top to bottom using trailing_zeros to
        // find a non-empty word at each level.
        for level in (1..self.levels.len()).rev() {
            let word = self.levels[level][word_idx];
            let bit = word.trailing_zeros() as usize;
            word_idx = (word_idx << 6) + bit;
        }

        let word = self.levels[0][word_idx];
        let bit = word.trailing_zeros() as usize;
        let idx = (word_idx << 6) + bit;
        if idx >= self.capacity {
            return None;
        }

        self.clear_bit(idx);
        Some(idx)
    }

    pub(crate) fn set_range(&mut self, start: usize, end: usize) {
        if start >= end || self.capacity == 0 {
            return;
        }
        debug_assert!(end <= self.capacity);

        // Fast path: set a contiguous range of bits.
        let word_idx = start >> 6;
        let end_word = (end - 1) >> 6;
        let start_bit = start & 63;
        let end_bit = (end - 1) & 63;

        if word_idx == end_word {
            let width = end_bit - start_bit + 1;
            let mask = if width == 64 {
                !0u64
            } else {
                ((1u64 << width) - 1) << start_bit
            };
            self.set_word_mask(word_idx, mask);
            return;
        }

        self.set_word_mask(word_idx, !0u64 << start_bit);
        for idx in (word_idx + 1)..end_word {
            self.set_word_mask(idx, !0u64);
        }
        let mask = if end_bit == 63 {
            !0u64
        } else {
            (1u64 << (end_bit + 1)) - 1
        };
        self.set_word_mask(end_word, mask);
    }

    fn propagate_set(&mut self, mut word_idx: usize) {
        for level in 1..self.levels.len() {
            let bit = word_idx & 63;
            word_idx >>= 6;
            let word = &mut self.levels[level][word_idx];
            let before = *word;
            *word |= 1u64 << bit;
            if before != 0 {
                break;
            }
        }
    }

    fn clear_bit(&mut self, idx: usize) {
        let mut word_idx = idx >> 6;
        let bit = idx & 63;
        let word = &mut self.levels[0][word_idx];
        *word &= !(1u64 << bit);
        if *word != 0 {
            return;
        }

        for level in 1..self.levels.len() {
            let bit = word_idx & 63;
            word_idx >>= 6;
            let word = &mut self.levels[level][word_idx];
            *word &= !(1u64 << bit);
            if *word != 0 {
                break;
            }
        }
    }

    fn set_word_mask(&mut self, word_idx: usize, mask: u64) {
        let word = &mut self.levels[0][word_idx];
        let before = *word;
        *word |= mask;
        if before == 0 && *word != 0 {
            self.propagate_set(word_idx);
        }
    }

    fn level_lengths(capacity: usize) -> Vec<usize> {
        if capacity == 0 {
            return Vec::new();
        }

        let mut lengths = Vec::new();
        let mut bits = capacity;
        loop {
            let words = (bits + 63) / 64;
            lengths.push(words);
            if words == 1 {
                break;
            }
            bits = words;
        }
        lengths
    }
}

#[cfg(test)]
mod tests {
    use super::LateFenceSlots;

    #[test]
    fn test_set_range_and_pop_all() {
        let mut slots = LateFenceSlots::new();
        slots.reset(130);
        slots.set_range(5, 130);

        let mut seen = vec![false; 130];
        let mut count = 0usize;
        while let Some(idx) = slots.pop() {
            assert!(idx >= 5 && idx < 130);
            assert!(!seen[idx], "duplicate pop for idx {}", idx);
            seen[idx] = true;
            count += 1;
        }

        assert_eq!(count, 125);
        assert!(slots.is_empty());
        assert!(seen[..5].iter().all(|v| !*v));
        assert!(seen[5..].iter().all(|v| *v));
    }

    #[test]
    fn test_push_pop_round_trip() {
        let mut slots = LateFenceSlots::new();
        slots.reset(128);
        slots.push(7);
        slots.push(64);
        slots.push(3);

        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, vec![3, 7, 64]);
        assert!(slots.is_empty());
    }

    #[test]
    fn test_reset_clears() {
        let mut slots = LateFenceSlots::new();
        slots.reset(64);
        slots.push(10);
        assert_eq!(slots.pop(), Some(10));

        slots.push(12);
        slots.reset(64);
        assert!(slots.pop().is_none());

        slots.set_range(60, 64);
        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, vec![60, 61, 62, 63]);
    }

    // Edge case tests
    #[test]
    fn test_empty_capacity() {
        let mut slots = LateFenceSlots::new();
        slots.reset(0);
        assert!(slots.is_empty());
        assert_eq!(slots.pop(), None);
        slots.push(0); // Should be no-op
        assert_eq!(slots.pop(), None);
        slots.set_range(0, 10); // Should be no-op
        assert_eq!(slots.pop(), None);
    }

    #[test]
    fn test_capacity_one() {
        let mut slots = LateFenceSlots::new();
        slots.reset(1);
        assert!(slots.is_empty());

        slots.push(0);
        assert!(!slots.is_empty());
        assert_eq!(slots.pop(), Some(0));
        assert!(slots.is_empty());
        assert_eq!(slots.pop(), None);
    }

    #[test]
    fn test_capacity_at_word_boundary_64() {
        let mut slots = LateFenceSlots::new();
        slots.reset(64);

        // Fill all 64 slots
        slots.set_range(0, 64);
        let mut count = 0;
        let mut seen = vec![false; 64];
        while let Some(idx) = slots.pop() {
            assert!(idx < 64);
            assert!(!seen[idx]);
            seen[idx] = true;
            count += 1;
        }
        assert_eq!(count, 64);
        assert!(slots.is_empty());
    }

    #[test]
    fn test_capacity_just_before_word_boundary_63() {
        let mut slots = LateFenceSlots::new();
        slots.reset(63);

        slots.set_range(0, 63);
        let mut count = 0;
        while let Some(idx) = slots.pop() {
            assert!(idx < 63);
            count += 1;
        }
        assert_eq!(count, 63);
        assert!(slots.is_empty());
    }

    #[test]
    fn test_capacity_just_after_word_boundary_65() {
        let mut slots = LateFenceSlots::new();
        slots.reset(65);

        slots.set_range(0, 65);
        let mut count = 0;
        let mut seen = vec![false; 65];
        while let Some(idx) = slots.pop() {
            assert!(idx < 65);
            assert!(!seen[idx]);
            seen[idx] = true;
            count += 1;
        }
        assert_eq!(count, 65);
        assert!(slots.is_empty());
    }

    #[test]
    fn test_capacity_128() {
        let mut slots = LateFenceSlots::new();
        slots.reset(128);

        slots.set_range(0, 128);
        let mut count = 0;
        while let Some(idx) = slots.pop() {
            assert!(idx < 128);
            count += 1;
        }
        assert_eq!(count, 128);
        assert!(slots.is_empty());
    }

    #[test]
    fn test_capacity_127() {
        let mut slots = LateFenceSlots::new();
        slots.reset(127);

        slots.set_range(0, 127);
        let mut count = 0;
        while let Some(idx) = slots.pop() {
            assert!(idx < 127);
            count += 1;
        }
        assert_eq!(count, 127);
        assert!(slots.is_empty());
    }

    #[test]
    fn test_capacity_129() {
        let mut slots = LateFenceSlots::new();
        slots.reset(129);

        slots.set_range(0, 129);
        let mut count = 0;
        while let Some(idx) = slots.pop() {
            assert!(idx < 129);
            count += 1;
        }
        assert_eq!(count, 129);
        assert!(slots.is_empty());
    }

    // set_range tests
    #[test]
    fn test_set_range_single_bit() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);
        slots.set_range(50, 51);

        assert_eq!(slots.pop(), Some(50));
        assert!(slots.is_empty());
    }

    #[test]
    fn test_set_range_empty_range() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);
        slots.set_range(50, 50);
        assert!(slots.is_empty());

        slots.set_range(60, 55); // start > end
        assert!(slots.is_empty());
    }

    #[test]
    fn test_set_range_within_single_word() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);
        slots.set_range(10, 20);

        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, (10..20).collect::<Vec<_>>());
    }

    #[test]
    fn test_set_range_spanning_two_words() {
        let mut slots = LateFenceSlots::new();
        slots.reset(200);
        slots.set_range(60, 70);

        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, (60..70).collect::<Vec<_>>());
    }

    #[test]
    fn test_set_range_spanning_multiple_words() {
        let mut slots = LateFenceSlots::new();
        slots.reset(300);
        slots.set_range(50, 200);

        let mut seen = vec![false; 300];
        let mut count = 0;
        while let Some(idx) = slots.pop() {
            assert!(idx >= 50 && idx < 200);
            assert!(!seen[idx]);
            seen[idx] = true;
            count += 1;
        }
        assert_eq!(count, 150);
    }

    #[test]
    fn test_set_range_starting_at_word_boundary() {
        let mut slots = LateFenceSlots::new();
        slots.reset(200);
        slots.set_range(64, 100);

        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, (64..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_set_range_ending_at_word_boundary() {
        let mut slots = LateFenceSlots::new();
        slots.reset(200);
        slots.set_range(50, 128);

        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, (50..128).collect::<Vec<_>>());
    }

    #[test]
    fn test_set_range_exact_word() {
        let mut slots = LateFenceSlots::new();
        slots.reset(200);
        slots.set_range(64, 128);

        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, (64..128).collect::<Vec<_>>());
    }

    #[test]
    fn test_set_range_full_capacity() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);
        slots.set_range(0, 100);

        let mut count = 0;
        while let Some(idx) = slots.pop() {
            assert!(idx < 100);
            count += 1;
        }
        assert_eq!(count, 100);
    }

    #[test]
    fn test_overlapping_set_ranges() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);
        slots.set_range(10, 30);
        slots.set_range(20, 40);

        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, (10..40).collect::<Vec<_>>());
    }

    #[test]
    fn test_multiple_disjoint_set_ranges() {
        let mut slots = LateFenceSlots::new();
        slots.reset(200);
        slots.set_range(10, 20);
        slots.set_range(50, 60);
        slots.set_range(100, 110);

        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();

        let mut expected = Vec::new();
        expected.extend(10..20);
        expected.extend(50..60);
        expected.extend(100..110);
        assert_eq!(out, expected);
    }

    // Push tests
    #[test]
    fn test_push_first_element() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);
        slots.push(0);
        assert_eq!(slots.pop(), Some(0));
        assert!(slots.is_empty());
    }

    #[test]
    fn test_push_last_element() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);
        slots.push(99);
        assert_eq!(slots.pop(), Some(99));
        assert!(slots.is_empty());
    }

    #[test]
    fn test_push_duplicate() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);
        slots.push(42);
        slots.push(42);
        slots.push(42);

        assert_eq!(slots.pop(), Some(42));
        assert!(slots.is_empty());
    }

    #[test]
    fn test_push_across_word_boundaries() {
        let mut slots = LateFenceSlots::new();
        slots.reset(200);
        slots.push(0);
        slots.push(63);
        slots.push(64);
        slots.push(127);
        slots.push(128);

        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, vec![0, 63, 64, 127, 128]);
    }

    #[test]
    fn test_push_many_elements() {
        let mut slots = LateFenceSlots::new();
        slots.reset(1000);

        let indices = vec![1, 5, 10, 50, 99, 100, 200, 500, 999];
        for &idx in &indices {
            slots.push(idx);
        }

        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, indices);
    }

    // Interleaved push/pop tests
    #[test]
    fn test_interleaved_push_pop() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);

        slots.push(10);
        slots.push(20);
        assert_eq!(slots.pop(), Some(10));

        slots.push(30);
        assert_eq!(slots.pop(), Some(20));
        assert_eq!(slots.pop(), Some(30));
        assert!(slots.is_empty());
    }

    #[test]
    fn test_push_pop_push_pop() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);

        for i in 0..10 {
            slots.push(i * 10);
            assert_eq!(slots.pop(), Some(i * 10));
            assert!(slots.is_empty());
        }
    }

    #[test]
    fn test_fill_drain_refill() {
        let mut slots = LateFenceSlots::new();
        slots.reset(50);

        // First fill
        slots.set_range(0, 50);
        let mut count = 0;
        while slots.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 50);
        assert!(slots.is_empty());

        // Second fill
        slots.set_range(10, 40);
        count = 0;
        while slots.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 30);
        assert!(slots.is_empty());
    }

    // Pop order tests
    #[test]
    fn test_pop_returns_lowest_index_first() {
        let mut slots = LateFenceSlots::new();
        slots.reset(200);

        slots.push(100);
        slots.push(50);
        slots.push(150);
        slots.push(25);

        assert_eq!(slots.pop(), Some(25));
        assert_eq!(slots.pop(), Some(50));
        assert_eq!(slots.pop(), Some(100));
        assert_eq!(slots.pop(), Some(150));
        assert!(slots.is_empty());
    }

    #[test]
    fn test_pop_order_with_set_range() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);
        slots.set_range(40, 60);

        let mut prev = None;
        while let Some(idx) = slots.pop() {
            if let Some(p) = prev {
                assert!(idx > p, "Expected increasing order");
            }
            prev = Some(idx);
        }
    }

    // Reset tests
    #[test]
    fn test_reset_to_larger_capacity() {
        let mut slots = LateFenceSlots::new();
        slots.reset(50);
        slots.set_range(0, 50);

        slots.reset(100);
        assert!(slots.is_empty());

        slots.set_range(0, 100);
        let mut count = 0;
        while slots.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 100);
    }

    #[test]
    fn test_reset_to_smaller_capacity() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);
        slots.set_range(0, 100);

        slots.reset(50);
        assert!(slots.is_empty());

        slots.set_range(0, 50);
        let mut count = 0;
        while slots.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 50);
    }

    #[test]
    fn test_reset_multiple_times() {
        let mut slots = LateFenceSlots::new();

        for cap in [10, 64, 100, 128, 200, 65, 50] {
            slots.reset(cap);
            assert!(slots.is_empty());
            slots.set_range(0, cap);
            let mut count = 0;
            while slots.pop().is_some() {
                count += 1;
            }
            assert_eq!(count, cap);
        }
    }

    #[test]
    fn test_reset_to_zero() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);
        slots.push(50);

        slots.reset(0);
        assert!(slots.is_empty());
        assert_eq!(slots.pop(), None);
    }

    // Large capacity tests
    #[test]
    fn test_large_capacity_4096() {
        let mut slots = LateFenceSlots::new();
        slots.reset(4096);

        slots.set_range(0, 4096);
        let mut count = 0;
        while slots.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 4096);
        assert!(slots.is_empty());
    }

    #[test]
    fn test_large_capacity_sparse() {
        let mut slots = LateFenceSlots::new();
        slots.reset(10000);

        for i in (0..10000).step_by(100) {
            slots.push(i);
        }

        let mut count = 0;
        let mut prev = None;
        while let Some(idx) = slots.pop() {
            if let Some(p) = prev {
                assert!(idx > p);
            }
            prev = Some(idx);
            count += 1;
        }
        assert_eq!(count, 100);
        assert!(slots.is_empty());
    }

    // Stress test with mixed operations
    #[test]
    fn test_mixed_operations() {
        let mut slots = LateFenceSlots::new();
        slots.reset(500);

        // Add some ranges
        slots.set_range(0, 50);
        slots.set_range(100, 150);
        slots.set_range(200, 250);

        // Pop some
        for _ in 0..30 {
            assert!(slots.pop().is_some());
        }

        // Add more individual slots
        slots.push(400);
        slots.push(450);
        slots.push(499);

        // Add another range
        slots.set_range(300, 320);

        // Drain and count
        let mut count = 0;
        let mut prev = None;
        while let Some(idx) = slots.pop() {
            if let Some(p) = prev {
                assert!(idx > p, "Expected increasing order: {} <= {}", p, idx);
            }
            prev = Some(idx);
            count += 1;
        }

        // Should have: (50-30) + 50 + 50 + 3 + 20 = 143
        assert_eq!(count, 143);
        assert!(slots.is_empty());
    }

    #[test]
    fn test_edge_case_63_64_boundary() {
        let mut slots = LateFenceSlots::new();
        slots.reset(128);

        // Test range that ends exactly at bit 63
        slots.set_range(60, 64);
        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, vec![60, 61, 62, 63]);

        // Test range that starts at bit 64
        slots.reset(128);
        slots.set_range(64, 68);
        out.clear();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, vec![64, 65, 66, 67]);

        // Test range spanning the boundary
        slots.reset(128);
        slots.set_range(62, 66);
        out.clear();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, vec![62, 63, 64, 65]);
    }

    #[test]
    fn test_all_boundary_positions() {
        let mut slots = LateFenceSlots::new();

        for boundary in [64, 128, 192, 256] {
            slots.reset(boundary + 10);

            // Test around each boundary
            slots.push(boundary - 1);
            slots.push(boundary);
            slots.push(boundary + 1);

            let mut out = Vec::new();
            while let Some(idx) = slots.pop() {
                out.push(idx);
            }
            out.sort_unstable();
            assert_eq!(out, vec![boundary - 1, boundary, boundary + 1]);
        }
    }

    #[test]
    fn test_set_range_ending_at_capacity() {
        let mut slots = LateFenceSlots::new();
        slots.reset(100);
        slots.set_range(90, 100);

        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, (90..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_single_word_full_range() {
        let mut slots = LateFenceSlots::new();
        slots.reset(64);
        slots.set_range(0, 64);

        let mut count = 0;
        while slots.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 64);
    }

    #[test]
    fn test_partial_word_at_end() {
        let mut slots = LateFenceSlots::new();
        slots.reset(67);
        slots.set_range(64, 67);

        let mut out = Vec::new();
        while let Some(idx) = slots.pop() {
            out.push(idx);
        }
        out.sort_unstable();
        assert_eq!(out, vec![64, 65, 66]);
    }
}
