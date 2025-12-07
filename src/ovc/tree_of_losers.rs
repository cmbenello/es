use std::mem;

use crate::ovc::offset_value_coding::SentinelValue;

pub struct LoserTree<T> {
    // The tree nodes.
    // Index 0: Stores the overall Winner.
    // Indices 1..size: Store the Losers of the tournament matches.
    nodes: Vec<Node<T>>,

    // The number of sources (k).
    // The conceptual leaves are located at indices [capacity .. 2*capacity].
    capacity: usize,
}

/// A node in the Loser Tree.
#[derive(Clone, Debug)]
struct Node<T> {
    key: T,       // The value of the LOSER at this node
    index: usize, // The run ID of the LOSER
}

impl<T> Node<T> {
    pub fn new(key: T, index: usize) -> Self {
        Node { key, index }
    }
}

impl<T: Ord + SentinelValue> LoserTree<T> {
    pub fn new(values: Vec<T>) -> Self {
        let size = values.len();
        if size == 0 {
            return Self {
                nodes: vec![],
                capacity: 0,
            };
        }

        let capacity = size.next_power_of_two();
        let mut nodes = Vec::with_capacity(capacity);

        // Initialize internal nodes (1..capacity) with Early Fence sentinels.
        // Index 0 is also initialized but will be overwritten at the end.
        for _ in 0..capacity {
            nodes.push(Node {
                key: T::early_fence(),
                index: usize::MAX,
            });
        }

        let mut lt = LoserTree { nodes, capacity };

        // 1. Process Real Values
        for (i, val) in values.into_iter().enumerate() {
            lt.pass(i, val);
        }

        // 2. Process Padding (Late Fence)
        for i in size..capacity {
            lt.pass(i, T::late_fence());
        }

        lt
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn peek(&self) -> Option<(&T, usize)> {
        if self.nodes.is_empty() {
            return None;
        }
        if self.nodes[0].key.is_late_fence() {
            return None;
        }
        debug_assert!(!self.nodes[0].key.is_early_fence());
        return Some((&self.nodes[0].key, self.nodes[0].index));
    }

    /// Replaces the current winner with `new_val`, replays the tournament,
    /// and returns the OLD winner value (ownership transferred).
    pub fn push(&mut self, new_val: T) -> T {
        if self.nodes.is_empty() {
            panic!("Cannot push to an empty LoserTree");
        }

        let source_idx = self.nodes[0].index;
        let old_node = self.pass(source_idx, new_val);
        return old_node.key;
    }

    /// Updates the value associated with `source_idx` to `new_val`.
    /// This supports both increasing and decreasing the key.
    /// Returns the old value that was previously stored for `source_idx`.
    pub fn update(&mut self, source_idx: usize, new_val: T) -> T {
        if self.nodes.is_empty() {
            panic!("Cannot update empty tree");
        }
        let old_node = self.pass(source_idx, new_val);
        old_node.key
    }

    /// Marks current winner as exhausted and returns the value that was just exhausted.
    pub fn mark_current_exhausted(&mut self) -> Option<T> {
        if self.nodes.is_empty() {
            return None;
        }

        let source_idx = self.nodes[0].index;
        let old_node = self.pass(source_idx, T::late_fence());

        if old_node.key.is_late_fence() {
            None
        } else {
            Some(old_node.key)
        }
    }

    /// Swaps the node at index `i` with the node at index `j`.
    ///
    /// # Safety
    /// Caller must ensure `i` and `j` are valid indices and `i != j`.
    #[inline(always)]
    fn swap_node(&mut self, i: usize, j: usize) {
        debug_assert!(i < self.nodes.len());
        debug_assert!(j < self.nodes.len());
        debug_assert!(i != j);

        unsafe {
            let ptr_i: *mut Node<T> = self.nodes.get_unchecked_mut(i);
            let ptr_j: *mut Node<T> = self.nodes.get_unchecked_mut(j);
            mem::swap(&mut (*ptr_i), &mut (*ptr_j));
        }
    }

    /// Core logic: Replay the tournament path from `index` up to the root.
    ///
    /// Supports arbitrary updates (decrease-key) via Phase 2 bubble-up.
    /// Uses efficient geometric check (bit shifting) to identify Former Winners.
    fn pass(&mut self, index: usize, key: T) -> Node<T> {
        let mut candidate = Node::new(key, index);
        let mut slot = Self::parent_index(self.leaf_index(index));
        // Tracks the height (level) of `slot`. Leaf is level 0. Parent is level 1.
        let mut level = 1;

        // --- PHASE 1: Standard Loser Tree Climb ---
        // Bubbles up from leaf. Standard "Play Match" logic.
        while slot != self.root_index() && self.nodes[slot].index != index {
            if candidate.key > self.nodes[slot].key {
                mem::swap(&mut candidate, &mut self.nodes[slot]);
            }
            slot = Self::parent_index(slot);
            level += 1;
        }

        // --- PHASE 2: Handle Updates / Final Placement ---
        let mut dest = slot;
        let mut dest_level = level; // Level of `dest`

        // Update Case: We found our own run ID in the tree.
        if candidate.index == index {
            while slot != self.root_index() {
                // Find the ancestor that holds the opponent (the former winner).
                // The former winner is the value in the ancestors that originates
                // from the subtree rooted at `dest` (which is at `dest_level`).
                loop {
                    slot = Self::parent_index(slot);
                    level += 1;

                    if slot == self.root_index() {
                        break;
                    }

                    // Geometric Property: If `opp_leaf` is a descendant of `dest`,
                    // shifting it up by `dest_level` must equal `dest`.
                    let opp_idx = self.nodes[slot].index;
                    if opp_idx != usize::MAX {
                        let opp_leaf = self.leaf_index(opp_idx);
                        if (opp_leaf >> dest_level) == dest {
                            break;
                        }
                    }
                }

                // If candidate is worse, we stop bubbling.
                if candidate.key > self.nodes[slot].key {
                    break;
                }

                // WE WIN
                // Swap content:
                // dest gets the Opponent (Old Winner -> New Loser).
                // slot gets the Candidate (New Winner -> Moving Up).
                self.swap_node(dest, slot);

                // Move our "active" position up to `slot`.
                dest = slot;
                dest_level = level;
            }
        }

        // Final Placement: Put the candidate into `dest`
        mem::swap(&mut self.nodes[dest], &mut candidate);
        candidate
    }

    // --- Navigation Helpers ---

    pub fn leaf_index(&self, run_id: usize) -> usize {
        self.capacity + run_id
    }

    pub fn root_index(&self) -> usize {
        0
    }

    pub fn parent_index(index: usize) -> usize {
        index / 2
    }

    #[cfg(test)]
    pub fn get_keys(&self) -> Vec<&T> {
        self.nodes.iter().map(|node| &node.key).collect()
    }

    /// Visualizes the tree structure in a human-readable format.
    /// Shows the tree layout with node indices, values, and source indices.
    pub fn visualize(&self) -> String
    where
        T: std::fmt::Display,
    {
        if self.nodes.is_empty() {
            return "Empty Tree".to_string();
        }

        let mut output = String::new();
        output.push_str(&format!(
            "=== Loser Tree (capacity={}) ===\n\n",
            self.capacity
        ));

        // Show the winner (root node at index 0)
        output.push_str(&format!(
            "Winner [0]: value={}, source={}\n\n",
            self.nodes[0].key, self.nodes[0].index
        ));

        // Calculate tree height
        let height = (self.capacity as f64).log2().ceil() as usize;

        // Display internal nodes level by level
        for level in 0..height {
            let start_idx = 1 << level; // 2^level
            let end_idx = (1 << (level + 1)).min(self.capacity);

            output.push_str(&format!("Level {} (Internal Nodes):\n", level));

            for idx in start_idx..end_idx {
                if idx < self.nodes.len() {
                    let node = &self.nodes[idx];
                    let value_str = if node.index == usize::MAX {
                        format!("{} (sentinel)", node.key)
                    } else {
                        format!("{}", node.key)
                    };

                    output.push_str(&format!(
                        "  [{}]: value={}, source={}\n",
                        idx, value_str, node.index
                    ));
                }
            }
            output.push('\n');
        }

        // Show conceptual leaves
        output.push_str("Conceptual Leaves (not stored):\n");
        output.push_str(&format!(
            "  Indices {}-{} map to sources 0-{}\n",
            self.capacity,
            2 * self.capacity - 1,
            self.capacity - 1
        ));

        output
    }

    /// Returns a compact single-line representation of the tree.
    /// Format: [winner] | [node1] [node2] ...
    pub fn compact_view(&self) -> String
    where
        T: std::fmt::Display,
    {
        if self.nodes.is_empty() {
            return "[]".to_string();
        }

        let mut parts = Vec::new();

        // Winner
        parts.push(format!(
            "[W:{}(s{})]",
            self.nodes[0].key, self.nodes[0].index
        ));

        // Internal nodes
        for (idx, node) in self.nodes.iter().enumerate().skip(1) {
            let display = if node.index == usize::MAX {
                format!("[{}:-]", idx)
            } else {
                format!("[{}:{}(s{})]", idx, node.key, node.index)
            };
            parts.push(display);
        }

        parts.join(" ")
    }

    /// Prints the tree structure as an ASCII tree diagram.
    /// This provides a visual hierarchical view of the tournament.
    pub fn ascii_tree(&self) -> String
    where
        T: std::fmt::Display,
    {
        if self.nodes.is_empty() {
            return "Empty Tree".to_string();
        }

        let mut output = String::new();
        output.push_str("=== Tournament Tree Structure ===\n\n");

        // Helper function to build tree recursively
        fn build_node<T: std::fmt::Display>(
            nodes: &[Node<T>],
            idx: usize,
            capacity: usize,
            prefix: String,
            is_last: bool,
        ) -> String {
            let mut result = String::new();

            // Current node display
            let connector = if is_last { "└── " } else { "├── " };

            if idx == 0 {
                result.push_str(&format!(
                    "WINNER: {} (source {})\n",
                    nodes[idx].key, nodes[idx].index
                ));
            } else if idx < nodes.len() {
                let node = &nodes[idx];
                let value_str = if node.index == usize::MAX {
                    format!("{} (sentinel)", node.key)
                } else {
                    format!("{}", node.key)
                };
                result.push_str(&format!(
                    "{}{}<{}> LOSER: {} (source {})\n",
                    prefix, connector, idx, value_str, node.index
                ));
            } else if idx >= capacity {
                // Conceptual leaf
                let source = idx - capacity;
                result.push_str(&format!(
                    "{}{}[Leaf {}] source {}\n",
                    prefix, connector, idx, source
                ));
                return result;
            } else {
                // Node is out of bounds but not a leaf, shouldn't happen
                return result;
            }

            // Recurse to children (only for internal nodes below capacity)
            // Note: For node 0, we need to start from node 1, not node 0
            if idx < capacity {
                let left_child = if idx == 0 { 1 } else { 2 * idx };
                let right_child = if idx == 0 { 1 } else { 2 * idx + 1 };

                // For root node (idx 0), only traverse once from node 1
                if idx == 0 {
                    result.push_str(&build_node(nodes, 1, capacity, String::new(), true));
                    return result;
                }

                // Only recurse if children indices are valid
                if left_child >= 2 * capacity && right_child >= 2 * capacity {
                    // Both children would be beyond the tree bounds
                    return result;
                }

                let child_prefix = format!("{}{}   ", prefix, if is_last { " " } else { "│" });

                if right_child < 2 * capacity {
                    result.push_str(&build_node(
                        nodes,
                        right_child,
                        capacity,
                        child_prefix.clone(),
                        false,
                    ));
                }
                if left_child < 2 * capacity {
                    result.push_str(&build_node(nodes, left_child, capacity, child_prefix, true));
                }
            }

            result
        }

        output.push_str(&build_node(
            &self.nodes,
            0,
            self.capacity,
            String::new(),
            true,
        ));
        output
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    #[derive(Debug, Eq, PartialEq, PartialOrd, Ord)]
    struct I32Value(i32);

    impl std::fmt::Display for I32Value {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl SentinelValue for I32Value {
        fn early_fence() -> Self {
            I32Value(i32::MIN)
        }
        fn late_fence() -> Self {
            I32Value(i32::MAX)
        }
        fn is_early_fence(&self) -> bool {
            self.0 == i32::MIN
        }
        fn is_late_fence(&self) -> bool {
            self.0 == i32::MAX
        }
    }

    // Tracks how many byte comparisons are evaluated for ordering.
    #[derive(Clone, Debug)]
    enum CountingBytes {
        EarlyFence,
        Value(Vec<u8>),
        LateFence,
    }

    static ORDER_COMPARISON_COUNT: AtomicUsize = AtomicUsize::new(0);

    impl CountingBytes {
        fn value(bytes: impl Into<Vec<u8>>) -> Self {
            Self::Value(bytes.into())
        }

        fn reset_comparisons() {
            ORDER_COMPARISON_COUNT.store(0, AtomicOrdering::Relaxed);
        }

        fn take_comparisons() -> usize {
            ORDER_COMPARISON_COUNT.swap(0, AtomicOrdering::Relaxed)
        }
    }

    impl PartialEq for CountingBytes {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (CountingBytes::Value(a), CountingBytes::Value(b)) => a == b,
                (CountingBytes::EarlyFence, CountingBytes::EarlyFence)
                | (CountingBytes::LateFence, CountingBytes::LateFence) => true,
                _ => false,
            }
        }
    }

    impl Eq for CountingBytes {}

    impl PartialOrd for CountingBytes {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for CountingBytes {
        fn cmp(&self, other: &Self) -> Ordering {
            match (self, other) {
                (CountingBytes::EarlyFence, CountingBytes::EarlyFence)
                | (CountingBytes::LateFence, CountingBytes::LateFence) => Ordering::Equal,
                (CountingBytes::EarlyFence, _) => Ordering::Less,
                (_, CountingBytes::EarlyFence) => Ordering::Greater,
                (CountingBytes::LateFence, _) => Ordering::Greater,
                (_, CountingBytes::LateFence) => Ordering::Less,
                (CountingBytes::Value(a), CountingBytes::Value(b)) => {
                    for (a_byte, b_byte) in a.iter().zip(b.iter()) {
                        ORDER_COMPARISON_COUNT.fetch_add(1, AtomicOrdering::Relaxed);
                        match a_byte.cmp(b_byte) {
                            Ordering::Equal => continue,
                            non_eq => return non_eq,
                        }
                    }
                    a.len().cmp(&b.len())
                }
            }
        }
    }

    impl SentinelValue for CountingBytes {
        fn early_fence() -> Self {
            CountingBytes::EarlyFence
        }
        fn late_fence() -> Self {
            CountingBytes::LateFence
        }
        fn is_early_fence(&self) -> bool {
            matches!(self, CountingBytes::EarlyFence)
        }
        fn is_late_fence(&self) -> bool {
            matches!(self, CountingBytes::LateFence)
        }
    }

    #[test]
    fn test_push_returns_ownership() {
        let values = vec![I32Value(10), I32Value(5), I32Value(20)];
        let mut tree = LoserTree::new(values);

        assert_eq!(tree.peek(), Some((&I32Value(5), 1)));
        let old = tree.push(I32Value(50));
        assert_eq!(old, I32Value(5));
        assert_eq!(tree.peek(), Some((&I32Value(10), 0)));
    }

    #[test]
    fn test_k_way_merge_correctness() {
        let list1 = vec![1, 10, 20];
        let list2 = vec![5, 15, 25];
        let list3 = vec![2, 8, 30];

        let mut iters = vec![list1.into_iter(), list2.into_iter(), list3.into_iter()];
        let mut initial_heads = Vec::new();
        for iter in iters.iter_mut() {
            if let Some(val) = iter.next() {
                initial_heads.push(I32Value(val));
            }
        }

        let mut tree = LoserTree::new(initial_heads);
        let mut result = Vec::new();

        while let Some((_, source_idx)) = tree.peek() {
            if let Some(next_val) = iters[source_idx].next() {
                let winner = tree.push(I32Value(next_val));
                result.push(winner);
            } else {
                if let Some(winner) = tree.mark_current_exhausted() {
                    result.push(winner);
                }
            }
        }

        assert_eq!(
            result,
            vec![1, 2, 5, 8, 10, 15, 20, 25, 30]
                .into_iter()
                .map(I32Value)
                .collect::<Vec<_>>()
        );
    }

    // =============================================================================
    // Update Functionality Tests
    // =============================================================================

    #[test]
    fn test_update_decrease_key_becomes_winner() {
        // Initial: [10, 20, 30, 40]
        // Tree Winner: 10 (idx 0)
        let values = vec![I32Value(10), I32Value(20), I32Value(30), I32Value(40)];
        let mut tree = LoserTree::new(values);

        assert_eq!(tree.peek().unwrap().0, &I32Value(10));

        // Update idx 3 (value 40) to 5. It should become the new winner.
        let old = tree.update(3, I32Value(5));
        assert_eq!(old, I32Value(40));

        let (val, idx) = tree.peek().unwrap();
        assert_eq!(val, &I32Value(5));
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_update_decrease_key_still_loser() {
        // Initial: [10, 50, 60, 70]
        // Winner: 10
        let values = vec![I32Value(10), I32Value(50), I32Value(60), I32Value(70)];
        let mut tree = LoserTree::new(values);

        // Update idx 1 (50) to 20. It's smaller than 50, but still larger than 10.
        // It should NOT become winner.
        let old = tree.update(1, I32Value(20));
        assert_eq!(old, I32Value(50));

        let (val, idx) = tree.peek().unwrap();
        assert_eq!(val, &I32Value(10)); // Winner unchanged
        assert_eq!(idx, 0);

        // If we pop 10, next winner should be 20.
        tree.push(I32Value(100)); // 0 becomes 100
        let (val, idx) = tree.peek().unwrap();
        assert_eq!(val, &I32Value(20)); // Our updated value wins now
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_update_increase_key() {
        // Initial: [10, 20, 30, 40]
        // Winner: 10
        let values = vec![I32Value(10), I32Value(20), I32Value(30), I32Value(40)];
        let mut tree = LoserTree::new(values);

        // Update idx 0 (Winner) to 100.
        // This is effectively the same as push(100).
        let old = tree.update(0, I32Value(100));
        assert_eq!(old, I32Value(10));

        // New winner should be 20 (idx 1).
        let (val, idx) = tree.peek().unwrap();
        assert_eq!(val, &I32Value(20));
        assert_eq!(idx, 1);

        // Update idx 1 (Winner) to 50.
        // New winner should be 30 (idx 2).
        tree.update(1, I32Value(50));
        let (val, idx) = tree.peek().unwrap();
        assert_eq!(val, &I32Value(30));
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_update_arbitrary_node_increase() {
        // Initial: [10, 20, 30, 40]
        let values = vec![I32Value(10), I32Value(20), I32Value(30), I32Value(40)];
        let mut tree = LoserTree::new(values);

        // Update idx 2 (30) to 200.
        // It wasn't the winner, and it still won't be.
        let old = tree.update(2, I32Value(200));
        assert_eq!(old, I32Value(30));

        // Winner is still 10.
        assert_eq!(tree.peek().unwrap().0, &I32Value(10));

        // Pop 10 -> 500
        tree.push(I32Value(500));
        // Next is 20
        assert_eq!(tree.peek().unwrap().0, &I32Value(20));
        // Pop 20 -> 500
        tree.push(I32Value(500));
        // Next is 40
        assert_eq!(tree.peek().unwrap().0, &I32Value(40));
        // Pop 40 -> 500
        tree.push(I32Value(500));
        // Next is 200 (our updated value)
        assert_eq!(tree.peek().unwrap().0, &I32Value(200));
        assert_eq!(tree.peek().unwrap().1, 2);
    }

    #[test]
    fn test_randomized_stress_updates() {
        // Simple Linear Congruential Generator for deterministic randomness
        struct SimpleRng {
            state: u64,
        }
        impl SimpleRng {
            fn new(seed: u64) -> Self {
                Self { state: seed }
            }
            fn next_u32(&mut self) -> u32 {
                self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (self.state >> 32) as u32
            }
            fn gen_range(&mut self, min: i32, max: i32) -> i32 {
                let range = (max - min) as u32;
                min + (self.next_u32() % range) as i32
            }
        }

        let k = 50;
        let iterations = 2000;
        let mut rng = SimpleRng::new(12345);

        // Ground truth state: A vector representing the current value at each source index.
        let mut current_values: Vec<i32> = (0..k).map(|_| rng.gen_range(0, 10000)).collect();

        // Initialize tree
        let tree_values: Vec<I32Value> = current_values.iter().map(|&x| I32Value(x)).collect();
        let mut tree = LoserTree::new(tree_values);

        for _ in 0..iterations {
            // 1. Verify Property: Tree Peek vs Ground Truth Min
            let (tree_min, tree_idx) = tree.peek().unwrap();

            // Find ground truth min
            let (_idx, gt_min) = current_values
                .iter()
                .enumerate()
                .min_by_key(|&(_, &val)| val)
                .unwrap();

            assert_eq!(
                tree_min.0, *gt_min,
                "Tree minimum value does not match ground truth minimum at iteration"
            );

            // Check that the returned index actually holds that value
            // (Note: We don't check tree_idx == gt_idx strictly because duplicate values might
            // result in different winner indices depending on tournament structure, but the VALUE must match).
            assert_eq!(
                current_values[tree_idx], tree_min.0,
                "Tree returned index does not hold the returned value in ground truth"
            );

            // 2. Perform Random Update
            // Pick a random source index
            let update_idx = rng.gen_range(0, k as i32) as usize;
            // Pick a new random value
            let new_val = rng.gen_range(0, 10000);

            // Update ground truth
            current_values[update_idx] = new_val;

            // Update tree
            tree.update(update_idx, I32Value(new_val));
        }
    }

    #[test]
    fn test_visualizer() {
        let values = vec![I32Value(15), I32Value(8), I32Value(23), I32Value(12)];
        let tree = LoserTree::new(values);

        // Test that visualizers don't panic and return non-empty strings
        let compact = tree.compact_view();
        assert!(!compact.is_empty());
        assert!(compact.contains("W:"));

        let visual = tree.visualize();
        assert!(!visual.is_empty());
        assert!(visual.contains("Winner"));
        assert!(visual.contains("Level"));

        let ascii = tree.ascii_tree();
        assert!(!ascii.is_empty());
        assert!(ascii.contains("WINNER"));
        assert!(ascii.contains("LOSER"));

        // Print for manual inspection during test
        println!("\n--- Compact View ---");
        println!("{}", compact);
        println!("\n--- Level View ---");
        println!("{}", visual);
        println!("\n--- ASCII Tree ---");
        println!("{}", ascii);

        // Test empty tree
        let empty_tree: LoserTree<I32Value> = LoserTree::new(vec![]);
        assert_eq!(empty_tree.visualize(), "Empty Tree");
        assert_eq!(empty_tree.compact_view(), "[]");
        assert_eq!(empty_tree.ascii_tree(), "Empty Tree");
    }

    #[test]
    fn test_geometric_subtree_property() {
        // Test the geometric property: (leaf_idx >> level) == subtree_root
        // This validates that we can determine if a leaf belongs to a subtree
        // using bit-shifting operations in a complete binary tree.

        // For a tree with capacity 8:
        // Leaves are at indices 8-15 (capacity + run_id)
        // Level 0: leaves (8-15)
        // Level 1: parents (4-7)
        // Level 2: parents (2-3)
        // Level 3: root (1)
        // Index 0: overall winner

        // Binary representation examples:
        // Leaf 8  = 0b1000, >> 1 = 0b0100 = 4
        // Leaf 9  = 0b1001, >> 1 = 0b0100 = 4
        // Leaf 10 = 0b1010, >> 1 = 0b0101 = 5
        // Leaf 11 = 0b1011, >> 1 = 0b0101 = 5

        println!("\n=== Testing Geometric Subtree Property ===");
        println!("Property: (leaf_idx >> level) == subtree_root");
        println!("This determines if a leaf belongs to a subtree\n");

        let capacity = 8;

        // Helper to print binary analysis
        let print_check = |leaf_idx: usize, level: usize, expected_root: usize| {
            let computed = leaf_idx >> level;
            let matches = computed == expected_root;
            println!(
                "Leaf {:2} (0b{:04b}) >> {} = {:2} (0b{:04b}) {} subtree_root {} - {}",
                leaf_idx,
                leaf_idx,
                level,
                computed,
                computed,
                if matches { "==" } else { "!=" },
                expected_root,
                if matches { "✓" } else { "✗" }
            );
            matches
        };

        // Test case 1: Leaf 8 (run_id 0) belongs to subtree rooted at 4 (level 1)
        println!("Test 1: Level 1 - Children of node 4");
        let leaf_idx = 8; // Leaf for run_id 0
        let subtree_root = 4;
        let level = 1;
        assert!(print_check(leaf_idx, level, subtree_root));

        // Test case 2: Leaf 9 also belongs to subtree rooted at 4 (level 1)
        let leaf_idx = 9; // Leaf for run_id 1
        assert!(print_check(leaf_idx, level, subtree_root));

        // Test case 3: Leaf 10 should NOT belong to subtree rooted at 4
        println!("\nTest 2: Negative check - Leaf 10 not in subtree 4");
        let leaf_idx = 10; // Leaf for run_id 2
        assert!(!print_check(leaf_idx, level, subtree_root));

        // Test case 4: Leaf 10 belongs to subtree rooted at 5 (level 1)
        println!("\nTest 3: Level 1 - Children of node 5");
        let subtree_root = 5;
        assert!(print_check(leaf_idx, level, subtree_root));
        assert!(print_check(11, level, subtree_root));

        // Test case 5: At level 2, leaves 8-11 belong to subtree rooted at 2
        println!("\nTest 4: Level 2 - All children of node 2");
        let level = 2;
        let subtree_root = 2;
        for run_id in 0..4 {
            let leaf_idx = capacity + run_id;
            assert!(print_check(leaf_idx, level, subtree_root));
        }

        // Test case 6: At level 2, leaves 12-15 do NOT belong to subtree rooted at 2
        println!("\nTest 5: Level 2 - Non-children of node 2");
        for run_id in 4..8 {
            let leaf_idx = capacity + run_id;
            assert!(!print_check(leaf_idx, level, subtree_root));
        }

        // Test case 7: At level 2, leaves 12-15 belong to subtree rooted at 3
        println!("\nTest 6: Level 2 - All children of node 3");
        let subtree_root = 3;
        for run_id in 4..8 {
            let leaf_idx = capacity + run_id;
            assert!(print_check(leaf_idx, level, subtree_root));
        }

        // Test case 8: At level 3, all leaves belong to subtree rooted at 1 (root)
        println!("\nTest 7: Level 3 - All leaves belong to root (node 1)");
        let level = 3;
        let subtree_root = 1;
        for run_id in 0..8 {
            let leaf_idx = capacity + run_id;
            assert!(print_check(leaf_idx, level, subtree_root));
        }

        // Test case 9: Different capacity (16)
        println!("\n=== Testing with capacity 16 ===");
        let capacity = 16;
        let level = 2;
        let subtree_root = 4; // Internal node at level 2

        // Subtree rooted at 4 should contain leaves 16-19 (run_id 0-3)
        println!("\nTest 8: Capacity 16, Level 2 - Children of node 4");
        for run_id in 0..4 {
            let leaf_idx = capacity + run_id;
            assert!(print_check(leaf_idx, level, subtree_root));
        }

        // Leaves 20-23 should NOT belong to subtree rooted at 4
        println!("\nTest 9: Capacity 16, Level 2 - Non-children of node 4");
        for run_id in 4..8 {
            let leaf_idx = capacity + run_id;
            assert!(!print_check(leaf_idx, level, subtree_root));
        }

        println!("\n✓ All geometric property tests passed!");
    }

    #[test]
    fn test_ord_comparisons_are_tracked() {
        CountingBytes::reset_comparisons();

        let mut tree = LoserTree::new(vec![
            CountingBytes::value(b"abc"),
            CountingBytes::value(b"abd"),
        ]);

        // Ignore the comparisons performed during tree construction.
        CountingBytes::reset_comparisons();
        let old = tree.update(0, CountingBytes::value(b"abe"));

        let comparisons = CountingBytes::take_comparisons();

        assert_eq!(old, CountingBytes::value(b"abc"));
        assert_eq!(comparisons, 3); // a==a, b==b, e>d

        let (winner, idx) = tree.peek().unwrap();
        assert_eq!(winner, &CountingBytes::value(b"abd"));
        assert_eq!(idx, 1);
    }
}
