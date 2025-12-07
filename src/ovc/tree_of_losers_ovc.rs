use std::cmp::Ordering;
use std::mem;

use crate::ovc::offset_value_coding::{OVC64Trait, OVCEntry, OVCU64, SentinelValue};

pub struct LoserTreeOVC<T> {
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

impl<T: OVC64Trait> LoserTreeOVC<T> {
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

        let mut lt = LoserTreeOVC { nodes, capacity };

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
    /// The OVC of the new_val must be correctly initialized by the caller.
    pub fn push(&mut self, new_val: T) -> T {
        if self.nodes.is_empty() {
            panic!("Cannot push to an empty LoserTree");
        }

        let source_idx = self.nodes[0].index;
        let old_node = self.pass(source_idx, new_val);
        return old_node.key;
    }

    /// Replaces the OVC of the current winner with `new_ovc`,
    /// and returns the OLD OVC value (ownership transferred).
    /// This is used for handling duplicate OVC values efficiently.
    pub fn replace_top_ovc(&mut self, new_ovc: OVCU64) -> OVCU64 {
        if self.nodes.is_empty() {
            panic!("Cannot replace OVC of an empty LoserTree");
        }

        let old_ovc = self.nodes[0].key.ovc().clone();
        *self.nodes[0].key.ovc_mut() = new_ovc;
        old_ovc
    }

    /// Updates the value associated with `source_idx` to `new_val`.
    /// This supports both increasing and decreasing the key.
    /// Returns the old value that was previously stored for `source_idx`.
    pub fn update(&mut self, source_idx: usize, new_val: T) -> T {
        if self.nodes.is_empty() {
            panic!("Cannot update empty tree");
        }
        let old_node = self.pass_for_update(source_idx, new_val, true);
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

    fn pass(&mut self, index: usize, key: T) -> Node<T> {
        let mut candidate = Node::new(key, index);
        let mut slot = Self::parent_index(self.leaf_index(index));

        while slot != self.root_index() {
            if self.nodes[slot].key.compare_and_update(&mut candidate.key) == Ordering::Less {
                mem::swap(&mut candidate, &mut self.nodes[slot]);
            }
            slot = Self::parent_index(slot);
        }

        let old_node_to_return = mem::replace(&mut self.nodes[slot], candidate);
        old_node_to_return
    }

    /// Updates the value at a specific index in the tree.
    ///
    /// Key differences from `pass`:
    /// - Initializes OVC to initial_value (unknown relationship to previous values)
    /// - Performs full comparison at root when candidate wins to maintain OVC correctness
    /// - Uses copy-down optimization to efficiently update multiple tree levels
    fn pass_for_update(&mut self, index: usize, key: T, mut full_comp: bool) -> Node<T> {
        let mut candidate = Node::new(key, index);

        // Initialize OVC - we don't know the relationship to previous value
        self.init_candidate_ovc(&mut candidate);

        let mut slot = Self::parent_index(self.leaf_index(index));
        let mut level = 1;

        // Phase 1: Climb up the tree until we hit root or find our own index
        while slot != self.root_index() && self.nodes[slot].index != index {
            if self.nodes[slot]
                .key
                .compare_and_update_with_mode(&mut candidate.key, full_comp)
                == Ordering::Less
            {
                mem::swap(&mut candidate, &mut self.nodes[slot]);
                full_comp = false;
            }
            slot = Self::parent_index(slot);
            level += 1;
        }

        // Special handling when we reach the root
        if slot == self.root_index() {
            return self.handle_root_update(candidate, slot, index);
        }

        // Phase 2: Copy-down for updates
        let old_node = self.save_old_node(slot);

        if candidate.index == index {
            let final_slot = self.bubble_up_with_copy_down(&mut candidate, slot, level, full_comp);
            slot = final_slot;
        }

        // Final placement
        self.place_candidate_at_dest(candidate, slot);

        old_node
    }

    /// Initialize candidate's OVC for update operations
    #[inline]
    fn init_candidate_ovc(&self, candidate: &mut Node<T>) {
        assert!(
            !candidate.key.ovc().is_early_fence(),
            "Candidate OVC should not be Early Fence"
        );
        if !candidate.key.is_late_fence() {
            *candidate.key.ovc_mut() = OVCU64::initial_value();
        }
    }

    /// Handle the case where candidate reaches root during update
    fn handle_root_update(&mut self, mut candidate: Node<T>, slot: usize, index: usize) -> Node<T> {
        if !candidate.key.is_late_fence() {
            if candidate.index == index {
                // Full comparison needed - candidate came from update
                if candidate
                    .key
                    .compare_and_update_with_mode(&mut self.nodes[slot].key, true)
                    == Ordering::Less
                {
                    // Candidate wins - reset OVC (could be smaller than old root)
                    *candidate.key.ovc_mut() = OVCU64::initial_value();
                } else {
                    // Candidate loses - inherit max OVC
                    candidate.key.max_ovc(&self.nodes[slot].key);
                }
            } else {
                // Candidate was already in tree - inherit max OVC
                candidate.key.max_ovc(&self.nodes[slot].key);
            }
        }

        mem::replace(&mut self.nodes[slot], candidate)
    }

    /// Save the old node value (will be returned to caller)
    #[inline]
    fn save_old_node(&self, dest: usize) -> Node<T> {
        unsafe { std::ptr::read(&self.nodes[dest]) }
    }

    /// Bubble candidate up the tree using copy-down optimization.
    /// Returns the final slot where candidate should be placed.
    fn bubble_up_with_copy_down(
        &mut self,
        candidate: &mut Node<T>,
        mut slot: usize,
        mut level: usize,
        full_comp: bool,
    ) -> usize {
        let mut dest = slot;
        let mut dest_level = level;

        while slot != self.root_index() {
            // Find next relevant opponent
            self.advance_to_next_opponent(&mut slot, &mut level, dest, dest_level);

            // Check if we still win
            if self.nodes[slot]
                .key
                .compare_and_update_with_mode(&mut candidate.key, full_comp)
                == Ordering::Less
            {
                break; // We lost, stop bubbling
            }

            // We win - copy opponent down and update OVC along path
            self.copy_down_and_update_ovc(dest, slot);

            // Move up the tree
            dest = slot;
            dest_level = level;
        }

        dest
    }

    /// Advance slot to the next relevant opponent in the tree
    fn advance_to_next_opponent(
        &self,
        slot: &mut usize,
        level: &mut usize,
        dest: usize,
        dest_level: usize,
    ) {
        loop {
            *slot = Self::parent_index(*slot);
            *level += 1;

            if *slot == self.root_index() {
                break;
            }

            // Check if this slot's value originated from our subtree
            let opp_idx = self.nodes[*slot].index;
            if opp_idx != usize::MAX {
                let opp_leaf = self.leaf_index(opp_idx);
                if (opp_leaf >> dest_level) == dest {
                    break; // Found a relevant opponent
                }
            }
        }
    }

    /// Copy node from slot to dest and update OVC values along the path
    fn copy_down_and_update_ovc(&mut self, dest: usize, slot: usize) {
        // Copy the winner down
        unsafe {
            let dest_ptr: *mut Node<T> = self.nodes.get_unchecked_mut(dest);
            let slot_ptr = self.nodes.get_unchecked(slot);
            std::ptr::write(dest_ptr, std::ptr::read(slot_ptr));
        }

        // Update OVC for all nodes between dest and slot
        let mut curr = Self::parent_index(dest);
        while curr != slot {
            unsafe {
                let curr_ptr: *mut Node<T> = self.nodes.get_unchecked_mut(curr);
                let slot_ptr = self.nodes.get_unchecked(slot);
                (*curr_ptr).key.max_ovc(&(*slot_ptr).key);
            }
            curr = Self::parent_index(curr);
        }
    }

    /// Place candidate at its final destination
    fn place_candidate_at_dest(&mut self, mut candidate: Node<T>, dest: usize) {
        if dest == self.root_index() {
            // Inherit max OVC when becoming root.
            // At this point, comparison between candidate and root has finished.
            // OVC(prev_root, candidate) = max(OVC(prev_root, current_root), OVC(current_root, candidate))
            candidate.key.max_ovc(&self.nodes[dest].key);
        }

        unsafe {
            let dest_ptr = self.nodes.get_unchecked_mut(dest);
            std::ptr::write(dest_ptr, candidate);
        }
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

pub fn merge_runs_with_tree_of_losers_with_ovc(
    mut runs: Vec<Box<impl Iterator<Item = OVCEntry>>>,
) -> Vec<OVCEntry> {
    let mut initial_entries = Vec::with_capacity(runs.len());
    for run in runs.iter_mut() {
        if let Some(val) = run.next() {
            initial_entries.push(val);
        } else {
            initial_entries.push(OVCEntry::late_fence());
        }
    }

    let mut tree = LoserTreeOVC::new(initial_entries);
    let mut output = Vec::new();

    while let Some((_, source_idx)) = tree.peek() {
        if let Some(next_val) = runs[source_idx].next() {
            output.push(tree.push(next_val));
        } else {
            if let Some(winner) = tree.mark_current_exhausted() {
                output.push(winner);
            }
        }
    }

    output
}

pub fn sort_with_tree_of_losers_with_ovc64(run: Vec<Vec<u8>>) -> Vec<OVCEntry> {
    let mut output = Vec::with_capacity(run.len());
    let mut tree = LoserTreeOVC::new(run.into_iter().map(|r| OVCEntry::new(r)).collect());

    while tree.peek().is_some() {
        output.push(tree.push(OVCEntry::late_fence()));
    }

    output
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use crate::ovc::{
        offset_value_coding::{OVCEntryWithCounter, encode_run_with_ovc64, encode_runs_with_ovc64},
        utils::generate_random_string_array,
    };

    use super::*;

    #[test]
    fn test_merge_with_tree_of_losers_with_ovc() {
        // let runs = vec![
        //     vec![vec![1, 2, 3], vec![3, 4, 5], vec![5, 6, 7]],
        //     vec![vec![2, 3, 4], vec![4, 5, 6], vec![6, 7, 8]],
        // ];
        // let runs = generate_vec_runs(num_runs, num_entries_per_run, 5);
        let runs = vec![
            // Run 0
            vec![
                vec![142, 123, 42, 54, 37],
                vec![146, 5, 243, 248, 37],
                vec![146, 237, 169, 149, 108],
            ],
            // Run 1
            vec![vec![146, 45, 111, 123, 14]],
        ];

        println!("Generated Runs:");
        for (i, run) in runs.iter().enumerate() {
            println!("Run {}: {:?}", i, run);
        }
        // Flatten the runs and sort them
        let mut expected_output = runs.clone().into_iter().flatten().collect::<Vec<_>>();
        expected_output.sort();

        // Encode the runs with OVC
        let ovc_runs = encode_runs_with_ovc64(&runs);
        println!("\nEncoded Runs with OVC:");
        for (i, run) in ovc_runs.iter().enumerate() {
            println!("Run {}:", i);
            for res in run {
                println!("  {:?}", res);
            }
        }

        // Create initial tree
        let initial_entries: Vec<OVCEntry> = ovc_runs.iter().map(|run| run[0].clone()).collect();
        let _tree = LoserTreeOVC::new(initial_entries);

        // Merge the runs with tree of losers with OVC
        let output = merge_runs_with_tree_of_losers_with_ovc(
            ovc_runs
                .into_iter()
                .map(|run| Box::new(run.into_iter()))
                .collect(),
        );

        // Check the output
        for (i, res) in output.iter().enumerate() {
            assert_eq!(res.get_key(), &expected_output[i]);
        }
    }

    #[test]
    fn test_sort_with_tree_of_losers_with_ovc() {
        let run_length = 10;
        let run = generate_random_string_array(run_length, 10);
        // Expected
        let mut expected = run.clone().into_iter().collect::<Vec<_>>();
        expected.sort();

        // Normalize the runs
        let normalized_run = run
            .into_iter()
            .map(|r| r.as_bytes().to_vec())
            .collect::<Vec<_>>();

        // Sort the runs with tree of losers with OVC
        let output: Vec<OVCEntry> = sort_with_tree_of_losers_with_ovc64(normalized_run);

        // Denormalize the output
        let mut result = Vec::new();
        for res in output {
            let denormalized = res
                .get_key()
                .into_iter()
                .map(|c| *c as char)
                .collect::<String>();
            result.push(denormalized);
        }

        assert_eq!(result, expected);
    }

    #[test]
    fn test_run_merge_simple() {
        let run1 = vec![vec![1, 2, 3], vec![3, 4, 5], vec![5, 6, 7]];

        let encoded_run1 = encode_run_with_ovc64(&run1);
        println!("Encoded Run 1:");
        for res in &encoded_run1 {
            println!("{:?}", res);
        }

        let run2 = vec![vec![2, 3, 4], vec![4, 5, 6], vec![6, 7, 8]];

        let encoded_run2 = encode_run_with_ovc64(&run2);
        println!("Encoded Run 2:");
        for res in &encoded_run2 {
            println!("{:?}", res);
        }

        let runs = vec![encoded_run1, encoded_run2];
        let mut expected_output = run1.into_iter().flatten().collect::<Vec<_>>();
        expected_output.sort();

        let output = merge_runs_with_tree_of_losers_with_ovc(
            runs.into_iter()
                .map(|run| Box::new(run.into_iter()))
                .collect(),
        );

        println!("Output\n");
        for res in &output {
            println!("{:?}", res);
        }
    }

    #[test]
    fn test_run_merge_4_runs() {
        let runs = vec![
            vec![vec![1, 2, 3], vec![5, 6, 7]],
            vec![vec![2, 3, 4], vec![5, 6, 7]],
            vec![vec![3, 4, 5], vec![6, 7, 8]],
            vec![vec![4, 5, 6], vec![6, 7, 8]],
        ];

        let encoded_runs = encode_runs_with_ovc64(&runs);

        let mut expected_output = runs.into_iter().flatten().collect::<Vec<_>>();
        expected_output.sort();

        let output = merge_runs_with_tree_of_losers_with_ovc(
            encoded_runs
                .into_iter()
                .map(|run| Box::new(run.into_iter()))
                .collect(),
        );

        println!("Output\n");
        for res in &output {
            println!("{:?}", res);
        }

        assert_eq!(output.len(), expected_output.len());
    }

    #[test]
    fn test_duplicated_values() {
        let runs = vec![
            vec![vec![1, 2, 3], vec![1, 2, 3]],
            vec![vec![1, 2, 3], vec![1, 2, 3]],
            vec![vec![1, 2, 3], vec![1, 2, 3]],
            vec![vec![1, 2, 3], vec![1, 2, 3]],
        ];

        let encoded_run = encode_runs_with_ovc64(&runs);

        println!("Encoded Run:");
        for res in &encoded_run {
            println!("{:?}", res);
        }

        let mut expected_output = runs.into_iter().flatten().collect::<Vec<_>>();
        expected_output.sort();

        let output = merge_runs_with_tree_of_losers_with_ovc(
            encoded_run
                .into_iter()
                .map(|run| Box::new(run.into_iter()))
                .collect(),
        );

        println!("Output\n");
        for res in &output {
            println!("{:?}", res);
        }

        assert_eq!(output.len(), expected_output.len());
    }

    #[test]
    fn test_sort_simple() {
        let random = vec![
            vec![8, 3, 5],
            vec![1, 2, 4],
            vec![6, 7, 9],
            vec![10, 11, 12],
            vec![2, 3, 1],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
        ];

        let output = sort_with_tree_of_losers_with_ovc64(random.clone());

        println!("Output\n");
        for res in &output {
            println!("{:?}", res);
        }

        let mut expected_output = random.clone();
        expected_output.sort();

        assert_eq!(output.len(), expected_output.len());
        for (i, res) in output.iter().enumerate() {
            assert_eq!(res.get_key(), expected_output[i]);
        }
    }

    #[test]
    fn test_sort_duplicated_values() {
        let random = vec![
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 4],
            vec![1, 2, 5],
            vec![1, 2, 1],
        ];

        let output = sort_with_tree_of_losers_with_ovc64(random.clone());

        println!("Output\n");
        for res in &output {
            println!("{:?}", res);
        }

        let mut expected_output = random.clone();
        expected_output.sort();

        assert_eq!(output.len(), expected_output.len());
        for (i, res) in output.iter().enumerate() {
            assert_eq!(res.get_key(), expected_output[i]);
        }
    }

    #[test]
    fn test_sort_different_length() {
        let random = vec![
            vec![1, 2, 3],
            vec![4],
            vec![8, 9],
            vec![10, 11, 12, 13, 14],
            vec![1],
            vec![4, 5, 6],
            vec![4, 5],
            vec![7, 8, 9, 10],
            vec![],
            vec![11],
        ];

        let output = sort_with_tree_of_losers_with_ovc64(random.clone());

        println!("Output\n");
        for res in &output {
            println!("{:?}", res);
        }

        let mut expected_output = random.clone();
        expected_output.sort();

        assert_eq!(output.len(), expected_output.len());
        for (i, res) in output.iter().enumerate() {
            assert_eq!(res.get_key(), expected_output[i]);
        }
    }

    #[test]
    fn test_merge_complicated_multiway() {
        // This test exercises a complex merge scenario with:
        // - 8 runs (requires 3 levels in the loser tree)
        // - Highly interleaved values across runs
        // - Variable-length keys (both short and long byte sequences)
        // - Duplicate prefixes that require deep comparison
        // - Mix of ascending sequences and jumps
        // - Different run lengths to stress exhaustion handling

        let runs = vec![
            // Run 0: Long keys with shared prefix [100, 50, ...]
            vec![
                vec![100, 50, 10, 20, 30],
                vec![100, 50, 15, 25, 35],
                vec![100, 50, 20, 30, 40],
                vec![150, 200, 250],
            ],
            // Run 1: Short keys that interleave early
            vec![vec![50], vec![75], vec![100], vec![125], vec![175]],
            // Run 2: Medium keys with duplicate prefix [100, 50, 15, ...]
            vec![
                vec![100, 50, 15, 20, 30],
                vec![100, 50, 15, 20, 31],
                vec![100, 50, 15, 20, 32],
            ],
            // Run 3: Keys that start low but jump high
            vec![vec![1, 2, 3], vec![200, 201, 202]],
            // Run 4: Single very long key
            vec![vec![100, 50, 15, 25, 35, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            // Run 5: Dense sequence in middle range
            vec![
                vec![100, 45],
                vec![100, 46],
                vec![100, 47],
                vec![100, 48],
                vec![100, 49],
            ],
            // Run 6: Another run with shared [100, 50] prefix
            vec![vec![100, 50, 10, 20, 31], vec![100, 50, 30, 40, 50]],
            // Run 7: High values to test late fence handling
            vec![vec![250, 251, 252], vec![253, 254, 255]],
        ];

        println!("\n=== Complicated Merge Test ===");
        println!("Number of runs: {}", runs.len());
        for (i, run) in runs.iter().enumerate() {
            println!("Run {}: {} keys", i, run.len());
            for (j, key) in run.iter().enumerate() {
                println!("  Key {}: {:?}", j, key);
            }
        }

        // Flatten and sort to get expected output
        let mut expected_output = runs.clone().into_iter().flatten().collect::<Vec<_>>();
        expected_output.sort();

        println!(
            "\nExpected sorted output ({} total keys):",
            expected_output.len()
        );
        for (i, key) in expected_output.iter().enumerate() {
            println!("  {}: {:?}", i, key);
        }

        // Encode runs with OVC
        let ovc_runs = encode_runs_with_ovc64(&runs);
        // Perform merge
        let output = merge_runs_with_tree_of_losers_with_ovc(
            ovc_runs
                .into_iter()
                .map(|run| Box::new(run.into_iter()))
                .collect(),
        );

        println!("\nMerged output ({} entries):", output.len());
        for (i, entry) in output.iter().enumerate() {
            println!("  {} key={:?}", i, entry.get_key());
        }

        // Verify correctness
        assert_eq!(
            output.len(),
            expected_output.len(),
            "Output length mismatch: got {}, expected {}",
            output.len(),
            expected_output.len()
        );

        for (i, (result_entry, expected_key)) in
            output.iter().zip(expected_output.iter()).enumerate()
        {
            assert_eq!(
                result_entry.get_key(),
                expected_key,
                "Mismatch at position {}: got {:?}, expected {:?}",
                i,
                result_entry.get_key(),
                expected_key
            );
        }

        println!("\n✓ Complicated merge test passed!");
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
            fn gen_bytes(&mut self, len: usize) -> Vec<u8> {
                (0..len).map(|_| (self.next_u32() % 256) as u8).collect()
            }
        }

        let k = 3;
        let iterations = 10;
        let mut rng = SimpleRng::new(12345);

        // Ground truth state: A vector representing the current value at each source index.
        let mut current_values: Vec<Vec<u8>> = (0..k).map(|_| rng.gen_bytes(8)).collect();

        // Initialize tree
        let tree_values: Vec<OVCEntry> = current_values
            .iter()
            .map(|x| OVCEntry::new(x.clone()))
            .collect();
        let mut tree = LoserTreeOVC::new(tree_values);

        for i in 0..iterations {
            // 1. Verify Property: Tree Peek vs Ground Truth Min
            let (tree_min, tree_idx) = tree.peek().unwrap();

            println!("Iteration {}", i);
            println!("{}", tree.ascii_tree());

            let mut sorted = current_values.clone();
            sorted.sort();
            for val in &sorted {
                println!("  {:?}", val);
            }

            // Find ground truth min
            let (_idx, gt_min) = current_values
                .iter()
                .enumerate()
                .min_by_key(|&(_, val)| val)
                .unwrap();

            assert_eq!(
                tree_min.get_key(),
                gt_min,
                "Tree minimum value does not match ground truth minimum at iteration"
            );

            // Check that the returned index actually holds that value
            // (Note: We don't check tree_idx == gt_idx strictly because duplicate values might
            // result in different winner indices depending on tournament structure, but the VALUE must match).
            assert_eq!(
                &current_values[tree_idx],
                tree_min.get_key(),
                "Tree returned index does not hold the returned value in ground truth"
            );

            // 2. Perform Random Update
            // Pick a random source index
            let update_idx = rng.gen_range(0, k as i32) as usize;
            // Pick a new random value
            let new_val = rng.gen_bytes(8);

            println!(
                "Updating index {} from {:?} to {:?}",
                update_idx, current_values[update_idx], new_val
            );

            // Update ground truth
            current_values[update_idx] = new_val.clone();

            // Update tree
            tree.update(update_idx, OVCEntry::new(new_val));
        }
    }

    // =============================================================================
    // Update Functionality Tests (similar to tree_of_losers.rs)
    // =============================================================================

    #[test]
    fn test_update_decrease_key_becomes_winner() {
        // Initial: [10, 20, 30, 40] (represented as single-byte Vec<u8>)
        // Tree Winner: [10] (idx 0)
        let values = vec![
            OVCEntry::new(vec![10]),
            OVCEntry::new(vec![20]),
            OVCEntry::new(vec![30]),
            OVCEntry::new(vec![40]),
        ];
        let mut tree = LoserTreeOVC::new(values);
        println!("{}", tree.ascii_tree());

        assert_eq!(tree.peek().unwrap().0.get_key(), &vec![10]);

        // Update idx 3 (value [40]) to [5]. It should become the new winner.
        let old = tree.update(3, OVCEntry::new(vec![5]));
        println!("{}", tree.ascii_tree());
        assert_eq!(old.get_key(), &vec![40]);

        let (val, idx) = tree.peek().unwrap();
        assert_eq!(val.get_key(), &vec![5]);
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_update_decrease_key_still_loser() {
        // Initial: [10, 50, 60, 70]
        // Winner: [10]
        let values = vec![
            OVCEntry::new(vec![10]),
            OVCEntry::new(vec![50]),
            OVCEntry::new(vec![60]),
            OVCEntry::new(vec![70]),
        ];
        let mut tree = LoserTreeOVC::new(values);
        println!("{}", tree.ascii_tree());

        // Update idx 1 ([50]) to [20]. It's smaller than [50], but still larger than [10].
        // It should NOT become winner.
        let old = tree.update(1, OVCEntry::new(vec![20]));
        println!("{}", tree.ascii_tree());
        assert_eq!(old.get_key(), &vec![50]);

        let (val, idx) = tree.peek().unwrap();
        assert_eq!(val.get_key(), &vec![10]); // Winner unchanged
        assert_eq!(idx, 0);

        // If we pop [10], next winner should be [20].
        tree.update(0, OVCEntry::new(vec![15])); // 0 becomes [100]
        println!("{}", tree.ascii_tree());
        let (val, idx) = tree.peek().unwrap();
        assert_eq!(val.get_key(), &vec![15]); // Our updated value wins now
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_update_increase_key() {
        // Initial: [10, 20, 30, 40]
        // Winner: [10]
        let values = vec![
            OVCEntry::new(vec![10]),
            OVCEntry::new(vec![20]),
            OVCEntry::new(vec![30]),
            OVCEntry::new(vec![40]),
        ];
        let mut tree = LoserTreeOVC::new(values);

        // Update idx 0 (Winner) to [100].
        // This is effectively the same as push([100]).
        let old = tree.update(0, OVCEntry::new(vec![100]));
        assert_eq!(old.get_key(), &vec![10]);

        // New winner should be [20] (idx 1).
        let (val, idx) = tree.peek().unwrap();
        assert_eq!(val.get_key(), &vec![20]);
        assert_eq!(idx, 1);

        // Update idx 1 (Winner) to [50].
        // New winner should be [30] (idx 2).
        tree.update(1, OVCEntry::new(vec![50]));
        let (val, idx) = tree.peek().unwrap();
        assert_eq!(val.get_key(), &vec![30]);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_update_arbitrary_node_increase() {
        // Initial: [10, 20, 30, 40]
        let values = vec![
            OVCEntry::new(vec![10]),
            OVCEntry::new(vec![20]),
            OVCEntry::new(vec![30]),
            OVCEntry::new(vec![40]),
        ];
        let mut tree = LoserTreeOVC::new(values);

        // Update idx 2 ([30]) to [200].
        // It wasn't the winner, and it still won't be.
        let old = tree.update(2, OVCEntry::new(vec![200]));
        assert_eq!(old.get_key(), &vec![30]);

        // Winner is still [10].
        assert_eq!(tree.peek().unwrap().0.get_key(), &vec![10]);

        // Pop [10] -> [500]
        tree.push(OVCEntry::new(vec![255, 255]));
        // Next is [20]
        assert_eq!(tree.peek().unwrap().0.get_key(), &vec![20]);
        // Pop [20] -> [500]
        tree.push(OVCEntry::new(vec![255, 255]));
        // Next is [40]
        assert_eq!(tree.peek().unwrap().0.get_key(), &vec![40]);
        // Pop [40] -> [500]
        tree.push(OVCEntry::new(vec![255, 255]));
        // Next is [200] (our updated value)
        assert_eq!(tree.peek().unwrap().0.get_key(), &vec![200]);
        assert_eq!(tree.peek().unwrap().1, 2);
    }

    #[test]
    fn test_update_multi_byte_keys() {
        // Test with multi-byte Vec<u8> keys to ensure OVC works correctly
        let values = vec![
            OVCEntry::new(vec![1, 2, 3]),
            OVCEntry::new(vec![2, 3, 4]),
            OVCEntry::new(vec![3, 4, 5]),
            OVCEntry::new(vec![4, 5, 6]),
        ];
        let mut tree = LoserTreeOVC::new(values);
        println!("{}", tree.ascii_tree());

        // Winner should be [1, 2, 3]
        assert_eq!(tree.peek().unwrap().0.get_key(), &vec![1, 2, 3]);

        // Update idx 2 to become the new minimum
        let old = tree.update(2, OVCEntry::new(vec![0, 9, 9]));
        assert_eq!(old.get_key(), &vec![3, 4, 5]);

        // New winner should be [0, 9, 9]
        let (val, idx) = tree.peek().unwrap();
        println!("{}", tree.ascii_tree());
        assert_eq!(val.get_key(), &vec![0, 9, 9]);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_update_with_shared_prefixes() {
        // Test updates with keys that share prefixes (important for OVC)
        let values = vec![
            OVCEntry::new(vec![100, 50, 10]),
            OVCEntry::new(vec![100, 50, 20]),
            OVCEntry::new(vec![100, 50, 30]),
            OVCEntry::new(vec![100, 50, 40]),
        ];
        let mut tree = LoserTreeOVC::new(values);

        // Winner is [100, 50, 10]
        assert_eq!(tree.peek().unwrap().0.get_key(), &vec![100, 50, 10]);

        // Update idx 3 to have the same prefix but smaller last byte
        let old = tree.update(3, OVCEntry::new(vec![100, 50, 5]));
        assert_eq!(old.get_key(), &vec![100, 50, 40]);

        // New winner should be [100, 50, 5]
        let (val, idx) = tree.peek().unwrap();
        assert_eq!(val.get_key(), &vec![100, 50, 5]);
        assert_eq!(idx, 3);

        // Update idx 1 to have a different prefix entirely
        tree.update(1, OVCEntry::new(vec![99, 255, 255]));

        // Winner should now be [99, 255, 255] since 99 < 100
        let (val, idx) = tree.peek().unwrap();
        assert_eq!(val.get_key(), &vec![99, 255, 255]);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_updates_produce_correct_encoded_run() {
        // This test verifies that when we:
        // 1. Create a tree with initial values
        // 2. Perform various updates
        // 3. Extract values by repeatedly calling push/pop
        // The resulting run matches what we'd get from encoding those values directly

        println!("\n=== Test: Updates Produce Correct Encoded Run ===\n");

        // Start with an initial set of values representing "sources"
        let initial_keys = vec![
            vec![10, 20, 30],
            vec![15, 25, 35],
            vec![20, 30, 40],
            vec![25, 35, 45],
        ];

        println!("Initial keys:");
        for (i, key) in initial_keys.iter().enumerate() {
            println!("  Source {}: {:?}", i, key);
        }

        // Track what values each source should have after updates
        let mut expected_values = initial_keys.clone();

        // Create tree with OVCEntry values
        let mut tree = LoserTreeOVC::new(
            initial_keys
                .iter()
                .map(|k| OVCEntry::new(k.clone()))
                .collect(),
        );

        // Perform a series of updates
        println!("\nPerforming updates:");

        // Update source 2 with a new value
        let new_val_2 = vec![5, 10, 15];
        println!(
            "  Update source 2: {:?} -> {:?}",
            expected_values[2], new_val_2
        );
        tree.update(2, OVCEntry::new(new_val_2.clone()));
        expected_values[2] = new_val_2;

        // Update source 0 with a new value
        let new_val_0 = vec![12, 22, 32];
        println!(
            "  Update source 0: {:?} -> {:?}",
            expected_values[0], new_val_0
        );
        tree.update(0, OVCEntry::new(new_val_0.clone()));
        expected_values[0] = new_val_0;

        // Update source 1 with a new value
        let new_val_1 = vec![8, 18, 28];
        println!(
            "  Update source 1: {:?} -> {:?}",
            expected_values[1], new_val_1
        );
        tree.update(1, OVCEntry::new(new_val_1.clone()));
        expected_values[1] = new_val_1;

        // Update source 3 with a new value
        let new_val_3 = vec![30, 40, 50];
        println!(
            "  Update source 3: {:?} -> {:?}",
            expected_values[3], new_val_3
        );
        tree.update(3, OVCEntry::new(new_val_3.clone()));
        expected_values[3] = new_val_3;

        println!("\nFinal expected values after updates:");
        for (i, key) in expected_values.iter().enumerate() {
            println!("  Source {}: {:?}", i, key);
        }

        // Now extract all values from the tree by repeatedly pushing late_fence
        let mut extracted_run: Vec<OVCEntry> = Vec::new();
        while tree.peek().is_some() {
            let popped = tree.push(OVCEntry::late_fence());
            if !popped.is_late_fence() {
                extracted_run.push(popped);
            }
        }

        println!("\nExtracted run from tree (after updates):");
        for (i, entry) in extracted_run.iter().enumerate() {
            println!("  [{}] {:?}", i, entry);
        }

        // Create the expected encoded run by encoding the final expected values
        let mut sorted_expected = expected_values.clone();
        sorted_expected.sort();

        let encoded_expected_run = encode_run_with_ovc64(&sorted_expected);

        println!("\nExpected encoded run (from sorted final values):");
        for (i, entry) in encoded_expected_run.iter().enumerate() {
            println!("  [{}] {:?}", i, entry);
        }

        // Verify that the extracted run matches the encoded expected run
        assert_eq!(
            extracted_run.len(),
            encoded_expected_run.len(),
            "Extracted run length doesn't match expected encoded run length"
        );

        for (i, (extracted, expected)) in extracted_run
            .iter()
            .zip(encoded_expected_run.iter())
            .enumerate()
        {
            assert_eq!(
                extracted.get_key(),
                expected.get_key(),
                "Mismatch at position {}: extracted {:?} != expected {:?}",
                i,
                extracted.get_key(),
                expected.get_key()
            );

            // Also verify OVC values match
            assert_eq!(
                extracted.get_ovc(),
                expected.get_ovc(),
                "OVC mismatch at position {}: extracted {:?} != expected {:?}",
                i,
                extracted.get_ovc(),
                expected.get_ovc()
            );
        }

        println!("\n✓ Test passed: Extracted run matches encoded expected run!");
    }

    #[test]
    fn test_root_ovc_preservation() {
        // Scenario: Sorted sequence
        // Tree capacity 4.
        let values = vec![
            OVCEntry::new(vec![0]),
            OVCEntry::new(vec![10]),
            OVCEntry::new(vec![20]),
            OVCEntry::new(vec![30]),
        ];

        let mut tree = LoserTreeOVC::new(values);

        // 1. Initial State: Root should be 10.
        let (root_ref, _) = tree.peek().unwrap();
        assert_eq!(root_ref.get_key(), &[0]);
        // Pop the new root. now 10 becomes root
        // Save a copy of the current root (Previous Root for the next step)
        let _prev_root = root_ref.clone();

        tree.push(OVCEntry::late_fence());

        let current_root = tree.peek().unwrap().0.clone();
        println!("Current root after popping 0: {:?}", current_root.get_ovc());

        // 2. Prepare Candidate: 15.
        // We want to replace 10 with 15.
        // 15 is larger than 10, so it preserves sorted order logic.
        let candidate = OVCEntry::new(vec![15]);

        // 4. Update the Tree
        // Update index 0 (which held 10) to 15.
        tree.update(1, candidate);

        // 5. Verify New Root
        let (new_root_ref, _) = tree.peek().unwrap();
        assert_eq!(new_root_ref.get_key(), &[15], "New root should be 15");

        // CRITICAL CHECK: Does the new root still hold the offset '5'?
        // Since 15 became the global winner, it never 'lost' inside the tree during update.
        // Therefore, its offset field should remain untouched by the tree logic.
        // Manually create a ovc
        println!("New root OVC: {}", new_root_ref.get_ovc());
    }

    #[test]
    fn test_update_with_smallest_value_ovc_initial() {
        // Test 1: Update with the smallest value
        // Expected: New root's OVC should be INITIAL_VALUE because we don't know
        // the relationship to the previous root (could be smaller)

        println!("\n=== Test 1: Update with Smallest Value ===");

        // Simple LCG for deterministic randomness
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
            fn gen_range(&mut self, min: usize, max: usize) -> usize {
                let range = (max - min) as u32;
                min + (self.next_u32() % range) as usize
            }
        }

        let mut rng = SimpleRng::new(42);

        // Create a tree with sorted values
        let values = vec![
            OVCEntry::new(vec![10]),
            OVCEntry::new(vec![20]),
            OVCEntry::new(vec![30]),
            OVCEntry::new(vec![40]),
            OVCEntry::new(vec![50]),
            OVCEntry::new(vec![60]),
            OVCEntry::new(vec![70]),
            OVCEntry::new(vec![80]),
        ];

        let mut tree = LoserTreeOVC::new(values);

        // ===== CASE 1: Root has initial_value OVC =====
        println!("\n--- Case 1: Root with initial_value OVC ---");
        println!("Initial tree:");
        println!("{}", tree.ascii_tree());

        let (initial_root, _) = tree.peek().unwrap();
        println!("Initial root: {:?}", initial_root);
        assert_eq!(initial_root.get_key(), &[10]);
        assert!(
            initial_root.get_ovc().is_initial_value(),
            "Initial root should have initial_value OVC"
        );

        // Pick a random non-root index to update
        let update_idx = rng.gen_range(1, 8);
        println!("\nUpdating index {} with smallest value [5]", update_idx);

        // Update with the smallest value (smaller than current root)
        tree.update(update_idx, OVCEntry::new(vec![5]));

        println!("\nTree after update:");
        println!("{}", tree.ascii_tree());

        // Verify new root is the smallest value
        let (new_root, new_idx) = tree.peek().unwrap();
        println!("\nNew root: {:?} from index {}", new_root, new_idx);
        assert_eq!(new_root.get_key(), &[5]);
        assert_eq!(new_idx, update_idx);

        // CRITICAL: The new root's OVC should be INITIAL_VALUE
        // because we don't know if it's smaller than the previous root
        assert!(
            new_root.get_ovc().is_initial_value(),
            "New root OVC should be initial_value when updated with smallest value. Got: {:?}",
            new_root.get_ovc()
        );

        println!("✓ Case 1 passed: New root has initial_value OVC as expected");

        // ===== CASE 2: Root has non-initial_value OVC =====
        println!("\n--- Case 2: Root with non-initial_value OVC ---");

        // Pop the current root [5] to make next value the root
        tree.push(OVCEntry::late_fence());

        let (current_root, _) = tree.peek().unwrap();
        println!("Current root after pop: {:?}", current_root);
        println!("Current root OVC: {:?}", current_root.get_ovc());
        assert_eq!(current_root.get_key(), &[10]);

        // After popping, [10] should have a normal_value OVC (based on comparison with [5])
        assert!(
            current_root.get_ovc().is_normal_value(),
            "Root should have normal or initial OVC. Got: {:?}",
            current_root.get_ovc()
        );

        // Pick another random index and update with smallest value
        let update_idx2 = rng.gen_range(1, 8);
        println!("\nUpdating index {} with smallest value [3]", update_idx2);

        tree.update(update_idx2, OVCEntry::new(vec![3]));

        println!("\nTree after second update:");
        println!("{}", tree.ascii_tree());

        // Verify new root
        let (new_root2, new_idx2) = tree.peek().unwrap();
        println!("\nNew root: {:?} from index {}", new_root2, new_idx2);
        assert_eq!(new_root2.get_key(), &[3]);
        assert_eq!(new_idx2, update_idx2);

        // Again, should be initial_value
        assert!(
            new_root2.get_ovc().is_initial_value(),
            "New root OVC should be initial_value. Got: {:?}",
            new_root2.get_ovc()
        );

        println!("✓ Case 2 passed: New root has initial_value OVC as expected");
    }

    #[test]
    fn test_update_between_root_and_others_ovc_inherited() {
        // Test 2: Update the ROOT's source index with value larger than current root
        // Expected: Value should NOT become new root (loses comparison), and should
        // inherit OVC based on comparison with previous root

        println!("\n=== Test 2: Update Root's Source Index ===");

        // Create tree with well-separated values
        let values = vec![
            OVCEntry::new(vec![10]),
            OVCEntry::new(vec![20]),
            OVCEntry::new(vec![30]),
            OVCEntry::new(vec![40]),
            OVCEntry::new(vec![50]),
            OVCEntry::new(vec![60]),
            OVCEntry::new(vec![70]),
            OVCEntry::new(vec![80]),
        ];

        let mut tree = LoserTreeOVC::new(values);

        // ===== CASE 1: Root has initial_value OVC =====
        println!("\n--- Case 1: Root with initial_value OVC ---");
        println!("Initial tree:");
        println!("{}", tree.ascii_tree());

        let (initial_root, root_idx) = tree.peek().unwrap();
        println!("Initial root: {:?} from index {}", initial_root, root_idx);
        assert_eq!(initial_root.get_key(), &[10]);
        assert_eq!(root_idx, 0);
        assert!(
            initial_root.get_ovc().is_initial_value(),
            "Initial root should have initial_value OVC"
        );

        // Update the ROOT's source index with a value LARGER than next smallest
        // (so it doesn't remain root)
        println!(
            "\nUpdating root's source index {} with value [25]",
            root_idx
        );
        let old_value = tree.update(root_idx, OVCEntry::new(vec![25]));
        println!("Old value: {:?}", old_value);
        assert_eq!(old_value.get_key(), &[10]);

        println!("\nTree after update:");
        println!("{}", tree.ascii_tree());

        // Verify new root is [20] (next smallest)
        let (new_root, new_idx) = tree.peek().unwrap();
        println!("\nNew root: {:?} from index {}", new_root, new_idx);
        assert_eq!(new_root.get_key(), &[20], "New root should be [20]");
        assert_eq!(new_idx, 1);

        // Extract all values to see the OVC encoding
        let mut extracted = Vec::new();
        while let Some((_val, _)) = tree.peek() {
            extracted.push(tree.push(OVCEntry::late_fence()));
        }

        println!("\nExtracted values with OVC:");
        for (i, entry) in extracted.iter().enumerate() {
            println!("  [{}] {:?}", i, entry);
        }

        // Find [25] in the extracted values
        let pos_25 = extracted.iter().position(|e| e.get_key() == &[25]).unwrap();
        println!("\n[25] is at position {}", pos_25);
        let ovc_25 = &extracted[pos_25];

        println!("OVC of [25]: {:?}", ovc_25.get_ovc());

        // Since [25] lost to other values during extraction,
        // it should have a valid OVC encoding
        assert!(
            ovc_25.get_ovc().is_normal_value() || ovc_25.get_ovc().is_initial_value(),
            "OVC should be normal_value or initial_value. Got: {:?}",
            ovc_25.get_ovc()
        );

        println!("✓ Case 1 passed: OVC encoding is correct");

        // ===== CASE 2: Root has non-initial_value OVC =====
        println!("\n--- Case 2: Root with non-initial_value OVC ---");

        // Create a fresh tree
        let values2 = vec![
            OVCEntry::new(vec![10]),
            OVCEntry::new(vec![20]),
            OVCEntry::new(vec![30]),
            OVCEntry::new(vec![40]),
            OVCEntry::new(vec![50]),
            OVCEntry::new(vec![60]),
            OVCEntry::new(vec![70]),
            OVCEntry::new(vec![80]),
        ];

        let mut tree2 = LoserTreeOVC::new(values2);

        // Pop a few values to get a root with non-initial OVC
        tree2.push(OVCEntry::late_fence()); // Pop [10]
        tree2.push(OVCEntry::late_fence()); // Pop [20]

        let (current_root, root_idx2) = tree2.peek().unwrap();
        println!(
            "Current root after pops: {:?} from index {}",
            current_root, root_idx2
        );
        println!("Current root OVC: {:?}", current_root.get_ovc());
        assert_eq!(current_root.get_key(), &[30]);
        assert_eq!(root_idx2, 2);

        // Root should have non-initial OVC after pops
        assert!(
            current_root.get_ovc().is_normal_value() || current_root.get_ovc().is_initial_value(),
            "Root should have valid OVC. Got: {:?}",
            current_root.get_ovc()
        );

        // Update the ROOT's source index with a value LARGER than next smallest
        println!(
            "\nUpdating root's source index {} with value [45]",
            root_idx2
        );
        let old_value2 = tree2.update(root_idx2, OVCEntry::new(vec![45]));
        println!("Old value: {:?}", old_value2);

        println!("\nTree after update:");
        println!("{}", tree2.ascii_tree());

        // Verify new root
        let (new_root2, new_idx2) = tree2.peek().unwrap();
        println!("\nNew root: {:?} from index {}", new_root2, new_idx2);
        assert_eq!(new_root2.get_key(), &[40], "New root should be [40]");

        println!("✓ Case 2 passed: Root update propagated correctly");
    }

    #[test]
    fn test_update_with_late_fence_small() {
        // Test 3: Update entries with late_fence
        // Expected: Late fence should propagate correctly, eventually making tree empty

        println!("\n=== Test 3: Update with Late Fence ===");

        // Create small tree
        let values = vec![
            OVCEntry::new(vec![10]),
            OVCEntry::new(vec![20]),
            OVCEntry::new(vec![30]),
            OVCEntry::new(vec![40]),
        ];

        let mut tree = LoserTreeOVC::new(values);

        println!("Initial tree:");
        println!("{}", tree.ascii_tree());

        let (initial_root, _) = tree.peek().unwrap();
        println!("Initial root OVC: {:?}", initial_root.get_ovc());
        assert!(
            initial_root.get_ovc().is_initial_value(),
            "Initial root should have initial_value OVC"
        );

        // Track which indices have been fenced
        let mut fenced = vec![false; 4];

        // Update random indices with late_fence
        let round = 0;
        let update_idx = round % 4;
        println!(
            "\nRound {}: Updating index {} with late_fence",
            round, update_idx
        );

        let old = tree.update(update_idx, OVCEntry::late_fence());
        println!("Old value: {:?}", old);

        // Mark as fenced
        if !old.is_late_fence() {
            fenced[update_idx] = true;
        }

        println!("\nTree after update:");
        println!("{}", tree.ascii_tree());

        // Count active (non-fenced) values
        let active_count = fenced.iter().filter(|&&f| !f).count();

        // Check peek
        if let Some((root, idx)) = tree.peek() {
            println!(
                "Current root: {:?} from index {} (active count: {})",
                root, idx, active_count
            );
            assert!(
                !root.is_late_fence(),
                "Root should never be late_fence when peek succeeds"
            );
            assert!(
                active_count > 0,
                "If peek succeeds, there should be active values"
            );
        } else {
            println!(
                "Tree is now empty (all values are late_fence, active count: {})",
                active_count
            );
            // Tree can be empty even if not all values are fenced, as long as
            // all remaining values are late_fence
        }
    }

    #[test]
    fn test_update_with_late_fence() {
        // Test 3: Update entries with late_fence
        // Expected: Late fence should propagate correctly, eventually making tree empty

        println!("\n=== Test 3: Update with Late Fence ===");

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
            fn gen_range(&mut self, min: usize, max: usize) -> usize {
                let range = (max - min) as u32;
                min + (self.next_u32() % range) as usize
            }
        }

        // ===== CASE 1: Root has initial_value OVC =====
        println!("\n--- Case 1: Root with initial_value OVC ---");

        let mut rng = SimpleRng::new(456);

        // Create small tree
        let values = vec![
            OVCEntry::new(vec![10]),
            OVCEntry::new(vec![20]),
            OVCEntry::new(vec![30]),
            OVCEntry::new(vec![40]),
        ];

        let mut tree = LoserTreeOVC::new(values);

        println!("Initial tree:");
        println!("{}", tree.ascii_tree());

        let (initial_root, _) = tree.peek().unwrap();
        println!("Initial root OVC: {:?}", initial_root.get_ovc());
        assert!(
            initial_root.get_ovc().is_initial_value(),
            "Initial root should have initial_value OVC"
        );

        // Track which indices have been fenced
        let mut fenced = vec![false; 4];

        // Update random indices with late_fence
        for round in 0..4 {
            let update_idx = rng.gen_range(0, 4);
            println!(
                "\nRound {}: Updating index {} with late_fence",
                round, update_idx
            );

            let old = tree.update(update_idx, OVCEntry::late_fence());
            println!("Old value: {:?}", old);

            // Mark as fenced
            if !old.is_late_fence() {
                fenced[update_idx] = true;
            }

            println!("\nTree after update:");
            println!("{}", tree.ascii_tree());

            // Count active (non-fenced) values
            let active_count = fenced.iter().filter(|&&f| !f).count();

            // Check peek
            if let Some((root, idx)) = tree.peek() {
                println!(
                    "Current root: {:?} from index {} (active count: {})",
                    root, idx, active_count
                );
                assert!(
                    !root.is_late_fence(),
                    "Root should never be late_fence when peek succeeds"
                );
                assert!(
                    active_count > 0,
                    "If peek succeeds, there should be active values"
                );
            } else {
                println!(
                    "Tree is now empty (all values are late_fence, active count: {})",
                    active_count
                );
                // Tree can be empty even if not all values are fenced, as long as
                // all remaining values are late_fence
            }
        }

        // Ensure all indices are fenced
        for idx in 0..4 {
            tree.update(idx, OVCEntry::late_fence());
        }

        println!("\nFinal tree (all late_fence):");
        println!("{}", tree.ascii_tree());

        // Tree should be empty now
        assert!(
            tree.peek().is_none(),
            "Tree should be empty when all values are late_fence"
        );

        println!("✓ Case 1 passed: Late fence handling works correctly");

        // ===== CASE 2: Root has non-initial_value OVC =====
        println!("\n--- Case 2: Root with non-initial_value OVC ---");

        let mut rng2 = SimpleRng::new(789);

        // Create another tree and pop some values first
        let values2 = vec![
            OVCEntry::new(vec![10]),
            OVCEntry::new(vec![20]),
            OVCEntry::new(vec![30]),
            OVCEntry::new(vec![40]),
        ];

        let mut tree2 = LoserTreeOVC::new(values2);

        // Pop first value to get a root with non-initial OVC
        tree2.push(OVCEntry::late_fence());

        let (current_root, _) = tree2.peek().unwrap();
        println!("Current root after pop: {:?}", current_root);
        println!("Current root OVC: {:?}", current_root.get_ovc());
        assert_eq!(current_root.get_key(), &[20]);

        // Track which indices have been fenced
        let mut fenced2 = vec![false; 4];
        fenced2[0] = true; // Already popped

        // Update random indices with late_fence
        for round in 0..4 {
            let update_idx = rng2.gen_range(1, 4);
            println!(
                "\nRound {}: Updating index {} with late_fence",
                round, update_idx
            );

            let old = tree2.update(update_idx, OVCEntry::late_fence());
            println!("Old value: {:?}", old);

            if !old.is_late_fence() {
                fenced2[update_idx] = true;
            }

            println!("\nTree after update:");
            println!("{}", tree2.ascii_tree());

            let active_count = fenced2.iter().filter(|&&f| !f).count();

            if let Some((root, idx)) = tree2.peek() {
                println!(
                    "Current root: {:?} from index {} (active count: {})",
                    root, idx, active_count
                );
                assert!(
                    !root.is_late_fence(),
                    "Root should never be late_fence when peek succeeds"
                );
            } else {
                println!("Tree is now empty (active count: {})", active_count);
            }
        }

        // Ensure all indices are fenced
        for idx in 0..4 {
            tree2.update(idx, OVCEntry::late_fence());
        }

        println!("\nFinal tree (all late_fence):");
        println!("{}", tree2.ascii_tree());

        assert!(
            tree2.peek().is_none(),
            "Tree should be empty when all values are late_fence"
        );

        println!("✓ Case 2 passed: Late fence handling works correctly");
    }

    #[test]
    fn test_ovc_entry_with_counter_counts_byte_comparisons() {
        // derive_ovc_from should count one comparison per byte
        OVCEntryWithCounter::reset_byte_comparisons();
        OVCEntryWithCounter::reset_ovc_comparisons();
        let mut current = OVCEntryWithCounter::new(b"abf".to_vec());
        let prev = OVCEntryWithCounter::new(b"abe".to_vec());
        assert!(current.derive_ovc_from(&prev));
        let derive_count = OVCEntryWithCounter::take_byte_comparisons();
        let derive_ovc_cmp = OVCEntryWithCounter::take_ovc_comparisons();
        assert_eq!(derive_count, 3);
        assert_eq!(derive_ovc_cmp, 0);

        // Tree operations should also tick the counter via compare_and_update/_with_mode
        let mut tree = LoserTreeOVC::new(vec![
            OVCEntryWithCounter::new(b"abc".to_vec()),
            OVCEntryWithCounter::new(b"abd".to_vec()),
        ]);

        // Ignore construction comparisons
        OVCEntryWithCounter::reset_byte_comparisons();
        OVCEntryWithCounter::reset_ovc_comparisons();
        let old = tree.push(OVCEntryWithCounter::new(b"abe".to_vec()));
        let comparisons = OVCEntryWithCounter::take_byte_comparisons();
        let ovc_comparisons = OVCEntryWithCounter::take_ovc_comparisons();

        assert_eq!(old.get_key(), b"abc");
        println!("Byte comparisons during push: {}", comparisons);
        println!("OVC comparisons during push: {}", ovc_comparisons);
        assert!(comparisons == 0); // push should not compare bytes
        assert!(ovc_comparisons >= 1);

        let (winner, idx) = tree.peek().unwrap();
        assert_eq!(winner.get_key(), b"abd");
        assert_eq!(idx, 1);
    }
}
