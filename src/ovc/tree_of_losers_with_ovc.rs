use crate::ovc::{offset_value_coding::OVCTrait, offset_value_coding_u64::OVCEntry64};
use std::{cmp::Ordering, usize};

fn prev_power_of_two(num: usize) -> usize {
    let mut size = 1;
    while size <= num {
        size *= 2;
    }
    size / 2
}

struct TolEntry<T: OVCTrait> {
    value: T,
    run_id: usize,
}

#[allow(dead_code)]
impl<T: OVCTrait> TolEntry<T> {
    pub fn new(value: T, run_id: usize) -> Self {
        Self { value, run_id }
    }

    pub fn early_fence() -> Self {
        Self {
            value: T::early_fence(),
            run_id: usize::MAX,
        }
    }

    pub fn is_early_fence(&self) -> bool {
        self.value.is_early_fence()
    }

    pub fn late_fence() -> Self {
        Self {
            value: T::late_fence(),
            run_id: usize::MAX,
        }
    }

    pub fn is_late_fence(&self) -> bool {
        self.value.is_late_fence()
    }
}

impl<T: std::fmt::Debug + OVCTrait> std::fmt::Debug for TolEntry<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.value.is_early_fence() {
            write!(f, "[EF]")
        } else if self.value.is_late_fence() {
            write!(f, "[LF]")
        } else {
            write!(f, "[R{}:{:?}]", self.run_id, self.value)
        }
    }
}

pub struct TreeOfLosersWithOVC<T: OVCTrait> {
    curr: usize,
    input_leaf_start: usize,
    entries: Vec<TolEntry<T>>,
}

impl<T: OVCTrait> TreeOfLosersWithOVC<T> {
    /// Creates a new TreeOfLosers structure for k-way merging
    ///
    /// # Arguments
    /// * `num_runs` - The number of input runs (sequences) to merge
    ///
    /// # Algorithm
    /// The tree is constructed as a tournament tree where internal nodes store losers
    /// and the root contains the overall winner. The tree structure is designed to handle
    /// any number of runs by creating a balanced tree with additional nodes as needed.
    pub fn new(num_runs: usize) -> Self {
        assert!(num_runs > 0);

        // Calculate the number of leaf nodes needed in the tree.
        // Each pair of runs shares a leaf node, so we need ceil(num_runs/2) leaves.
        let input_leaf_nodes = num_runs.div_ceil(2);
        assert!(input_leaf_nodes > 0);

        // Find the largest power of 2 that is smaller or equal to the number of leaf nodes.
        // This forms the base of our nearly-complete binary tree structure.
        //
        // Example 1: If num_runs=5, then input_leaf_nodes=3 (ceil(5/2))
        //           base_leaf_nodes=2 (largest power of 2 ≤ 3)
        //
        // Initial tree (before inserting any runs):
        // ================= Tree of Losers (Pretty) =================
        // Root (Winner): EF
        //
        // └── [EF]
        //     ├── [EF] <-- runs (0, 1)
        //     └── [EF]
        //         ├── [LF] <-- run 4
        //         └── [EF] <-- runs (2, 3)
        //
        // After inserting values [20, 10, 30, 15, 25]:
        // ================= Tree of Losers (Pretty) =================
        // Root (Winner): (10, 1)
        //
        // └── [R3:15]
        //     ├── [R0:20] <-- runs (0, 1)
        //     └── [R4:25]
        //         ├── [LF] <-- run 4
        //         └── [R2:30] <-- runs (2, 3)

        // Base leaf nodes refer to the number of leaf nodes in the largest complete
        // binary subtree contained within our tree.
        let base_leaf_nodes = prev_power_of_two(input_leaf_nodes);

        // Calculate total number of nodes in the tree.
        // Formula: 2 * base_leaf_nodes + (input_leaf_nodes - base_leaf_nodes) * 2
        //
        // This creates a tree where:
        // - 2 * base_leaf_nodes form a complete binary tree
        // - Remaining 2 * (input_leaf_nodes - base_leaf_nodes) are additional nodes
        //
        // For num_runs=5 example:
        // - base_leaf_nodes = 2
        // - input_leaf_nodes = 3
        // - num_nodes = 2*2 + (3-2)*2 = 4 + 2 = 6
        let num_nodes = 2 * base_leaf_nodes + (input_leaf_nodes - base_leaf_nodes) * 2;
        let mut entries = Vec::with_capacity(num_nodes);

        // Fill the tree with early fences and late fences.
        for _ in 0..num_runs {
            entries.push(TolEntry::early_fence());
        }

        for _ in num_runs..num_nodes {
            entries.push(TolEntry::late_fence());
        }

        Self {
            curr: 0,
            input_leaf_start: num_nodes - input_leaf_nodes,
            entries,
        }
    }

    pub fn top_run_id(&mut self) -> Option<usize> {
        if self.entries[0].value.is_late_fence() {
            None
        } else if self.entries[0].value.is_early_fence() {
            let curr = self.curr;
            self.curr += 1;
            Some(curr)
        } else {
            Some(self.entries[0].run_id)
        }
    }

    pub fn node_index(&self, run_id: usize) -> usize {
        // Calculate the index of the node in the tree corresponding to the run_id.
        // The input_leaf_start is the starting index for the leaf nodes in the entries vector.
        self.input_leaf_start + run_id / 2
    }

    pub fn pop_and_insert(&mut self, run_id: usize, value: Option<T>) -> Option<T> {
        // which leaf node to insert the new entry.
        let mut index = self.node_index(run_id);

        // If value is None, insert a late fence. Otherwise, insert the value.
        // If the value is a late fence, make the run_id to be usize::MAX.
        let mut entry = value.map_or_else(TolEntry::late_fence, |value| TolEntry { value, run_id });

        while index > 0 {
            // The smaller goes to the next level, the larger stays in the current level.
            // If the entry in the location is smaller than the entry, swap them.
            // If the entry in the location is larger than the entry, do nothing.
            // If the entries are equal, we do not swap them, and
            // * The existing entry's OVC will be DUPLICATE
            // * The current entry's OVC will not change
            if self.entries[index]
                .value
                .compare_and_update(&mut entry.value)
                == Ordering::Less
            {
                std::mem::swap(&mut self.entries[index], &mut entry);
            }
            index /= 2;
        }
        // The top unconditionally gets popped
        std::mem::swap(&mut self.entries[0], &mut entry);

        if entry.value.is_early_fence() {
            None
        } else {
            Some(entry.value)
        }
    }
}

// This would only work if the tree stored ALL values upfront
impl<T: OVCTrait> Iterator for TreeOfLosersWithOVC<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(run_id) = self.top_run_id() {
            // Just pop without inserting anything new
            self.pop_and_insert(run_id, None)
        } else {
            None
        }
    }
}

impl<T: std::fmt::Debug + OVCTrait> TreeOfLosersWithOVC<T> {
    pub fn print(&self) {
        if self.entries.is_empty() {
            return;
        }
        println!("================= Tree of losers =================");
        println!("{:?} ", self.entries[0]);
        let mut index = 1;
        let mut level = 0;
        while index < self.entries.len() {
            for _ in 0..(1 << level) {
                if index >= self.entries.len() {
                    break;
                }
                print!("{:?} ", self.entries[index]);
                index += 1;
            }
            println!();
            level += 1;
        }
    }

    /// Pretty prints the tree structure with visual indentation and tree branches
    pub fn pretty_print(&self) {
        if self.entries.is_empty() {
            println!("Empty tree");
            return;
        }

        println!("================= Tree of Losers (Pretty) =================");
        println!("Root (Winner): {:?}", self.entries[0]);
        println!();

        // Calculate which runs map to which leaf nodes
        let num_runs = self
            .entries
            .iter()
            .filter(|e| !e.value.is_late_fence())
            .count();

        // Helper function to format entry with run mapping info
        let format_entry = |index: usize, entry: &TolEntry<T>| -> String {
            let base_str = if entry.value.is_early_fence() {
                "[EF]".to_string()
            } else if entry.value.is_late_fence() {
                "[LF]".to_string()
            } else {
                format!("[R{}:{:?}]", entry.run_id, entry.value)
            };

            // Add run mapping for leaf nodes
            if self.is_leaf_node(index) {
                let run_ids = self.get_runs_for_leaf(index, num_runs);
                if !run_ids.is_empty() {
                    let runs_str = if run_ids.len() == 1 {
                        format!("run {}", run_ids[0])
                    } else {
                        format!("runs ({}, {})", run_ids[0], run_ids[1])
                    };
                    format!("{} <-- {}", base_str, runs_str)
                } else {
                    base_str
                }
            } else {
                base_str
            }
        };

        // Print tree using recursive approach
        self.print_subtree(1, "", true, &format_entry);
    }

    /// Check if a node index represents a leaf node
    fn is_leaf_node(&self, index: usize) -> bool {
        let left_child = index * 2;
        left_child >= self.entries.len()
    }

    /// Get the run IDs that map to a specific leaf node
    fn get_runs_for_leaf(&self, leaf_index: usize, num_runs: usize) -> Vec<usize> {
        let mut runs = Vec::new();

        // Reverse the node_index calculation to find which runs map here
        //     node_index = input_leaf_start + run_id / 2
        //     run_id / 2 = node_index - input_leaf_start
        let offset = leaf_index as isize - self.input_leaf_start as isize;

        if offset >= 0 {
            let offset = offset as usize;
            // Each leaf position can handle 2 runs
            let base_run = offset * 2;
            if base_run < num_runs {
                runs.push(base_run);
            }
            if base_run + 1 < num_runs {
                runs.push(base_run + 1);
            }
        }

        runs
    }

    /// Helper method to recursively print subtree
    fn print_subtree<F>(&self, index: usize, prefix: &str, is_last: bool, format_fn: &F)
    where
        F: Fn(usize, &TolEntry<T>) -> String,
    {
        if index >= self.entries.len() {
            return;
        }

        // Print current node
        let connector = if is_last { "└── " } else { "├── " };
        println!(
            "{}{}{}",
            prefix,
            connector,
            format_fn(index, &self.entries[index])
        );

        // Prepare prefix for children
        let child_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

        // Print children (right child first)
        let left_child = index * 2;
        let right_child = index * 2 + 1;

        if right_child < self.entries.len() {
            self.print_subtree(
                right_child,
                &child_prefix,
                left_child >= self.entries.len(),
                format_fn,
            );
        }

        if left_child < self.entries.len() {
            self.print_subtree(left_child, &child_prefix, true, format_fn);
        }
    }
}

pub fn merge_runs_with_tree_of_losers_with_ovc<T: std::fmt::Debug + OVCTrait>(
    mut runs: Vec<Box<impl Iterator<Item = T>>>,
) -> Vec<T> {
    // Create a tree of losers.
    let mut tree = TreeOfLosersWithOVC::new(runs.len());

    // Output
    let mut output = Vec::new();

    // Fill the tree with the first entry of each run.
    // If the top entry is a late fence, it will return None.
    while let Some(run_id) = tree.top_run_id() {
        // println!("\nTop run ID: {}", run_id);
        // tree.pretty_print();
        // Pop the top entry and insert the next entry from the run.
        // It the top entry is a early fence, it will return None.
        if let Some(val) = tree.pop_and_insert(run_id, runs[run_id].next()) {
            output.push(val);
        }
    }

    output
}

pub fn sort_with_tree_of_losers_with_ovc64(mut run: Vec<Vec<u8>>) -> Vec<OVCEntry64> {
    let num_runs = run.len();
    let mut tree = TreeOfLosersWithOVC::new(num_runs);

    // Output
    let mut output = Vec::new();

    while let Some(run_id) = tree.top_run_id() {
        // println!("\nTop run ID: {}", run_id);
        // tree.pretty_print();
        if let Some(val) = tree.pop_and_insert(run_id, run.pop().map(|r| r.into())) {
            output.push(val);
        }
    }

    output
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ovc::{
        offset_value_coding_u64::{encode_run_with_ovc64, encode_runs_with_ovc64},
        utils::{
            generate_random_string_array, generate_runs, generate_string_runs, generate_vec_runs,
        },
    };

    #[test]
    fn test_merge_with_tree_of_losers_with_ovc() {
        // let runs = vec![
        //     vec![vec![1, 2, 3], vec![3, 4, 5], vec![5, 6, 7]],
        //     vec![vec![2, 3, 4], vec![4, 5, 6], vec![6, 7, 8]],
        // ];
        let num_runs = 10;
        let num_entries_per_run = 10;
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
        let output = sort_with_tree_of_losers_with_ovc64(normalized_run);

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
}
