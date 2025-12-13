use crate::ovc::offset_value_coding::{OVC64Trait, OVCKeyValue, OVCU64, SentinelValue};
use crate::ovc::tree_of_losers_ovc::LoserTreeOVC;

// K-way merge iterator
pub struct MergeWithOVC<I: Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)>> {
    // Tree of losers with OVC
    tree: LoserTreeOVC<OVCKeyValue>,
    // Iterators for each run
    iterators: Vec<I>,
}

impl<I: Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)>> MergeWithOVC<I> {
    pub fn new(mut iterators: Vec<I>) -> Self {
        if iterators.is_empty() || iterators.len() == 1 {
            panic!("MergeWithOVC requires at least two iterators");
        }

        let mut initial = Vec::with_capacity(iterators.len());
        for iter in iterators.iter_mut() {
            if let Some(val) = iter.next().map(|e| e.into()) {
                initial.push(val);
            } else {
                initial.push(OVCKeyValue::late_fence());
            }
        }

        let tree = LoserTreeOVC::new(initial);

        Self { tree, iterators }
    }
}

impl<I: Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)>> Iterator for MergeWithOVC<I> {
    type Item = (OVCU64, Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        let (_, source_idx) = self.tree.peek()?;
        if let Some(mut next_item) = self.iterators[source_idx]
            .next()
            .map(|e: Self::Item| OVCKeyValue::from(e))
        {
            if next_item.ovc().is_duplicate_value() {
                // For duplicate OVC, we swap the OVC value in the top of the tree
                // and return the item directly
                *next_item.ovc_mut() = self.tree.replace_top_ovc(*next_item.ovc());
                return Some(next_item.take());
            }
            let winner = self.tree.push(next_item);
            return Some(winner.take());
        }
        self.tree
            .mark_current_exhausted()
            .map(|winner| winner.take())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ovc::offset_value_coding::OVC64Trait;

    // Helper function to encode a sorted run with proper OVC values
    fn encode_run_with_ovc(data: Vec<(Vec<u8>, Vec<u8>)>) -> Vec<(OVCU64, Vec<u8>, Vec<u8>)> {
        assert!(data.is_sorted_by_key(|(k, _)| k.clone()));

        let mut result = Vec::with_capacity(data.len());

        for (i, (key, value)) in data.into_iter().enumerate() {
            let mut new_entry = OVCKeyValue::new(key, value);
            if i > 0 {
                new_entry.derive_ovc_from(&result[i - 1]);
            }
            result.push(new_entry);
        }

        result.into_iter().map(|e| e.take()).collect()
    }

    #[test]
    #[should_panic(expected = "MergeWithOVC requires at least two iterators")]
    fn test_merge_empty_iterators() {
        let iterators: Vec<std::vec::IntoIter<(OVCU64, Vec<u8>, Vec<u8>)>> = vec![];
        let mut merger = MergeWithOVC::new(iterators);

        assert!(merger.next().is_none());
    }

    #[test]
    #[should_panic(expected = "MergeWithOVC requires at least two iterators")]
    fn test_merge_single_iterator() {
        let data = encode_run_with_ovc(vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"b".to_vec(), b"2".to_vec()),
            (b"c".to_vec(), b"3".to_vec()),
        ]);

        let iterators = vec![data.into_iter()];
        let merger = MergeWithOVC::new(iterators);

        let results: Vec<_> = merger.collect();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].1, b"a");
        assert_eq!(results[1].1, b"b");
        assert_eq!(results[2].1, b"c");
    }

    #[test]
    fn test_merge_two_sorted_iterators() {
        let data1 = encode_run_with_ovc(vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"c".to_vec(), b"3".to_vec()),
            (b"e".to_vec(), b"5".to_vec()),
        ]);

        let data2 = encode_run_with_ovc(vec![
            (b"b".to_vec(), b"2".to_vec()),
            (b"d".to_vec(), b"4".to_vec()),
            (b"f".to_vec(), b"6".to_vec()),
        ]);

        let iterators = vec![data1.into_iter(), data2.into_iter()];
        let merger = MergeWithOVC::new(iterators);

        let results: Vec<_> = merger.collect();

        assert_eq!(results.len(), 6);
        assert_eq!(results[0].1, b"a");
        assert_eq!(results[1].1, b"b");
        assert_eq!(results[2].1, b"c");
        assert_eq!(results[3].1, b"d");
        assert_eq!(results[4].1, b"e");
        assert_eq!(results[5].1, b"f");
    }

    #[test]
    fn test_merge_multiple_iterators() {
        let data1 = encode_run_with_ovc(vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"d".to_vec(), b"4".to_vec()),
        ]);

        let data2 = encode_run_with_ovc(vec![
            (b"b".to_vec(), b"2".to_vec()),
            (b"e".to_vec(), b"5".to_vec()),
        ]);

        let data3 = encode_run_with_ovc(vec![
            (b"c".to_vec(), b"3".to_vec()),
            (b"f".to_vec(), b"6".to_vec()),
        ]);

        let iterators = vec![data1.into_iter(), data2.into_iter(), data3.into_iter()];
        let merger = MergeWithOVC::new(iterators);

        let results: Vec<_> = merger.collect();

        assert_eq!(results.len(), 6);
        for i in 0..6 {
            let expected_key = vec![b'a' + i as u8];
            assert_eq!(results[i].1, expected_key);
        }
    }

    #[test]
    fn test_merge_with_duplicates() {
        let data1 = encode_run_with_ovc(vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"b".to_vec(), b"3".to_vec()),
            (b"c".to_vec(), b"5".to_vec()),
        ]);

        let data2 = encode_run_with_ovc(vec![
            (b"a".to_vec(), b"2".to_vec()),
            (b"b".to_vec(), b"4".to_vec()),
            (b"d".to_vec(), b"6".to_vec()),
        ]);

        let iterators = vec![data1.into_iter(), data2.into_iter()];
        let merger = MergeWithOVC::new(iterators);

        let results: Vec<_> = merger.collect();

        assert_eq!(results.len(), 6);
        // Verify all 'a' keys come before 'b' keys, etc.
        assert_eq!(results[0].1, b"a");
        assert_eq!(results[1].1, b"a");
        assert_eq!(results[2].1, b"b");
        assert_eq!(results[3].1, b"b");
        assert_eq!(results[4].1, b"c");
        assert_eq!(results[5].1, b"d");
    }

    #[test]
    fn test_merge_uneven_iterators() {
        let data1 = encode_run_with_ovc(vec![(b"a".to_vec(), b"1".to_vec())]);

        let data2 = encode_run_with_ovc(vec![
            (b"b".to_vec(), b"2".to_vec()),
            (b"c".to_vec(), b"3".to_vec()),
            (b"d".to_vec(), b"4".to_vec()),
            (b"e".to_vec(), b"5".to_vec()),
        ]);

        let data3 = encode_run_with_ovc(vec![
            (b"f".to_vec(), b"6".to_vec()),
            (b"g".to_vec(), b"7".to_vec()),
        ]);

        let iterators = vec![data1.into_iter(), data2.into_iter(), data3.into_iter()];
        let merger = MergeWithOVC::new(iterators);

        let results: Vec<_> = merger.collect();

        assert_eq!(results.len(), 7);
        for i in 0..7 {
            let expected_key = vec![b'a' + i as u8];
            assert_eq!(results[i].1, expected_key);
        }
    }

    #[test]
    fn test_ovc_values_preserved() {
        let data1 = encode_run_with_ovc(vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"c".to_vec(), b"3".to_vec()),
        ]);

        let data2 = encode_run_with_ovc(vec![
            (b"b".to_vec(), b"2".to_vec()),
            (b"d".to_vec(), b"4".to_vec()),
        ]);

        let iterators = vec![data1.into_iter(), data2.into_iter()];
        let merger = MergeWithOVC::new(iterators);

        let results: Vec<_> = merger.collect();

        assert_eq!(results.len(), 4);
        // OVC values are based on the actual key content and position
        // So we just verify that sorting is correct and values exist
        assert_eq!(results[0].1, b"a");
        assert_eq!(results[1].1, b"b");
        assert_eq!(results[2].1, b"c");
        assert_eq!(results[3].1, b"d");
    }

    #[test]
    fn test_merge_large_values() {
        let large_value = vec![b'x'; 1000];

        let data1 = encode_run_with_ovc(vec![
            (b"a".to_vec(), large_value.clone()),
            (b"c".to_vec(), large_value.clone()),
        ]);

        let data2 = encode_run_with_ovc(vec![
            (b"b".to_vec(), large_value.clone()),
            (b"d".to_vec(), large_value.clone()),
        ]);

        let iterators = vec![data1.into_iter(), data2.into_iter()];
        let merger = MergeWithOVC::new(iterators);

        let results: Vec<_> = merger.collect();

        assert_eq!(results.len(), 4);
        // Verify sorting and that large values are preserved
        assert_eq!(results[0].1, b"a");
        assert_eq!(results[0].2.len(), 1000);
        assert_eq!(results[1].1, b"b");
        assert_eq!(results[1].2.len(), 1000);
    }

    #[test]
    fn test_merge_empty_and_nonempty() {
        let data1: Vec<(OVCU64, Vec<u8>, Vec<u8>)> = vec![];

        let data2 = encode_run_with_ovc(vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"b".to_vec(), b"2".to_vec()),
        ]);

        let data3: Vec<(OVCU64, Vec<u8>, Vec<u8>)> = vec![];

        let iterators = vec![data1.into_iter(), data2.into_iter(), data3.into_iter()];
        let merger = MergeWithOVC::new(iterators);

        let results: Vec<_> = merger.collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1, b"a");
        assert_eq!(results[1].1, b"b");
    }

    #[test]
    fn test_ovc_after_merge() {
        let data = vec![
            vec![vec![1, 1, 1], vec![6, 6, 4], vec![8, 8, 7]],
            vec![vec![1, 1, 2], vec![6, 6, 5], vec![8, 8, 8]],
            vec![vec![1, 1, 3], vec![6, 6, 6], vec![8, 8, 9]],
        ];

        let mut flatten = data.clone().into_iter().flatten().collect::<Vec<_>>();
        flatten.sort();
        let encoded =
            encode_run_with_ovc(flatten.into_iter().map(|k| (k, b"val".to_vec())).collect());

        let iterators = (0..data.len())
            .map(|i| {
                let keys = data[i].clone();
                let key_value_pairs: Vec<(Vec<u8>, Vec<u8>)> =
                    keys.into_iter().map(|k| (k, b"val".to_vec())).collect();
                encode_run_with_ovc(key_value_pairs).into_iter()
            })
            .collect::<Vec<_>>();

        let merger = MergeWithOVC::new(iterators.into_iter().map(|v| v.into_iter()).collect());

        let results: Vec<_> = merger.collect();

        assert_eq!(results.len(), encoded.len());
        for i in 0..results.len() {
            assert_eq!(results[i], encoded[i]);
        }
    }

    #[test]
    fn test_merge_with_consecutive_duplicates_in_runs() {
        // This test specifically targets the duplicate OVC handling in the merge
        // When a run contains consecutive duplicate keys, their OVC values are marked
        // with DuplicateValue flag, and the merge must handle this correctly

        // Run 1: has consecutive duplicates "a", "a", "a" and "c", "c"
        let data1 = encode_run_with_ovc(vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"a".to_vec(), b"2".to_vec()), // duplicate key
            (b"a".to_vec(), b"3".to_vec()), // duplicate key
            (b"c".to_vec(), b"5".to_vec()),
            (b"c".to_vec(), b"6".to_vec()), // duplicate key
        ]);

        // Run 2: has consecutive duplicates "b", "b" and "d", "d", "d"
        let data2 = encode_run_with_ovc(vec![
            (b"b".to_vec(), b"4".to_vec()),
            (b"b".to_vec(), b"4.5".to_vec()), // duplicate key
            (b"d".to_vec(), b"7".to_vec()),
            (b"d".to_vec(), b"8".to_vec()), // duplicate key
            (b"d".to_vec(), b"9".to_vec()), // duplicate key
        ]);

        // Verify that the runs have duplicate OVC values
        assert!(data1[1].0.is_duplicate_value()); // second "a"
        assert!(data1[2].0.is_duplicate_value()); // third "a"
        assert!(data1[4].0.is_duplicate_value()); // second "c"
        assert!(data2[1].0.is_duplicate_value()); // second "b"
        assert!(data2[3].0.is_duplicate_value()); // second "d"
        assert!(data2[4].0.is_duplicate_value()); // third "d"

        let iterators = vec![data1.into_iter(), data2.into_iter()];
        let merger = MergeWithOVC::new(iterators);

        let results: Vec<_> = merger.collect();

        // Should have all 10 items
        assert_eq!(results.len(), 10);

        // Verify correct ordering: keys should be sorted
        // Count occurrences of each key
        let a_count = results.iter().filter(|(_, k, _)| k == b"a").count();
        let b_count = results.iter().filter(|(_, k, _)| k == b"b").count();
        let c_count = results.iter().filter(|(_, k, _)| k == b"c").count();
        let d_count = results.iter().filter(|(_, k, _)| k == b"d").count();

        assert_eq!(a_count, 3);
        assert_eq!(b_count, 2);
        assert_eq!(c_count, 2);
        assert_eq!(d_count, 3);

        // Verify that the result is sorted (all "a"s before "b"s, etc.)
        for i in 0..results.len() - 1 {
            assert!(
                results[i].1 <= results[i + 1].1,
                "Results not sorted at index {}: {:?} > {:?}",
                i,
                results[i].1,
                results[i + 1].1
            );
        }

        // Reconstruct what the merged result should look like and verify OVC consistency
        let mut all_items: Vec<(Vec<u8>, Vec<u8>)> = vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"a".to_vec(), b"2".to_vec()),
            (b"a".to_vec(), b"3".to_vec()),
            (b"b".to_vec(), b"4".to_vec()),
            (b"b".to_vec(), b"4.5".to_vec()),
            (b"c".to_vec(), b"5".to_vec()),
            (b"c".to_vec(), b"6".to_vec()),
            (b"d".to_vec(), b"7".to_vec()),
            (b"d".to_vec(), b"8".to_vec()),
            (b"d".to_vec(), b"9".to_vec()),
        ];

        // Sort the items by key (and value for stability)
        all_items.sort();
        let expected = encode_run_with_ovc(all_items);

        // The OVC values should be consistent with a freshly encoded run
        for i in 0..results.len() {
            assert_eq!(
                results[i].0, expected[i].0,
                "OVC mismatch at index {}: got {:?}, expected {:?}",
                i, results[i].0, expected[i].0
            );
        }
    }

    #[test]
    fn test_merge_multiple_runs_with_duplicates() {
        // Test merging 3+ runs where each has consecutive duplicates
        let data1 = encode_run_with_ovc(vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"a".to_vec(), b"2".to_vec()),
            (b"d".to_vec(), b"4".to_vec()),
        ]);

        let data2 = encode_run_with_ovc(vec![
            (b"b".to_vec(), b"3".to_vec()),
            (b"b".to_vec(), b"3.5".to_vec()),
            (b"e".to_vec(), b"5".to_vec()),
        ]);

        let data3 = encode_run_with_ovc(vec![
            (b"c".to_vec(), b"3.7".to_vec()),
            (b"c".to_vec(), b"3.8".to_vec()),
            (b"c".to_vec(), b"3.9".to_vec()),
        ]);

        // Verify duplicate OVC values exist in input
        assert!(data1[1].0.is_duplicate_value());
        assert!(data2[1].0.is_duplicate_value());
        assert!(data3[1].0.is_duplicate_value());
        assert!(data3[2].0.is_duplicate_value());

        let iterators = vec![data1.into_iter(), data2.into_iter(), data3.into_iter()];
        let merger = MergeWithOVC::new(iterators);

        let results: Vec<_> = merger.collect();

        assert_eq!(results.len(), 9);

        // Verify the output is sorted
        for i in 0..results.len() - 1 {
            assert!(
                results[i].1 <= results[i + 1].1,
                "Results not sorted at index {}: {:?} > {:?}",
                i,
                results[i].1,
                results[i + 1].1
            );
        }

        // Verify correct key counts
        assert_eq!(results[0].1, b"a");
        assert_eq!(results[1].1, b"a");
        assert_eq!(results[2].1, b"b");
        assert_eq!(results[3].1, b"b");
        assert_eq!(results[4].1, b"c");
        assert_eq!(results[5].1, b"c");
        assert_eq!(results[6].1, b"c");
        assert_eq!(results[7].1, b"d");
        assert_eq!(results[8].1, b"e");

        // Verify OVC consistency by sorting all items and comparing
        let mut all_items = vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"a".to_vec(), b"2".to_vec()),
            (b"b".to_vec(), b"3".to_vec()),
            (b"b".to_vec(), b"3.5".to_vec()),
            (b"c".to_vec(), b"3.7".to_vec()),
            (b"c".to_vec(), b"3.8".to_vec()),
            (b"c".to_vec(), b"3.9".to_vec()),
            (b"d".to_vec(), b"4".to_vec()),
            (b"e".to_vec(), b"5".to_vec()),
        ];
        all_items.sort();
        let expected = encode_run_with_ovc(all_items);

        for i in 0..results.len() {
            assert_eq!(
                results[i].0, expected[i].0,
                "OVC mismatch at index {}: got {:?}, expected {:?}",
                i, results[i].0, expected[i].0
            );
        }
    }
}
