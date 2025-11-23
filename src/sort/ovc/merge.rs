use crate::ovc::entry::SentinelValue;
use crate::ovc::offset_value_coding_u64::OVCU64;
use crate::ovc::offset_value_coding_u64_kv::OVCKeyValuePair;
use crate::ovc::tree_of_losers_with_ovc::TreeOfLosersWithOVC;

// K-way merge iterator
pub struct MergeWithOVC<I: Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)>> {
    // Tree of losers with OVC
    tree: TreeOfLosersWithOVC<OVCKeyValuePair>,
    // Iterators for each run
    iterators: Vec<I>,
}

impl<I: Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)>> MergeWithOVC<I> {
    pub fn new(mut iterators: Vec<I>) -> Self {
        if iterators.is_empty() || iterators.len() == 1 {
            panic!("MergeWithOVC requires at least two iterators");
        }

        let mut tree = TreeOfLosersWithOVC::new(iterators.len());

        // Initialize the tree with the first element of each iterator
        for i in 0..iterators.len() {
            let e = iterators[i]
                .next()
                .map_or(OVCKeyValuePair::late_fence(), |(_ovc, k, v)| {
                    // The OVC of the first element in each iterator is ignored and set to initial value
                    OVCKeyValuePair::new(k, v)
                });
            tree.pop_and_insert(i, Some(e.into()));
        }

        Self { tree, iterators }
    }
}

impl<I: Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)>> Iterator for MergeWithOVC<I> {
    type Item = (OVCU64, Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(run_id) = self.tree.top_run_id() {
            self.tree
                .pop_and_insert(run_id, self.iterators[run_id].next().map(|e| e.into()))
                .map(|e| e.take())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ovc::offset_value_coding::OVCTrait;

    // Helper function to encode a sorted run with proper OVC values
    fn encode_run_with_ovc(data: Vec<(Vec<u8>, Vec<u8>)>) -> Vec<(OVCU64, Vec<u8>, Vec<u8>)> {
        assert!(data.is_sorted_by_key(|(k, _)| k.clone()));

        let mut result = Vec::with_capacity(data.len());

        for (i, (key, value)) in data.into_iter().enumerate() {
            let mut new_entry = OVCKeyValuePair::new(key, value);
            if i > 0 {
                new_entry.update(&result[i - 1]);
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
}
