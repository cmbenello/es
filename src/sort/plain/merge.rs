use std::cmp::Ordering;
use std::collections::BinaryHeap;

// K-way merge iterator
pub struct MergeIterator<I: Iterator<Item = (Vec<u8>, Vec<u8>)>> {
    // Min heap of (key, value, source_index)
    heap: BinaryHeap<HeapEntry>,
    // Iterators for each run
    iterators: Vec<I>,
}

struct HeapEntry {
    key: Vec<u8>,
    value: Vec<u8>,
    source: usize,
}

// Implement reverse ordering for min-heap
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap
        other
            .key
            .cmp(&self.key)
            .then_with(|| other.source.cmp(&self.source))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for HeapEntry {}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.source == other.source
    }
}

impl<I: Iterator<Item = (Vec<u8>, Vec<u8>)>> MergeIterator<I> {
    pub fn new(mut iterators: Vec<I>) -> Self {
        let mut heap = BinaryHeap::new();

        // Initialize heap with first element from each iterator
        for (i, iter) in iterators.iter_mut().enumerate() {
            if let Some((key, value)) = iter.next() {
                heap.push(HeapEntry {
                    key,
                    value,
                    source: i,
                });
            }
        }

        Self { heap, iterators }
    }
}

impl<I: Iterator<Item = (Vec<u8>, Vec<u8>)>> Iterator for MergeIterator<I> {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        // Get minimum element
        let entry = self.heap.pop()?;

        // Try to get next element from the same source
        if let Some((key, value)) = self.iterators[entry.source].next() {
            self.heap.push(HeapEntry {
                key,
                value,
                source: entry.source,
            });
        }

        Some((entry.key, entry.value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order_preserving_encoding::*;

    #[test]
    fn test_merge_empty_iterators() {
        let iterators: Vec<std::vec::IntoIter<(Vec<u8>, Vec<u8>)>> = vec![];
        let mut merger = MergeIterator::new(iterators);
        assert_eq!(merger.next(), None);
    }

    #[test]
    fn test_merge_single_iterator() {
        let data = vec![
            (vec![1, 2, 3], vec![10, 20, 30]),
            (vec![4, 5, 6], vec![40, 50, 60]),
            (vec![7, 8, 9], vec![70, 80, 90]),
        ];
        let iterators = vec![data.into_iter()];
        let mut merger = MergeIterator::new(iterators);

        assert_eq!(merger.next(), Some((vec![1, 2, 3], vec![10, 20, 30])));
        assert_eq!(merger.next(), Some((vec![4, 5, 6], vec![40, 50, 60])));
        assert_eq!(merger.next(), Some((vec![7, 8, 9], vec![70, 80, 90])));
        assert_eq!(merger.next(), None);
    }

    #[test]
    fn test_merge_two_sorted_iterators() {
        let data1 = vec![
            (vec![1], vec![10]),
            (vec![3], vec![30]),
            (vec![5], vec![50]),
        ];
        let data2 = vec![
            (vec![2], vec![20]),
            (vec![4], vec![40]),
            (vec![6], vec![60]),
        ];

        let iterators = vec![data1.into_iter(), data2.into_iter()];
        let merger = MergeIterator::new(iterators);

        let results: Vec<_> = merger.collect();
        assert_eq!(results.len(), 6);

        // Verify sorted order
        for i in 0..results.len() {
            assert_eq!(results[i].0, vec![i as u8 + 1]);
            assert_eq!(results[i].1, vec![(i as u8 + 1) * 10]);
        }
    }

    #[test]
    fn test_merge_multiple_iterators_with_gaps() {
        let data1 = vec![(vec![1], vec![1]), (vec![7], vec![7])];
        let data2 = vec![(vec![3], vec![3]), (vec![8], vec![8])];
        let data3 = vec![(vec![2], vec![2]), (vec![5], vec![5]), (vec![9], vec![9])];
        let data4 = vec![(vec![4], vec![4]), (vec![6], vec![6])];

        let iterators = vec![
            data1.into_iter(),
            data2.into_iter(),
            data3.into_iter(),
            data4.into_iter(),
        ];
        let merger = MergeIterator::new(iterators);

        let results: Vec<_> = merger.collect();
        assert_eq!(results.len(), 9);

        // Verify sorted order
        for i in 0..results.len() {
            assert_eq!(results[i].0, vec![i as u8 + 1]);
            assert_eq!(results[i].1, vec![i as u8 + 1]);
        }
    }

    #[test]
    fn test_merge_with_duplicate_keys() {
        let data1 = vec![
            (vec![1], vec![10]),
            (vec![2], vec![20]),
            (vec![2], vec![21]),
            (vec![4], vec![40]),
        ];
        let data2 = vec![
            (vec![2], vec![22]),
            (vec![3], vec![30]),
            (vec![4], vec![41]),
        ];

        let iterators = vec![data1.into_iter(), data2.into_iter()];
        let merger = MergeIterator::new(iterators);

        let results: Vec<_> = merger.collect();
        assert_eq!(results.len(), 7);

        // Check that all values with key [2] are together
        let key2_values: Vec<_> = results
            .iter()
            .filter(|(k, _)| k == &vec![2])
            .map(|(_, v)| v.clone())
            .collect();
        assert_eq!(key2_values.len(), 3);
        assert!(key2_values.contains(&vec![20]));
        assert!(key2_values.contains(&vec![21]));
        assert!(key2_values.contains(&vec![22]));
    }

    #[test]
    fn test_merge_empty_and_non_empty_iterators() {
        let data1 = vec![(vec![1], vec![10]), (vec![3], vec![30])];
        let data2: Vec<(Vec<u8>, Vec<u8>)> = vec![];
        let data3 = vec![(vec![2], vec![20]), (vec![4], vec![40])];

        let iterators = vec![data1.into_iter(), data2.into_iter(), data3.into_iter()];
        let merger = MergeIterator::new(iterators);

        let results: Vec<_> = merger.collect();
        assert_eq!(results.len(), 4);
        assert_eq!(results[0], (vec![1], vec![10]));
        assert_eq!(results[1], (vec![2], vec![20]));
        assert_eq!(results[2], (vec![3], vec![30]));
        assert_eq!(results[3], (vec![4], vec![40]));
    }

    #[test]
    fn test_merge_with_order_preserving_encoded_integers() {
        // Create sorted data with order-preserving encoding
        let data1 = vec![
            (i32_to_order_preserving_bytes(-100).to_vec(), vec![1]),
            (i32_to_order_preserving_bytes(0).to_vec(), vec![2]),
            (i32_to_order_preserving_bytes(50).to_vec(), vec![3]),
        ];
        let data2 = vec![
            (i32_to_order_preserving_bytes(-50).to_vec(), vec![4]),
            (i32_to_order_preserving_bytes(25).to_vec(), vec![5]),
            (i32_to_order_preserving_bytes(100).to_vec(), vec![6]),
        ];

        let iterators = vec![data1.into_iter(), data2.into_iter()];
        let merger = MergeIterator::new(iterators);

        let results: Vec<_> = merger.collect();
        assert_eq!(results.len(), 6);

        // Decode keys to verify order
        let decoded_keys: Vec<i32> = results
            .iter()
            .map(|(k, _)| decode_i32(k).unwrap())
            .collect();

        assert_eq!(decoded_keys, vec![-100, -50, 0, 25, 50, 100]);
    }

    #[test]
    fn test_merge_with_string_keys() {
        let data1 = vec![
            (b"apple".to_vec(), vec![1]),
            (b"cherry".to_vec(), vec![3]),
            (b"elderberry".to_vec(), vec![5]),
        ];
        let data2 = vec![
            (b"banana".to_vec(), vec![2]),
            (b"date".to_vec(), vec![4]),
            (b"fig".to_vec(), vec![6]),
        ];

        let iterators = vec![data1.into_iter(), data2.into_iter()];
        let merger = MergeIterator::new(iterators);

        let results: Vec<_> = merger.collect();
        assert_eq!(results.len(), 6);

        let keys: Vec<String> = results
            .iter()
            .map(|(k, _)| String::from_utf8(k.clone()).unwrap())
            .collect();

        assert_eq!(
            keys,
            vec!["apple", "banana", "cherry", "date", "elderberry", "fig"]
        );
    }

    #[test]
    fn test_merge_large_number_of_iterators() {
        // Create 10 iterators, each with 5 elements
        let mut all_iterators = Vec::new();

        for i in 0..10 {
            let mut data = Vec::new();
            for j in 0..5 {
                let key_val = (i * 5 + j) as u8;
                let key = vec![key_val];
                let value = vec![key_val.saturating_mul(10)];
                data.push((key, value));
            }
            all_iterators.push(data.into_iter());
        }

        let merger = MergeIterator::new(all_iterators);
        let results: Vec<_> = merger.collect();

        assert_eq!(results.len(), 50);

        // Verify all elements are in order
        for i in 0..50 {
            assert_eq!(results[i].0, vec![i as u8]);
            assert_eq!(results[i].1, vec![(i as u8).saturating_mul(10)]);
        }
    }

    #[test]
    fn test_merge_preserves_stable_ordering() {
        // When keys are equal, elements from earlier iterators should come first
        let data1 = vec![
            (vec![1], vec![10]),
            (vec![2], vec![20]),
            (vec![3], vec![30]),
        ];
        let data2 = vec![
            (vec![1], vec![11]),
            (vec![2], vec![21]),
            (vec![3], vec![31]),
        ];
        let data3 = vec![
            (vec![1], vec![12]),
            (vec![2], vec![22]),
            (vec![3], vec![32]),
        ];

        let iterators = vec![data1.into_iter(), data2.into_iter(), data3.into_iter()];
        let merger = MergeIterator::new(iterators);

        let results: Vec<_> = merger.collect();
        assert_eq!(results.len(), 9);

        // For each key, verify the values come in source order
        let key1_values: Vec<_> = results
            .iter()
            .filter(|(k, _)| k == &vec![1])
            .map(|(_, v)| v.clone())
            .collect();
        assert_eq!(key1_values, vec![vec![10], vec![11], vec![12]]);

        let key2_values: Vec<_> = results
            .iter()
            .filter(|(k, _)| k == &vec![2])
            .map(|(_, v)| v.clone())
            .collect();
        assert_eq!(key2_values, vec![vec![20], vec![21], vec![22]]);
    }

    #[test]
    fn test_heap_entry_ordering() {
        // Test the HeapEntry ordering directly
        // Note: HeapEntry implements reverse ordering for min-heap
        let entry1 = HeapEntry {
            key: vec![1, 2, 3],
            value: vec![],
            source: 0,
        };
        let entry2 = HeapEntry {
            key: vec![1, 2, 4],
            value: vec![],
            source: 1,
        };
        let entry3 = HeapEntry {
            key: vec![1, 2, 3],
            value: vec![],
            source: 1,
        };

        // Due to reverse ordering for min-heap:
        // Smaller keys should be "greater" in heap ordering
        assert!(entry1 > entry2); // [1,2,3] > [1,2,4] in heap (reversed)
        assert!(entry1 > entry3); // source 0 > source 1 when keys are equal (reversed)
        assert!(entry3 > entry2); // [1,2,3] > [1,2,4] in heap (reversed)
    }

    #[test]
    fn test_merge_with_composite_keys() {
        // Test with multi-column keys (like in TPC-H lineitem)
        let data1 = vec![
            // (orderkey, linenumber) as concatenated bytes
            (vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], vec![10]), // orderkey=1, linenumber=1
            (vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3], vec![12]), // orderkey=1, linenumber=3
            (vec![0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2], vec![22]), // orderkey=2, linenumber=2
        ];
        let data2 = vec![
            (vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2], vec![11]), // orderkey=1, linenumber=2
            (vec![0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1], vec![21]), // orderkey=2, linenumber=1
            (vec![0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1], vec![31]), // orderkey=3, linenumber=1
        ];

        let iterators = vec![data1.into_iter(), data2.into_iter()];
        let merger = MergeIterator::new(iterators);

        let results: Vec<_> = merger.collect();
        assert_eq!(results.len(), 6);

        // Verify the merge maintained proper ordering
        // Expected order: (1,1), (1,2), (1,3), (2,1), (2,2), (3,1)
        assert_eq!(results[0].1, vec![10]); // (1,1)
        assert_eq!(results[1].1, vec![11]); // (1,2)
        assert_eq!(results[2].1, vec![12]); // (1,3)
        assert_eq!(results[3].1, vec![21]); // (2,1)
        assert_eq!(results[4].1, vec![22]); // (2,2)
        assert_eq!(results[5].1, vec![31]); // (3,1)
    }

    #[test]
    fn test_merge_with_float_encoded_keys() {
        // Test with order-preserving encoded floats
        let data1 = vec![
            (f64_to_order_preserving_bytes(-10.5).to_vec(), vec![1]),
            (f64_to_order_preserving_bytes(0.0).to_vec(), vec![3]),
            (f64_to_order_preserving_bytes(15.75).to_vec(), vec![5]),
        ];
        let data2 = vec![
            (f64_to_order_preserving_bytes(-5.25).to_vec(), vec![2]),
            (f64_to_order_preserving_bytes(10.0).to_vec(), vec![4]),
            (f64_to_order_preserving_bytes(20.5).to_vec(), vec![6]),
        ];

        let iterators = vec![data1.into_iter(), data2.into_iter()];
        let merger = MergeIterator::new(iterators);

        let results: Vec<_> = merger.collect();
        assert_eq!(results.len(), 6);

        // Verify values come out in correct order
        assert_eq!(results[0].1, vec![1]); // -10.5
        assert_eq!(results[1].1, vec![2]); // -5.25
        assert_eq!(results[2].1, vec![3]); // 0.0
        assert_eq!(results[3].1, vec![4]); // 10.0
        assert_eq!(results[4].1, vec![5]); // 15.75
        assert_eq!(results[5].1, vec![6]); // 20.5
    }
}
