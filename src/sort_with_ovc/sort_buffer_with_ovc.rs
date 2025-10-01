use crate::ovc::{
    offset_value_coding_u64::OVCU64, offset_value_coding_u64_kv::OVCKeyValuePair,
    tree_of_losers_with_ovc::TreeOfLosersWithOVC,
};

pub struct SortBufferOVC {
    data: Vec<(Vec<u8>, Vec<u8>)>,
    memory_used: usize,
    memory_limit: usize,
}

impl SortBufferOVC {
    pub fn new(memory_limit: usize) -> Self {
        Self {
            data: Vec::new(),
            memory_used: 0,
            memory_limit,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn entry_size(key: &[u8], value: &[u8]) -> usize {
        // key.len() + value.len() + std::mem::size_of::<u32>() * 2 + std::mem::size_of::<u64>() // key and value lengths and OVC size.
        8 + key.len() + value.len() // Ignore OVC size for simplicity
    }

    pub fn has_space(&self, key: &[u8], value: &[u8]) -> bool {
        let entry_size = Self::entry_size(key, value);
        self.memory_used + entry_size <= self.memory_limit
    }

    pub fn append(&mut self, key: Vec<u8>, value: Vec<u8>) -> bool {
        let entry_size = Self::entry_size(&key, &value);

        if self.memory_used + entry_size > self.memory_limit {
            return false;
        }

        self.data.push((key, value));
        self.memory_used += entry_size;
        true
    }

    pub fn sorted_iter(&mut self) -> Box<dyn Iterator<Item = (OVCU64, Vec<u8>, Vec<u8>)>> {
        if self.data.is_empty() {
            return Box::new(std::iter::empty());
        } else if self.data.len() == 1 {
            let (key, value) = self.data.remove(0);
            let ovc_entry = OVCKeyValuePair::new(key, value);
            return Box::new(std::iter::once(ovc_entry.take()));
        }

        let mut tree = TreeOfLosersWithOVC::<OVCKeyValuePair>::new(self.data.len());
        self.data
            .drain(..)
            .enumerate()
            .for_each(|(i, (key, value))| {
                let ovc_entry = OVCKeyValuePair::new(key, value);
                tree.pop_and_insert(i, Some(ovc_entry));
            });
        Box::new(tree.into_iter().map(|e| e.take()))
    }

    pub fn reset(&mut self) {
        self.data.clear();
        self.memory_used = 0;
    }
}

#[cfg(test)]
mod tests {
    use crate::ovc::{
        offset_value_coding_u64::encode_run_with_ovc64, utils::generate_random_array,
    };

    use super::*;

    #[test]
    fn test_new_buffer() {
        let buffer = SortBufferOVC::new(1024);
        assert!(buffer.is_empty());
        assert_eq!(buffer.memory_used, 0);
        assert_eq!(buffer.memory_limit, 1024);
    }

    #[test]
    fn test_append_and_has_space() {
        let mut buffer = SortBufferOVC::new(1024);

        let key1 = b"key1".to_vec();
        let value1 = b"value1".to_vec();

        // Should have space initially
        assert!(buffer.has_space(&key1, &value1));

        // Append should succeed
        assert!(buffer.append(key1.clone(), value1.clone()));
        assert!(!buffer.is_empty());

        // Memory should be tracked
        let expected_size = SortBufferOVC::entry_size(&key1, &value1);
        assert_eq!(buffer.memory_used, expected_size);
    }

    #[test]
    fn test_memory_limit() {
        // Create buffer with very small limit
        let mut buffer = SortBufferOVC::new(50);

        let small_key = b"k".to_vec();
        let small_value = b"v".to_vec();

        // Should be able to add at least one small entry
        assert!(buffer.append(small_key.clone(), small_value.clone()));

        // Large entry should not fit
        let large_key = vec![b'x'; 100];
        let large_value = vec![b'y'; 100];
        assert!(!buffer.has_space(&large_key, &large_value));
        assert!(!buffer.append(large_key, large_value));
    }

    #[test]
    fn test_sorted_iter() {
        let mut buffer = SortBufferOVC::new(1024);

        // Add entries in reverse order
        buffer.append(b"c".to_vec(), b"3".to_vec());
        buffer.append(b"a".to_vec(), b"1".to_vec());
        buffer.append(b"b".to_vec(), b"2".to_vec());

        // Get sorted iterator
        let results: Vec<_> = buffer.sorted_iter().collect();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].1, b"a".to_vec());
        assert_eq!(results[0].2, b"1".to_vec());
        assert_eq!(results[1].1, b"b".to_vec());
        assert_eq!(results[1].2, b"2".to_vec());
        assert_eq!(results[2].1, b"c".to_vec());
        assert_eq!(results[2].2, b"3".to_vec());

        // Buffer should be empty after sorted_iter (drain)
        buffer.reset();
        assert!(buffer.is_empty());
        assert_eq!(buffer.memory_used, 0);
    }

    #[test]
    fn test_reset() {
        let mut buffer = SortBufferOVC::new(1024);

        // Add some data
        buffer.append(b"key1".to_vec(), b"value1".to_vec());
        buffer.append(b"key2".to_vec(), b"value2".to_vec());

        assert!(!buffer.is_empty());
        assert!(buffer.memory_used > 0);

        // Reset
        buffer.reset();

        assert!(buffer.is_empty());
        assert_eq!(buffer.memory_used, 0);

        // Should be able to add data again
        assert!(buffer.append(b"new_key".to_vec(), b"new_value".to_vec()));
    }

    #[test]
    fn test_duplicate_keys() {
        let mut buffer = SortBufferOVC::new(1024);

        buffer.append(b"b".to_vec(), b"2".to_vec());
        buffer.append(b"a".to_vec(), b"1".to_vec());
        buffer.append(b"b".to_vec(), b"3".to_vec());
        buffer.append(b"a".to_vec(), b"4".to_vec());

        let results: Vec<_> = buffer.sorted_iter().collect();

        assert_eq!(results.len(), 4);
        // All 'a' keys should come first
        assert_eq!(results[0].1, b"a".to_vec());
        assert_eq!(results[1].1, b"a".to_vec());
        // Then 'b' keys
        assert_eq!(results[2].1, b"b".to_vec());
        assert_eq!(results[3].1, b"b".to_vec());
    }

    #[test]
    fn test_entry_size_calculation() {
        let key = b"test_key".to_vec();
        let value = b"test_value".to_vec();

        let size = SortBufferOVC::entry_size(&key, &value);

        // Size should include key + value + metadata
        let expected = key.len() + value.len() + std::mem::size_of::<u32>() * 2; // key and value lengths
        // + std::mem::size_of::<u64>(); // OVC size

        assert_eq!(size, expected);
    }

    #[test]
    fn test_large_dataset() {
        let mut buffer = SortBufferOVC::new(10240);

        // Add 100 entries
        for i in (0..100).rev() {
            let key = format!("key_{:03}", i);
            let value = format!("value_{}", i);
            buffer.append(key.into_bytes(), value.into_bytes());
        }

        let results: Vec<_> = buffer.sorted_iter().collect();

        assert_eq!(results.len(), 100);

        // Verify sorted order
        for i in 0..100 {
            let expected_key = format!("key_{:03}", i);
            assert_eq!(results[i].1, expected_key.as_bytes());
        }
    }

    #[test]
    fn test_empty_buffer_sorted_iter() {
        let mut buffer = SortBufferOVC::new(1024);

        let results: Vec<_> = buffer.sorted_iter().collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_single_entry() {
        let mut buffer = SortBufferOVC::new(1024);

        buffer.append(b"only".to_vec(), b"one".to_vec());

        let results: Vec<_> = buffer.sorted_iter().collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, b"only".to_vec());
        assert_eq!(results[0].2, b"one".to_vec());
    }

    #[test]
    fn test_ovc_sorted_iter() {
        let num_entries = 100;
        let key_size = 10;
        let random_data = (0..num_entries)
            .into_iter()
            .map(|_| generate_random_array::<u8>(key_size))
            .collect::<Vec<_>>();
        let mut sorted_data = random_data.clone();
        sorted_data.sort();
        let encoded = encode_run_with_ovc64(&sorted_data);

        let mut buffer = SortBufferOVC::new(1024 * 1024); // Enough memory for all entries
        for key in random_data {
            buffer.append(key, b"value".to_vec());
        }

        let results: Vec<_> = buffer.sorted_iter().collect();
        assert_eq!(results.len(), num_entries);
        for i in 0..num_entries {
            println!("OVC: {}", results[i].0);
            assert_eq!(results[i].0, encoded[i].get_ovc());
            assert_eq!(results[i].1, encoded[i].get_key());
            assert_eq!(results[i].2, b"value".to_vec());
        }
    }
}
