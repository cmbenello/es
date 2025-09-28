pub struct SortBuffer {
    data: Vec<(Vec<u8>, Vec<u8>)>,
    memory_used: usize,
    memory_limit: usize,
}

impl SortBuffer {
    pub fn new(memory_limit: usize) -> Self {
        Self {
            data: Vec::new(),
            memory_used: 0,
            memory_limit,
        }
    }
}

impl SortBuffer {
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn memory_used(&self) -> usize {
        return self.memory_used;
    }

    pub fn has_space(&self, key: &[u8], value: &[u8]) -> bool {
        let entry_size = key.len() + value.len() + std::mem::size_of::<u32>() * 2; // key and value lengths
        self.memory_used + entry_size <= self.memory_limit
    }

    pub fn append(&mut self, key: Vec<u8>, value: Vec<u8>) -> bool {
        let entry_size = key.len() + value.len() + std::mem::size_of::<u32>() * 2; // key and value lengths

        if self.memory_used + entry_size > self.memory_limit {
            println!(
                "SortBuffer: Not enough space to append entry. Used: {}, Entry size: {}, Limit: {}",
                self.memory_used, entry_size, self.memory_limit
            );
            return false;
        }

        self.data.push((key, value));
        self.memory_used += entry_size;
        true
    }

    pub fn sorted_iter(&mut self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>> {
        self.data.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let data = std::mem::take(&mut self.data);
        Box::new(data.into_iter())
    }

    pub fn reset(&mut self) {
        self.data.clear();
        self.memory_used = 0;
    }
}
