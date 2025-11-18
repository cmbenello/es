/// Item in the heap, tagged with generation to implement freezing.
#[derive(Clone, Debug)]
pub struct Item {
    pub g: u64, // current run == current_gen, future runs have gen > current_gen
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

impl Item {
    pub fn new(g: u64, key: Vec<u8>, value: Vec<u8>) -> Self {
        Self { g, key, value }
    }
}

impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        self.g == other.g && self.key == other.key && self.value == other.value
    }
}

impl Eq for Item {}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Item {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.g.cmp(&other.g).then_with(|| self.key.cmp(&other.key))
    }
}
