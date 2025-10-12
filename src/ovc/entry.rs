use std::cmp::Ordering;

#[allow(dead_code)]
pub(crate) struct Entry<T: Ord + SentinelValue> {
    pub(crate) value: T,
    pub(crate) run_id: usize,
}

impl<T: std::fmt::Debug + Ord + SentinelValue> std::fmt::Debug for Entry<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.run_id != usize::MAX {
            write!(f, "({:?}, {})", self.value, self.run_id)
        } else {
            write!(f, "{:?}", self.value)
        }
    }
}

#[allow(dead_code)]
impl<T: Ord + SentinelValue> Entry<T> {
    pub fn new(value: T, run_id: usize) -> Self {
        Entry { value, run_id }
    }

    pub fn new_early_fence() -> Self {
        Entry {
            value: T::early_fence(),
            run_id: usize::MAX,
        }
    }

    pub fn new_late_fence() -> Self {
        Entry {
            value: T::late_fence(),
            run_id: usize::MAX,
        }
    }
}

pub trait SentinelValue {
    fn early_fence() -> Self;
    fn late_fence() -> Self;

    fn is_early_fence(&self) -> bool;
    fn is_late_fence(&self) -> bool;
}

#[derive(Clone)]
pub enum Sentineled<T: Ord> {
    Early,
    Normal(T),
    Late,
}

impl<T: Ord + std::fmt::Display> std::fmt::Display for Sentineled<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sentineled::Early => write!(f, "EF"),
            Sentineled::Normal(value) => write!(f, "{}", value),
            Sentineled::Late => write!(f, "LF"),
        }
    }
}

impl<T: Ord + std::fmt::Debug> std::fmt::Debug for Sentineled<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sentineled::Early => write!(f, "EF"),
            Sentineled::Normal(value) => write!(f, "{:?}", value),
            Sentineled::Late => write!(f, "LF"),
        }
    }
}

impl<T: Ord> Sentineled<T> {
    pub fn new(value: T) -> Self {
        Sentineled::Normal(value)
    }

    pub fn inner(self) -> T {
        match self {
            Sentineled::Normal(value) => value,
            _ => panic!("Cannot get inner value from a sentinel"),
        }
    }
}

impl<T: Ord> PartialEq for Sentineled<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Sentineled::Normal(a), Sentineled::Normal(b)) => a == b,
            _ => false,
        }
    }
}

impl<T: Ord> Eq for Sentineled<T> {}

impl<T: Ord> PartialOrd for Sentineled<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> Ord for Sentineled<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Sentineled::Normal(a), Sentineled::Normal(b)) => a.cmp(b),
            (Sentineled::Early, Sentineled::Early) | (Sentineled::Late, Sentineled::Late) => {
                Ordering::Equal
            }
            (Sentineled::Early, _) => Ordering::Less,
            (Sentineled::Late, _) => Ordering::Greater,
            (Sentineled::Normal(_), Sentineled::Early) => Ordering::Greater,
            (Sentineled::Normal(_), Sentineled::Late) => Ordering::Less,
        }
    }
}

impl<T: Ord> SentinelValue for Sentineled<T> {
    fn early_fence() -> Self {
        Sentineled::Early
    }

    fn late_fence() -> Self {
        Sentineled::Late
    }

    fn is_early_fence(&self) -> bool {
        matches!(self, Sentineled::Early)
    }

    fn is_late_fence(&self) -> bool {
        matches!(self, Sentineled::Late)
    }
}

#[allow(dead_code)]
pub(crate) fn next_power_of_two(num: usize) -> usize {
    let mut size = 1;
    while size < num {
        size *= 2;
    }
    size
}
