use crate::rand::small_thread_rng;
use crate::sketch::{CDF, MergeableSampler, Quantile, QuantileSampler};
use rand::Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

/// A mergeable Reservoir Sampler using Priority Sampling.
#[derive(Clone)]
pub struct ReservoirSampler<T> {
    k: usize,
    /// Total items processed (needed for scaling rank estimates)
    total_seen: usize,
    /// Min-Heap of (priority, item)
    heap: BinaryHeap<WeightedItem<T>>,
}

#[derive(Clone)]
struct WeightedItem<T> {
    priority: f64,
    value: T,
}

// (Ord implementations omitted for brevity, assuming same as previous response:
// WeightedItem creates a Min-Heap based on priority)
impl<T> PartialEq for WeightedItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl<T> Eq for WeightedItem<T> {}
impl<T> PartialOrd for WeightedItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.priority.partial_cmp(&self.priority)
    }
}
impl<T> Ord for WeightedItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<T> ReservoirSampler<T>
where
    T: Clone + PartialOrd,
{
    pub fn new(k: usize) -> Self {
        ReservoirSampler {
            k,
            total_seen: 0,
            heap: BinaryHeap::with_capacity(k),
        }
    }

    fn update_internal(&mut self, item: T) {
        self.total_seen += 1;
        let mut rng = small_thread_rng();
        let priority: f64 = rng.random();

        if self.heap.len() < self.k {
            self.heap.push(WeightedItem {
                priority,
                value: item,
            });
        } else if let Some(min_item) = self.heap.peek() {
            if priority > min_item.priority {
                self.heap.pop();
                self.heap.push(WeightedItem {
                    priority,
                    value: item,
                });
            }
        }
    }

    fn merge_internal(&mut self, other: &ReservoirSampler<T>) {
        self.total_seen += other.total_seen;
        for weighted_item in &other.heap {
            if self.heap.len() < self.k {
                self.heap.push(weighted_item.clone());
            } else if let Some(min_item) = self.heap.peek() {
                if weighted_item.priority > min_item.priority {
                    self.heap.pop();
                    self.heap.push(weighted_item.clone());
                }
            }
        }
    }

    // --- NEW METHODS START HERE ---

    /// Returns the total number of items seen by this sampler (and merged samplers).
    pub fn count_internal(&self) -> usize {
        self.total_seen
    }

    /// Estimates the rank of `x` in the total stream.
    /// Rank = Number of items <= x
    pub fn rank_internal(&self, x: &T) -> usize {
        if self.heap.is_empty() {
            return 0;
        }

        // Count how many items in the reservoir are <= x
        let mut r = 0;
        for wi in &self.heap {
            if &wi.value <= x {
                r += 1;
            }
        }

        // Scale up: (ReservoirRank / ReservoirSize) * TotalSeen
        // We use floating point math to avoid early truncation, then cast back.
        let scaling_factor = self.total_seen as f64 / self.heap.len() as f64;
        (r as f64 * scaling_factor) as usize
    }

    /// Estimates the quantile of `x` (Rank / Total).
    /// Returns value in [0.0, 1.0]
    pub fn quantile_internal(&self, x: &T) -> f64 {
        if self.total_seen == 0 {
            return 0.0;
        }
        let rank = self.rank_internal(x);
        rank as f64 / self.total_seen as f64
    }

    /// Returns the full CDF approximation.
    pub fn cdf_internal(&self) -> CDF<T> {
        // Extract values
        let mut samples: Vec<T> = self.heap.iter().map(|wi| wi.value.clone()).collect();

        // Sort to determine order
        samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let len = samples.len();
        if len == 0 {
            return CDF(vec![]);
        }

        let mut quantiles = Vec::with_capacity(len);

        // In a uniform reservoir sample, every item represents an equal
        // chunk of the original probability mass (1 / len).
        for (i, v) in samples.into_iter().enumerate() {
            // q is the cumulative probability up to this item
            let q = (i + 1) as f64 / len as f64;
            quantiles.push(Quantile { q, v });
        }

        CDF(quantiles)
    }
}

impl<T> QuantileSampler<T> for ReservoirSampler<T>
where
    T: PartialOrd + Clone,
{
    fn update(&mut self, x: T) {
        self.update_internal(x);
    }

    fn rank(&self, x: T) -> usize {
        self.rank_internal(&x)
    }

    fn count(&self) -> usize {
        self.count_internal()
    }

    fn quantile(&self, x: T) -> f64 {
        self.quantile_internal(&x)
    }

    fn cdf(&self) -> CDF<T> {
        self.cdf_internal()
    }
}

impl<T> MergeableSampler<T> for ReservoirSampler<T>
where
    T: PartialOrd + Clone,
{
    fn merge(&mut self, other: &Self) {
        self.merge_internal(other);
    }
}

mod tests {
    use crate::sketch::kll::KLL;

    use super::*;

    #[test]
    fn test_reservoir_sampler_basic() {
        let mut sampler = ReservoirSampler::new(100);

        for i in 0..1000 {
            sampler.update(i as f64);
        }

        assert_eq!(sampler.count(), 1000);
        assert_eq!(sampler.heap.len(), 100);

        // Test that rank is reasonable
        let rank500 = sampler.rank(500.0);
        assert!(
            rank500 > 400 && rank500 < 600,
            "rank(500) should be ~500, got {}",
            rank500
        );
    }

    #[test]
    fn test_reservoir_sampler_quantile() {
        let mut sampler = ReservoirSampler::new(200);

        for i in 0..10000 {
            sampler.update(i as f64);
        }

        let cdf = sampler.cdf();

        // Reservoir sampling with 200 samples from 10000 items has significant variance
        let median = cdf.query(0.5);
        assert!(
            median > 4000.0 && median < 6000.0,
            "Median should be ~5000, got {}",
            median
        );

        let p90 = cdf.query(0.9);
        assert!(
            p90 > 8000.0 && p90 < 9800.0,
            "90th percentile should be ~9000, got {}",
            p90
        );
    }

    #[test]
    fn test_reservoir_sampler_merge() {
        let mut sampler1 = ReservoirSampler::new(50);
        let mut sampler2 = ReservoirSampler::new(50);

        for i in 0..500 {
            sampler1.update(i as f64);
        }

        for i in 500..1000 {
            sampler2.update(i as f64);
        }

        sampler1.merge(&sampler2);

        assert_eq!(sampler1.count(), 1000);

        let cdf = sampler1.cdf();
        let median = cdf.query(0.5);
        assert!(
            median > 400.0 && median < 600.0,
            "After merge, median should be ~500, got {}",
            median
        );
    }

    #[test]
    fn test_reservoir_sampler_uniform() {
        // Test that reservoir sampling maintains uniform distribution
        let mut sampler = ReservoirSampler::new(100);
        let mut rng = small_thread_rng();

        for _ in 0..10000 {
            sampler.update(rng.random_range(0.0..100.0));
        }

        let cdf = sampler.cdf();

        // Check that quantiles are roughly uniform
        // With only 100 samples from 10000 items, variance is higher
        let q25 = cdf.query(0.25);
        let q50 = cdf.query(0.50);
        let q75 = cdf.query(0.75);

        assert!(
            q25 > 10.0 && q25 < 40.0,
            "25th percentile should be ~25, got {}",
            q25
        );
        assert!(
            q50 > 35.0 && q50 < 65.0,
            "50th percentile should be ~50, got {}",
            q50
        );
        assert!(
            q75 > 60.0 && q75 < 90.0,
            "75th percentile should be ~75, got {}",
            q75
        );
    }

    #[test]
    fn test_reservoir_sampler_trait() {
        // Test using the QuantileSampler trait
        let mut sampler: Box<dyn QuantileSampler<f64>> = Box::new(ReservoirSampler::new(100));

        for i in 0..1000 {
            sampler.update(i as f64);
        }

        assert_eq!(sampler.count(), 1000);

        // let rank = sampler.rank(500.0);
        // assert!(
        //     rank > 400 && rank < 600,
        //     "rank(500) should be ~500, got {}",
        //     rank
        // );
    }
}
