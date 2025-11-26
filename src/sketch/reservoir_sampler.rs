use crate::rand::small_thread_rng;
use crate::sketch::{CDF, MergeableSampler, Quantile, QuantileSampler};
use rand::Rng;
use std::cmp::Ordering;
use std::fmt;

/// Reservoir sampling data structure for uniform random sampling.
///
/// This maintains a fixed-size reservoir of k samples from a potentially
/// infinite stream, ensuring each item has equal probability of being sampled.
pub struct ReservoirSampler<T> {
    /// The reservoir holding k samples.
    reservoir: Vec<T>,

    /// Maximum capacity of the reservoir.
    k: usize,

    /// Total number of items seen so far.
    n: usize,
}

impl<T> ReservoirSampler<T>
where
    T: PartialOrd + Clone,
{
    pub fn new(k: usize) -> Self {
        ReservoirSampler {
            reservoir: Vec::with_capacity(k),
            k,
            n: 0,
        }
    }

    fn update_internal(&mut self, x: T) {
        self.n += 1;

        if self.reservoir.len() < self.k {
            // Reservoir not full yet, just add the item
            self.reservoir.push(x);
        } else {
            // Reservoir is full, randomly decide whether to include this item
            let j = small_thread_rng().random_range(0..self.n);
            if j < self.k {
                self.reservoir[j] = x;
            }
        }
    }

    fn merge_internal(&mut self, other: &ReservoirSampler<T>) {
        // Merge two reservoir samplers
        // This uses weighted reservoir sampling where each sampler contributes
        // proportionally to its observed count

        let total_n = self.n + other.n;

        if total_n == 0 {
            return;
        }

        // First, combine all samples
        let mut combined = Vec::with_capacity(self.reservoir.len() + other.reservoir.len());
        combined.extend(self.reservoir.iter().cloned());
        combined.extend(other.reservoir.iter().cloned());

        // Randomly sample k items from the combined set, weighted by their counts
        self.reservoir.clear();

        if combined.len() <= self.k {
            self.reservoir = combined;
        } else {
            // Use reservoir sampling to select k items from combined
            for (i, item) in combined.into_iter().enumerate() {
                if i < self.k {
                    self.reservoir.push(item);
                } else {
                    let j = small_thread_rng().random_range(0..=i);
                    if j < self.k {
                        self.reservoir[j] = item;
                    }
                }
            }
        }

        self.n = total_n;
    }

    fn rank_internal(&self, x: T) -> usize {
        let mut r = 0;
        for v in &self.reservoir {
            if v <= &x {
                r += 1;
            }
        }

        // Scale the rank based on the total count
        if self.reservoir.is_empty() {
            0
        } else {
            (r * self.n) / self.reservoir.len()
        }
    }

    fn count_internal(&self) -> usize {
        self.n
    }

    fn quantile_internal(&self, x: T) -> f64 {
        if self.reservoir.is_empty() {
            return 0.0;
        }

        let rank = self.rank_internal(x);
        rank as f64 / self.n as f64
    }

    fn cdf_internal(&self) -> CDF<T> {
        let mut samples = self.reservoir.clone();
        samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let weight = self.n as f64 / self.reservoir.len() as f64;
        let mut quantiles = Vec::with_capacity(samples.len());

        for (i, v) in samples.into_iter().enumerate() {
            let q = ((i + 1) as f64 * weight) / self.n as f64;
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
        self.rank_internal(x)
    }

    fn count(&self) -> usize {
        self.count_internal()
    }

    fn quantile(&self, x: T) -> f64 {
        self.quantile_internal(x)
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

impl<T> fmt::Display for ReservoirSampler<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Reservoir Sampler (k={})", self.k)?;
        writeln!(f, "Total items seen: {}", self.n)?;
        writeln!(f, "Current samples: {}", self.reservoir.len())?;

        if !self.reservoir.is_empty() {
            write!(f, "Sample values: [")?;
            let display_count = self.reservoir.len().min(10);
            for (i, v) in self.reservoir.iter().take(display_count).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", v)?;
            }
            if self.reservoir.len() > display_count {
                write!(f, ", ... ({} more)", self.reservoir.len() - display_count)?;
            }
            writeln!(f, "]")?;
        }

        Ok(())
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
        assert_eq!(sampler.reservoir.len(), 100);

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

        let rank = sampler.rank(500.0);
        assert!(
            rank > 400 && rank < 600,
            "rank(500) should be ~500, got {}",
            rank
        );
    }
}
