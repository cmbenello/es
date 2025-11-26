use std::cmp::Ordering;

pub mod kll;
pub mod reservoir_sampler;

/// Common trait for sampling-based data structures that provide quantile estimation.
///
/// This trait defines the core operations for any sampling technique (KLL sketch,
/// reservoir sampling, etc.) used for approximate quantile queries.
pub trait QuantileSampler<T> {
    /// Add a new value to the sampler.
    fn update(&mut self, x: T);

    /// Get the rank (count of items ≤ x) for a given value.
    fn rank(&self, x: T) -> usize;

    /// Get the total count of items seen so far.
    fn count(&self) -> usize;

    /// Get the quantile (fraction of items ≤ x) for a given value.
    fn quantile(&self, x: T) -> f64;

    /// Get the cumulative distribution function.
    fn cdf(&self) -> CDF<T>;
}

/// Extension trait for samplers that support merging operations.
/// This is separate from QuantileSampler because `merge` uses `&Self`,
/// which prevents the trait from being object-safe (dyn-compatible).
pub trait MergeableSampler<T>: QuantileSampler<T> {
    /// Merge another sampler into this one.
    fn merge(&mut self, other: &Self);
}

#[derive(Clone, Debug)]
pub struct Quantile<T> {
    pub q: f64,
    pub v: T,
}

#[derive(Clone, Debug)]
pub struct CDF<T>(Vec<Quantile<T>>);

impl<T> CDF<T>
where
    T: PartialOrd + Clone,
{
    /// Find the size of the CDF.
    pub fn size(&self) -> usize {
        self.0.len()
    }

    /// Find the quantile for a given value x.
    /// Returns the fraction of values less than or equal to x.
    pub fn quantile(&self, x: T) -> f64 {
        // Find the rightmost position where value <= x
        let idx = self
            .0
            .binary_search_by(|q| {
                // Use match to handle all cases properly
                match q.v.partial_cmp(&x) {
                    Some(Ordering::Less) | Some(Ordering::Equal) => Ordering::Less,
                    Some(Ordering::Greater) => Ordering::Greater,
                    None => Ordering::Greater, // Handle NaN by treating as greater
                }
            })
            .unwrap_or_else(|e| e);

        // idx is the position where we'd insert a value > x
        // So idx-1 is the last position with value <= x
        if idx == 0 { 0.0 } else { self.0[idx - 1].q }
    }

    /// Query the CDF for a specific quantile p (0.0 to 1.0).
    /// Returns the smallest value whose cumulative probability is >= p.
    pub fn query(&self, p: f64) -> T {
        // Handle edge cases
        if self.0.is_empty() {
            panic!("Cannot query empty CDF");
        }

        // Find the first position where cumulative probability >= p
        let idx = self
            .0
            .binary_search_by(|q| {
                if q.q < p {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_or_else(|e| e);

        // idx is the first position with cumulative probability >= p
        if idx == self.0.len() {
            // p is greater than all cumulative probabilities, return the last value
            self.0[self.0.len() - 1].v.clone()
        } else {
            self.0[idx].v.clone()
        }
    }
}

mod tests {
    use std::collections::HashMap;

    use super::kll::*;
    use super::reservoir_sampler::*;
    use super::*;

    #[test]
    fn test_range_partition_quality_uniform() {
        let num_partitions = 10;
        let total_items = 10000;

        let mut kll = KLL::new(200);
        let mut reservoir = ReservoirSampler::new(200);

        let mut data = Vec::new();
        for i in 0..total_items {
            let v = i as f64;
            data.push(v);
            kll.update(v);
            reservoir.update(v);
        }

        let kll_cdf = kll.cdf();
        let res_cdf = reservoir.cdf();

        let mut kll_boundaries = vec![f64::NEG_INFINITY];
        let mut res_boundaries = vec![f64::NEG_INFINITY];

        for i in 1..num_partitions {
            let quantile = i as f64 / num_partitions as f64;
            kll_boundaries.push(kll_cdf.query(quantile));
            res_boundaries.push(res_cdf.query(quantile));
        }
        kll_boundaries.push(f64::INFINITY);
        res_boundaries.push(f64::INFINITY);

        let count_partition = |boundaries: &[f64]| -> Vec<usize> {
            let mut counts = vec![0; num_partitions];
            for &v in &data {
                for i in 0..num_partitions {
                    if v > boundaries[i] && v <= boundaries[i + 1] {
                        counts[i] += 1;
                        break;
                    }
                }
            }
            counts
        };

        let kll_counts = count_partition(&kll_boundaries);
        let res_counts = count_partition(&res_boundaries);

        let optimal_size = total_items / num_partitions;

        let calculate_imbalance = |counts: &[usize]| {
            let max = *counts.iter().max().unwrap();
            ((max as f64 - optimal_size as f64) / optimal_size as f64 * 100.0).abs()
        };

        let kll_imbalance = calculate_imbalance(&kll_counts);
        let res_imbalance = calculate_imbalance(&res_counts);

        assert!(
            kll_imbalance < 10.0,
            "KLL max imbalance should be < 10% for uniform data, got {:.2}%",
            kll_imbalance
        );

        assert!(
            res_imbalance < 60.0,
            "Reservoir max imbalance should be < 60% for uniform data, got {:.2}%",
            res_imbalance
        );
    }

    #[test]
    fn test_range_partition_quality_zipf() {
        use crate::fastzipf::FastZipf;
        use std::collections::HashMap;

        let num_partitions = 40;
        let skew = 0.80;
        let nr_items = 100000;
        let num_samples = 1_000_000;

        let mut zipf = FastZipf::new(skew, nr_items);
        let mut kll = KLL::new(200);
        let mut reservoir = ReservoirSampler::new(600);

        let mut data = Vec::with_capacity(num_samples);
        let mut freq_map: HashMap<u64, usize> = HashMap::new();

        for _ in 0..num_samples {
            let v = zipf.sample() as f64;
            data.push(v);
            *freq_map.entry(v as u64).or_insert(0) += 1;
            kll.update(v);
            reservoir.update(v);
        }

        // Find top 10 most frequent values
        let mut freq_vec: Vec<(u64, usize)> = freq_map.into_iter().collect();
        freq_vec.sort_by(|a, b| b.1.cmp(&a.1));

        println!("Top 10 most frequent values:");
        for (value, count) in freq_vec.iter().take(10) {
            let pct = *count as f64 / num_samples as f64 * 100.0;
            println!("  Value {}: {} occurrences ({:.2}%)", value, count, pct);
        }

        let kll_cdf = kll.cdf();
        let res_cdf = reservoir.cdf();

        let mut kll_boundaries = vec![f64::NEG_INFINITY];
        let mut res_boundaries = vec![f64::NEG_INFINITY];

        for i in 1..num_partitions {
            let quantile = i as f64 / num_partitions as f64;
            kll_boundaries.push(kll_cdf.query(quantile));
            res_boundaries.push(res_cdf.query(quantile));
        }
        kll_boundaries.push(f64::INFINITY);
        res_boundaries.push(f64::INFINITY);

        let count_partition = |boundaries: &[f64]| -> Vec<usize> {
            let mut counts = vec![0; num_partitions];
            for &v in &data {
                for i in 0..num_partitions {
                    if v > boundaries[i] && v <= boundaries[i + 1] {
                        counts[i] += 1;
                        break;
                    }
                }
            }
            counts
        };

        let kll_counts = count_partition(&kll_boundaries);
        let res_counts = count_partition(&res_boundaries);

        let optimal_size = num_samples / num_partitions;
        println!(
            "\nOptimal partition size: {}({}%)",
            optimal_size,
            100.0 / num_partitions as f64
        );

        let calculate_imbalance = |counts: &[usize]| {
            let max = *counts.iter().max().unwrap();
            ((max as f64 - optimal_size as f64) / optimal_size as f64 * 100.0).abs()
        };

        let kll_imbalance = calculate_imbalance(&kll_counts);
        let res_imbalance = calculate_imbalance(&res_counts);

        println!("\nKLL imbalance: {:.2}%", kll_imbalance);
        println!("Reservoir imbalance: {:.2}%", res_imbalance);
    }

    #[test]
    fn test_merged_sketch_partition_quality_zipf() {
        use crate::fastzipf::FastZipf;

        let num_partitions = 40;
        let skew = 0.80;
        let nr_items = 100000;
        let samples_per_chunk = 250_000;

        let mut zipf = FastZipf::new(skew, nr_items);

        // Generate all data first
        let total_samples = num_partitions * samples_per_chunk;
        let mut all_data = Vec::with_capacity(total_samples);
        for _ in 0..total_samples {
            all_data.push(zipf.sample() as f64);
        }

        // Top 5 most frequent values
        let mut freq_map = HashMap::new();
        for &v in &all_data {
            *freq_map.entry(v as u64).or_insert(0) += 1;
        }
        let mut freq_vec: Vec<(u64, usize)> = freq_map.into_iter().collect();
        freq_vec.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // Build sketches independently on each chunk
        let mut kll_sketches = Vec::new();
        let mut res_sketches = Vec::new();
        for chunk_idx in 0..num_partitions {
            let mut kll = KLL::new(200);
            let mut reservoir = ReservoirSampler::new(600);
            let start = chunk_idx * samples_per_chunk;
            let end = start + samples_per_chunk;

            for &v in &all_data[start..end] {
                kll.update(v);
                reservoir.update(v);
            }
            kll_sketches.push(kll);
            res_sketches.push(reservoir);
        }

        // Merge all sketches
        let mut merged_kll = kll_sketches.remove(0);
        for sketch in kll_sketches.drain(..) {
            merged_kll.merge(&sketch);
        }

        let mut merged_res = res_sketches.remove(0);
        for sketch in res_sketches.drain(..) {
            merged_res.merge(&sketch);
        }

        // Get boundaries from merged sketches
        let kll_cdf = merged_kll.cdf();
        let res_cdf = merged_res.cdf();

        let mut kll_boundaries = vec![f64::NEG_INFINITY];
        let mut res_boundaries = vec![f64::NEG_INFINITY];

        for i in 1..num_partitions {
            let quantile = i as f64 / num_partitions as f64;
            kll_boundaries.push(kll_cdf.query(quantile));
            res_boundaries.push(res_cdf.query(quantile));
        }
        kll_boundaries.push(f64::INFINITY);
        res_boundaries.push(f64::INFINITY);

        // Count how data is distributed with these boundaries
        let count_partition = |boundaries: &[f64]| -> Vec<usize> {
            let mut counts = vec![0; num_partitions];
            for &v in &all_data {
                for i in 0..num_partitions {
                    if v > boundaries[i] && v <= boundaries[i + 1] {
                        counts[i] += 1;
                        break;
                    }
                }
            }
            counts
        };

        let kll_counts = count_partition(&kll_boundaries);
        let res_counts = count_partition(&res_boundaries);

        let optimal_size = total_samples / num_partitions;

        let calculate_imbalance = |counts: &[usize]| {
            let max = *counts.iter().max().unwrap();
            ((max as f64 - optimal_size as f64) / optimal_size as f64 * 100.0).abs()
        };

        let kll_imbalance = calculate_imbalance(&kll_counts);
        let res_imbalance = calculate_imbalance(&res_counts);

        println!("\nMerged sketch test (Zipf skew={}):", skew);
        println!("  Total samples: {}", total_samples);
        println!("  Num partitions: {}", num_partitions);
        println!("  Optimal partition size: {} ({:.2}%)", optimal_size, 100.0 / num_partitions as f64);
        println!("\nTop 5 most frequent values:");
        for (value, count) in freq_vec.iter().take(5) {
            let pct = *count as f64 / total_samples as f64 * 100.0;
            println!("  Value {}: {} occurrences ({:.2}%)", value, count, pct);
        }
        println!("\nKLL (size=200):");
        println!("  Min partition size: {}", kll_counts.iter().min().unwrap());
        println!("  Max partition size: {}", kll_counts.iter().max().unwrap());
        println!("  Imbalance: {:.2}%", kll_imbalance);
        println!("\nReservoir (size=600):");
        println!("  Min partition size: {}", res_counts.iter().min().unwrap());
        println!("  Max partition size: {}", res_counts.iter().max().unwrap());
        println!("  Imbalance: {:.2}%", res_imbalance);
    }
}
