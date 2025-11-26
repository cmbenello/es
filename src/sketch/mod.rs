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
    use super::kll::*;
    use super::reservoir_sampler::*;
    use super::*;

    #[test]
    fn test_compare_kll_vs_reservoir() {
        // Compare KLL sketch and reservoir sampling on the same data
        let mut kll = KLL::new(100);
        let mut reservoir = ReservoirSampler::new(100);

        for i in 0..5000 {
            let v = i as f64;
            kll.update(v);
            reservoir.update(v);
        }

        println!("KLL Sketch:");
        println!("{}", kll);
        println!("\nReservoir Sampler:");
        println!("{}", reservoir);

        let kll_cdf = kll.cdf();
        let res_cdf = reservoir.cdf();

        // Both should give reasonable estimates
        let kll_median = kll_cdf.query(0.5);
        let res_median = res_cdf.query(0.5);

        println!("KLL median: {}", kll_median);
        println!("Reservoir median: {}", res_median);

        // KLL should be close to 2500 (has better accuracy guarantees)
        assert!(
            kll_median > 2300.0 && kll_median < 2700.0,
            "KLL median: {}",
            kll_median
        );

        // Reservoir sampling has higher variance with only 100 samples from 5000 items
        // We use a more relaxed bound here
        assert!(
            res_median > 1500.0 && res_median < 3500.0,
            "Reservoir median: {}",
            res_median
        );
    }

    #[test]
    fn test_compare_kll_vs_reservoir_skewed() {
        // Compare KLL sketch and reservoir sampling on skewed data
        // This generates a heavily right-skewed distribution:
        // - 90% of values are in [0, 1000)
        // - 10% of values are in [1000, 10000)
        let mut kll = KLL::new(100);
        let mut reservoir = ReservoirSampler::new(100);

        // Add 4500 values in the range [0, 1000) - 90% of data
        for i in 0..4500 {
            let v = (i % 1000) as f64;
            kll.update(v);
            reservoir.update(v);
        }

        // Add 500 values in the range [1000, 10000) - 10% of data (tail)
        for i in 0..500 {
            let v = 1000.0 + (i * 18) as f64; // Spread across [1000, 10000)
            kll.update(v);
            reservoir.update(v);
        }

        println!("\n=== SKEWED DATA TEST ===");
        println!("KLL Sketch:");
        println!("{}", kll);
        println!("\nReservoir Sampler:");
        println!("{}", reservoir);

        let kll_cdf = kll.cdf();
        let res_cdf = reservoir.cdf();

        // Test median (should be around 500 since 90% of data is in [0, 1000))
        let kll_median = kll_cdf.query(0.5);
        let res_median = res_cdf.query(0.5);

        println!("\nMedian (p50):");
        println!("  KLL median: {}", kll_median);
        println!("  Reservoir median: {}", res_median);
        println!("  Expected: ~500 (middle of bulk data)");

        // Test 90th percentile (should be near upper end of bulk, ~900-1000)
        let kll_p90 = kll_cdf.query(0.9);
        let res_p90 = res_cdf.query(0.9);

        println!("\n90th percentile:");
        println!("  KLL p90: {}", kll_p90);
        println!("  Reservoir p90: {}", res_p90);
        println!("  Expected: ~900-1000 (end of bulk data)");

        // Test 99th percentile (should be in the tail, somewhere in [1000, 10000))
        let kll_p99 = kll_cdf.query(0.99);
        let res_p99 = res_cdf.query(0.99);

        println!("\n99th percentile:");
        println!("  KLL p99: {}", kll_p99);
        println!("  Reservoir p99: {}", res_p99);
        println!("  Expected: somewhere in [1000, 10000) (tail)");

        // KLL should handle skewed data well with its deterministic guarantees
        assert!(
            kll_median > 300.0 && kll_median < 700.0,
            "KLL median should be ~500, got {}",
            kll_median
        );

        assert!(
            kll_p90 > 700.0 && kll_p90 < 1200.0,
            "KLL p90 should be ~900-1000, got {}",
            kll_p90
        );

        assert!(
            kll_p99 > 1000.0,
            "KLL p99 should be in tail (>1000), got {}",
            kll_p99
        );

        // Reservoir sampling has uniform sampling probability, so:
        // - It likely has ~90 samples from [0, 1000) and ~10 from [1000, 10000)
        // - Median estimate will have high variance
        // - Tail estimates will be very rough (only ~10 samples)
        assert!(
            res_median > 0.0 && res_median < 1500.0,
            "Reservoir median should be roughly in bulk range, got {}",
            res_median
        );

        // The p90 for reservoir is less predictable due to sampling variance
        assert!(
            res_p90 > 500.0,
            "Reservoir p90 should be > 500, got {}",
            res_p90
        );
    }

    #[test]
    fn test_range_partition_quality_uniform() {
        // Test how well quantile-based partitioning works with uniform data
        let num_partitions = 10;
        let total_items = 10000;

        let mut kll = KLL::new(200);
        let mut reservoir = ReservoirSampler::new(200);

        // Generate uniform data
        let mut data = Vec::new();
        for i in 0..total_items {
            let v = i as f64;
            data.push(v);
            kll.update(v);
            reservoir.update(v);
        }

        println!("\n=== RANGE PARTITION QUALITY (Uniform Data) ===");
        println!("Total items: {}", total_items);
        println!("Target partitions: {}", num_partitions);
        println!("Optimal partition size: {}", total_items / num_partitions);

        // Get partition boundaries from both methods
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

        // Count actual items in each partition
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

        // Calculate statistics
        let optimal_size = total_items / num_partitions;

        let calculate_stats = |counts: &[usize], name: &str| {
            let min = *counts.iter().min().unwrap();
            let max = *counts.iter().max().unwrap();
            let avg = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
            let variance: f64 = counts
                .iter()
                .map(|&c| {
                    let diff = c as f64 - avg;
                    diff * diff
                })
                .sum::<f64>()
                / counts.len() as f64;
            let stddev = variance.sqrt();
            let max_imbalance =
                ((max as f64 - optimal_size as f64) / optimal_size as f64 * 100.0).abs();

            println!("\n{} partitioning:", name);
            println!("  Partition sizes: {:?}", counts);
            println!("  Min: {}, Max: {}, Avg: {:.1}", min, max, avg);
            println!("  Stddev: {:.2}", stddev);
            println!("  Max imbalance: {:.2}%", max_imbalance);
            println!(
                "  Range: {} ({}% of optimal)",
                max - min,
                (max - min) as f64 / optimal_size as f64 * 100.0
            );

            max_imbalance
        };

        let kll_imbalance = calculate_stats(&kll_counts, "KLL");
        let res_imbalance = calculate_stats(&res_counts, "Reservoir");

        // For uniform data, partitions should be very balanced
        assert!(
            kll_imbalance < 10.0,
            "KLL max imbalance should be < 10% for uniform data, got {:.2}%",
            kll_imbalance
        );

        // Reservoir sampling will have more variance
        // With only 200 samples from 10000 items (2% sampling rate), significant variance is expected
        assert!(
            res_imbalance < 60.0,
            "Reservoir max imbalance should be < 60% for uniform data, got {:.2}%",
            res_imbalance
        );
    }

    #[test]
    fn test_range_partition_quality_skewed() {
        // Test how well quantile-based partitioning works with skewed data
        let num_partitions = 10;
        let total_items = 100000;

        let mut kll = KLL::new(200);
        let mut reservoir = ReservoirSampler::new(600);

        let mut data = Vec::new();

        // 90000 items in [0, 1000)
        for i in 0..90000 {
            let v = (i % 1000) as f64;
            data.push(v);
            kll.update(v);
            reservoir.update(v);
        }

        // 10000 items in [1000, 10000)
        for i in 0..10000 {
            let v = 1000.0 + (i % 9000) as f64;
            data.push(v);
            kll.update(v);
            reservoir.update(v);
        }

        println!("\n=== RANGE PARTITION QUALITY (Skewed Data) ===");
        println!("Total items: {}", total_items);
        println!("Target partitions: {}", num_partitions);
        println!("Optimal partition size: {}", total_items / num_partitions);
        println!("Data distribution: 80% in [0, 1000), 20% in [1000, 10000)");
        println!("KLL Sketch:");
        println!("{}", kll);
        println!("\nReservoir Sampler:");
        println!("{}", reservoir);

        // Get partition boundaries from both methods
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

        println!(
            "\nKLL boundaries: {:?}",
            &kll_boundaries[1..kll_boundaries.len() - 1]
        );
        println!(
            "Reservoir boundaries: {:?}",
            &res_boundaries[1..res_boundaries.len() - 1]
        );

        // Count actual items in each partition
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

        // Calculate statistics
        let optimal_size = total_items / num_partitions;

        let calculate_stats = |counts: &[usize], name: &str| {
            let min = *counts.iter().min().unwrap();
            let max = *counts.iter().max().unwrap();
            let avg = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
            let variance: f64 = counts
                .iter()
                .map(|&c| {
                    let diff = c as f64 - avg;
                    diff * diff
                })
                .sum::<f64>()
                / counts.len() as f64;
            let stddev = variance.sqrt();
            let max_imbalance =
                ((max as f64 - optimal_size as f64) / optimal_size as f64 * 100.0).abs();

            println!("\n{} partitioning:", name);
            println!("  Partition sizes: {:?}", counts);
            println!("  Min: {}, Max: {}, Avg: {:.1}", min, max, avg);
            println!("  Stddev: {:.2}", stddev);
            println!("  Max imbalance: {:.2}%", max_imbalance);
            println!(
                "  Range: {} ({}% of optimal)",
                max - min,
                (max - min) as f64 / optimal_size as f64 * 100.0
            );

            max_imbalance
        };

        let kll_imbalance = calculate_stats(&kll_counts, "KLL");
        let res_imbalance = calculate_stats(&res_counts, "Reservoir");

        // For skewed data with good quantile estimates, partitions should still be balanced
        // because we're partitioning by quantiles, not by value ranges
        assert!(
            kll_imbalance < 15.0,
            "KLL max imbalance should be < 15% for skewed data, got {:.2}%",
            kll_imbalance
        );

        // Reservoir sampling will have more variance due to limited samples
        // In practice, with only 200 samples from 10000 items, variance can be significant
        // On skewed data, uniform sampling struggles to capture the distribution accurately
        // This can lead to very poor partition quality (>60% imbalance is possible)
        assert!(
            res_imbalance < 80.0,
            "Reservoir max imbalance should be < 80% for skewed data, got {:.2}%",
            res_imbalance
        );

        // Demonstrate that KLL provides much better partition quality
        println!("\nPartition quality comparison:");
        println!("  KLL imbalance: {:.2}%", kll_imbalance);
        println!("  Reservoir imbalance: {:.2}%", res_imbalance);
        println!(
            "  KLL is {:.1}x better at balancing partitions",
            res_imbalance / kll_imbalance
        );
    }

    #[test]
    fn test_compare_kll_vs_reservoir_zipf() {
        use crate::fastzipf::FastZipf;

        // Compare KLL sketch and reservoir sampling on Zipf-distributed data
        // Using skew of 0.99 (close to 1.0, highly skewed)
        let skew = 0.99;
        let nr_items = 100000; // Universe size
        let num_samples = 1_000_000; // Number of items to sample

        let mut zipf = FastZipf::new(skew, nr_items);

        let mut kll = KLL::new(200);
        let mut reservoir = ReservoirSampler::new(200);

        println!("\n=== ZIPF DISTRIBUTION TEST (skew={}) ===", skew);
        println!(
            "Sampling {} items from Zipf distribution over {} unique values",
            num_samples, nr_items
        );

        let start = std::time::Instant::now();

        // Track actual frequency distribution for comparison
        let mut actual_counts = vec![0usize; nr_items];

        for _ in 0..num_samples {
            let v = zipf.sample() as f64;
            actual_counts[v as usize] += 1;
            kll.update(v);
            reservoir.update(v);
        }

        let elapsed = start.elapsed();
        println!("Data generation and sampling took: {:?}", elapsed);

        // Find the most frequent items
        let mut freq_items: Vec<(usize, usize)> = actual_counts
            .iter()
            .enumerate()
            .map(|(idx, &count)| (idx, count))
            .collect();
        freq_items.sort_by(|a, b| b.1.cmp(&a.1));

        println!("\nTop 10 most frequent items:");
        for i in 0..10.min(freq_items.len()) {
            let (item, count) = freq_items[i];
            let pct = count as f64 / num_samples as f64 * 100.0;
            println!("  Item {}: {} occurrences ({:.2}%)", item, count, pct);
        }

        println!("\nKLL Sketch:");
        println!("{}", kll);
        println!("\nReservoir Sampler:");
        println!("{}", reservoir);

        let kll_cdf = kll.cdf();
        let res_cdf = reservoir.cdf();

        // Test various quantiles
        let quantiles = [0.5, 0.9, 0.95, 0.99];

        println!("\nQuantile estimates:");
        for &q in &quantiles {
            let kll_val = kll_cdf.query(q);
            let res_val = res_cdf.query(q);

            // Calculate actual quantile from the true distribution
            let mut sorted_items: Vec<(usize, usize)> = actual_counts
                .iter()
                .enumerate()
                .map(|(idx, &count)| (idx, count))
                .filter(|(_, count)| *count > 0)
                .collect();
            sorted_items.sort_by_key(|(idx, _)| *idx);

            let mut cumulative = 0usize;
            let target = (q * num_samples as f64) as usize;
            let mut actual_val = 0.0;
            for (idx, count) in sorted_items {
                cumulative += count;
                if cumulative >= target {
                    actual_val = idx as f64;
                    break;
                }
            }

            println!(
                "  p{:.0}: KLL={:.0}, Reservoir={:.0}, Actual={:.0}",
                q * 100.0,
                kll_val,
                res_val,
                actual_val
            );

            let kll_error = ((kll_val - actual_val).abs() / actual_val * 100.0).min(100.0);
            let res_error = ((res_val - actual_val).abs() / actual_val * 100.0).min(100.0);

            println!(
                "    KLL error: {:.2}%, Reservoir error: {:.2}%",
                kll_error, res_error
            );
        }

        // Basic sanity checks - Zipf is highly skewed, so most values are concentrated at the low end
        let kll_median = kll_cdf.query(0.5);
        let res_median = res_cdf.query(0.5);

        // With skew of 0.99, the distribution is highly skewed but the median
        // is still in the hundreds range (not single digits)
        // The median should be much smaller than the universe size
        assert!(
            kll_median < nr_items as f64 / 100.0,
            "KLL median should be small relative to universe size for highly skewed Zipf, got {}",
            kll_median
        );

        // Reservoir might have higher variance but should still capture the skew
        assert!(
            res_median < nr_items as f64 / 10.0,
            "Reservoir median should be reasonably small for Zipf, got {}",
            res_median
        );
    }
}
