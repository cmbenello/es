use es::fastzipf::FastZipf;
use es::sketch::{MergeableSampler, QuantileSampler};
use es::sketch::kll::KLL;
use es::sketch::reservoir_sampler::ReservoirSampler;
use std::collections::HashMap;

fn main() {
    let num_partitions = 40;
    let skew_values = vec![0.5, 0.6, 0.7, 0.8, 0.9, 0.99];
    let nr_items = 100000;
    let samples_per_chunk = 250_000;
    let num_trials = 10;

    // CSV header
    println!("skew,trial,sketch_type,min_partition_size,max_partition_size,imbalance_pct,variance,top1_pct,top2_pct,top3_pct");

    for &skew in &skew_values {
        for trial in 0..num_trials {
            let mut zipf = FastZipf::new(skew, nr_items);

            // Generate all data first
            let total_samples = num_partitions * samples_per_chunk;
            let mut all_data = Vec::with_capacity(total_samples);
            for _ in 0..total_samples {
                all_data.push(zipf.sample() as f64);
            }

            // Calculate frequency of top 3 values
            let mut freq_map = HashMap::new();
            for &v in &all_data {
                *freq_map.entry(v as u64).or_insert(0) += 1;
            }
            let mut freq_vec: Vec<(u64, usize)> = freq_map.into_iter().collect();
            freq_vec.sort_unstable_by(|a, b| b.1.cmp(&a.1));

            let top1_pct = freq_vec.get(0).map(|(_, count)| *count as f64 / total_samples as f64 * 100.0).unwrap_or(0.0);
            let top2_pct = freq_vec.get(1).map(|(_, count)| *count as f64 / total_samples as f64 * 100.0).unwrap_or(0.0);
            let top3_pct = freq_vec.get(2).map(|(_, count)| *count as f64 / total_samples as f64 * 100.0).unwrap_or(0.0);

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

            let calculate_stats = |counts: &[usize]| -> (usize, usize, f64, f64) {
                let min = *counts.iter().min().unwrap();
                let max = *counts.iter().max().unwrap();
                let imbalance = ((max as f64 - optimal_size as f64) / optimal_size as f64 * 100.0).abs();

                // Calculate variance
                let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
                let variance = counts.iter()
                    .map(|&c| {
                        let diff = c as f64 - mean;
                        diff * diff
                    })
                    .sum::<f64>() / counts.len() as f64;

                (min, max, imbalance, variance)
            };

            let (kll_min, kll_max, kll_imbalance, kll_variance) = calculate_stats(&kll_counts);
            let (res_min, res_max, res_imbalance, res_variance) = calculate_stats(&res_counts);

            // Output KLL stats
            println!("{},{},KLL,{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                skew, trial, kll_min, kll_max, kll_imbalance, kll_variance, top1_pct, top2_pct, top3_pct);

            // Output Reservoir stats
            println!("{},{},RS,{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                skew, trial, res_min, res_max, res_imbalance, res_variance, top1_pct, top2_pct, top3_pct);

            eprintln!("Completed skew={}, trial={}", skew, trial);
        }
    }
}
