use es::fastzipf::FastZipf;
use es::sketch::kll::KLL;
use es::sketch::reservoir_sampler::ReservoirSampler;
use es::sketch::{MergeableSampler, QuantileSampler};
use std::collections::HashMap;

fn main() {
    let num_partitions = 40;
    let skew_values = vec![0.5, 0.6, 0.7, 0.8, 0.9, 0.99];
    let nr_items = 100000;
    let samples_per_chunk = 250_000;
    let num_trials = 10;

    // CSV header
    println!(
        "skew,trial,sketch_type,min_partition_size,max_partition_size,imbalance_pct,variance,top1_pct,top2_pct,top3_pct,total_items,unique_items,total_preagg_entries"
    );

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

            let top1_pct = freq_vec
                .get(0)
                .map(|(_, count)| *count as f64 / total_samples as f64 * 100.0)
                .unwrap_or(0.0);
            let top2_pct = freq_vec
                .get(1)
                .map(|(_, count)| *count as f64 / total_samples as f64 * 100.0)
                .unwrap_or(0.0);
            let top3_pct = freq_vec
                .get(2)
                .map(|(_, count)| *count as f64 / total_samples as f64 * 100.0)
                .unwrap_or(0.0);

            // Build sketches independently on each chunk
            let mut kll_sketches = Vec::new();
            let mut res_sketches = Vec::new();
            let mut preagg_kll_sketches = Vec::new();
            let mut preagg_res_sketches = Vec::new();
            let mut chunk_freq_maps = Vec::new();

            for chunk_idx in 0..num_partitions {
                let mut kll = KLL::new(200);
                let mut reservoir = ReservoirSampler::new(600);
                let mut preagg_kll = KLL::new(200);
                let mut preagg_reservoir = ReservoirSampler::new(600);

                let start = chunk_idx * samples_per_chunk;
                let end = start + samples_per_chunk;

                // Standard approach: feed all samples
                for &v in &all_data[start..end] {
                    kll.update(v);
                    reservoir.update(v);
                }

                // Preaggregation approach: aggregate locally first
                let mut chunk_freqs = HashMap::new();
                for &v in &all_data[start..end] {
                    *chunk_freqs.entry(v as u64).or_insert(0) += 1;
                }

                // Feed only unique values into sketch (treating each unique value once)
                for &unique_val in chunk_freqs.keys() {
                    preagg_kll.update(unique_val as f64);
                    preagg_reservoir.update(unique_val as f64);
                }

                kll_sketches.push(kll);
                res_sketches.push(reservoir);
                preagg_kll_sketches.push(preagg_kll);
                preagg_res_sketches.push(preagg_reservoir);
                chunk_freq_maps.push(chunk_freqs);
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

            let mut merged_preagg_kll = preagg_kll_sketches.remove(0);
            for sketch in preagg_kll_sketches.drain(..) {
                merged_preagg_kll.merge(&sketch);
            }

            let mut merged_preagg_res = preagg_res_sketches.remove(0);
            for sketch in preagg_res_sketches.drain(..) {
                merged_preagg_res.merge(&sketch);
            }

            // Get boundaries from merged sketches
            let kll_cdf = merged_kll.cdf();
            let res_cdf = merged_res.cdf();
            let preagg_kll_cdf = merged_preagg_kll.cdf();
            let preagg_res_cdf = merged_preagg_res.cdf();

            let mut kll_boundaries = vec![f64::NEG_INFINITY];
            let mut res_boundaries = vec![f64::NEG_INFINITY];
            let mut preagg_kll_boundaries = vec![f64::NEG_INFINITY];
            let mut preagg_res_boundaries = vec![f64::NEG_INFINITY];

            for i in 1..num_partitions {
                let quantile = i as f64 / num_partitions as f64;
                kll_boundaries.push(kll_cdf.query(quantile));
                res_boundaries.push(res_cdf.query(quantile));
                preagg_kll_boundaries.push(preagg_kll_cdf.query(quantile));
                preagg_res_boundaries.push(preagg_res_cdf.query(quantile));
            }
            kll_boundaries.push(f64::INFINITY);
            res_boundaries.push(f64::INFINITY);
            preagg_kll_boundaries.push(f64::INFINITY);
            preagg_res_boundaries.push(f64::INFINITY);

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

            // For preaggregated approach, count aggregated entries per partition
            // Each chunk contributes its unique items independently (naive combination, no global dedup)
            let count_partition_preagg = |boundaries: &[f64], freq_maps: &[HashMap<u64, usize>]| -> Vec<usize> {
                let mut counts = vec![0; num_partitions];
                for freq_map in freq_maps {
                    for &val in freq_map.keys() {
                        for i in 0..num_partitions {
                            if val as f64 > boundaries[i] && val as f64 <= boundaries[i + 1] {
                                counts[i] += 1; // Count each aggregated entry (duplicates across chunks counted separately)
                                break;
                            }
                        }
                    }
                }
                counts
            };

            let kll_counts = count_partition(&kll_boundaries);
            let res_counts = count_partition(&res_boundaries);
            // For preaggregated: partition the naive combination of aggregated results
            let preagg_kll_counts = count_partition_preagg(&preagg_kll_boundaries, &chunk_freq_maps);
            let preagg_res_counts = count_partition_preagg(&preagg_res_boundaries, &chunk_freq_maps);

            // Standard approaches partition the original dataset
            let optimal_size = total_samples / num_partitions;

            // Preaggregated approaches partition aggregated entries (sum of unique items per chunk)
            // This counts duplicates across chunks separately (naive combination)
            let total_preagg_entries: usize = chunk_freq_maps.iter()
                .map(|freq_map| freq_map.len())
                .sum();
            let optimal_preagg_size = total_preagg_entries / num_partitions;

            // Also track globally unique items for reporting
            let mut global_unique_set = std::collections::HashSet::new();
            for freq_map in &chunk_freq_maps {
                for &val in freq_map.keys() {
                    global_unique_set.insert(val);
                }
            }
            let total_unique_items = global_unique_set.len();

            let calculate_stats = |counts: &[usize], opt_size: usize| -> (usize, usize, f64, f64) {
                let min = *counts.iter().min().unwrap();
                let max = *counts.iter().max().unwrap();
                let imbalance =
                    ((max as f64 - opt_size as f64) / opt_size as f64 * 100.0).abs();

                // Calculate variance
                let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
                let variance = counts
                    .iter()
                    .map(|&c| {
                        let diff = c as f64 - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / counts.len() as f64;

                (min, max, imbalance, variance)
            };

            let (kll_min, kll_max, kll_imbalance, kll_variance) = calculate_stats(&kll_counts, optimal_size);
            let (res_min, res_max, res_imbalance, res_variance) = calculate_stats(&res_counts, optimal_size);
            let (preagg_kll_min, preagg_kll_max, preagg_kll_imbalance, preagg_kll_variance) = calculate_stats(&preagg_kll_counts, optimal_preagg_size);
            let (preagg_res_min, preagg_res_max, preagg_res_imbalance, preagg_res_variance) = calculate_stats(&preagg_res_counts, optimal_preagg_size);

            // Output KLL stats
            println!(
                "{},{},KLL,{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{}",
                skew,
                trial,
                kll_min,
                kll_max,
                kll_imbalance,
                kll_variance,
                top1_pct,
                top2_pct,
                top3_pct,
                total_samples,
                total_unique_items,
                total_preagg_entries
            );

            // Output Reservoir stats
            println!(
                "{},{},RS,{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{}",
                skew,
                trial,
                res_min,
                res_max,
                res_imbalance,
                res_variance,
                top1_pct,
                top2_pct,
                top3_pct,
                total_samples,
                total_unique_items,
                total_preagg_entries
            );

            // Output Preagg KLL stats
            println!(
                "{},{},PREAGG_KLL,{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{}",
                skew,
                trial,
                preagg_kll_min,
                preagg_kll_max,
                preagg_kll_imbalance,
                preagg_kll_variance,
                top1_pct,
                top2_pct,
                top3_pct,
                total_samples,
                total_unique_items,
                total_preagg_entries
            );

            // Output Preagg Reservoir stats
            println!(
                "{},{},PREAGG_RS,{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{}",
                skew,
                trial,
                preagg_res_min,
                preagg_res_max,
                preagg_res_imbalance,
                preagg_res_variance,
                top1_pct,
                top2_pct,
                top3_pct,
                total_samples,
                total_unique_items,
                total_preagg_entries
            );

            eprintln!("Completed skew={}, trial={}", skew, trial);
        }
    }
}
