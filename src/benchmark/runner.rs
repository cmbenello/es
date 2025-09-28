use super::input::BenchmarkInputProvider;
use super::types::{BenchmarkConfig, BenchmarkResult, BenchmarkStats};
use super::verification::OutputVerifier;
use crate::sort_policy::{Policy, PolicyParameters, SortConfig, get_all_policies};
use crate::{
    ExternalSorter, ExternalSorterWithOVC, RunsOutput, RunsOutputWithOVC, SortInput, SortOutput,
    SortStats,
};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;

pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    input_provider: Box<dyn BenchmarkInputProvider>,
    verifier: Option<Box<dyn OutputVerifier>>,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig, input_provider: Box<dyn BenchmarkInputProvider>) -> Self {
        Self {
            config,
            input_provider,
            verifier: None,
        }
    }

    pub fn set_verifier(&mut self, verifier: Box<dyn OutputVerifier>) {
        self.verifier = Some(verifier);
    }

    pub fn run_benchmarks(&self) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
        // Get data size estimation for policy configuration
        let dataset_mb = self.input_provider.estimate_data_size_mb()?;

        // Get policies based on configuration
        let policies = get_all_policies(SortConfig {
            memory_mb: self.config.memory_mb as f64,
            dataset_mb,
            page_size_kb: 64.0,
            max_threads: self.config.threads as f64,
        });

        let mut all_results = Vec::new();

        // Print benchmark header
        self.print_benchmark_header(dataset_mb)?;

        // Run benchmarks for each policy
        for (policy, params) in policies {
            println!("Running benchmark for policy: {}", policy.name());
            println!("Parameters: {}", params);
            println!("{}", "=".repeat(80));

            // Perform warmup runs
            if self.config.warmup_runs > 0 {
                self.run_warmup_runs(&policy, &params)?;
            }

            // Run actual benchmark
            let result = self.run_policy_benchmark(&policy, &params)?;
            all_results.push(result);
        }

        Ok(all_results)
    }

    fn print_benchmark_header(&self, dataset_mb: f64) -> Result<(), Box<dyn std::error::Error>> {
        let total_entries = self.input_provider.get_entry_count();

        println!("\n=== BENCHMARK MODE ===");
        println!("{}", self.input_provider.get_description());
        if let Some(entries) = total_entries {
            println!("Total entries: {}", entries);
        }
        println!("Estimated data size: {:.2} MB", dataset_mb);
        println!("Threads: {}", self.config.threads);
        println!("Memory limit: {} MB", self.config.memory_mb);
        println!("OVC enabled: {}", self.config.ovc);
        println!("Temporary directory: {:?}", self.config.temp_dir);
        println!("Warmup runs: {}", self.config.warmup_runs);
        println!("Runs per configuration: {}", self.config.benchmark_runs);
        println!("Verify output: {}", self.config.verify);
        println!();

        Ok(())
    }

    fn run_warmup_runs(
        &self,
        _policy: &Policy,
        params: &PolicyParameters,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("  Performing {} warmup run(s)...", self.config.warmup_runs);

        for warmup in 1..=self.config.warmup_runs {
            print!("    Warmup {}/{}: ", warmup, self.config.warmup_runs);

            let temp_dir = self.config.temp_dir.join(format!(
                "warmup_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs()
            ));
            std::fs::create_dir_all(&temp_dir)?;

            let input = self.input_provider.create_sort_input()?;

            let (run_gen_stats, merge_stats) = if self.config.ovc {
                let (runs, sketch, run_gen_stats) = ExternalSorterWithOVC::run_generation(
                    input,
                    params.run_gen_threads as usize,
                    (params.run_size_mb * 1024.0 * 1024.0) as usize,
                    self.config.sketch_size,
                    self.config.sketch_sampling_interval,
                    self.config.run_indexing_interval,
                    &temp_dir,
                )?;

                let (_merged_runs, merge_stats) = ExternalSorterWithOVC::merge(
                    runs,
                    params.merge_threads as usize,
                    sketch,
                    &temp_dir,
                )?;
                drop(_merged_runs);

                (run_gen_stats, merge_stats)
            } else {
                let (runs, sketch, run_gen_stats) = ExternalSorter::run_generation(
                    input,
                    params.run_gen_threads as usize,
                    (params.run_size_mb * 1024.0 * 1024.0) as usize,
                    self.config.sketch_size,
                    self.config.sketch_sampling_interval,
                    self.config.run_indexing_interval,
                    &temp_dir,
                )?;

                let (_merged_runs, merge_stats) =
                    ExternalSorter::merge(runs, params.merge_threads as usize, sketch, &temp_dir, Some(self.config.boundary_imbalance_factor))?;
                drop(_merged_runs);

                (run_gen_stats, merge_stats)
            };

            println!(
                "{:.2}s",
                run_gen_stats.time_ms as f64 / 1000.0 + merge_stats.time_ms as f64 / 1000.0
            );

            // Clean up
            std::fs::remove_dir_all(&temp_dir)?;
            self.sync_filesystem()?;
        }

        println!("  Warmup complete.\n");
        Ok(())
    }

    fn run_policy_benchmark(
        &self,
        policy: &Policy,
        params: &PolicyParameters,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let mut accumulated_stats = BenchmarkStats::new();
        let mut valid_runs = 0;
        let mut actual_entries = self.input_provider.get_entry_count().unwrap_or(0);

        for run in 1..=self.config.benchmark_runs {
            print!("  Run {}/{}: ", run, self.config.benchmark_runs);

            let temp_dir = self.config.temp_dir.join(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs()
                    .to_string(),
            );
            std::fs::create_dir_all(&temp_dir)?;

            let input = self.input_provider.create_sort_input()?;

            let output = self.run_single_sort(input, params, &temp_dir)?;

            // Capture timing values before using output
            let run_gen_time_ms = output.stats().run_generation_time_ms.unwrap_or(0);
            let merge_time_ms = output.stats().merge_time_ms.unwrap_or(0);
            let merge_entry_sum = output.stats().merge_entry_num.iter().sum::<u64>() as usize;

            println!("{}", output.stats());

            // Accumulate statistics
            accumulated_stats.total_time +=
                run_gen_time_ms as f64 / 1000.0 + merge_time_ms as f64 / 1000.0;
            accumulated_stats.runs_count += output.stats().num_runs;

            if let Some(rg_time) = output.stats().run_generation_time_ms {
                accumulated_stats.run_gen_time += rg_time as f64 / 1000.0;
            }
            if let Some(m_time) = output.stats().merge_time_ms {
                accumulated_stats.merge_time += m_time as f64 / 1000.0;
            }

            // Accumulate I/O stats
            if let Some(ref io) = output.stats().run_generation_io_stats {
                accumulated_stats.run_gen_read_ops += io.read_ops;
                accumulated_stats.run_gen_read_mb += io.read_bytes as f64 / 1_000_000.0;
                accumulated_stats.run_gen_write_ops += io.write_ops;
                accumulated_stats.run_gen_write_mb += io.write_bytes as f64 / 1_000_000.0;
            }
            if let Some(ref io) = output.stats().merge_io_stats {
                accumulated_stats.merge_read_ops += io.read_ops;
                accumulated_stats.merge_read_mb += io.read_bytes as f64 / 1_000_000.0;
                accumulated_stats.merge_write_ops += io.write_ops;
                accumulated_stats.merge_write_mb += io.write_bytes as f64 / 1_000_000.0;
            }

            // Calculate imbalance factor
            if output.stats().merge_entry_num.len() > 1 {
                let min_entries = *output.stats().merge_entry_num.iter().min().unwrap_or(&0);
                let max_entries = *output.stats().merge_entry_num.iter().max().unwrap_or(&0);
                if min_entries > 0 {
                    let imbalance = max_entries as f64 / min_entries as f64;
                    accumulated_stats.imbalance_sum += imbalance;
                    accumulated_stats.imbalance_count += 1;
                }
            } else if output.stats().merge_entry_num.len() == 1 {
                accumulated_stats.imbalance_sum += 1.0;
                accumulated_stats.imbalance_count += 1;
            }

            valid_runs += 1;
            println!(
                "{:.2}s",
                run_gen_time_ms as f64 / 1000.0 + merge_time_ms as f64 / 1000.0
            );

            // Update actual entries from merge stats if not known
            if actual_entries == 0 {
                actual_entries = merge_entry_sum;
            }

            // Verify if requested (only on first run)
            if self.config.verify && run == 1 {
                if let Some(ref verifier) = self.verifier {
                    println!("    Verifying sorted output...");
                    verifier.verify(&*output)?;
                    println!("    Verification passed!");
                } else {
                    println!("    Warning: Verification requested but no verifier provided");
                }
            }

            drop(output); // Release resources
            println!();

            // Clean up
            std::fs::remove_dir_all(&temp_dir)?;
            self.sync_filesystem()?;
        }

        // Convert accumulated stats to final result
        let result = accumulated_stats.to_benchmark_result(
            policy.name(),
            &self.config,
            params.run_size_mb,
            params.run_gen_threads as usize,
            params.merge_threads as usize,
            actual_entries,
            valid_runs,
        );

        Ok(result)
    }

    fn run_single_sort(
        &self,
        input: Box<dyn SortInput>,
        params: &PolicyParameters,
        temp_dir: &Path,
    ) -> Result<Box<dyn SortOutput>, Box<dyn std::error::Error>> {
        let output = if self.config.ovc {
            let (runs, sketch, run_gen_stats) = ExternalSorterWithOVC::run_generation(
                input,
                params.run_gen_threads as usize,
                (params.run_size_mb * 1024.0 * 1024.0) as usize,
                self.config.sketch_size,
                self.config.sketch_sampling_interval,
                self.config.run_indexing_interval,
                temp_dir,
            )?;

            let (merged_runs, merge_stats) = ExternalSorterWithOVC::merge(
                runs,
                params.merge_threads as usize,
                sketch,
                temp_dir,
            )?;

            let stats = SortStats {
                num_runs: run_gen_stats.num_runs,
                runs_info: run_gen_stats.runs_info,
                run_generation_time_ms: Some(run_gen_stats.time_ms),
                merge_entry_num: merge_stats.merge_entry_num,
                merge_time_ms: Some(merge_stats.time_ms),
                run_generation_io_stats: run_gen_stats.io_stats,
                merge_io_stats: merge_stats.io_stats,
            };

            Box::new(RunsOutputWithOVC {
                runs: merged_runs,
                stats: stats.clone(),
            }) as Box<dyn SortOutput>
        } else {
            let (runs, sketch, run_gen_stats) = ExternalSorter::run_generation(
                input,
                params.run_gen_threads as usize,
                (params.run_size_mb * 1024.0 * 1024.0) as usize,
                self.config.sketch_size,
                self.config.sketch_sampling_interval,
                self.config.run_indexing_interval,
                temp_dir,
            )?;

            let (merged_runs, merge_stats) =
                ExternalSorter::merge(runs, params.merge_threads as usize, sketch, temp_dir, Some(self.config.boundary_imbalance_factor))?;

            let stats = SortStats {
                num_runs: run_gen_stats.num_runs,
                runs_info: run_gen_stats.runs_info,
                run_generation_time_ms: Some(run_gen_stats.time_ms),
                merge_entry_num: merge_stats.merge_entry_num,
                merge_time_ms: Some(merge_stats.time_ms),
                run_generation_io_stats: run_gen_stats.io_stats,
                merge_io_stats: merge_stats.io_stats,
            };

            Box::new(RunsOutput {
                runs: merged_runs,
                stats: stats.clone(),
            }) as Box<dyn SortOutput>
        };

        Ok(output)
    }


    fn sync_filesystem(&self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let dir_fd = File::open(&self.config.temp_dir).map_err(|e| {
                format!("Failed to open directory {:?}: {}", self.config.temp_dir, e)
            })?;
    
            #[cfg(target_os = "linux")]
            {
                libc::syncfs(dir_fd.as_raw_fd());
            }
    
            #[cfg(any(target_os = "macos", target_os = "freebsd"))]
            {
                libc::fsync(dir_fd.as_raw_fd());
            }
        }
        Ok(())
    }
}
