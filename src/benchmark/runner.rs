use super::input::BenchmarkInputProvider;
use super::types::{BenchmarkConfig, BenchmarkResult};
use super::verification::OutputVerifier;
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

    /// New single-configuration benchmark entrypoint
    /// Runs warmups (if configured) and the requested number of runs using the provided params
    pub fn run_configuration(&self) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let dataset_mb = self.input_provider.estimate_data_size_mb()?;
        self.print_benchmark_header(dataset_mb)?;

        println!("Running benchmark for config: {}", self.config.config_name);
        println!(
            "Parameters: Run Gen Threads: {}, Run Size: {:.2} MB, Run Gen Memory: {:.1} MB, Merge Threads: {}, Merge Fanin: {}, Merge Memory: {:.1} MB, Imbalance Factor: {:.1}",
            self.config.run_gen_threads,
            self.config.run_size_mb,
            self.config.run_gen_memory_mb,
            self.config.merge_threads,
            self.config.merge_fanin,
            self.config.merge_memory_mb,
            self.config.imbalance_factor,
        );
        println!("{}", "=".repeat(80));

        if self.config.warmup_runs > 0 {
            self.run_warmup_runs()?;
        }

        let per_run_stats = self.run_policy_benchmark()?;
        Ok(BenchmarkResult {
            config: self.config.clone(),
            stats: per_run_stats,
        })
    }

    fn print_benchmark_header(&self, dataset_mb: f64) -> Result<(), Box<dyn std::error::Error>> {
        let total_entries = self.input_provider.get_entry_count();

        println!("\n=== BENCHMARK MODE ===");
        println!("{}", self.input_provider.get_description());
        if let Some(entries) = total_entries {
            println!("Total entries: {}", entries);
        }
        println!("Estimated data size: {:.2} MB", dataset_mb);
        println!("Run generation threads: {}", self.config.run_gen_threads);
        println!("Run size (MB): {:.2}", self.config.run_size_mb);
        println!(
            "Run generation memory (MB): {:.1}",
            self.config.run_gen_memory_mb
        );
        println!("Merge threads: {}", self.config.merge_threads);
        println!("Merge fan-in: {}", self.config.merge_fanin);
        println!("Merge memory (MB): {:.1}", self.config.merge_memory_mb);
        println!("OVC enabled: {}", self.config.ovc);
        println!("Temporary directory: {:?}", self.config.temp_dir);
        println!("Warmup runs: {}", self.config.warmup_runs);
        println!("Runs per configuration: {}", self.config.benchmark_runs);
        println!(
            "Cooldown between runs (s): {}",
            self.config.cooldown_seconds
        );
        println!("Verify output: {}", self.config.verify);
        println!();

        Ok(())
    }

    fn run_warmup_runs(&self) -> Result<(), Box<dyn std::error::Error>> {
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

            let (run_gen_stats, multi_merge_stats) = if self.config.ovc {
                let (runs, sketch, run_gen_stats) = ExternalSorterWithOVC::run_generation(
                    input,
                    self.config.run_gen_threads as usize,
                    (self.config.run_size_mb * 1024.0 * 1024.0) as usize,
                    self.config.sketch_size,
                    self.config.sketch_sampling_interval,
                    self.config.run_indexing_interval,
                    &temp_dir,
                )?;

                let (_merged_runs, multi_merge_stats) = ExternalSorterWithOVC::multi_merge(
                    runs,
                    self.config.merge_fanin,
                    self.config.merge_threads as usize,
                    &sketch,
                    self.config.imbalance_factor,
                    &temp_dir,
                )?;
                drop(_merged_runs);

                (run_gen_stats, multi_merge_stats)
            } else {
                let (runs, sketch, run_gen_stats) = ExternalSorter::run_generation(
                    input,
                    self.config.run_gen_threads as usize,
                    (self.config.run_size_mb * 1024.0 * 1024.0) as usize,
                    self.config.sketch_size,
                    self.config.sketch_sampling_interval,
                    self.config.run_indexing_interval,
                    &temp_dir,
                )?;

                let mergeable_runs: Vec<crate::sort::sorter::MergeableRun> = runs;

                let (_merged_run, per_merge_stats) = ExternalSorter::multi_merge(
                    mergeable_runs,
                    self.config.merge_fanin,
                    self.config.merge_threads as usize,
                    &sketch,
                    self.config.imbalance_factor,
                    &temp_dir,
                )?;
                drop(_merged_run);

                (run_gen_stats, per_merge_stats)
            };

            let total_merge_time_ms: u128 = multi_merge_stats.iter().map(|s| s.time_ms).sum();

            println!(
                "{:.2}s",
                run_gen_stats.time_ms as f64 / 1000.0 + total_merge_time_ms as f64 / 1000.0
            );

            // Clean up
            std::fs::remove_dir_all(&temp_dir)?;
            self.sync_filesystem()?;

            // Cooldown between warmup runs (except after the last warmup)
            if self.config.cooldown_seconds > 0 && warmup < self.config.warmup_runs {
                println!(
                    "    Cooling down for {}s before next warmup...",
                    self.config.cooldown_seconds
                );
                std::thread::sleep(std::time::Duration::from_secs(self.config.cooldown_seconds));
            }
        }

        println!("  Warmup complete.\n");
        Ok(())
    }

    fn run_policy_benchmark(&self) -> Result<Vec<SortStats>, Box<dyn std::error::Error>> {
        let mut per_run_stats: Vec<SortStats> = Vec::new();

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

            let output = self.run_single_sort(input, &temp_dir)?;

            println!("{}", output.stats());

            // Collect per-run SortStats and store
            let stats_this_run = output.stats();
            let run_gen_time_ms = stats_this_run.run_gen_stats.time_ms;
            let merge_time_ms: u128 = stats_this_run
                .per_merge_stats
                .iter()
                .map(|m| m.time_ms)
                .sum();
            per_run_stats.push(stats_this_run.clone());
            println!(
                "{:.2}s",
                run_gen_time_ms as f64 / 1000.0 + merge_time_ms as f64 / 1000.0
            );

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

            // Cooldown between benchmark runs (except after the last run)
            if self.config.cooldown_seconds > 0 && run < self.config.benchmark_runs {
                println!(
                    "  Cooling down for {}s before next run...",
                    self.config.cooldown_seconds
                );
                std::thread::sleep(std::time::Duration::from_secs(self.config.cooldown_seconds));
            }
        }

        Ok(per_run_stats)
    }

    fn run_single_sort(
        &self,
        input: Box<dyn SortInput>,
        temp_dir: &Path,
    ) -> Result<Box<dyn SortOutput>, Box<dyn std::error::Error>> {
        let output = if self.config.ovc {
            let (runs, sketch, run_gen_stats) = ExternalSorterWithOVC::run_generation(
                input,
                self.config.run_gen_threads as usize,
                (self.config.run_size_mb * 1024.0 * 1024.0) as usize,
                self.config.sketch_size,
                self.config.sketch_sampling_interval,
                self.config.run_indexing_interval,
                temp_dir,
            )?;

            let (merged_run, multi_merge_stats) = ExternalSorterWithOVC::multi_merge(
                runs,
                self.config.merge_fanin,
                self.config.merge_threads as usize,
                &sketch,
                self.config.imbalance_factor,
                temp_dir,
            )?;

            Box::new(RunsOutputWithOVC {
                run: merged_run,
                stats: SortStats::new(run_gen_stats, multi_merge_stats),
            }) as Box<dyn SortOutput>
        } else {
            let (runs, sketch, run_gen_stats) = ExternalSorter::run_generation(
                input,
                self.config.run_gen_threads as usize,
                (self.config.run_size_mb * 1024.0 * 1024.0) as usize,
                self.config.sketch_size,
                self.config.sketch_sampling_interval,
                self.config.run_indexing_interval,
                temp_dir,
            )?;

            let (merged_run, multi_merge_stats) = ExternalSorter::multi_merge(
                runs,
                self.config.merge_fanin,
                self.config.merge_threads as usize,
                &sketch,
                self.config.imbalance_factor,
                temp_dir,
            )?;

            Box::new(RunsOutput {
                run: merged_run,
                stats: SortStats::new(run_gen_stats, multi_merge_stats),
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
