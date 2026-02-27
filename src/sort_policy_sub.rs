// ---------------------------------------------------------------------------
// Practical Resource-Configuration Planner (from paper §Resource Config.)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlannerRegime {
    /// Analytical thread-bound side: T_max² ≤ 2·M_eff²/(D·P).
    /// K=1 is still checked against the merge engine's global fan-in budget.
    ThreadBound,
    /// Analytical memory-bound side: T_max² > 2·M_eff²/(D·P).
    MemoryBound,
}

const REPLACEMENT_SELECTION_RUN_EXPANSION: f64 = 2.0;

#[derive(Debug, Clone)]
pub struct PlannerConfig {
    /// Dataset size in MB (D)
    pub dataset_mb: f64,
    /// Available memory budget in MB (M, upper bound)
    pub memory_mb: f64,
    /// Maximum thread count (T_max)
    pub max_threads: usize,
    /// I/O page / buffer size in KB (P, default: 64)
    pub page_size_kb: f64,
    /// Safety margin for the K=1 single-step merge constraint.
    /// Applied to run_size and T_merge targets so they sit safely below their
    /// theoretical limits (avoids operating right on the feasibility boundary).
    /// (default: 0.05)
    pub safety_slack: f64,
    /// Fraction of M permanently occupied by the sparse index.
    /// Subtracted upfront: M_eff = M × (1 − sparse_index_fraction).
    /// M_eff is the effective memory budget used for BOTH run-generation
    /// buffers and merge I/O buffers — it is not a safety margin.
    /// (default: 0.05)
    pub sparse_index_fraction: f64,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            dataset_mb: 0.0,
            memory_mb: 0.0,
            max_threads: 1,
            page_size_kb: 64.0,
            safety_slack: 0.05,
            sparse_index_fraction: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlannerResult {
    pub run_gen_threads: usize,
    pub merge_threads: usize,
    /// Per-thread run-generation memory target (= run size) in MB
    pub rg_buf_mb: f64,
    /// Global merge fan-in for each merge operation
    pub merge_fanin: usize,
    pub regime: PlannerRegime,
    pub num_runs: usize,
    pub run_gen_memory_mb: f64,
    pub merge_memory_mb: f64,
    /// True when the computed fanin achieves a single merge step
    pub is_single_step: bool,
}

impl std::fmt::Display for PlannerResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let regime = match self.regime {
            PlannerRegime::ThreadBound => "thread-bound",
            PlannerRegime::MemoryBound => "memory-bound",
        };
        write!(
            f,
            "[Planner/{regime}] T_gen={rg}, T_merge={rm}, rg_buf={rs:.1} MB, \
             fanin={fi}, runs={nr}, run_gen_mem={rgm:.1} MB, merge_mem={mm:.1} MB, \
             single_step={ss}",
            regime = regime,
            rg = self.run_gen_threads,
            rm = self.merge_threads,
            rs = self.rg_buf_mb,
            fi = self.merge_fanin,
            nr = self.num_runs,
            rgm = self.run_gen_memory_mb,
            mm = self.merge_memory_mb,
            ss = self.is_single_step,
        )
    }
}

/// Derive resource-efficient sort parameters from the given budget tuple
/// `(D, M, T_max, P)` following the two-regime policy:
///
/// * **Thread-bound** (`T_max² ≤ 2·M_eff²/(D·P)`): symmetric `T_gen = T_merge
///   = T_max`; choose the *smallest* run size (working-set) that keeps K=1
///   under the global fan-in budget.
/// * **Memory-bound** (`T_max² > 2·M_eff²/(D·P)`): preserve `T_gen = T_max`;
///   reduce `T_merge` until the hyperbola constraint is satisfied (with safety
///   slack); choose `run_size = M_eff/T_gen`.
///
/// **M_eff vs safety_slack**: `sparse_index_fraction` is subtracted from M
/// upfront to get M_eff = M × (1 − sparse_index_fraction), the effective
/// memory available for both run-generation buffers and merge I/O buffers.
/// `safety_slack` is an independent margin applied on top of M_eff to keep
/// run_size and T_merge safely below the K=1 single-step merge boundary.
///
/// In both regimes the merge fan-in is set to the maximum value allowed by
/// the selected `T_merge` and the memory budget. Estimated run count is used
/// only to predict whether the merge will likely be single-step (`is_single_step`).
/// This avoids overfitting fan-in to estimated run count, which can differ from
/// actual runs (for example with replacement selection effects).
///
/// Note: `merge_fanin` is **global** fan-in per merge operation (the merge
/// engine receives one `fanin` value and merges up to that many runs in one
/// operation), not per-thread fan-in.
pub fn plan_resource_efficient(config: &PlannerConfig) -> PlannerResult {
    let d = config.dataset_mb;
    let m = config.memory_mb;
    let t_max = config.max_threads as f64;
    let p = config.page_size_kb / 1024.0; // KB → MB
    let slack = config.safety_slack;
    let sparse_reserve = config.sparse_index_fraction;

    // Effective memory after reserving the sparse-index fraction.
    // The sparse index permanently occupies sparse_index_fraction of M during
    // BOTH run generation and merging, so we subtract it once here and use
    // m_eff for all subsequent budget calculations.
    let m_eff = m * (1.0 - sparse_reserve);

    // Feasibility threshold: T_max² ≤ 2·M_eff²/(D·P)
    let threshold = 2.0 * m_eff * m_eff / (d * p);
    let regime = if t_max * t_max > threshold {
        PlannerRegime::MemoryBound
    } else {
        PlannerRegime::ThreadBound
    };

    let max_fanin_in_budget_for = |t_merge: usize| -> usize {
        ((m_eff / (t_merge as f64 * p)).floor() as usize)
            .saturating_sub(1) // reserve 1 output buffer per thread
            .max(2)
    };

    // -----------------------------------------------------------------------
    // Phase 1: derive T_gen, T_merge, rg_buf_mb
    //
    // All memory calculations use M_eff (sparse index already excluded).
    // safety_slack is an additional margin on top of M_eff to keep run_size
    // and T_merge away from the K=1 single-step merge boundary.
    //
    // Memory-bound regime (T_max² > threshold):
    //   T_gen  = T_max                              (no reduction — generation is cheap)
    //   T_merge = floor((threshold / T_gen) × (1 − slack))
    //             ^^^ [slack] keeps T_gen·T_merge safely below the K=1 hyperbola
    //   run_size = (M_eff / T_gen) × (1 − slack)
    //              ^^^ [slack] each thread fills ~95% of its effective memory share
    //
    // Thread-bound regime (T_max² ≤ threshold):
    //   T_gen = T_merge = T_max                     (symmetric; full parallelism)
    //
    //   Derivation of rg_buf_min (K=1 lower bound):
    //     num_runs ≈ D / (2·rg_buf_mb)
    //     K=1 requires num_runs ≤ merge_fanin
    //     merge_fanin ≤ max_fanin_in_budget
    //   therefore:
    //     D / (2·rg_buf_mb) ≤ max_fanin_in_budget
    //     rg_buf_mb ≥ D / (2·max_fanin_in_budget) = rg_buf_min
    //
    //   rg_buf_mb = min(rg_buf_min × (1 + slack),  ← [slack] nudge above floor
    //                  (M_eff / T_gen) × (1 − slack)) ← [slack] cap below per-thread budget
    // -----------------------------------------------------------------------
    let (t_gen_f, t_merge_f, rg_buf_mb) = match regime {
        PlannerRegime::MemoryBound => {
            let t_gen = t_max;
            let t_merge = ((threshold / t_gen) * (1.0 - slack)) // [slack] stays below hyperbola
                .floor()
                .clamp(1.0, t_max);
            let run_size = (m_eff / t_gen) * (1.0 - slack); // [slack] stays below per-thread budget
            (t_gen, t_merge, run_size)
        }
        PlannerRegime::ThreadBound => {
            let t_gen = t_max;
            let t_merge = t_max;
            let max_fanin_in_budget = max_fanin_in_budget_for(t_merge as usize);
            let rg_buf_min = d / (REPLACEMENT_SELECTION_RUN_EXPANSION * max_fanin_in_budget as f64); // global-fanin K=1 floor
            let run_size = (rg_buf_min * (1.0 + slack)) // [slack] sits above floor
                .min((m_eff / t_gen) * (1.0 - slack)); // [slack] capped below per-thread budget
            (t_gen, t_merge, run_size)
        }
    };

    let t_gen = t_gen_f.round() as usize;
    let t_merge = t_merge_f.round() as usize;

    // -----------------------------------------------------------------------
    // Phase 2: derive merge_fanin
    //
    //   num_runs           = ceil(D / (2·rg_buf_mb))
    //   required_fanin_K1  = num_runs
    //
    //   max_fanin_in_budget = floor(M_eff / (T_merge·P)) − 1
    //     - M_eff: sparse index fraction already excluded (see above)
    //     - ÷ (T_merge·P): total I/O pages available, split across merge threads
    //     - − 1: each thread needs 1 output buffer beyond its fanin input buffers,
    //            so (fanin + 1)·T_merge·P ≤ M_eff  ⟹  fanin ≤ floor(M_eff/(T_merge·P)) − 1
    //   No slack here — M_eff is already the hard physical limit for I/O buffers.
    //
    //   merge_fanin = max_fanin_in_budget
    //     Keep fan-in at the maximum budget-limited value for the chosen
    //     merge thread count; do not shrink it to estimated num_runs.
    //
    //   is_single_step = (num_runs <= merge_fanin)
    //     This remains an estimate because actual run count can differ.
    // -----------------------------------------------------------------------
    let estimated_run_length_mb =
        (rg_buf_mb * REPLACEMENT_SELECTION_RUN_EXPANSION).max(f64::EPSILON);
    let num_runs = ((d / estimated_run_length_mb).ceil() as usize).max(1);
    let max_fanin_in_budget = max_fanin_in_budget_for(t_merge);
    let merge_fanin = max_fanin_in_budget;
    let is_single_step = num_runs <= merge_fanin;

    let run_gen_memory_mb = t_gen as f64 * rg_buf_mb;
    let merge_memory_mb = t_merge as f64 * merge_fanin as f64 * p;

    PlannerResult {
        run_gen_threads: t_gen,
        merge_threads: t_merge,
        rg_buf_mb,
        merge_fanin,
        regime,
        num_runs,
        run_gen_memory_mb,
        merge_memory_mb,
        is_single_step,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_bound_uses_global_fanin_for_single_step() {
        let plan = plan_resource_efficient(&PlannerConfig {
            dataset_mb: 204_800.0, // 200 GiB
            memory_mb: 28_800.0,   // 28.8 GiB budget
            max_threads: 16,
            page_size_kb: 64.0,
            ..PlannerConfig::default()
        });

        assert_eq!(plan.regime, PlannerRegime::ThreadBound);
        assert!(plan.is_single_step);
        assert!(plan.num_runs <= plan.merge_fanin);
        assert!(plan.rg_buf_mb > 1.0);
    }

    #[test]
    fn single_step_flag_matches_global_fanin_condition() {
        let thread_bound = PlannerConfig {
            dataset_mb: 204_800.0,
            memory_mb: 28_800.0,
            max_threads: 16,
            page_size_kb: 64.0,
            ..PlannerConfig::default()
        };
        let memory_bound = PlannerConfig {
            dataset_mb: 204_800.0,
            memory_mb: 256.0,
            max_threads: 16,
            page_size_kb: 64.0,
            ..PlannerConfig::default()
        };

        for cfg in [thread_bound, memory_bound] {
            let plan = plan_resource_efficient(&cfg);
            assert_eq!(plan.is_single_step, plan.num_runs <= plan.merge_fanin);
        }
    }

    #[test]
    fn planner_keeps_merge_fanin_at_budget_max() {
        let cfg = PlannerConfig {
            dataset_mb: 204_800.0,
            memory_mb: 28_800.0,
            max_threads: 16,
            page_size_kb: 64.0,
            ..PlannerConfig::default()
        };
        let plan = plan_resource_efficient(&cfg);

        let p = cfg.page_size_kb / 1024.0;
        let m_eff = cfg.memory_mb * (1.0 - cfg.sparse_index_fraction);
        let expected_max_fanin = ((m_eff / (plan.merge_threads as f64 * p)).floor() as usize)
            .saturating_sub(1)
            .max(2);

        assert_eq!(plan.merge_fanin, expected_max_fanin);
    }

    #[test]
    fn estimated_runs_use_fixed_replacement_selection_expansion() {
        let cfg = PlannerConfig {
            dataset_mb: 10_000.0,
            memory_mb: 2_048.0,
            max_threads: 8,
            page_size_kb: 64.0,
            ..PlannerConfig::default()
        };
        let plan = plan_resource_efficient(&cfg);

        let expected_runs = (cfg.dataset_mb
            / (plan.rg_buf_mb * REPLACEMENT_SELECTION_RUN_EXPANSION))
            .ceil() as usize;
        assert_eq!(plan.num_runs, expected_runs.max(1));
    }
}
