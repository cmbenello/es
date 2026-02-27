// ---------------------------------------------------------------------------
// Practical Resource-Configuration Planner (from paper §Resource Config.)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlannerRegime {
    /// Analytical thread-bound side:
    /// T_max² ≤ (RS expansion)·M_eff²/(D·P).
    /// K=1 is still checked against the merge engine's global fan-in budget.
    ThreadBound,
    /// Analytical memory-bound side:
    /// T_max² > (RS expansion)·M_eff²/(D·P).
    MemoryBound,
}

/// Fixed usable-memory ratio (rho) used by planner equations.
/// Matches the paper's implementation setting.
const PLANNER_RHO: f64 = 0.8;
/// Conservative RS expansion factor `E` used by planner estimates.
/// `E = 1.0` means no assumed expansion (worst-case for single-step feasibility).
const RS_EXPANSION_E: f64 = 1.0;

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
    /// Fraction of M permanently occupied by the sparse index.
    /// Subtracted after rho scaling:
    /// M_eff = (rho × M) × (1 − sparse_index_fraction).
    /// M_eff is the effective memory budget used for BOTH run-generation
    /// buffers and merge I/O buffers — it is not a safety margin.
    /// (default: 0.05)
    pub sparse_index_fraction: f64,
    /// Minimum per-thread run-generation buffer target (MB).
    /// Planner will reduce run-generation threads to honor this floor when
    /// possible (best-effort under extremely small memory budgets).
    /// (default: 40.0)
    pub min_rg_buf_mb: f64,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            dataset_mb: 0.0,
            memory_mb: 0.0,
            max_threads: 1,
            page_size_kb: 64.0,
            sparse_index_fraction: 0.05,
            min_rg_buf_mb: 40.0,
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
/// * **Thread-bound** (`T_max² ≤ E·M_eff²/(D·P)`, `E` = RS expansion):
///   symmetric `T_gen = T_merge
///   = T_max`; choose the *smallest* run size (working-set) that keeps K=1
///   under the global fan-in budget.
/// * **Memory-bound** (`T_max² > E·M_eff²/(D·P)`): start from `T_gen = T_max`,
///   then reduce `T_gen` if needed to satisfy `min_rg_buf_mb`; reduce
///   `T_merge` until the hyperbola constraint is satisfied.
///
/// **M_eff**: planner first applies a fixed usable-memory
/// ratio `rho=0.8`, then subtracts `sparse_index_fraction`:
/// `M_eff = (rho × M) × (1 − sparse_index_fraction)`.
/// This is the effective
/// memory available for both run-generation buffers and merge I/O buffers.
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
    let sparse_reserve = config.sparse_index_fraction;
    let min_rg_buf_mb = config.min_rg_buf_mb.max(0.0);

    // Effective memory:
    //   M_eff = (rho × M) × (1 - sparse_index_fraction)
    // We first apply fixed usable-memory ratio rho, then subtract sparse-index
    // reserve used across both run generation and merge phases.
    let m_eff = (m * PLANNER_RHO) * (1.0 - sparse_reserve);

    // Feasibility threshold:
    //   T_max² ≤ E·M_eff²/(D·P), where E is RS run-expansion assumption.
    let threshold = RS_EXPANSION_E * m_eff * m_eff / (d * p);
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
    let max_tgen_for_min_rg = |min_rg: f64| -> usize {
        if min_rg <= 0.0 {
            return config.max_threads.max(1);
        }
        // Per-thread cap is M_eff/T_gen in both regimes.
        let cap = (m_eff / min_rg).floor() as usize;
        cap.clamp(1, config.max_threads.max(1))
    };

    // -----------------------------------------------------------------------
    // Phase 1: derive T_gen, T_merge, rg_buf_mb
    //
    // All memory calculations use M_eff (sparse index already excluded).
    // Memory-bound regime (T_max² > threshold):
    //   Start from T_gen = T_max, but if that would make rg_buf_mb too small,
    //   reduce T_gen so rg_buf_mb reaches at least min_rg_buf_mb.
    //   T_merge = floor(threshold / T_gen)
    //   rg_buf_mb = max(min_rg_buf_mb, smallest feasible under T_gen)
    //              then capped by (M_eff / T_gen)
    //
    // Thread-bound regime (T_max² ≤ threshold):
    //   Start from T_gen = T_merge = T_max (symmetric), but if needed reduce
    //   T_gen to satisfy min_rg_buf_mb.
    //
    //   Derivation of rg_buf_min (K=1 lower bound, worst-case):
    //     num_runs ≈ D / rg_buf_mb     (worst-case, no expansion)
    //     K=1 requires num_runs ≤ merge_fanin
    //     merge_fanin ≤ max_fanin_in_budget
    //   therefore:
    //     D / rg_buf_mb ≤ max_fanin_in_budget
    //     rg_buf_mb ≥ D / max_fanin_in_budget = rg_buf_min
    //
    //   rg_buf_target = max(min_rg_buf_mb, rg_buf_min)
    //   rg_buf_mb     = min(rg_buf_target, (M_eff / T_gen))
    // -----------------------------------------------------------------------
    let (t_gen_f, t_merge_f, rg_buf_mb) = match regime {
        PlannerRegime::MemoryBound => {
            let t_gen = t_max.min(max_tgen_for_min_rg(min_rg_buf_mb) as f64);
            let t_merge = (threshold / t_gen).floor().clamp(1.0, t_max);
            let run_size_cap = m_eff / t_gen;
            // t_gen is already reduced (best-effort) to honor min_rg_buf_mb.
            // If memory is too small even at T_gen=1, this falls back to cap.
            let run_size = run_size_cap;
            (t_gen, t_merge, run_size)
        }
        PlannerRegime::ThreadBound => {
            let t_gen = t_max.min(max_tgen_for_min_rg(min_rg_buf_mb) as f64);
            let t_merge = t_max;
            let max_fanin_in_budget = max_fanin_in_budget_for(t_merge as usize);
            // Worst-case K=1 floor (no assumed RS expansion):
            // rg_buf_min_worst_case = D / max_fanin_in_budget
            let rg_buf_min_worst_case = d / (RS_EXPANSION_E * max_fanin_in_budget as f64);
            let run_size_target = rg_buf_min_worst_case.max(min_rg_buf_mb);
            let run_size = run_size_target.min(m_eff / t_gen); // capped below per-thread budget
            (t_gen, t_merge, run_size)
        }
    };

    let t_gen = t_gen_f.round() as usize;
    let t_merge = t_merge_f.round() as usize;

    // -----------------------------------------------------------------------
    // Phase 2: derive merge_fanin
    //
    //   num_runs           = ceil(D / rg_buf_mb)
    //   required_fanin_K1  = num_runs
    //
    //   max_fanin_in_budget = floor(M_eff / (T_merge·P)) − 1
    //     - M_eff: sparse index fraction already excluded (see above)
    //     - ÷ (T_merge·P): total I/O pages available, split across merge threads
    //     - − 1: each thread needs 1 output buffer beyond its fanin input buffers,
    //            so (fanin + 1)·T_merge·P ≤ M_eff  ⟹  fanin ≤ floor(M_eff/(T_merge·P)) − 1
    //   M_eff is the hard physical limit for I/O buffers.
    //
    //   merge_fanin = max_fanin_in_budget
    //     Keep fan-in at the maximum budget-limited value for the chosen
    //     merge thread count; do not shrink it to estimated num_runs.
    //
    //   is_single_step = (num_runs <= merge_fanin)
    //     This remains an estimate because actual run count can differ.
    // -----------------------------------------------------------------------
    let estimated_run_length_mb = (rg_buf_mb * RS_EXPANSION_E).max(f64::EPSILON);
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
        let m_eff = (cfg.memory_mb * PLANNER_RHO) * (1.0 - cfg.sparse_index_fraction);
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

        let expected_runs = (cfg.dataset_mb / (plan.rg_buf_mb * RS_EXPANSION_E)).ceil() as usize;
        assert_eq!(plan.num_runs, expected_runs.max(1));
    }

    #[test]
    fn memory_bound_shrinks_run_gen_threads_for_min_rg_buf() {
        let cfg = PlannerConfig {
            dataset_mb: 204_800.0,
            memory_mb: 200.0,
            max_threads: 16,
            page_size_kb: 64.0,
            min_rg_buf_mb: 40.0,
            ..PlannerConfig::default()
        };
        let plan = plan_resource_efficient(&cfg);

        assert_eq!(plan.regime, PlannerRegime::MemoryBound);
        assert!(plan.run_gen_threads < cfg.max_threads);
        assert!(plan.rg_buf_mb >= cfg.min_rg_buf_mb);
    }
}
