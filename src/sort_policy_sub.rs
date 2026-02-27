// ---------------------------------------------------------------------------
// Practical Resource-Configuration Planner (from paper §Resource Config.)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlannerRegime {
    /// T_max² ≤ 2·ρ²·M²/(D·P) — symmetric T_max is single-step feasible.
    ThreadBound,
    /// T_max² > 2·ρ²·M²/(D·P) — must reduce merge parallelism.
    MemoryBound,
}

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
    /// Merge-partition imbalance factor ρ (default: 1.0)
    pub imbalance_factor: f64,
    /// Fractional safety slack applied to all resource targets (default: 0.05)
    pub safety_slack: f64,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            dataset_mb: 0.0,
            memory_mb: 0.0,
            max_threads: 1,
            page_size_kb: 64.0,
            imbalance_factor: 1.0,
            safety_slack: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlannerResult {
    pub run_gen_threads: usize,
    pub merge_threads: usize,
    /// Per-thread run-generation memory target (= run size) in MB
    pub run_size_mb: f64,
    /// Per-thread merge fan-in (set for K=1 single step when feasible)
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
            "[Planner/{regime}] T_gen={rg}, T_merge={rm}, run_size={rs:.1} MB, \
             fanin={fi}, runs={nr}, run_gen_mem={rgm:.1} MB, merge_mem={mm:.1} MB, \
             single_step={ss}",
            regime = regime,
            rg = self.run_gen_threads,
            rm = self.merge_threads,
            rs = self.run_size_mb,
            fi = self.merge_fanin,
            nr = self.num_runs,
            rgm = self.run_gen_memory_mb,
            mm = self.merge_memory_mb,
            ss = self.is_single_step,
        )
    }
}

/// Derive resource-efficient sort parameters from the given budget tuple
/// `(D, M, T_max, P, ρ)` following the two-regime policy:
///
/// * **Thread-bound** (`T_max² ≤ 2ρ²M²/(D·P)`): symmetric `T_gen = T_merge
///   = T_max`; choose the *smallest* run size (working-set) that keeps K=1.
/// * **Memory-bound** (`T_max² > 2ρ²M²/(D·P)`): preserve `T_gen = T_max`;
///   reduce `T_merge` until the hyperbola constraint is satisfied (with safety
///   slack); choose `run_size = M/T_gen`.
///
/// In both regimes the merge fan-in is set to achieve a single merge step
/// (K=1) when the memory budget allows; otherwise the maximum budget-limited
/// fan-in is used and the engine will perform multi-step merging automatically.
pub fn plan_resource_efficient(config: &PlannerConfig) -> PlannerResult {
    let d = config.dataset_mb;
    let m = config.memory_mb;
    let t_max = config.max_threads as f64;
    let p = config.page_size_kb / 1024.0; // KB → MB
    let rho = config.imbalance_factor;
    let slack = config.safety_slack;

    // Feasibility threshold: T_max² ≤ 2ρ²M²/(D·P)
    let threshold = 2.0 * rho * rho * m * m / (d * p);
    let regime = if t_max * t_max > threshold {
        PlannerRegime::MemoryBound
    } else {
        PlannerRegime::ThreadBound
    };

    let (t_gen_f, t_merge_f, run_size_mb) = match regime {
        PlannerRegime::MemoryBound => {
            // Preserve T_gen = T_max; reduce T_merge until
            // T_gen · T_merge ≤ threshold (with slack).
            let t_gen = t_max;
            let t_merge = ((threshold / t_gen) * (1.0 - slack))
                .floor()
                .clamp(1.0, t_max);
            // run_size = M / T_gen (use full budget per run-gen thread, less slack)
            let run_size = (m / t_gen) * (1.0 - slack);
            (t_gen, t_merge, run_size)
        }
        PlannerRegime::ThreadBound => {
            // Symmetric: T_gen = T_merge = T_max.
            // Minimum run size for K=1: D·P/M  (from merge_memory ≤ M).
            // Add slack so the estimate sits above the theoretical floor.
            let t_gen = t_max;
            let t_merge = t_max;
            let run_size_min = d * p / m;
            // Cap at M/T_gen so run-gen memory stays within budget.
            let run_size = (run_size_min * (1.0 + slack)).min((m / t_gen) * (1.0 - slack));
            (t_gen, t_merge, run_size)
        }
    };

    let t_gen = t_gen_f.round() as usize;
    let t_merge = t_merge_f.round() as usize;

    // Number of runs and per-thread merge fan-in for K=1.
    let num_runs = ((d / run_size_mb).ceil() as usize).max(1);
    let fanin_for_single_step = num_runs.div_ceil(t_merge).max(2);
    // Maximum fan-in that fits in the memory budget.
    let max_fanin_in_budget = ((m / (t_merge as f64 * p)).floor() as usize).max(2);
    let merge_fanin = fanin_for_single_step.min(max_fanin_in_budget);
    let is_single_step = merge_fanin >= fanin_for_single_step;

    let run_gen_memory_mb = t_gen as f64 * run_size_mb;
    let merge_memory_mb = t_merge as f64 * merge_fanin as f64 * p;

    PlannerResult {
        run_gen_threads: t_gen,
        merge_threads: t_merge,
        run_size_mb,
        merge_fanin,
        regime,
        num_runs,
        run_gen_memory_mb,
        merge_memory_mb,
        is_single_step,
    }
}
