// ---------------------------------------------------------------------------
// Practical Resource-Configuration Planner (from paper §Resource Config.)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlannerRegime {
    /// T_max² ≤ 2·ρ²·M_eff²/(D·P) — symmetric T_max is single-step feasible.
    ThreadBound,
    /// T_max² > 2·ρ²·M_eff²/(D·P) — must reduce merge parallelism.
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
            imbalance_factor: 1.0,
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
/// * **Thread-bound** (`T_max² ≤ 2ρ²·M_eff²/(D·P)`): symmetric `T_gen = T_merge
///   = T_max`; choose the *smallest* run size (working-set) that keeps K=1.
/// * **Memory-bound** (`T_max² > 2ρ²·M_eff²/(D·P)`): preserve `T_gen = T_max`;
///   reduce `T_merge` until the hyperbola constraint is satisfied (with safety
///   slack); choose `run_size = M_eff/T_gen`.
///
/// **M_eff vs safety_slack**: `sparse_index_fraction` is subtracted from M
/// upfront to get M_eff = M × (1 − sparse_index_fraction), the effective
/// memory available for both run-generation buffers and merge I/O buffers.
/// `safety_slack` is an independent margin applied on top of M_eff to keep
/// run_size and T_merge safely below the K=1 single-step merge boundary.
///
/// In both regimes the merge fan-in is set to achieve a single merge step
/// (K=1) when the budget allows; otherwise the maximum budget-limited fan-in
/// is used and the engine performs multi-step merging automatically.
pub fn plan_resource_efficient(config: &PlannerConfig) -> PlannerResult {
    let d = config.dataset_mb;
    let m = config.memory_mb;
    let t_max = config.max_threads as f64;
    let p = config.page_size_kb / 1024.0; // KB → MB
    let rho = config.imbalance_factor;
    let slack = config.safety_slack;
    let sparse_reserve = config.sparse_index_fraction;

    // Effective memory after reserving the sparse-index fraction.
    // The sparse index permanently occupies sparse_index_fraction of M during
    // BOTH run generation and merging, so we subtract it once here and use
    // m_eff for all subsequent budget calculations.
    let m_eff = m * (1.0 - sparse_reserve);

    // Feasibility threshold: T_max² ≤ 2ρ²·M_eff²/(D·P)
    let threshold = 2.0 * rho * rho * m_eff * m_eff / (d * p);
    let regime = if t_max * t_max > threshold {
        PlannerRegime::MemoryBound
    } else {
        PlannerRegime::ThreadBound
    };

    // -----------------------------------------------------------------------
    // Phase 1: derive T_gen, T_merge, run_size_mb
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
    //   run_size_min = D·P / M_eff                  (theoretical K=1 floor)
    //   run_size = min(run_size_min × (1 + slack),  ← [slack] nudge above floor
    //                  (M_eff / T_gen) × (1 − slack)) ← [slack] cap below per-thread budget
    // -----------------------------------------------------------------------
    let (t_gen_f, t_merge_f, run_size_mb) = match regime {
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
            let run_size_min = d * p / m_eff; // theoretical K=1 floor
            let run_size = (run_size_min * (1.0 + slack)) // [slack] sits above floor
                .min((m_eff / t_gen) * (1.0 - slack)); // [slack] capped below per-thread budget
            (t_gen, t_merge, run_size)
        }
    };

    let t_gen = t_gen_f.round() as usize;
    let t_merge = t_merge_f.round() as usize;

    // -----------------------------------------------------------------------
    // Phase 2: derive merge_fanin
    //
    //   num_runs           = ceil(D / run_size_mb)
    //   fanin_single_step  = ceil(num_runs / T_merge)   target for K=1 merge
    //
    //   max_fanin_in_budget = floor(M_eff / (T_merge·P)) − 1
    //     - M_eff: sparse index fraction already excluded (see above)
    //     - ÷ (T_merge·P): total I/O pages available, split across merge threads
    //     - − 1: each thread needs 1 output buffer beyond its fanin input buffers,
    //            so (fanin + 1)·T_merge·P ≤ M_eff  ⟹  fanin ≤ floor(M_eff/(T_merge·P)) − 1
    //   No slack here — M_eff is already the hard physical limit for I/O buffers.
    //
    //   merge_fanin = min(fanin_single_step, max_fanin_in_budget)
    //     If budget allows K=1, use exactly fanin_single_step.
    //     Otherwise cap at max_fanin_in_budget; engine does multi-step merging automatically.
    // -----------------------------------------------------------------------
    let num_runs = ((d / run_size_mb).ceil() as usize).max(1);
    let fanin_for_single_step = num_runs.div_ceil(t_merge).max(2);
    let max_fanin_in_budget = ((m_eff / (t_merge as f64 * p)).floor() as usize)
        .saturating_sub(1) // reserve 1 output buffer per thread
        .max(2);
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
