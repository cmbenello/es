# Resource Planner Policy

This document describes how the planner in `src/sort_policy_sub.rs` chooses:

- `rg_buf_mb` (per-thread run-generation buffer target)
- `merge_fanin` (global fan-in per merge operation)
- `merge_threads`

## Inputs

Given:

- `D`: dataset size in MB (`dataset_mb`)
- `M`: memory budget in MB (`memory_mb`)
- `T_max`: max threads (`max_threads`)
- `P`: page size in MB (`page_size_kb / 1024`)
- `rho`: usable-memory ratio (fixed at `0.8` in planner)
- `sparse_index_fraction`: reserved memory fraction for sparse index
- `min_rg_buf_mb`: minimum per-thread run-generation buffer target (default `10 MB`)

Derived:

- `M_eff = (rho * M) * (1 - sparse_index_fraction)`

## Step 1: Regime Detection

Compute:

- `E = replacement-selection run-expansion assumption` (planner uses `E = 1`)
- `threshold = E * M_eff^2 / (D * P)`

Regime:

- `ThreadBound` if `T_max^2 <= threshold`
- `MemoryBound` if `T_max^2 > threshold`

## Simplified Heuristic (Worst-Case K=1)

Use this compact version when planning:

1. `T_merge = T_max`
2. `max_fanin_in_budget = floor(M_eff / (T_merge * P)) - 1`
3. `rg_buf_min = D / max_fanin_in_budget`  
   This is the K=1 bound under `E = 1`.
4. `rg_buf_target = max(min_rg_buf_mb, rg_buf_min)`
5. If `rg_buf_target > (M_eff / T_gen)`, reduce `T_gen` until it fits.
6. Keep `merge_fanin = max_fanin_in_budget`.

This keeps estimates simple and consistent with `E = 1`.

## Step 2: Choose `T_gen`, `T_merge`, and `rg_buf_mb`

### Memory-bound

- Start from `T_gen = T_max`
- If `(M_eff / T_gen) < min_rg_buf_mb`, shrink `T_gen` to:
  - `T_gen = floor(M_eff / min_rg_buf_mb)` (clamped to `[1, T_max]`)
- `T_merge = floor(threshold / T_gen)`
- Clamp `T_merge` to `[1, T_max]`
- `rg_buf_mb = M_eff / T_gen`

### Thread-bound

- Start from `T_gen = T_merge = T_max`
- If needed, shrink `T_gen` with the same floor rule as above so
  `rg_buf_mb` can reach at least `min_rg_buf_mb`
- `max_fanin_in_budget = floor(M_eff / (T_merge * P)) - 1` (minimum 2)
- Estimated runs use `num_runs ≈ D / rg_buf_mb` under `E = 1`.
- K=1 requirement: `num_runs <= merge_fanin`
- Budget cap: `merge_fanin <= max_fanin_in_budget`
- Sufficient K=1 condition under budget:
  - `D / rg_buf_mb <= max_fanin_in_budget`
  - `rg_buf_mb >= D / max_fanin_in_budget`
- Define:
  - `rg_buf_min = D / max_fanin_in_budget` (K=1 lower bound under `E = 1`)
- Final choice:
  - `rg_buf_target = max(min_rg_buf_mb, rg_buf_min)`
  - `rg_buf_mb = min(rg_buf_target, M_eff / T_gen)`

## Step 3: Choose `merge_fanin` and `single_step`

Compute:

- `num_runs = ceil(D / rg_buf_mb)`
- `required_fanin_k1 = num_runs`
- `max_fanin_in_budget = floor(M_eff / (T_merge * P)) - 1` (minimum 2)

Then:

- `merge_fanin = max_fanin_in_budget`
- `is_single_step = (num_runs <= merge_fanin)`

## Worked Example (Lineitem, 28.8 GB Budget)

Inputs from a real run setup:

- `D = 199659.46 MB`
- `M = 29491 MB`
- `T_max = 16`
- `P = 64 KiB = 0.0625 MB`
- `rho = 0.8` (fixed in planner)
- `sparse_index_fraction = 0.05`

Step-by-step:

1. Effective memory:
   - `M_eff = (rho * M) * (1 - sparse_index_fraction)`
   - `M_eff = (0.8 * 29491) * 0.95 = 22413.16 MB`
2. Threshold:
   - `threshold = E * M_eff^2 / (D * P)` with `E = 1`
   - `threshold = 40256.52`
   - `T_max^2 = 256 <= 40256.52`, so regime is `ThreadBound`.
3. Thread-bound settings:
   - `T_gen = T_merge = 16`
   - `max_fanin_in_budget = floor(22413.16 / (16 * 0.0625)) - 1 = 22412`
   - `rg_buf_min = D / max_fanin_in_budget = 8.9086 MB` (under `E = 1`)
   - `rg_buf_target = max(min_rg_buf_mb, rg_buf_min) = max(10.0, 8.9086) = 10.0 MB`
   - `rg_buf_mb = min(rg_buf_target, M_eff / T_gen) = min(10.0, 1400.82) = 10.0 MB`
4. Estimated runs and single-step flag:
   - `num_runs = ceil(D / rg_buf_mb) = ceil(199659.46 / 10.0) = 19966`
   - `merge_fanin = 22412`
   - `is_single_step = (19966 <= 22412) = true`

Rounded planner-style values for this example:

- `T_gen=16, T_merge=16, rg_buf=10.0 MB, fanin=22412, runs=19966, run_gen_mem=160.0 MB, merge_mem=22412.0 MB, single_step=true`

Observed value from one real benchmark run with the same input/budget:

- `actual generated runs = 42818` (higher than estimate; effective expansion in that run was about `D / (actual_runs * rg_buf_mb) ~= 1.21` instead of `2.0`)

## Worked Example (GenSort 200 GiB, 28.8 GB Budget)

Inputs:

- `D = 204800.00 MB` (200 GiB)
- `M = 29491 MB`
- `T_max = 16`
- `P = 64 KiB = 0.0625 MB`
- `rho = 0.8` (fixed in planner)
- `sparse_index_fraction = 0.05`

Step-by-step:

1. Effective memory:
   - `M_eff = (rho * M) * (1 - sparse_index_fraction)`
   - `M_eff = (0.8 * 29491) * 0.95 = 22413.16 MB`
2. Threshold:
   - `threshold = E * M_eff^2 / (D * P)` with `E = 1`
   - `threshold = 39246.07`
   - `T_max^2 = 256 <= 39246.07`, so regime is `ThreadBound`.
3. Thread-bound settings:
   - `T_gen = T_merge = 16`
   - `max_fanin_in_budget = floor(22413.16 / (16 * 0.0625)) - 1 = 22412`
   - `rg_buf_min = D / max_fanin_in_budget = 9.1380 MB` (under `E = 1`)
   - `rg_buf_target = max(min_rg_buf_mb, rg_buf_min) = max(10.0, 9.1380) = 10.0 MB`
   - `rg_buf_mb = min(rg_buf_target, M_eff / T_gen) = min(10.0, 1400.82) = 10.0 MB`
4. Estimated runs and single-step flag:
   - `num_runs = ceil(D / rg_buf_mb) = ceil(204800 / 10.0) = 20480`
   - `merge_fanin = 22412`
   - `is_single_step = (20480 <= 22412) = true`

Rounded planner-style values for this example:

- `T_gen=16, T_merge=16, rg_buf=10.0 MB, fanin=22412, runs=20480, run_gen_mem=160.0 MB, merge_mem=22412.0 MB, single_step=true`

Observed value from an actual GenSort benchmark log we have (same dataset/budget
family, but different older planner setting: `rg_buf=0.46 MB`):

- `actual generated runs = 419619` (with `rg_buf=0.46 MB`, implied effective expansion `D / (actual_runs * rg_buf_mb) ~= 1.06`)

Note:

- The exact worked-config (`rg_buf=10.0 MB`) actual run count is not recorded in
  this document yet; add it after running that exact configuration.

## Important Semantics

- `merge_fanin` is **global fan-in per merge operation**.
- It is **not** per-thread fan-in.
- `merge_fanin` is kept at the budget-limited maximum for the chosen
  `merge_threads`; it is not reduced to match estimated run count.
- Planner uses `E = 1` for the run estimate (`num_runs ≈ D / rg_buf_mb`).

This matches merge engine behavior (`fanin >= runs.len()` means one merge pass).
