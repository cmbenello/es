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
- `slack`: safety slack (`safety_slack`)
- `sparse_index_fraction`: reserved memory fraction for sparse index

Derived:

- `M_eff = M * (1 - sparse_index_fraction)`

## Step 1: Regime Detection

Compute:

- `threshold = 2 * M_eff^2 / (D * P)`

Regime:

- `ThreadBound` if `T_max^2 <= threshold`
- `MemoryBound` if `T_max^2 > threshold`

## Step 2: Choose `T_gen`, `T_merge`, and `rg_buf_mb`

`T_gen` is always set to `T_max`.

### Memory-bound

- `T_merge = floor((threshold / T_gen) * (1 - slack))`
- Clamp `T_merge` to `[1, T_max]`
- `rg_buf_mb = (M_eff / T_gen) * (1 - slack)`

### Thread-bound

- `T_merge = T_max`
- `max_fanin_in_budget = floor(M_eff / (T_merge * P)) - 1` (minimum 2)
- Replacement-selection estimate: `num_runs ≈ D / (2 * rg_buf_mb)`
- K=1 requirement: `num_runs <= merge_fanin`
- Budget cap: `merge_fanin <= max_fanin_in_budget`
- Sufficient K=1 condition under budget:
  - `D / (2 * rg_buf_mb) <= max_fanin_in_budget`
  - `rg_buf_mb >= D / (2 * max_fanin_in_budget)`
- Define:
  - `rg_buf_min = D / (2 * max_fanin_in_budget)`
- Final choice:
  - `rg_buf_mb = min(rg_buf_min * (1 + slack), (M_eff / T_gen) * (1 - slack))`

## Step 3: Choose `merge_fanin` and `single_step`

Compute:

- `num_runs = ceil(D / (2 * rg_buf_mb))`
- `required_fanin_k1 = num_runs`
- `max_fanin_in_budget = floor(M_eff / (T_merge * P)) - 1` (minimum 2)

Then:

- `merge_fanin = max_fanin_in_budget`
- `is_single_step = (num_runs <= merge_fanin)`

## Important Semantics

- `merge_fanin` is **global fan-in per merge operation**.
- It is **not** per-thread fan-in.
- `merge_fanin` is kept at the budget-limited maximum for the chosen
  `merge_threads`; it is not reduced to match estimated run count.
- Fixed `2.0` models classic replacement-selection behavior on random
  keys (average run length about 2x heap size).

This matches merge engine behavior (`fanin >= runs.len()` means one merge pass).
