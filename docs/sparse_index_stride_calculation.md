# Sparse Index Stride Calculation

This document explains how sparse-index sampling stride is computed today.

Relevant code paths:

- `src/sort/core/engine.rs`
- `src/sort/core/run_format.rs`
- `src/sort/plain/run.rs`
- `src/sort/ovc/run.rs`
- `src/replacement_selection/tol_mm.rs`
- `src/replacement_selection/tol_mm_ovc.rs`

## Overview

Stride is selected by mode:

- `CountBalanced` and `KeyOnly`: record-count stride (`IndexingInterval::Records`)
- `SizeBalanced`: byte stride (`IndexingInterval::Bytes`)

Then stride is adapted dynamically from:

- per-thread sparse-index byte budget (5% rule), and
- running average key/record size estimates.

The system does not enforce a hard stop when budget is reached. Budget is used
to pick sampling density.

## 1. Budget Assignment (5% Rule)

`INDEX_BUDGET_PCT = 5` in `engine.rs`.

### Run generation

Each worker thread gets an equal index budget:

```text
thread_index_budget = ceil(run_gen_mem * 0.05)
```

This is `~5% of run_gen_mem` per run-generation thread.

If input size estimate is available (`SortInput::estimated_size_bytes()`), each
thread also gets:

```text
estimated_thread_data_bytes = ceil(estimated_total_data_bytes / worker_count)
```

### Merge

Merge-side budget uses merge buffer memory:

```text
merge_memory_bytes = merge_threads * merge_fanin * DEFAULT_BUFFER_SIZE
total_index_budget = merge_memory_bytes * 0.05
thread_index_budget = ceil(total_index_budget / merge_threads)
estimated_thread_data_bytes = ceil(total_merged_run_bytes / merge_threads)
```

In the static `SorterCore::multi_merge(...)` API, base stride starts at:

- `1000 bytes` for `SizeBalanced`
- `1000 records` otherwise

and then receives the same budget/estimate fields.

Merge output runs are also bootstrapped from input-run sparse-index averages
before appending merged records:

- Each input run can expose `(avg_key_bytes, avg_record_bytes, sample_count)`.
- Merge combines them as weighted averages using `sample_count` as weight.
- The merged output run starts with this bootstrap so first stride decisions do
  not start from zero estimates.

## 2. Initial Average Size Bootstrap

Before run output starts, replacement selection performs initial fill. During
that phase it accumulates:

- `bootstrap_key_bytes += key_len`
- `bootstrap_record_bytes += 8 + key_len + value_len`
- `bootstrap_count += 1`

(`8` is key/value length headers.)

After fill:

```text
avg_key_bytes = bootstrap_key_bytes / bootstrap_count
avg_record_bytes = bootstrap_record_bytes / bootstrap_count
```

These are passed through `RunSink::set_sparse_index_bootstrap(...)` and applied
to each new run before appends begin.

## 3. Dynamic Stride Formula

Implemented in both `Run` and `RunWithOVC` (`dynamic_stride()`).

Given:

- `budget_bytes` = per-thread sparse-index budget
- `estimated_total_data_bytes` = per-thread total data estimate
- `avg_key_bytes`, `avg_record_bytes` = running averages
- `index_entry_overhead = size_of::<IndexEntry>()`

Compute:

```text
avg_index_entry_bytes = max(1, index_entry_overhead + avg_key_bytes)
target_samples = max(1, budget_bytes / avg_index_entry_bytes)
```

Then stride by mode:

- Record mode:

```text
estimated_records = max(1, ceil(estimated_total_data_bytes / avg_record_bytes))
stride_records = ceil(estimated_records / target_samples)
```

- Byte mode:

```text
stride_bytes = ceil(estimated_total_data_bytes / target_samples)
```

Final stride is clamped to at least `1`.

## 4. When Recalculation Happens

Sampling decision:

- always sample first record (`sparse_index.is_empty()`), then
- sample when current position reaches the next threshold:
  - record mode: `total_entries >= next_index_at_entry`
  - byte mode: `total_bytes >= next_index_at_bytes`

For every appended record (sampled or not), the run updates cumulative size
totals:

- `observed_key_bytes_total += key_len`
- `observed_record_bytes_total += record_bytes`
- `observed_record_count += 1`

Current averages used by `dynamic_stride()` are computed from:

- bootstrap estimate from initial fill, and
- these observed totals.

When a sample is taken:

1. push sparse index entry (`key`, `file_offset`)
2. recompute dynamic stride
3. set next threshold:
   - `next_index_at_entry = sampled_record_index + stride`
   - `next_index_at_bytes = sampled_file_offset + stride`

So stride is not globally fixed; it updates at each sample point.

## 5. Fallback Behavior

If budget or estimated total data is unavailable, dynamic stride returns `None`,
and the configured base interval is used unchanged.

All interval values are normalized with `.max(1)`.
