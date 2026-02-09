# Merge Partitioning With Sparse Indexes

This document explains how merge range boundaries are computed in:

- `src/sort/core/engine.rs`
- `src/sort/core/run_format.rs`

## Sparse Index Format

Each physical run stores:

```text
IndexEntry {
  key: Vec<u8>,
  file_offset: usize,   // byte offset relative to that run start
}
```

`MultiSparseIndexes` is a zero-copy logical concatenation of one or more run indexes.

```text
segments
  seg0 run_id=20 entries: [ (a,0), (d,100), (f,200) ]
  seg1 run_id=21 entries: [ (g,0), (h,120) ]

logical view (global index)
  i=0  (a,20,0)
  i=1  (d,20,100)
  i=2  (f,20,200)
  i=3  (g,21,0)
  i=4  (h,21,120)
```

Ordering key is always a total order:

```text
(key, run_id, file_offset)
```

This is implemented via `cmp_key_run_offset`.

## Boundary Vector Shape

For `N` output partitions, boundary array length is `N + 1`:

```text
bounds[0]   = None           // -infinity
bounds[1]   = first split
...
bounds[N-1] = last split
bounds[N]   = None           // +infinity
```

Thread `t` merges:

```text
lower_inc = boundary(t)
upper_exc = boundary(t + 1)
```

## Partition Types

`merge_once_with_hooks` supports:

- `RangeOnly`: partition by key only. After selecting boundaries, `run_id` and `offset`
  are set to `0` when passed to `scan_range`.
- `RangeCnt`: partition by **entry count** (uses count-based boundary search).
- `RangeSize`: partition by **byte size** (uses size-based boundary search).

`RangeOnly` and `RangeCnt` both use the count-based search. `RangeSize` uses the size-based search.

Current implementation does not build a prefix array. Each thread computes its
two boundary targets directly from boundary index and imbalance factor:

```text
ratio(i) = None                                         if i == 0 or i == N
ratio(i) = heavy + (i - 1) * each                       otherwise

heavy = min(max(r, 1) / N, 1)
each  = (1 - heavy) / (N - 1)

target(i) = None                                         if ratio(i) is None
target(i) = floor(total * ratio(i))                      otherwise

thread t:
  lower_target = target(t)
  upper_target = target(t + 1)
  lower = boundary_from_target(lower_target)
  upper = boundary_from_target(upper_target)
```

## Boundary Target Math

`r` controls first-partition heaviness:

```text
heavy = min(max(r, 1) / N, 1)
each  = (1 - heavy) / (N - 1)
```

Interior boundary `i` (`1 <= i < N`) uses:

```text
ratio(i)  = heavy + (i - 1) * each
target(i) = floor(total * ratio(i))
```

Outside boundaries are open:

```text
target(0) = None
target(N) = None
```

`total` means entry count (`count mode`) or bytes (`size mode`).

Count mode uses **sparse index entry count**, not full record count. The total is:

```text
total_sparse = sum_k index[k].len()
```

If the run indexing interval is `N`, each sparse index entry represents roughly `N`
records, so count-mode targets are approximate when `N > 1`.

## Double Binary Search (Core Idea)

Both `select_boundary_by_count` and `select_boundary_by_size` use:

```text
outer loop (value domain):
  choose pivot from the midpoint of the largest active range

inner loop (index domain):
  for each sparse index k:
    pos[k] = upper_bound(index[k], pivot) in [lo[k], hi[k])
```

Pivot selection details:

- Count mode: largest range is `hi[k] - lo[k]` (entry count).
- Size mode: largest range is `bytes_before(index[k], hi[k]) - bytes_before(index[k], lo[k])`.
- The pivot is the midpoint *by index* in the selected range: `mid = lo[idx] + len / 2`.
- If there is a tie for largest range, we pick the lowest index `k` (deterministic).
- Size mode two-phase pivot:
  - Phase 1 (bytes): choose the largest byte range to move the byte objective quickly.
  - Phase 2 (count): if the byte phase stalls (no `lo/hi` movement), switch to
    largest entry range to shrink windows and bound the candidate set size.

Why "largest range" works well (intuition):

Count mode (entry target):

```text
Array A: 100000 entries (tiny values)
Array B: 2 entries

Largest entry range = Array A
Pivot comes from Array A, so each iteration cuts the 100000-entry range roughly in half.
This avoids wasting iterations on the 2-entry array.
```

Size mode (byte target):

```text
Array A: 100000 entries, total 1 MB
Array B: 2 entries, total 1 GB
Target: 500 MB

Largest byte range = Array B
Pivot comes from Array B, so the byte objective moves by ~GB/2 per step.
This converges quickly to the 500 MB boundary.
```

Size mode stall example (can still happen):

```text
index0: offsets [0, 1]        // 1 byte total
index1: offsets [0, 1000]     // 1000 bytes total
target_bytes = 10

Largest byte range = index1
pivot = midpoint of index1 -> entry at offset 1000

upper_bound(index0, pivot) = hi0
upper_bound(index1, pivot) = hi1
bytes_before(hi0) + bytes_before(hi1) >= target_bytes
=> choose "go left", but hi does not change => stall
```

Even with strictly monotonic offsets, byte targets can stall because the byte objective
can cross the target without shrinking any index range.

Update rule:

```text
if objective(pos) < target:
  lo = pos
else:
  hi = pos
```

Progress detection:

```text
made_progress = any(lo[k] changes) or any(hi[k] changes)
if !made_progress: break
```

In code this is tracked by comparing `lo[idx]` or `hi[idx]` against the new `pos[idx]`
inside the update loop. If every update writes back the same value, the iteration is
considered stalled.

Why we do not return the pivot after a stall:

- The pivot is a *sampling key* (midpoint of the largest active range), not guaranteed to be
  the smallest key whose global rank/bytes reaches the target.
- A stall means `lo/hi` did not change, so the pivot can be outside the true boundary.

Example (unique keys, size mode, includes run_id/offset):

```text
index0: [ (a,1,0), (b,1,1) ]
index1: [ (c,2,0), (d,2,1000) ]
target_bytes = 10

lo = [0, 0], hi = [2, 2]
largest byte range = index1 -> pivot = midpoint = (d,2,1000)

upper_bound(index0, pivot) = 2
upper_bound(index1, pivot) = 2
bytes = 1 + 1000 (>= target), hi = pos == hi (no change) => stall
```

Returning `pivot=(d,2,1000)` would be wrong because the smallest key with bytes >= 10
is `(c,2,0)`. The candidate scan over `[lo, hi)` finds that earliest boundary.

Vector vs scalar notes:

- `lo` and `hi` are `Vec<usize>` with one entry per sparse index.
- `pos` is also a `Vec<usize>` with the per-index upper_bound result.
- `mid` is a scalar, computed for the selected pivot range as `(lo[idx] + hi[idx]) / 2`.

Objective:

- Count mode: `sum(pos[k])`
- Size mode: `sum(bytes_before(index[k], pos[k]))`

Note: `pos[k]` is a sparse-index position. In count mode the objective and targets
are in **sparse entry units**; with indexing interval `N`, `pos[k] * N` is the
approximate record count.

Finalization:

1. Gather candidate entries from `lo[k]`.
2. Sort candidates by `(key, run_id, offset)`.
3. Return first candidate that satisfies objective `>= target`.

Candidate set size:

- If the loop terminates naturally with `hi[k] - lo[k] <= 1` for every `k`, then the
  candidate set is at most `K` entries (one per index).
- If we stall early, the candidate set can be larger: it is bounded by
  `sum_k (hi[k] - lo[k])`. Uniqueness does not prevent a stall in size mode, so there
  is no strict small bound in that case.

## Bug Found And Fixed

A correctness bug was found while adding complex tests (duplicates, range-partitioned runs, skewed sizes).

### Symptom

In some distributions, the outer loop stopped because an iteration made no progress:

```text
pos == lo  (or pos == hi) for every index
=> changed = false
=> loop stops
```

Then boundary finalization looked only at `lo[k]` candidates. That set can be too small, so it could:

- pick a later boundary than expected, or
- return `None` even though a valid boundary existed.

### Why It Happened

The unresolved search window after a stall is still `[lo[k], hi[k])`.
Using only `lo[k]` ignores other valid candidates inside that unresolved region.

### Fix

When a stall is detected, candidate expansion now uses the whole unresolved windows:

```text
for each index k:
  candidates += entries in [lo[k], hi[k])
```

Then we do the same final global check (sorted by `(key, run_id, offset)`) and choose the first candidate with objective `>= target`.

This fix is applied in both:

- `select_boundary_by_count`
- `select_boundary_by_size`

## Per-Thread Boundary Selection

Yes, each merge thread computes boundaries independently.

Implementation in `merge_once_with_hooks`:

1. For thread `t`, compute only two targets:
   - lower: boundary `t`
   - upper: boundary `t + 1`
2. Call boundary search twice (`count` mode or `size` mode), once per target.

No `Vec<Option<usize>>` target array is allocated anymore.

## Why "No Progress" Can Happen Even With Unique `(key, run_id, offset)`

Uniqueness alone does not guarantee shrinking in this outer loop because the pivot is a sampled existing key
(midpoint of the largest active range), not a numeric midpoint in value space. That pivot can still map to the
same boundary positions.

### Count Mode (Entry Target)

With strict uniqueness and largest‑range pivot selection, a stall should **not** happen in count mode:

- We always pick the midpoint of the largest active **entry** range.
- For that range, `upper_bound(pivot)` is strictly inside `(lo, hi]`, so either `lo` or `hi` must move.
- Therefore `made_progress` is guaranteed true each iteration.

If a stall occurs in count mode, it indicates one of:

- non‑unique compare keys (e.g., not using `(key, run_id, offset)`),
- corrupted index ordering,
- or a bug in `upper_bound`/range bounds.

### Size Mode (Byte Target)

Stalls can still happen in size mode, even with unique keys, because the decision is based on **bytes**, not index rank.

Two common patterns:

1. **Pivot at the edge but bytes already exceed the target**
   - `upper_bound` returns `hi` for every index
   - `bytes >= target` so we choose the "go left" update
   - `hi` does not change -> stall

2. **Pivot at the edge but bytes still below the target**
   - `upper_bound` returns `lo` for every index
   - `bytes < target` so we choose the "go right" update
   - `lo` does not change -> stall

This is possible when a single index has very large offset jumps or when byte ranges are heavily skewed, so the
byte objective flips without shrinking index ranges.

```text
pos[k] == lo[k] for all k
or
pos[k] == hi[k] for all k
```

so an update writes the same values back.

Example with unique values:

```text
index0: [ 9, 10 ]   // largest active range, midpoint is 10
index1: [ 1 ]
index2: [ 2 ]

pivot: 10 (global max)
upper_bound(index0, 10) = hi0
upper_bound(index1, 10) = hi1
upper_bound(index2, 10) = hi2
=> no interval shrinks on "hi = pos"
```

The algorithm now handles this safely by:

1. breaking if an iteration makes no progress
2. evaluating candidates from unresolved windows `[lo[k], hi[k])`

## Size Objective Detail

`bytes_before(index, pos)` uses sparse index file offsets:

```text
pos == 0           -> 0
pos >= len(index)  -> total_bytes(index)
otherwise          -> global_offset_at(pos)
```

`global_offset_at` includes segment base bytes for range-partitioned runs, so
size accumulation remains valid across multi-segment logical indexes.

## How MultiSparseIndexes Is Created

`MultiSparseIndexes` is built from one or more **segments**:

- Each segment corresponds to a single physical run and stores:
  - `run_id`
  - `entries` (`&[IndexEntry]`)
  - `total_bytes` for that run
  - `base_bytes` (prefix sum of prior segments)

Creation flow:

1. `MergeableRun::sparse_indexes()` builds a `Vec<SparseIndexSegment>`:
   - `Single` run: one segment.
   - `RangePartitioned` runs: one segment per run.
2. `MultiSparseIndexes::from_segments()`:
   - Drops empty segments.
   - Computes `segment_ends` (prefix sum of entry counts).
   - Computes `base_bytes` for each segment (prefix sum of bytes).
   - Sets `total_len` and `total_bytes`.

This yields a *logical* concatenation without copying `IndexEntry`.

## bytes_before and entry_before

**Entry count before** a position `pos` is just `pos` itself, because the logical
index space is the concatenation of entries.

**bytes_before(index, pos)** is computed as:

```text
pos == 0           -> 0
pos >= len(index)  -> total_bytes(index)
otherwise          -> global_offset_at(pos)
```

`global_offset_at(pos)` maps `pos` to `(segment.base_bytes + entry.file_offset)`,
so byte accumulation stays correct across multiple segments.
