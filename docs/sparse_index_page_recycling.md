# Sparse Index Page Recycling

This document explains how sparse index memory is reused between input runs
and the output runs of a merge step.

Relevant code paths:

- `src/sort/core/sparse_index.rs` — `SparseIndex`, `SparseIndexPage`, `SparseIndexPagePool`
- `src/sort/core/engine.rs` — `merge_once_with_hooks`
- `src/sort/core/run_format.rs` — `FormatSortHooks::merge_range`
- `src/sort/ovc/sorter.rs` — `OvcRunFormat::create_merge_sources_and_execute`

For how sparse-index sampling stride is chosen, see
`docs/sparse_index_stride_calculation.md`.

## Motivation

During merge, each input run carries a sparse index that maps sampled keys to
file offsets. The merge step uses these indexes for two purposes:

1. **Partition boundary computation** — each merge thread walks the combined
   sparse indexes to find its key-range boundaries.
2. **Seek** — each thread uses the sparse index of each input run to find the
   starting file offset for its assigned key range.

After seek, the input sparse indexes are no longer needed. Meanwhile, each
output run must build its own sparse index as it writes merged data. The old
approach freed the input indexes and allocated fresh memory for the outputs.
Page recycling eliminates that waste: it collects the 64KB pages from input
sparse indexes and redistributes them to output runs before the merge begins
writing.

## Page-Backed Sparse Index

Sparse index entries are no longer stored as `Vec<IndexEntry>` with
heap-allocated keys. Instead, entries are serialized into fixed-size 64KB
pages (`SparseIndexPage`).

### Entry layout (contiguous, never spans page boundaries)

```text
[file_offset: u64 (8 bytes)] [key_len: u16 (2 bytes)] [key: [u8; key_len]]
```

### Indexed access

A separate `pos: Vec<u32>` stores packed handles for O(1) entry lookup:

```text
handle = (page_idx << 16) | byte_offset
```

- **page_idx** (upper 16 bits): up to 65,536 pages = 4 GB of page storage
- **byte_offset** (lower 16 bits): up to 65,536 = full 64KB page

Given a handle, `key(i)` and `file_offset(i)` decode the handle and read
directly from the page bytes — no per-entry heap allocation.

## Recycling Protocol

```text
                          time ──────────────────────────────────►

  Thread 0 ┌─ partition ─┬─ seek ─┬─ release ─┬ barrier ┬─ take ─┬─ merge ──►
  Thread 1 ├─ partition ─┼─ seek ─┼─ release ─┤  wait   ├─ take ─┼─ merge ──►
  Thread 2 ├─ partition ─┼─ seek ─┼─ release ─┤         ├─ take ─┼─ merge ──►
  Thread 3 └─ partition ─┴─ seek ─┴─ release ─┘         └─ take ─┴─ merge ──►
                                                  ▲
                                         all pages now in pool
```

### Step by step

1. **Pre-merge setup** (`engine.rs: merge_once_with_hooks`)

   Before spawning merge threads, the engine:
   - Sets the sparse index reader count on each input run to `merge_threads`
     (atomic refcount).
   - Creates a shared `SparseIndexPagePool` (empty, `Mutex`-protected).
   - Creates a `Barrier(merge_threads)`.

2. **Partition boundary computation** (each thread)

   Each thread reads the combined sparse indexes (`MultiSparseIndexes`) to
   compute its `(lower_inc, upper_exc)` key-range bounds. This is read-only
   access to input sparse indexes.

3. **Seek / source creation** (each thread)

   Each thread calls `scan_range_as_source` (OVC path) or `scan_range` (plain
   path) on each input run for its key range. This binary-searches the sparse
   index to find the starting file offset, then opens an aligned reader at that
   position. After this step, the sparse index is no longer needed.

4. **Release pages to pool** (each thread)

   Each thread calls `release_sparse_index_to_pool(pool)` on each input run.
   This atomically decrements the run's refcount. The last thread to decrement
   (refcount reaches 0) drains the run's pages via `sparse_index.take_buffer()`
   and pushes them into the `SparseIndexPagePool`. Pages have their `used`
   counter reset to 0 so they can be rewritten.

   ```rust
   // In run.release_sparse_index_to_pool:
   let prev = refcount.fetch_sub(1, AcqRel);
   if prev == 1 {
       pool.return_pages(sparse_index.take_buffer());
   }
   ```

5. **Barrier** (all threads synchronize)

   `barrier.wait()` ensures all threads have finished releasing before any
   thread starts taking. After the barrier, the pool contains all recycled
   pages from every input run.

6. **Redistribute pages** (each thread)

   Each thread takes its fair share of pages from the pool:

   ```
   total      = pool.len()
   per_thread = total / merge_threads
   extra      = total % merge_threads
   my_count   = per_thread + (1 if thread_id < extra else 0)
   my_pages   = pool.take(my_count)
   ```

   The taken pages are seeded into the output run's sparse index via
   `seed_buffer(my_pages)`, which adds pre-allocated (but reset) pages to
   the output's page buffer.

7. **Merge** (each thread)

   The merge loop writes records to the output run. As the output's sparse
   index samples new entries, it writes them into the seeded pages. If the
   output exhausts its seeded pages, it allocates fresh pages from the heap
   (this is the normal `push()` overflow path — no failure, just a new
   `SparseIndexPage::new()`).

## Thread Safety

| Component | Synchronization | Reason |
|---|---|---|
| `SparseIndex` (per-run) | `UnsafeCell` + atomic refcount | Single writer during build; read-only during merge; last-releaser clears |
| `SparseIndexPagePool` | `Mutex<Vec<SparseIndexPage>>` | Multiple threads call `return_pages` and `take` concurrently |
| Reader count | `AtomicUsize` per run | Tracks how many threads still need the sparse index |
| Phase separation | `std::sync::Barrier` | Ensures all releases complete before any takes |

## Oversized Keys

Keys larger than 64KB minus 10 bytes of header cannot fit in a single page.
When `record_sample` encounters such a key, it truncates it to the maximum
length that fits. This is safe because sparse index keys are only used for
seek lower-bounding — a prefix is always a valid (conservative) lower bound.
