# Zero-Copy OVC Merge

## Overview

`ZeroCopyMergeWithOVC` performs k-way merge of sorted runs without allocating
`Vec<u8>` for values. Keys live in the loser tree; values flow directly from
each source's `AlignedReader` to the output `AlignedWriter`. Key buffers are
recycled through the tree so that after warm-up no allocations occur in the
merge hot path.

## On-Disk Record Format

Each run stores records sequentially in one of two forms:

```
Normal / Initial record:
+--------+-----------+-----------+---------------+-------+
| ovc 4B | key_len 4B| val_len 4B| truncated_key | value |
+--------+-----------+-----------+---------------+-------+

Duplicate-key record (ovc flag = DuplicateValue):
+--------+-----------+-------+
| ovc 4B | val_len 4B| value |
+--------+-----------+-------+
```

For normal records, `truncated_key` stores only the suffix bytes starting at
`ovc.offset()`. The full key is `prev_key[..offset] ++ truncated_key`. For
duplicate records, the key is identical to the previous record's key and no
key bytes are stored.

## Architecture

```
                        ZeroCopyMergeWithOVC
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │   ┌──────────────────────────────────────┐               │
  │   │        LoserTreeOVC<OVCKey32>        │               │
  │   │                                      │               │
  │   │  Each node stores:                   │               │
  │   │    - ovc:   OVCU32                   │               │
  │   │    - key:   Vec<u8>  (full key)      │               │
  │   │    - value_len: usize                │               │
  │   │                                      │               │
  │   │  NO value bytes in the tree.         │               │
  │   └──────────────────────────────────────┘               │
  │                                                          │
  │   spare_key: Vec<u8>   ← recycled key buffer             │
  │                                                          │
  │   sources: Vec<RunIteratorWithOVC>                       │
  │     ┌─────────┐ ┌─────────┐ ┌─────────┐                 │
  │     │ Source 0 │ │ Source 1 │ │ Source 2 │  ...           │
  │     │ (reader) │ │ (reader) │ │ (reader) │               │
  │     └─────────┘ └─────────┘ └─────────┘                 │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  Output Run   │
                  │ (AlignedWriter│)
                  └───────────────┘
```

### Key types

- **`OVCKey32`** — Tree entry holding `{ovc, key: Vec<u8>, value_len}`.
  No value bytes. Implements `OVCTreeKey` so the loser tree can compare
  keys using OVC-accelerated comparisons.

- **`MergeSource`** — Trait with three methods that split record reading
  into two phases:
  1. `next_key_into(&mut key_buf)` — reads header + key, stops before
     value bytes, returns `(ovc, value_len)`.
  2. `copy_value_to(writer, len)` — copies value bytes from source reader
     directly to the provided writer.
  3. `skip_value(len)` — seeks past value bytes without reading.

- **`RunIteratorWithOVC`** — Implements `MergeSource`. Wraps an
  `AlignedReader` positioned within a run's byte range.

## The Merge Loop

```
merge_into(output):
    loop:
        (winner, src_idx) = tree.peek()       // who has the smallest key?
        if no winner: break                    // all sources exhausted

        output.write_header_and_key(winner)    // OVC + key_len + val_len + key
        sources[src_idx].copy_value_to(output) // value bytes: reader → writer

        // advance the winning source
        inner loop:
            next = sources[src_idx].next_key_into(spare_key)
            if exhausted:
                tree.mark_current_exhausted()
                spare_key = old_winner.take_key()
                break
            if next is duplicate:
                output immediately, continue inner loop
            else:
                new_entry = OVCKey32(next_ovc, spare_key, val_len)
                old_winner = tree.push(new_entry)
                spare_key = old_winner.take_key()  // recycle!
                break
```

## Detailed Step-by-Step Example

Consider a 3-way merge with three source runs. After initialization,
the tree has read the first key from each source:

```
Sources (AlignedReaders positioned at value bytes of first record):

  Source 0:  "apple"  [val=100B]  "cherry" [val=80B]  ...
                      ▲ reader here

  Source 1:  "banana" [val=200B]  "date"   [val=50B]  ...
                      ▲ reader here

  Source 2:  "apricot"[val=150B]  "fig"    [val=90B]  ...
                      ▲ reader here

Tree (OVCKey32 entries):
  Winner = Source 0, key="apple", value_len=100
  Losers = [Source 2 "apricot", Source 1 "banana"]

spare_key = []  (empty after init)
```

### Iteration 1: emit "apple"

```
Step 1: Write header+key to output
  output ← [ovc|key_len|val_len|"apple"]

Step 2: Copy value directly from Source 0's reader to output writer
  Source 0 reader ────(100 bytes)────► output writer
  No Vec<u8> allocated for the value.

  Source 0:  "apple"  [val=100B]  "cherry" [val=80B]  ...
                                  ▲ reader now here (past value, at next record)

Step 3: Advance Source 0
  Read next key from Source 0 into spare_key → spare_key = "cherry", val_len=80
  Not a duplicate, so:
    new_entry = OVCKey32 { ovc, key: "cherry" (moved from spare_key), val_len: 80 }
    old_winner = tree.push(new_entry)   // tree returns old winner's OVCKey32
    spare_key = old_winner.take_key()   // recycle the old "apple" buffer
```

After iteration 1:

```
Tree:
  Winner = Source 2, key="apricot", value_len=150
  Losers = [Source 1 "banana", Source 0 "cherry"]

spare_key = "apple" buffer (capacity reused, content overwritten next time)

Output so far:
  [ovc|klen|vlen|"apple"|<100 value bytes>]
```

### Iteration 2: emit "apricot"

Same pattern: write header+key, copy value from Source 2, read next key
from Source 2 into spare_key, push to tree, recycle.

## Key Buffer Lifecycle

The critical optimization: key `Vec<u8>` buffers cycle through the system
without allocation after warm-up.

```
    spare_key                OVCKey32::new(...)              tree node
  ┌───────────┐   move    ┌─────────────────┐   push    ┌──────────────┐
  │ Vec<u8>   │ ────────► │ OVCKey32 {      │ ────────► │ tree[i].key  │
  │ (has cap) │           │   key: Vec<u8>  │           │ (stored)     │
  └───────────┘           └─────────────────┘           └──────────────┘
        ▲                                                      │
        │                     old winner                       │
        │                 ┌─────────────────┐                  │
        └──── take_key ◄──│ OVCKey32 {      │ ◄── tree.push() │
                          │   key: Vec<u8>  │     returns old  │
                          └─────────────────┘                  │

  After warm-up, all key buffers have sufficient capacity.
  resize() is a no-op (just adjusts len, no realloc).
```

In pseudocode:

```
spare_key.resize(new_key_len, 0)    // reuses existing capacity
reader.read_exact(&mut spare_key)   // fills buffer
new_entry = OVCKey32::new(ovc, mem::take(&mut spare_key), vlen)  // spare_key → tree
old_winner = tree.push(new_entry)   // tree returns evicted entry
spare_key = old_winner.take_key()   // recycle the buffer back
```

Total live key buffers = k (one per tree slot) + 1 (spare). No allocator
calls after the first pass through all sources.

## Duplicate Key Handling

When a source yields consecutive duplicate keys (OVC flag = `DuplicateValue`),
they are output immediately in an inner loop without pushing to the tree.
This is valid because duplicates from the same source are always adjacent in
sorted order.

```
outer loop:
    winner = tree.peek() → (key="dog", source=1)
    output header+key for "dog"
    copy value from source 1

    inner loop:
        read next from source 1 → "dog" (duplicate!)
        ┌──────────────────────────────────────────────────┐
        │ Swap OVC:                                        │
        │   swapped_ovc = tree.replace_top_ovc(dup_ovc)    │
        │   This gives us the correct output OVC for this  │
        │   duplicate relative to the previously emitted   │
        │   record in the merged output.                   │
        │                                                  │
        │ Output immediately:                              │
        │   output.append_header_and_key(swapped_ovc, key) │
        │   source[1].copy_value_to(output, val_len)       │
        └──────────────────────────────────────────────────┘
        read next from source 1 → "elephant" (not duplicate)
        push "elephant" to tree, recycle key buffer
        break inner loop

    continue outer loop (tree picks next winner)
```

This avoids pushing/popping the tree for consecutive duplicates, which is
important for workloads with many duplicate keys.

## Value Transfer: Reader to Writer

Values are never materialized into a `Vec<u8>`. `AlignedReader` implements
`BufRead`, so `copy_value_to` passes slices from the reader's internal 64KB
buffer directly to the writer -- no intermediate buffer at all:

```
Source AlignedReader                          Output AlignedWriter
┌──────────────────┐                         ┌──────────────────┐
│  64KB internal   │     &[u8] slice          │  64KB internal   │
│  buffer          │ ────────────────────────► │  buffer          │
│  [===========##] │  fill_buf() → write_all  │  [####========]  │
│       ▲ consumed │                          │                  │
└──────────────────┘                          └──────────────────┘
      refills from disk                            flushes to disk
      as slices are consumed                       as buffer fills
```

The loop:

```rust
while remaining > 0 {
    let buf = reader.fill_buf()?;       // borrow reader's internal buffer
    let n = buf.len().min(remaining);
    writer.write_all(&buf[..n])?;       // write slice directly to output
    reader.consume(n);                  // advance reader past consumed bytes
    remaining -= n;
}
```

Zero heap allocation, zero intermediate copies. Data moves from the
reader's page-aligned I/O buffer directly into the writer's I/O buffer
in up to 64KB slices.

## MergeSource Trait

```rust
pub trait MergeSource: Send {
    /// Read next record's header + key into `key` buffer.
    /// Returns (ovc, value_len). Reader stops at value start.
    fn next_key_into(&mut self, key: &mut Vec<u8>) -> Option<(OVCU32, usize)>;

    /// Copy value bytes directly from source reader to writer.
    fn copy_value_to(&mut self, writer: &mut dyn Write, len: usize);

    /// Skip value bytes (seek past without reading).
    fn skip_value(&mut self, len: usize);
}
```

The two-phase read design (`next_key_into` then `copy_value_to`) is the
key insight. After reading a key, the reader is positioned exactly at the
value bytes. The merge decides what to do with the value:

- **Winner**: `copy_value_to` → transfers to output writer.
- **Out of range** (during initial seek): `skip_value` → seeks past.

## Pipeline Integration

The zero-copy merge is wired into the external sort pipeline via the
`RunFormat::direct_merge` trait method:

```
merge_once_with_hooks (engine.rs)
  │
  ├── partition work across threads using sparse indexes
  ├── set_sparse_index_readers(run, thread_count)
  │
  └── per thread: merge_range (run_format.rs)
        │
        ├── create output run file
        │
        ├── F::direct_merge(runs, output, bounds, io_tracker)
        │     │
        │     ├── create RunIteratorWithOVC sources from each input run
        │     │     (uses sparse index to seek to start position)
        │     │
        │     ├── release_sparse_index() on all input runs
        │     │     (refcount-based: last thread to release frees memory)
        │     │
        │     └── ZeroCopyMergeWithOVC::new(sources).merge_into(output)
        │
        └── finalize output run
```

For non-OVC formats, `direct_merge` returns `false` and the pipeline
falls back to the iterator-based merge path.

## Memory Savings

### Before (iterator-based merge)

```
Per record in flight:
  - key:   Vec<u8>  (heap alloc per record)
  - value: Vec<u8>  (heap alloc per record)

Total live allocations ≈ k * 2 Vec<u8>  (in tree nodes)
                       + N * 2 Vec<u8>  (iterator yields fresh Vecs)
```

### After (zero-copy merge)

```
Per tree slot:
  - key: Vec<u8>  (recycled, no alloc after warm-up)
  - value_len: usize  (just a number, no data)

Total live allocations = k + 1 key buffers  (k tree slots + 1 spare)
                       + 0 value buffers     (values flow reader → writer)
                       + 1 stack [u8; 8192]  (transfer chunk)
```

For a 128-way merge with 1KB values, this eliminates ~128KB of live value
buffers and all per-record allocation overhead.

## Files

| File | Role |
|------|------|
| `src/ovc/offset_value_coding_32.rs` | `OVCKey32` type (key-only tree entry) |
| `src/ovc/tree_of_losers_ovc.rs` | `OVCTreeKey` impl for `OVCKey32` |
| `src/sort/ovc/merge.rs` | `MergeSource` trait, `ZeroCopyMergeWithOVC` |
| `src/sort/ovc/run.rs` | `MergeSource` impl for `RunIteratorWithOVC`, `append_header_and_key`, `writer()` |
| `src/sort/ovc/sorter.rs` | `direct_merge` impl for `OvcRunFormat` |
| `src/sort/core/run_format.rs` | `RunFormat::direct_merge` trait method, `merge_range` integration |
