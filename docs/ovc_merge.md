# OVC Multi-Merge Redesign Plan (Key-Only Heads, Lazy Value Read)

## 1) Problem Statement

The current OVC merge path materializes each record as owned `(OVCU32, Vec<u8>, Vec<u8>)`
while scanning, then wraps that into `OVCKeyValue32` for tree operations.
This causes:

1. Per-record allocation churn (key/value `Vec` in iterator hot path).
2. Eager value reads for records that are not emitted yet.
3. Higher memory pressure during high fan-in, multi-thread merge.

In high fan-in settings, this contributes to RSS growth significantly above the
intended merge memory target.

## 2) Existing OVC On-Disk Layout

OVC run records are encoded in two forms:

1. Duplicate key:
   - `ovc (4B)` + `value_len (4B)` + `value`
2. Non-duplicate key:
   - `ovc (4B)` + `truncated_key_len (4B)` + `value_len (4B)` + `truncated_key` + `value`

The duplicate form stores no key bytes.
The non-duplicate form stores only suffix bytes; prefix is implied by prior key in the same run.

## 3) Design Goals

1. Keep changes minimal and localized to OVC merge path first.
2. Avoid raw-pointer record APIs in tree nodes.
3. Avoid per-record key/value materialization in merge hot path.
4. Read value bytes only when a record is emitted (winner path).
5. Preserve correctness for duplicate/truncated-key OVC semantics.
6. Keep compatibility with current `AlignedReader` and `RunSink` APIs.

### Why not `OvcHead { ovc, key: Vec<u8>, value: Vec<u8> }` as-is?

A stream-local reusable head object with `Vec` buffers is already better than
per-record fresh allocation. However, eagerly reading `value` for every active
head still wastes memory bandwidth and memory footprint because only the winner
is emitted next.

Therefore this plan keeps key metadata in head state but defers value bytes
until winner emission.

## 4) Core Idea

Use per-run reusable key state and per-run head metadata.

- `last_key_by_run: Vec<Vec<u8>>` (length = K runs)
  - Stores last emitted full key for each run.
  - Reused and resized in place.
- `heads_by_run: Vec<OvcHeadMeta>` (length = K runs)
  - Stores metadata for current candidate record per run.
  - Caches only key suffix bytes and value location/length.
- Value is loaded only when that run wins and record is emitted.

This keeps merge comparison key-centric and lazily defers value I/O.

## 5) Proposed Per-Run Metadata

```rust
struct OvcHeadMeta {
    ovc: OVCU32,
    // Key representation:
    // full_key = last_key_by_run[run_id][..prefix_len] + suffix
    prefix_len: usize,
    suffix: Vec<u8>, // empty for duplicate
    is_duplicate: bool,

    // Value location:
    value_pos: u64,  // absolute file offset where value bytes start
    value_len: usize,

    // Stable tie-break / bounds:
    run_id: u32,
    record_offset: usize, // offset within run
    exhausted: bool,
}
```

Notes:
- `suffix` is reused per run (`clear`, `reserve`, `resize` as needed).
- `value_pos` allows deferred value read without keeping value in memory.

## 6) Merge Reader Behavior

### 6.1 `load_head(run_idx)`

`load_head` parses only the record header and key-related bytes:

1. Read `ovc`.
2. If duplicate:
   - read `value_len`
   - set `is_duplicate = true`, `prefix_len = last_key.len()`, `suffix = []`
   - set `value_pos = reader.position()`
3. Else:
   - read `truncated_key_len`
   - read `value_len`
   - read `truncated_key` into reusable `suffix`
   - set `prefix_len = ovc.offset()`
   - set `value_pos = reader.position()`

At this point, **value bytes are not read**.

Range/bound filtering:
- If record is outside `[lower_inc, upper_exc)`, skip value by seeking
  `value_pos + value_len`, and continue to next record.
- Otherwise, keep head active with reader parked at `value_pos`.

### 6.2 Page Boundary Handling

No special page-pointer API is required.
`AlignedReader::read_exact` already refills across internal 64KB buffers;
values spanning multiple pages are handled correctly.

## 7) Comparison Strategy

Tree comparisons use head metadata.

Fast path:
- compare `ovc` first.

Fallback path when OVC cannot decide:
- compare logical keys without full-file reads:
  - key view: `last_key[..prefix_len] + suffix`
- lexicographic compare over two segments.

Final tie-break:
- `(run_id, record_offset)` for deterministic ordering.

### Why this is safe

For each run, OVC encodes relation to previous key in that same run.
Keeping `last_key_by_run` (last emitted key per run) is sufficient to decode
duplicate/truncated-key representation for comparison and output reconstruction.

## 8) Emit Path (Winner Only)

When run `r` wins:

1. Reconstruct full key into `last_key_by_run[r]`:
   - duplicate: unchanged
   - non-duplicate:
     - `last_key.resize(prefix_len + suffix_len, 0)`
     - copy suffix into `[prefix_len..]`
2. Read value lazily:
   - `value_buf_by_run[r].resize(value_len, 0)`
   - `reader.read_exact(&mut value_buf[..])`
3. Emit:
   - `sink.push_record_with_ovc(ovc, &last_key_by_run[r], &value_buf_by_run[r])`
4. Advance stream:
   - immediately call `load_head(r)` to prepare next candidate.

This ensures value memory is used only on winner path.

## 9) Why Not Raw Pointers

Raw pointers in tree nodes increase unsafe/lifetime complexity and are not needed.
The minimal, robust design keeps:

- compact metadata in heads,
- owned reusable buffers per run,
- borrowing via slices only at compare/emit time.

This follows the spirit of replacement-selection memory optimization while
staying simpler for merge integration.

## 10) Integration Scope (Minimal First)

Phase 1 (OVC-only merge path):
- Introduce OVC merge cursor/head loader.
- Keep plain merge unchanged.
- Keep existing run generation path unchanged.

Phase 2:
- Wire new OVC merge into `RunFormat`/merge hooks.
- Retain old path behind a flag during validation.

Phase 3 (optional):
- Unify plain and OVC buffered merge architecture if desired.

## 11) Correctness Invariants

1. `last_key_by_run[r]` is always the last emitted full key for run `r`.
2. Active head for run `r` is decoded relative to `last_key_by_run[r]`.
3. Reader for run `r` is parked at current head's value start until emit/skip.
4. Deterministic total ordering uses `(logical_key, run_id, record_offset)`.

## 12) Expected Benefits

1. Eliminate per-record value materialization for non-winner candidates.
2. Remove most per-record heap churn in OVC merge.
3. Reduce peak merge RSS and allocator pressure.
4. Preserve OVC compare acceleration with key-aware fallback.

## 13) Test Plan

1. Duplicate-heavy runs:
   - verify ordering and OVC correctness.
2. Truncated-key runs:
   - verify logical comparison without full-key read from disk.
3. Large values (>64KB):
   - verify correctness across internal page boundaries.
4. Bounds filtering:
   - verify lower/upper handling with deferred value read.
5. Determinism:
   - repeated runs produce same order for equal keys.
6. Allocation profile:
   - compare allocation count and RSS against current path.

## 14) Open Decisions

1. Keep existing `LoserTreeOVC` with an adapter type vs. specialized merge tree
   operating directly on `OvcHeadMeta`.
2. Whether to mutate OVC metadata during fallback comparisons for additional speed.
3. Whether to add chunked value emission API in sink for very large values.
