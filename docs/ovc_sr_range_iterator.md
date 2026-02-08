# OVC_SR Range Iterator (RunWithOVCSR)

This document explains how range iteration works in `RunWithOVCSR` (OVC sparse-run format), including the `KeyRunIdOffsetBound` ordering and how the sparse index is used to find the start point.

**Overview**
- Each run has a `run_id` assigned at creation time.
- Each record is ordered by its file offset within the run.
- The iterator compares by a `KeyRunIdOffsetBound` tuple `(key, run_id, offset)`.
- Range iteration returns records whose keys satisfy: `lower_bound <= (key, run_id, offset) < upper_bound`.

**Bounds Format**
- Bounds are passed as `Option<(&[u8], u32, usize)>` (a `KeyRunIdOffsetBound` tuple):
  - `key: &[u8]`
  - `run_id: u32`
  - `offset: usize` (bytes from run start)
- Use `None` for unbounded sides.

**Sparse Index Entries**
- The sparse index stores entries at fixed intervals (`indexing_interval`).
- Each index entry stores:
  - `key` for the record at that index point.
  - `file_offset` pointing to that record, measured from the start of the run.
- The indexed record is a sync point for OVC decoding (the record can be decoded independently).

**Sparse Index In Practice (ASCII)**
Run file (logical record order, not bytes):
```
idx:    0    1    2    3    4    5    6    7    8    9   10   11   12
key:    a    a    a    b    b    c    c    c    d    e    e    f    g
offset: 0   12   24   36   48   60   72   84   96  108  120  132  144

indexing_interval = 4
```

Sparse index entries (every 4th record):
```
entry @ idx 0  -> key=a  file_offset=O0
entry @ idx 4  -> key=b  file_offset=O48
entry @ idx 8  -> key=d  file_offset=O96
entry @ idx 12 -> key=g  file_offset=O144
```

Example lower bound: `(key=c, offset=72)`
```
find_start_position selects the last index entry < lower_bound:
  (key=b) @ O48

iterator starts scanning at O48 and skips forward until >= (c,72)
```

**Iterator Start Positions And Terms**
- `lower_bound` (input): the user-supplied lower bound as `(key, run_id, offset)`.
- `start_key` (from sparse index): the key from the chosen sparse index entry (full key).
- `start_suffix` (from sparse index): computed from `run_id` + `start_offset` (run-relative).
- `start_offset` (from sparse index): run-relative offset of the indexed record.
- `aligned_offset` (I/O alignment): absolute file offset (page-aligned) used for direct I/O.
- `run_start` (run start): absolute file offset where this run begins (`start_bytes`).

**What Actually Happens At Start**
1. Binary search sparse index with `lower_bound`.
2. Pick the last entry with `entry < lower_bound` (by `KeyRunIdOffsetBound` order).
3. Seek to `aligned_offset` and skip `align_skip_bytes` to reach `start_offset`.
4. Initialize:
   - `prev_key = start_key`
5. Begin scanning forward in file order, comparing `(key, run_id, offset)` to bounds.

**Summary Of Key Roles**
- `lower_bound`: the logical start of the scan (KeyRunIdOffsetBound order).
- `start_key/start_suffix`: a *nearby* position used to begin scanning.
- `run_start`/`start_offset`: the byte position where scanning actually begins.
- The iterator **always** scans forward from the sparse start to reach the true lower bound.

**Finding The Start Position**
- `find_start_position(lower_bound)` performs a binary search over the sparse index.
- It returns the last entry with `entry < lower_bound` in KeyRunIdOffsetBound order.
- The iterator starts from that entry’s `file_offset`.
- This may start slightly before the true lower bound, but ensures the start is close and still correct.

**Iterator State**
- `run_id`: constant for the run.
- `current_offset`: derived from the current byte position (`file_pos - run_start`).
- `prev_key`: initialized to the sparse index key (full key).
- `file_pos`: tracks file position, with direct-I/O alignment handled by `align_down`.

**Range Scan Algorithm (High-Level)**
1. Read `lower_bound` and `upper_bound` `(key, run_id, offset)` tuples, if provided.
2. Seek to the sparse index entry chosen by `find_start_position`.
4. For each record in file order:
   - Decode the key (or reuse `prev_key` for duplicates).
   - Compare `(key, run_id, current_file_offset)` to bounds.
   - Emit if `>= lower_bound` and `< upper_bound`.
   - Offsets advance naturally as the reader moves forward.

**KeyRunIdOffsetBound Comparison Semantics**
- The comparison is lexicographic:
  - `(key_a, run_id_a, offset_a) < (key_b, run_id_b, offset_b)` if:
    - `key_a < key_b`, or
    - `key_a == key_b` and `(run_id_a, offset_a) < (run_id_b, offset_b)`
- This means a record is included if:
  - `key` is greater than the lower bound key, or
  - `key` equals the lower bound key and `(run_id, offset) >= lower`

**Duplicate Keys (OVC DuplicateValue)**
- Duplicate records do not store key bytes.
- The iterator uses `prev_key` as the current key.
- The current file offset is used as the offset component for comparisons.

**Invariants And Assumptions**
- Bounds are always `KeyRunIdOffsetBound` tuples (unless empty).
- Indexed records are OVC sync points so decoding at a sparse index entry is safe.
- The file offset is monotonic within each run.
- Suffix overflow is acceptable for your dataset sizes.
- Splitting between runs is acceptable and can be desirable for IO locality and thread focus.

**Why This Works**
- The `KeyRunIdOffsetBound` order creates a total order among duplicates.
- Sparse index reduces seek overhead while keeping correctness.
- Entry numbers ensure deterministic ordering within a run.
- Run IDs allow partitions to separate or group runs predictably.

**Practical Consequences**
- A partition may start or end between runs, which is intentional and helps keep per-thread IO localized.
- If a lower bound lands in the middle of a run, the iterator will skip until the bound is met using `KeyRunIdOffsetBound` comparisons.
