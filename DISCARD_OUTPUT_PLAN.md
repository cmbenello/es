# Plan: Add Discard Final Output Option

## Goal

Add configuration option to discard (not write) the final merged output while still executing the full merge algorithm and collecting statistics. This enables benchmarking the sort algorithm performance without final I/O overhead.

## Use Case

When benchmarking or testing sort performance, we want to measure:
- Run generation time
- Merge algorithm performance
- Memory usage and thread scaling

Without the cost of:
- Final output file I/O
- Disk space for final result

## Architecture

### Multi-Level Merge Structure

The external sort performs merges in levels:

```
Initial: [100 runs] (from run generation phase)
         ↓
Level 1: [100 runs] → merge (fanin=10) → [10 runs] ✅ MUST WRITE (input to next level)
         ↓
Level 2: [10 runs] → merge (fanin=10) → [1 run] ❌ CAN DISCARD (final output)
```

**Key Insight**: Only the FINAL merge can be discarded. All intermediate merges must write to disk because their output becomes input to the next merge level.

### Components to Add

#### 1. Configuration Flag

Add to `SorterCore`:
```rust
discard_final_output: bool
```

#### 2. Discard RunSink

New `RunSink` implementation that counts records but doesn't write:
```rust
pub struct DiscardRunSink {
    record_count: usize,
    sketch: Sketch<Vec<u8>>,
}
```

#### 3. Empty SortOutput

New `SortOutput` implementation for when output is discarded:
```rust
pub struct EmptySortOutput {
    stats: SortStats,
}
```

#### 4. Modified Merge Logic

Pass `discard_output` flag through merge functions:
- `multi_level_merge()` - detect final merge
- `merge_range()` - choose sink based on flag

## Implementation Steps

### Step 1: Add Configuration Flag
- File: `src/sort/plain/sorter.rs`
- Add `discard_final_output: bool` field to `SorterCore`
- Add `set_discard_final_output(&mut self, bool)` setter
- Initialize to `false` in constructor

### Step 2: Create DiscardRunSink
- File: `src/sort/run_sink.rs`
- Implement `RunSink` trait
- `push_record()` - increment counter, update sketch, no write
- `finalize()` - return empty marker

### Step 3: Create EmptySortOutput
- File: `src/lib.rs` or new file
- Implement `SortOutput` trait
- `iter()` - return empty iterator
- `stats()` - return statistics

### Step 4: Modify Merge Logic
- File: `src/sort/core/engine.rs`
- Add `discard_final_output` parameter to `multi_level_merge()`
- Pass `false` for intermediate merges (in loop)
- Pass flag for final merge (after loop)
- File: `src/sort/core/run_format.rs`
- Add `discard_output` parameter to `merge_range()`
- Choose `DiscardRunSink` vs `RunWriterSink` based on flag

### Step 5: Update Main Sort Method
- File: `src/sort/plain/sorter.rs`
- Pass `discard_final_output` to merge functions
- Return `EmptySortOutput` when flag is true

### Step 6: Update CLI and Tests
- File: `examples/gen_sort_cli.rs`
- Add command-line flag for discard mode
- Add basic tests to verify statistics still collected

## Benefits

- ✅ Measures pure algorithm performance
- ✅ Eliminates final I/O overhead in benchmarks
- ✅ Still collects full statistics
- ✅ Preserves all intermediate merge behavior
- ✅ Backward compatible (default: false)

## Non-Goals

- Does NOT keep final output in memory
- Does NOT skip intermediate merges
- Does NOT affect run generation phase
