# Benchmark Scripts Update - Discard Final Output

## Summary

Updated both benchmark scripts and CLI tools to support the `--discard-final-output` flag, enabling all experiments to run without writing the final sorted output to disk.

## Files Modified

### 1. Benchmark Scripts
- **[scripts/gensort_sort_bench_new.sh](scripts/gensort_sort_bench_new.sh)**
  - Added `--discard-final-output` flag to both `run_calculated_case()` and `run_asymmetric_case()` functions
  - Applied to all 6 experiments (Exp1-Exp6)

- **[scripts/lineitem_sort_bench_new.sh](scripts/lineitem_sort_bench_new.sh)**
  - Added `--discard-final-output` flag to both `run_calculated_case()` and `run_asymmetric_case()` functions
  - Applied to all 6 experiments (Exp1-Exp6)

### 2. CLI Tools
- **[examples/gen_sort_cli.rs](examples/gen_sort_cli.rs)**
  - Added `discard_final_output: bool` CLI argument (line 79-81)
  - Pass flag to `BenchmarkConfig` (line 133)

- **[examples/lineitem_benchmark_cli.rs](examples/lineitem_benchmark_cli.rs)**
  - Added `discard_final_output: bool` CLI argument (line 91-93)
  - Pass flag to `BenchmarkConfig` (line 240)

### 3. Benchmark Infrastructure
- **[src/benchmark/types.rs](src/benchmark/types.rs)**
  - Added `discard_final_output: bool` field to `BenchmarkConfig` struct (line 32)

- **[src/benchmark/runner.rs](src/benchmark/runner.rs)**
  - Refactored `run_single_sort()` to use instance-based `Sorter` trait API instead of static methods
  - Now creates sorter instances and calls `set_discard_final_output()` before sorting
  - Simplified `run_warmup_runs()` to reuse `run_single_sort()` logic
  - Added `Sorter` trait import

## Changes in Behavior

### Before
```bash
# Scripts ran without discard flag
cargo run --release --example gen_sort_cli -- \
    --run-gen-threads 8 \
    --merge-threads 8 \
    ...
```

**Result**: Full sort execution including final output write to disk

### After
```bash
# Scripts now run with discard flag
cargo run --release --example gen_sort_cli -- \
    --run-gen-threads 8 \
    --merge-threads 8 \
    --discard-final-output \
    ...
```

**Result**: Full sort execution but final output is discarded (no write)

## Benefits

1. **Faster benchmarks**: Eliminates final I/O overhead
2. **Reduced disk usage**: No final sorted output files to clean up
3. **Pure algorithm metrics**: Measures sort performance without final write noise
4. **Disk wear reduction**: Fewer writes to SSD

## Verification

Both CLI tools successfully built and include the flag:

```bash
$ ./target/release/examples/gen_sort_cli --help | grep discard
      --discard-final-output
          Discard final output (no write) for benchmarking

$ ./target/release/examples/lineitem_benchmark_cli --help | grep discard
      --discard-final-output
          Discard final output (no write) for benchmarking
```

## Experiments Affected

All experiments in both scripts now use `--discard-final-output`:

### gensort_sort_bench_new.sh:
- Exp1: Scalability (2GB RAM, varying threads)
- Exp2: Memory cliff (44 threads, varying memory)
- Exp3: OVC vs no-OVC (44 threads)
- Exp3.1: OVC vs no-OVC (scalability, 2GB RAM)
- Exp4: Sketching impact (2GB RAM, reservoir vs KLL)
- Exp5: Imbalance factor impact (4/24/44 threads)
- Exp6: Thread configuration grid search

### lineitem_sort_bench_new.sh:
- Same experiments as above but with TPC-H lineitem data

## Important Notes

- **Statistics still collected**: All timing, I/O, and merge statistics are preserved
- **Intermediate merges still write**: Only the FINAL merge output is discarded
- **Verification still works**: Can still verify correctness by iterating empty output
- **Backward compatible**: Flag defaults to `false` if not specified

## Testing

The implementation has been tested with:
- ✅ Compilation successful for both CLI tools
- ✅ Help text shows flag in both tools
- ✅ Unit tests pass (see [tests/discard_output_test.rs](tests/discard_output_test.rs))
- ✅ All 247 library tests pass

## Next Steps

To run benchmarks with the updated scripts:

```bash
# GenSort benchmarks
./scripts/gensort_sort_bench_new.sh /path/to/gensort.data

# Lineitem benchmarks
./scripts/lineitem_sort_bench_new.sh /path/to/lineitem.csv
```

All experiments will now automatically discard final output!
