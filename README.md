# CrocSort — Resource-Efficient, Skew-Resilient Parallel External Merge Sort

Research code for **CrocSort** (VLDB 2026): a parallel external merge sort in Rust that
completes under tight memory budgets where PostgreSQL, DuckDB, and ClickHouse abort —
running at worst ~50% faster than state-of-the-art systems on the configurations all
systems can complete, and finishing sort configurations existing systems cannot run at all.

> R. Otaki\*, C. Benello\*, F. Zhao, A. J. Elmore, G. Graefe.
> *CrocSort: Resource-Efficient, Skew-Resilient Parallel External Merge Sort.* VLDB 2026.
> (\*equal contribution)

A follow-up paper (under review) builds a lightweight model that predicts near-optimal
sort configurations (memory, phase-specific thread counts) on cloud and disaggregated
hardware, transferring calibration across machines to within 3–5% of optimal.

## What's here

- `src/` — the sorter: run generation and merge phases with jointly planned memory
  budgets and per-phase parallelism; skew-resilient partitioning; direct I/O by default
  (`--features buffered_io` switches to buffered I/O)
- `dataset_generator/` — input generation across skewed and uniform key distributions
- `planner/` — configuration-planning experiments (the follow-up paper's model)
- `benchmark_results/`, `docs/`, `examples/` — measurements, notes, and usage examples
- Comparison harnesses against DuckDB and friends (`duckdb_sort.sh`, plotting scripts)

## Building

```bash
cargo build --release
./target/release/es --help
```

Reads/writes Parquet via Arrow; benchmarks were run on Linux with direct I/O.

## Status

Active research code backing the paper — interfaces move; expect benchmark scripts and
experiment logs alongside the library. Questions welcome via issues.
