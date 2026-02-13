# KVBin Dataset Generators

Three example binaries produce large synthetic datasets in the **KVBin** format for
benchmarking the external sorter.  All are fully parallelised and produce deterministic
output given the same `--seed`.

| Binary | Dataset name | Dup skew | Size skew | Isolates |
|---|---|---|---|---|
| `gen_freq_key_kvbin` | `freq_key` | Yes (50 % → one key) | No (uniform 512 B) | duplicate splitting only |
| `gen_heavy_key_kvbin` | `heavy_key` | Yes (10 % → one key) | Yes (32 KiB vs 128 B) | dup + size skew combined |
| `gen_heavy_range_kvbin` | `heavy_range` | No (rare dups) | Yes (32 KiB vs 128 B) | size skew only |

---

## KVBin binary format

Every record is a simple length-prefixed key/value pair, all integers little-endian:

```
┌──────────┬─────────────┬──────────┬─────────────────┐
│ klen: u32│ key (klen B)│ vlen: u32│ value (vlen B)  │
└──────────┴─────────────┴──────────┴─────────────────┘
```

Both generators always write an 8-byte `u64` key (`klen = 8`).

## Sparse index format (`.idx`)

A companion `.idx` file holds a flat array of `u64` little-endian byte offsets.  Entry *i*
is the byte offset in the `.kvbin` file at which the *i*-th ~4 MiB block begins.  This
allows O(1) random seeks without scanning the whole dataset.

```
.kvbin                         .idx
│ block 0  (≈4 MiB) │──────► offset_0 = 0
│ block 1  (≈4 MiB) │──────► offset_1
│ block 2  (≈4 MiB) │──────► offset_2
│  …                │         …
```

### How the generators create the index

Each generator uses a dedicated **writer thread** that receives `Block` structs (each ~4 MiB
of serialised records) from the worker threads over a bounded channel.  For every block the
writer receives it:

1. Appends the current cumulative byte offset (`bytes_written`) to the `.idx` file as a
   little-endian `u64`.
2. Writes the block's raw bytes to the `.kvbin` file.
3. Advances `bytes_written` by the block's length.

Because each worker flushes its buffer whenever it reaches the `INDEX_STRIDE_BYTES` threshold
(4 MiB), every index entry corresponds to roughly one 4 MiB block.  The index is written
incrementally — no second pass over the data is needed.

> **Note:** Block boundaries do not necessarily align with record boundaries; a single record
> is never split across blocks, but the last record in a block may push the block slightly
> past the 4 MiB target.  The index stores exact byte offsets, so readers can seek precisely.

---

## `gen_freq_key_kvbin` — frequency skew only (no size skew)

### Purpose

Isolates **duplicate splitting** as the sole partition-planning challenge.  Every record
carries the same payload size, so byte volume is directly proportional to row count.  This
lets Count-Balanced succeed — demonstrating that it correctly handles duplicate keys via the
virtual order `(key, run_id, offset)` — while Key-Only still fails.

### Skew model

| Row category | Key | Value size |
|---|---|---|
| Heavy-hitter (`dup_frac` of rows, default **50 %**) | `heavy_key` (default `0`) | `payload` (default **512 B**) |
| Normal (remaining rows) | Uniform random `u64` ≠ `heavy_key` | `payload` (default **512 B**) |

All records are the same size.  Byte-Balanced degenerates to Count-Balanced here since all
record weights are equal.

### Policy predictions

| Policy | Outcome | Reason |
|---|---|---|
| Key-Only | **FAIL** | Cannot split `heavy_key`; one partition absorbs 50 % of rows = 50 % of bytes |
| Count-Balanced | **WORKS** | Splits within `heavy_key` via virtual order; equal counts = equal bytes |
| Byte-Balanced | **WORKS** | Equivalent to Count-Balanced when all records are the same size |

### Default dataset (~200 GiB)

```
record = 4 + 8 + 4 + 512 = 528 B
rows   = 200 × 2^30 / 528 ≈ 406 720 387
```

### CLI reference

```
cargo run --release --example gen_freq_key_kvbin -- \
  --out      datasets/freq_key.kvbin       \
  --idx      datasets/freq_key.kvbin.idx   \
  --rows     406720387                     \
  --dup-frac 0.50                          \
  --heavy-key 0                            \
  --payload  512                           \
  --threads  0                             \   # 0 = use all logical CPUs
  --seed     1
```

| Flag | Default | Description |
|---|---|---|
| `--out` | *(required)* | Output `.kvbin` file path |
| `--idx` | *(required)* | Output `.idx` file path |
| `--rows` | `406 720 387` | Total records to write |
| `--dup-frac` | `0.50` | Fraction of rows that get `heavy_key` |
| `--heavy-key` | `0` | The duplicated key value |
| `--payload` | `512` | Value size (bytes) for every row |
| `--threads` | `0` (auto) | Worker thread count |
| `--seed` | `1` | PRNG seed (each thread derives its own stream) |

---

## `gen_heavy_key_kvbin` — duplicate heavy-hitter + size skew

### Purpose

Combines both failure modes in one dataset: one key dominates row count **and** that key's
records are 256× larger than all others.  Key-Only fails on the duplicate split; Count-Balanced
fixes that but still produces badly unequal byte volumes; only Byte-Balanced handles both.

Use this dataset to show Byte-Balanced's robustness when neither simpler policy suffices.

### Skew model

| Row category | Key | Value size |
|---|---|---|
| Heavy-hitter (`dup_frac` of rows, default **10 %**) | `heavy_key` (default `0`) | `heavy_payload` (default **32 KiB**) |
| Normal (remaining rows) | Uniform random `u64` ≠ `heavy_key` | `other_payload` (default **128 B**) |

### Policy predictions

| Policy | Outcome | Reason |
|---|---|---|
| Key-Only | **FAIL** | Cannot split `heavy_key`; one partition absorbs 10 % of rows ≈ 96.2 % of bytes |
| Count-Balanced | **FAIL** | Splits within `heavy_key` ✓ but equal row count still means 256× byte imbalance between heavy and normal partitions |
| Byte-Balanced | **WORKS** | Splits within `heavy_key` via virtual order and weights boundaries by record size |

### Default dataset (~200 GiB)

```
avg_record = 0.1 × (4 + 8 + 4 + 32 768) + 0.9 × (4 + 8 + 4 + 128)
           = 3 408 B
rows       = 200 × 2^30 / 3 408 ≈ 63 013 489
```

### CLI reference

```
cargo run --release --example gen_heavy_key_kvbin -- \
  --out   datasets/heavy_key.kvbin       \
  --idx   datasets/heavy_key.kvbin.idx   \
  --rows  63013489                     \
  --dup-frac      0.10                 \
  --heavy-key     0                    \
  --heavy-payload 32768                \
  --other-payload 128                  \
  --threads 0                          \   # 0 = use all logical CPUs
  --seed    1
```

| Flag | Default | Description |
|---|---|---|
| `--out` | *(required)* | Output `.kvbin` file path |
| `--idx` | *(required)* | Output `.idx` file path |
| `--rows` | `63 013 489` | Total records to write |
| `--dup-frac` | `0.10` | Fraction of rows that get `heavy_key` |
| `--heavy-key` | `0` | The duplicated key value |
| `--heavy-payload` | `32768` | Value size (bytes) for heavy-hitter rows |
| `--other-payload` | `128` | Value size (bytes) for normal rows |
| `--threads` | `0` (auto) | Worker thread count |
| `--seed` | `1` | PRNG seed (each thread derives its own stream) |

---

## `gen_heavy_range_kvbin` — hot key range + size skew

### Purpose

Models a workload where a small contiguous range of keys accounts for a disproportionate
share of total data volume, but individual keys within that range are not heavily
duplicated.  This stresses partition-size imbalance and skewed I/O without the
duplicate-key complexity.

### Skew model

| Row category | Key range | Value size |
|---|---|---|
| Hot (`hot_frac` of rows, default **20 %**) | Uniform in `[0, hot_range)` (default `[0, 65 535]`) | `hot_payload` (default **32 KiB**) |
| Cold (remaining rows) | Uniform in `[hot_range, u64::MAX]` | `cold_payload` (default **128 B**) |

Hot keys are sampled uniformly from the range `[0, hot_range)`.  Despite only 20 % of
**rows** being hot, they account for ~97 % of total **bytes** with the default payloads.

### Policy predictions

| Policy | Outcome | Reason |
|---|---|---|
| Key-Only | **FAIL** | Count-based boundaries put ~20 % of rows into hot partitions, but those hold ~97 % of bytes |
| Count-Balanced | **FAIL** | Same as Key-Only — rare duplicates mean tie-breaking adds no value; size skew still dominates |
| Byte-Balanced | **WORKS** | Targets equal byte volume; ~97 % of bytes fill ~97 % of partitions regardless of row count |

### Default dataset (~200 GiB)

```
avg_record = 0.2 × (4 + 8 + 4 + 32 768) + 0.8 × (4 + 8 + 4 + 128)
           = 6 672 B
rows       = 200 × 2^30 / 6 672 ≈ 32 186 505
```

### CLI reference

```
cargo run --release --example gen_heavy_range_kvbin -- \
  --out   datasets/heavy_range.kvbin       \
  --idx   datasets/heavy_range.kvbin.idx   \
  --rows  32186505                       \
  --hot-frac    0.20                     \
  --hot-payload 32768                    \
  --cold-payload 128                     \
  --hot-range   65536                    \
  --threads 0                            \
  --seed    1
```

| Flag | Default | Description |
|---|---|---|
| `--out` | *(required)* | Output `.kvbin` file path |
| `--idx` | *(required)* | Output `.idx` file path |
| `--rows` | `32 186 505` | Total records to write |
| `--hot-frac` | `0.20` | Fraction of rows drawn from the hot key range |
| `--hot-payload` | `32768` | Value size (bytes) for hot rows |
| `--cold-payload` | `128` | Value size (bytes) for cold rows |
| `--hot-range` | `65536` | Upper bound (exclusive) of the hot key range |
| `--threads` | `0` (auto) | Worker thread count |
| `--seed` | `1` | PRNG seed (each thread derives its own stream) |

---

## Architecture

Both generators share the same parallelism model:

```
main thread
│
├─ writer thread  ──── receives Block structs over a bounded channel, writes .kvbin + .idx
│
└─ N worker threads (thread::scope)
   │  each owns an independent SplitMix64 PRNG stream (seeded from global seed + thread_id)
   │  workers compete for row ranges via AtomicU64 (50 000 rows per chunk)
   └─ when buffer ≥ 4 MiB → send Block to writer
```

**Worker → writer back-pressure**: the channel capacity is `threads * 2`, so at most
`threads * 2 × ~4 MiB` of data can be buffered in-flight before workers stall, keeping
memory bounded.

**No zero-initialisation overhead**: payload buffers are extended with `reserve` +
`unsafe set_len` before `fill_payload` overwrites every byte, avoiding the double-write
that `Vec::resize` would cause at 32 KiB per hot record.

---

## Dataset comparison

| Property | `freq_key` | `heavy_key` | `heavy_range` |
|---|---|---|---|
| Key duplicates | Extreme (50 % → one key) | Extreme (10 % → one key) | Negligible (uniform in 65 k range) |
| Size skew | None (uniform 512 B) | Yes (32 KiB vs 128 B) | Yes (32 KiB vs 128 B) |
| Key-Only | FAIL (dup) | FAIL (dup + size) | FAIL (size) |
| Count-Balanced | **WORKS** | FAIL (size) | FAIL (size) |
| Byte-Balanced | **WORKS** | **WORKS** | **WORKS** |
| Primary stress | Duplicate splitting only | Dup splitting + size skew | Range-partition byte imbalance |
