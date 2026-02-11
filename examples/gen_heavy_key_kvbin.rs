/// Generate a synthetic KVBin dataset with a **duplicate heavy-hitter + size skew**.
///
/// # Dataset characteristics
///
/// A configurable fraction of rows (`--dup-frac`, default 50 %) all share the same key
/// (`--heavy-key`, default 0) and carry a large payload (`--heavy-payload`, default 32 KiB).
/// Every other row gets a uniform-random `u64` key (guaranteed ≠ heavy-key) with a small
/// payload (`--other-payload`, default 128 B).
///
/// This dataset combines **both** failure modes for partition planners:
/// - Key-Only cannot split within the heavy key → catastrophic imbalance.
/// - Count-Balanced splits within the heavy key via virtual order `(key, run_id, offset)`,
///   but the 256× payload difference still causes severe byte imbalance.
/// - Byte-Balanced addresses both simultaneously.
///
/// Contrast with `freq_key` (uniform sizes, dup skew only) and `heavy_range` (size skew only).
///
/// Default parameters target ~200 GiB of raw data with ~13 M rows:
/// ```text
/// avg_record ≈ 0.5×(4+8+4+32768) + 0.5×(4+8+4+128) = 16 460 B
/// 200 GiB / 16 460 B ≈ 13 043 510 rows
/// ```
///
/// # KVBin record format
///
/// ```text
/// ┌──────────┬─────────────┬──────────┬─────────────────┐
/// │ klen: u32│ key (klen B)│ vlen: u32│ value (vlen B)  │
/// └──────────┴─────────────┴──────────┴─────────────────┘
/// ```
/// All integers are little-endian.  The key is always 8 bytes (a `u64`).
///
/// # Sparse index format (`.idx`)
///
/// A sequence of `u64` little-endian byte offsets.  Entry *i* holds the byte offset in the
/// `.kvbin` file at which the *i*-th ~4 MiB block starts, allowing O(1) random seeks.
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use clap::Parser;
use crossbeam::channel;

/// Approximate block size used both to flush worker buffers and to write one sparse-index entry.
const INDEX_STRIDE_BYTES: usize = 4 * 1024 * 1024;
/// Number of rows each worker grabs from the shared counter per iteration (reduces contention).
const CHUNK_ROWS: u64 = 50_000;

#[derive(Parser, Debug)]
struct Args {
    /// Output .kvbin path
    #[arg(long)]
    out: PathBuf,

    /// Output .idx path (u64 offsets, no keys)
    #[arg(long)]
    idx: PathBuf,

    /// Total rows to generate.
    /// Default targets ~200 GiB: avg_record ≈ 0.5×(4+8+4+32768) + 0.5×(4+8+4+128) = 16 460 B
    #[arg(long, default_value_t = 13_043_510)]
    rows: u64,

    /// Fraction of rows with the heavy-hitter key
    #[arg(long, default_value_t = 0.50)]
    dup_frac: f64,

    /// Heavy-hitter key value
    #[arg(long, default_value_t = 0)]
    heavy_key: u64,

    /// Payload size (bytes) for heavy-hitter rows
    #[arg(long, default_value_t = 32 * 1024)]
    heavy_payload: usize,

    /// Payload size (bytes) for normal rows
    #[arg(long, default_value_t = 128)]
    other_payload: usize,

    /// Worker threads (0 = auto)
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Seed for PRNG
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

/// Fast non-cryptographic PRNG (splitmix64).  Each thread seeds its own instance from the
/// global seed XOR a thread-specific offset so workers produce independent streams.
#[derive(Clone)]
struct SplitMix64 {
    x: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { x: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.x = self.x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Returns a uniform float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        let v = self.next_u64() >> 11;
        (v as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

/// Fill `buf` with deterministic pseudo-random bytes (avoids accidental compression artifacts).
fn fill_payload(buf: &mut [u8], rng: &mut SplitMix64) {
    let mut i = 0;
    while i + 8 <= buf.len() {
        let v = rng.next_u64().to_le_bytes();
        buf[i..i + 8].copy_from_slice(&v);
        i += 8;
    }
    if i < buf.len() {
        let v = rng.next_u64().to_le_bytes();
        let remain = buf.len() - i;
        buf[i..].copy_from_slice(&v[..remain]);
    }
}

#[derive(Clone, Copy)]
struct WorkerConfig {
    dup_frac: f64,
    heavy_key: u64,
    heavy_payload: usize,
    other_payload: usize,
    seed: u64,
}

struct Block {
    data: Vec<u8>,
    rows: u64,
}

fn resolve_threads(requested: usize) -> usize {
    if requested == 0 {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1)
    } else {
        requested.max(1)
    }
}

fn append_record(buf: &mut Vec<u8>, key: u64, payload_len: usize, rng: &mut SplitMix64) {
    let klen: u32 = 8;
    let vlen: u32 = payload_len as u32;

    buf.extend_from_slice(&klen.to_le_bytes());
    buf.extend_from_slice(&key.to_le_bytes());
    buf.extend_from_slice(&vlen.to_le_bytes());

    // Reserve space without zeroing — fill_payload writes every byte, so the zero-init
    // from resize() would be wasted work (noticeable at 32 KiB per hot record).
    let start = buf.len();
    buf.reserve(payload_len);
    // SAFETY: fill_payload unconditionally writes all bytes in the slice.
    unsafe { buf.set_len(start + payload_len) };
    fill_payload(&mut buf[start..], rng);
}

fn write_blocks(
    rx: channel::Receiver<Block>,
    out_path: PathBuf,
    idx_path: PathBuf,
) -> io::Result<u64> {
    let out_f = File::create(out_path)?;
    let mut out = BufWriter::with_capacity(8 * 1024 * 1024, out_f);

    let idx_f = File::create(idx_path)?;
    let mut idx = BufWriter::with_capacity(256 * 1024, idx_f);

    let mut bytes_written: u64 = 0;
    let mut rows_written: u64 = 0;

    for block in rx.iter() {
        idx.write_all(&bytes_written.to_le_bytes())?;
        out.write_all(&block.data)?;
        bytes_written += block.data.len() as u64;
        rows_written += block.rows;
    }

    out.flush()?;
    idx.flush()?;
    Ok(rows_written)
}

fn worker_loop(
    thread_id: usize,
    cfg: WorkerConfig,
    total_rows: u64,
    next_row: &AtomicU64,
    tx: channel::Sender<Block>,
) -> io::Result<()> {
    let mut rng = SplitMix64::new(
        cfg.seed
            .wrapping_add((thread_id as u64 + 1) * 0x9E3779B97F4A7C15),
    );
    let max_record = 4 + 8 + 4 + cfg.heavy_payload;
    let mut buf = Vec::with_capacity(INDEX_STRIDE_BYTES + max_record);
    let mut rows_in_buf: u64 = 0;

    loop {
        let start = next_row.fetch_add(CHUNK_ROWS, Ordering::Relaxed);
        if start >= total_rows {
            break;
        }
        let mut remaining = (total_rows - start).min(CHUNK_ROWS);

        while remaining > 0 {
            remaining -= 1;

            let is_dup = rng.next_f64() < cfg.dup_frac;
            let key: u64 = if is_dup {
                cfg.heavy_key
            } else {
                let mut k = rng.next_u64();
                if k == cfg.heavy_key {
                    k = k.wrapping_add(1);
                }
                k
            };
            let payload_len: usize = if is_dup {
                cfg.heavy_payload
            } else {
                cfg.other_payload
            };

            append_record(&mut buf, key, payload_len, &mut rng);
            rows_in_buf += 1;

            if buf.len() >= INDEX_STRIDE_BYTES {
                tx.send(Block {
                    data: std::mem::take(&mut buf),
                    rows: rows_in_buf,
                })
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "writer thread ended"))?;
                buf = Vec::with_capacity(INDEX_STRIDE_BYTES + max_record);
                rows_in_buf = 0;
            }
        }
    }

    if rows_in_buf > 0 {
        tx.send(Block {
            data: buf,
            rows: rows_in_buf,
        })
        .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "writer thread ended"))?;
    }

    Ok(())
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let threads = resolve_threads(args.threads);

    // Back-pressure: allow at most threads*2 blocks (~4 MiB each) in-flight before workers stall.
    let (tx, rx) = channel::bounded::<Block>(threads * 2);
    let writer_out = args.out.clone();
    let writer_idx = args.idx.clone();
    let writer_handle = thread::spawn(move || write_blocks(rx, writer_out, writer_idx));

    let next_row = AtomicU64::new(0);
    let cfg = WorkerConfig {
        dup_frac: args.dup_frac,
        heavy_key: args.heavy_key,
        heavy_payload: args.heavy_payload,
        other_payload: args.other_payload,
        seed: args.seed,
    };

    thread::scope(|s| -> io::Result<()> {
        let mut handles = Vec::with_capacity(threads);
        for thread_id in 0..threads {
            let tx = tx.clone();
            let next_row = &next_row;
            handles.push(s.spawn(move || worker_loop(thread_id, cfg, args.rows, next_row, tx)));
        }

        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "worker thread panicked",
                    ));
                }
            }
        }
        Ok(())
    })?;

    drop(tx);

    let rows_written = match writer_handle.join() {
        Ok(Ok(rows)) => rows,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "writer thread panicked",
            ));
        }
    };

    if rows_written != args.rows {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "row count mismatch: wrote {rows_written}, expected {}",
                args.rows
            ),
        ));
    }

    Ok(())
}
