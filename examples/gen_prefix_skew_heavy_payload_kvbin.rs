use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;

const INDEX_STRIDE_BYTES: usize = 4 * 1024 * 1024;
const HEAVY_GROUPS: usize = 16;
const MEDIUM_GROUPS: usize = 1000;

const HEAVY_SHARE_NUM: u64 = 60;
const MEDIUM_SHARE_NUM: u64 = 30;

const HEAVY_PREFIX_LEN: usize = 200;
const HEAVY_SHARED_HEAD_LEN: usize = 192;
const HEAVY_TAIL_LEN: usize = 8;
const HEAVY_NEAR_DIFF_MIN: usize = 4;
const HEAVY_NEAR_DIFF_MAX: usize = 8;

const MEDIUM_PREFIX_MIN: usize = 128;
const MEDIUM_PREFIX_MAX: usize = 200;
const MEDIUM_SHARED_HEAD_LEN: usize = 96;
const MEDIUM_SUFFIX_MIN: usize = 8;
const MEDIUM_SUFFIX_MAX: usize = 40;

const RANDOM_KEY_LEN: usize = 256;

const HEAVY_KEY_LEN: usize = HEAVY_PREFIX_LEN + HEAVY_TAIL_LEN;
const MM_MAX_PAYLOAD_BYTES: usize = (64 * 1024) - 8; // extent minus header/footer
const MM_MANAGED_SLICE_OVERHEAD_BYTES: usize = 4; // key_len:u16 + value_len:u16
const MAX_SORTABLE_HEAVY_VALUE_BYTES: usize =
    MM_MAX_PAYLOAD_BYTES - MM_MANAGED_SLICE_OVERHEAD_BYTES - HEAVY_KEY_LEN;
const DEFAULT_HEAVY_PAYLOAD_MIN: usize = 8 * 1024;
const DEFAULT_HEAVY_PAYLOAD_MAX: usize = MAX_SORTABLE_HEAVY_VALUE_BYTES;
const MEDIUM_PAYLOAD_MIN: usize = 256;
const MEDIUM_PAYLOAD_MAX: usize = 2 * 1024;
const RANDOM_PAYLOAD_MIN: usize = 64;
const RANDOM_PAYLOAD_MAX: usize = 256;

const DEFAULT_BURST_START_PROB: f64 = 0.28;
const BURST_GROUPS_MIN: usize = 2;
const BURST_GROUPS_MAX: usize = 5;
const BURST_ROUNDS_MIN: usize = 2;
const BURST_ROUNDS_MAX: usize = 6;

#[derive(Parser, Debug)]
#[command(name = "gen_prefix_skew_heavy_payload_kvbin")]
#[command(about = "Generate PrefixSkew-HeavyPayload in KVBin format")]
struct Args {
    /// Output .kvbin path
    #[arg(long)]
    out: PathBuf,

    /// Output .idx path (u64 offsets, no keys)
    #[arg(long)]
    idx: PathBuf,

    /// Scale factor. Target bytes = sf * gib_per_sf * 2^30.
    /// Ignored when --rows is provided.
    #[arg(long, default_value_t = 1.0)]
    sf: f64,

    /// Dataset GiB represented by SF=1.0.
    #[arg(long, default_value_t = 1.0)]
    gib_per_sf: f64,

    /// Explicit row count override. If set, --sf is ignored.
    #[arg(long)]
    rows: Option<u64>,

    /// Heavy-group payload minimum (bytes).
    #[arg(long, default_value_t = DEFAULT_HEAVY_PAYLOAD_MIN)]
    heavy_payload_min: usize,

    /// Heavy-group payload maximum (bytes). Default is capped to keep records
    /// sortable by the current 64 KiB memory-manager extent model.
    #[arg(long, default_value_t = DEFAULT_HEAVY_PAYLOAD_MAX)]
    heavy_payload_max: usize,

    /// Probability of starting a short heavy-group burst at each row.
    #[arg(long, default_value_t = DEFAULT_BURST_START_PROB)]
    burst_start_prob: f64,

    /// Seed for deterministic generation.
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

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

    fn next_f64(&mut self) -> f64 {
        let v = self.next_u64() >> 11;
        (v as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn usize_bounded(&mut self, upper_exclusive: usize) -> usize {
        debug_assert!(upper_exclusive > 0);
        (self.next_u64() % (upper_exclusive as u64)) as usize
    }

    fn u64_bounded(&mut self, upper_exclusive: u64) -> u64 {
        debug_assert!(upper_exclusive > 0);
        self.next_u64() % upper_exclusive
    }

    fn usize_range_inclusive(&mut self, min: usize, max: usize) -> usize {
        debug_assert!(min <= max);
        min + self.usize_bounded(max - min + 1)
    }
}

#[derive(Clone)]
struct HeavyGroup {
    prefix: [u8; HEAVY_PREFIX_LEN],
    duplicate_tail: [u8; HEAVY_TAIL_LEN],
}

#[derive(Clone)]
struct MediumGroup {
    prefix: Vec<u8>,
}

#[derive(Default)]
struct BurstState {
    groups: Vec<usize>,
    cursor: usize,
    steps_left: usize,
}

impl BurstState {
    fn next_heavy_group(
        &mut self,
        heavy_dup_remaining: &[u64; HEAVY_GROUPS],
        heavy_near_remaining: &[u64; HEAVY_GROUPS],
    ) -> Option<usize> {
        while self.steps_left > 0 && !self.groups.is_empty() {
            let group = self.groups[self.cursor % self.groups.len()];
            self.cursor += 1;
            self.steps_left -= 1;
            if heavy_dup_remaining[group] + heavy_near_remaining[group] > 0 {
                return Some(group);
            }
        }
        self.groups.clear();
        self.cursor = 0;
        self.steps_left = 0;
        None
    }
}

fn fill_random_slice(buf: &mut [u8], rng: &mut SplitMix64) {
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

fn append_random_bytes(buf: &mut Vec<u8>, len: usize, rng: &mut SplitMix64) {
    let start = buf.len();
    buf.reserve(len);
    // SAFETY: fill_random_slice fully writes the range.
    unsafe { buf.set_len(start + len) };
    fill_random_slice(&mut buf[start..], rng);
}

fn append_heavy_record(
    buf: &mut Vec<u8>,
    rng: &mut SplitMix64,
    group: &HeavyGroup,
    duplicate: bool,
    heavy_payload_min: usize,
    heavy_payload_max: usize,
) {
    let payload_len = rng.usize_range_inclusive(heavy_payload_min, heavy_payload_max);
    let key_len = HEAVY_KEY_LEN;

    buf.extend_from_slice(&(key_len as u32).to_le_bytes());
    buf.extend_from_slice(&group.prefix);

    let tail_start = buf.len();
    buf.extend_from_slice(&group.duplicate_tail);
    if !duplicate {
        let diff_len = rng.usize_range_inclusive(HEAVY_NEAR_DIFF_MIN, HEAVY_NEAR_DIFF_MAX);
        let diff_start = tail_start + HEAVY_TAIL_LEN - diff_len;
        fill_random_slice(&mut buf[diff_start..tail_start + HEAVY_TAIL_LEN], rng);
        if buf[tail_start..tail_start + HEAVY_TAIL_LEN] == group.duplicate_tail {
            buf[tail_start + HEAVY_TAIL_LEN - 1] ^= 1;
        }
    }

    buf.extend_from_slice(&(payload_len as u32).to_le_bytes());
    append_random_bytes(buf, payload_len, rng);
}

fn append_medium_record(buf: &mut Vec<u8>, rng: &mut SplitMix64, group: &MediumGroup) {
    let suffix_len = rng.usize_range_inclusive(MEDIUM_SUFFIX_MIN, MEDIUM_SUFFIX_MAX);
    let key_len = group.prefix.len() + suffix_len;
    let payload_len = rng.usize_range_inclusive(MEDIUM_PAYLOAD_MIN, MEDIUM_PAYLOAD_MAX);

    buf.extend_from_slice(&(key_len as u32).to_le_bytes());
    buf.extend_from_slice(&group.prefix);
    append_random_bytes(buf, suffix_len, rng);

    buf.extend_from_slice(&(payload_len as u32).to_le_bytes());
    append_random_bytes(buf, payload_len, rng);
}

fn append_random_record(buf: &mut Vec<u8>, rng: &mut SplitMix64) {
    let payload_len = rng.usize_range_inclusive(RANDOM_PAYLOAD_MIN, RANDOM_PAYLOAD_MAX);
    buf.extend_from_slice(&(RANDOM_KEY_LEN as u32).to_le_bytes());
    append_random_bytes(buf, RANDOM_KEY_LEN, rng);
    buf.extend_from_slice(&(payload_len as u32).to_le_bytes());
    append_random_bytes(buf, payload_len, rng);
}

fn fisher_yates_shuffle(values: &mut [usize], rng: &mut SplitMix64) {
    if values.len() <= 1 {
        return;
    }
    for i in (1..values.len()).rev() {
        let j = rng.usize_bounded(i + 1);
        values.swap(i, j);
    }
}

fn distribute_counts(total: u64, buckets: usize, rng: &mut SplitMix64) -> Vec<u64> {
    let mut counts = vec![total / buckets as u64; buckets];
    let extras = (total % buckets as u64) as usize;
    if extras == 0 {
        return counts;
    }

    let mut order: Vec<usize> = (0..buckets).collect();
    fisher_yates_shuffle(&mut order, rng);
    for i in 0..extras {
        counts[order[i]] += 1;
    }
    counts
}

fn init_heavy_groups(rng: &mut SplitMix64) -> [HeavyGroup; HEAVY_GROUPS] {
    let mut shared_head = [0u8; HEAVY_SHARED_HEAD_LEN];
    fill_random_slice(&mut shared_head, rng);

    std::array::from_fn(|group_id| {
        let mut prefix = [0u8; HEAVY_PREFIX_LEN];
        prefix[..HEAVY_SHARED_HEAD_LEN].copy_from_slice(&shared_head);
        let tag =
            (rng.next_u64() ^ (group_id as u64).wrapping_mul(0x9E3779B97F4A7C15)).to_le_bytes();
        prefix[HEAVY_SHARED_HEAD_LEN..].copy_from_slice(&tag);

        let mut duplicate_tail = [0u8; HEAVY_TAIL_LEN];
        fill_random_slice(&mut duplicate_tail, rng);

        HeavyGroup {
            prefix,
            duplicate_tail,
        }
    })
}

fn init_medium_groups(rng: &mut SplitMix64) -> Vec<MediumGroup> {
    let mut shared_head = [0u8; MEDIUM_SHARED_HEAD_LEN];
    fill_random_slice(&mut shared_head, rng);

    let mut groups = Vec::with_capacity(MEDIUM_GROUPS);
    for group_id in 0..MEDIUM_GROUPS {
        let prefix_len = rng.usize_range_inclusive(MEDIUM_PREFIX_MIN, MEDIUM_PREFIX_MAX);
        let mut prefix = vec![0u8; prefix_len];

        let shared_len = shared_head.len().min(prefix_len.saturating_sub(4));
        if shared_len > 0 {
            prefix[..shared_len].copy_from_slice(&shared_head[..shared_len]);
        }
        if shared_len < prefix_len {
            fill_random_slice(&mut prefix[shared_len..], rng);
        }

        let tag = (group_id as u32).to_le_bytes();
        let tag_pos = prefix_len - 4;
        prefix[tag_pos..].copy_from_slice(&tag);
        groups.push(MediumGroup { prefix });
    }

    groups
}

fn estimate_rows_for_sf(
    sf: f64,
    gib_per_sf: f64,
    heavy_payload_min: usize,
    heavy_payload_max: usize,
) -> u64 {
    let target_bytes = sf * gib_per_sf * (1u64 << 30) as f64;
    let heavy_avg_key = HEAVY_KEY_LEN as f64;
    let medium_avg_prefix = (MEDIUM_PREFIX_MIN + MEDIUM_PREFIX_MAX) as f64 / 2.0;
    let medium_avg_suffix = (MEDIUM_SUFFIX_MIN + MEDIUM_SUFFIX_MAX) as f64 / 2.0;
    let medium_avg_key = medium_avg_prefix + medium_avg_suffix;

    let heavy_avg_payload = (heavy_payload_min + heavy_payload_max) as f64 / 2.0;
    let medium_avg_payload = (MEDIUM_PAYLOAD_MIN + MEDIUM_PAYLOAD_MAX) as f64 / 2.0;
    let random_avg_payload = (RANDOM_PAYLOAD_MIN + RANDOM_PAYLOAD_MAX) as f64 / 2.0;

    let avg_record_bytes = 8.0
        + 0.60 * (heavy_avg_key + heavy_avg_payload)
        + 0.30 * (medium_avg_key + medium_avg_payload)
        + 0.10 * (RANDOM_KEY_LEN as f64 + random_avg_payload);

    (target_bytes / avg_record_bytes).round().max(1.0) as u64
}

fn try_start_burst(
    burst: &mut BurstState,
    rng: &mut SplitMix64,
    heavy_dup_remaining: &[u64; HEAVY_GROUPS],
    heavy_near_remaining: &[u64; HEAVY_GROUPS],
) {
    let mut available = Vec::with_capacity(HEAVY_GROUPS);
    for group_id in 0..HEAVY_GROUPS {
        if heavy_dup_remaining[group_id] + heavy_near_remaining[group_id] > 0 {
            available.push(group_id);
        }
    }
    if available.len() < BURST_GROUPS_MIN {
        return;
    }

    fisher_yates_shuffle(&mut available, rng);
    let max_groups = available.len().min(BURST_GROUPS_MAX);
    let group_count = if max_groups == BURST_GROUPS_MIN {
        BURST_GROUPS_MIN
    } else {
        rng.usize_range_inclusive(BURST_GROUPS_MIN, max_groups)
    };

    let rounds = rng.usize_range_inclusive(BURST_ROUNDS_MIN, BURST_ROUNDS_MAX);
    burst.groups = available.into_iter().take(group_count).collect();
    burst.cursor = rng.usize_bounded(group_count);
    burst.steps_left = group_count * rounds;
}

fn pick_weighted_heavy_group(
    rng: &mut SplitMix64,
    heavy_dup_remaining: &[u64; HEAVY_GROUPS],
    heavy_near_remaining: &[u64; HEAVY_GROUPS],
    heavy_remaining: u64,
) -> usize {
    let mut r = rng.u64_bounded(heavy_remaining);
    for group_id in 0..HEAVY_GROUPS {
        let w = heavy_dup_remaining[group_id] + heavy_near_remaining[group_id];
        if r < w {
            return group_id;
        }
        r -= w;
    }
    HEAVY_GROUPS - 1
}

fn pick_live_medium_group(rng: &mut SplitMix64, medium_remaining: &[u64]) -> usize {
    loop {
        let idx = rng.usize_bounded(MEDIUM_GROUPS);
        if medium_remaining[idx] > 0 {
            return idx;
        }
    }
}

fn flush_block(
    out: &mut BufWriter<File>,
    idx: &mut BufWriter<File>,
    block: &mut Vec<u8>,
    bytes_written: &mut u64,
    index_entries: &mut u64,
) -> io::Result<()> {
    if block.is_empty() {
        return Ok(());
    }
    idx.write_all(&bytes_written.to_le_bytes())?;
    out.write_all(block)?;
    *bytes_written += block.len() as u64;
    *index_entries += 1;
    block.clear();
    Ok(())
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    if !(args.sf.is_finite() && args.sf > 0.0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--sf must be a finite positive value",
        ));
    }
    if !(args.gib_per_sf.is_finite() && args.gib_per_sf > 0.0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--gib-per-sf must be a finite positive value",
        ));
    }
    if !(args.burst_start_prob.is_finite() && (0.0..=1.0).contains(&args.burst_start_prob)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--burst-start-prob must be in [0, 1]",
        ));
    }
    if args.heavy_payload_min == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--heavy-payload-min must be > 0",
        ));
    }
    if args.heavy_payload_min > args.heavy_payload_max {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--heavy-payload-min must be <= --heavy-payload-max",
        ));
    }
    if args.heavy_payload_max > MAX_SORTABLE_HEAVY_VALUE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "--heavy-payload-max={} is too large for the current sorter memory model; max supported is {} for heavy key length {}",
                args.heavy_payload_max, MAX_SORTABLE_HEAVY_VALUE_BYTES, HEAVY_KEY_LEN
            ),
        ));
    }

    let rows = args.rows.unwrap_or_else(|| {
        estimate_rows_for_sf(
            args.sf,
            args.gib_per_sf,
            args.heavy_payload_min,
            args.heavy_payload_max,
        )
    });
    if rows == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "rows must be greater than 0",
        ));
    }

    let heavy_rows = rows.saturating_mul(HEAVY_SHARE_NUM) / 100;
    let medium_rows = rows.saturating_mul(MEDIUM_SHARE_NUM) / 100;
    let random_rows = rows - heavy_rows - medium_rows;

    let heavy_dup_total = heavy_rows / 2;
    let heavy_near_total = heavy_rows - heavy_dup_total;

    let mut setup_rng = SplitMix64::new(args.seed ^ 0xD1CE_B00C_A11C_E5ED);
    let heavy_groups = init_heavy_groups(&mut setup_rng);
    let medium_groups = init_medium_groups(&mut setup_rng);

    let dup_counts_vec = distribute_counts(heavy_dup_total, HEAVY_GROUPS, &mut setup_rng);
    let near_counts_vec = distribute_counts(heavy_near_total, HEAVY_GROUPS, &mut setup_rng);
    let medium_counts = distribute_counts(medium_rows, MEDIUM_GROUPS, &mut setup_rng);

    let mut heavy_dup_remaining = [0u64; HEAVY_GROUPS];
    let mut heavy_near_remaining = [0u64; HEAVY_GROUPS];
    heavy_dup_remaining.copy_from_slice(&dup_counts_vec);
    heavy_near_remaining.copy_from_slice(&near_counts_vec);

    let mut medium_remaining = medium_counts;
    let mut random_remaining = random_rows;
    let mut heavy_remaining = heavy_rows;
    let mut medium_remaining_total = medium_rows;

    let out_f = File::create(&args.out)?;
    let mut out = BufWriter::with_capacity(8 * 1024 * 1024, out_f);
    let idx_f = File::create(&args.idx)?;
    let mut idx = BufWriter::with_capacity(256 * 1024, idx_f);

    let mut rng = SplitMix64::new(args.seed ^ 0xA5A5_A5A5_5A5A_5A5A);
    let mut burst = BurstState::default();

    let mut block = Vec::with_capacity(INDEX_STRIDE_BYTES + args.heavy_payload_max + 1024);
    let mut bytes_written: u64 = 0;
    let mut index_entries: u64 = 0;
    let mut rows_written: u64 = 0;

    while rows_written < rows {
        let mut heavy_group_from_burst =
            burst.next_heavy_group(&heavy_dup_remaining, &heavy_near_remaining);

        if heavy_group_from_burst.is_none()
            && heavy_remaining > 0
            && rng.next_f64() < args.burst_start_prob
        {
            try_start_burst(
                &mut burst,
                &mut rng,
                &heavy_dup_remaining,
                &heavy_near_remaining,
            );
            heavy_group_from_burst =
                burst.next_heavy_group(&heavy_dup_remaining, &heavy_near_remaining);
        }

        if let Some(group_id) = heavy_group_from_burst {
            let dup_left = heavy_dup_remaining[group_id];
            let near_left = heavy_near_remaining[group_id];
            let choose_dup = if dup_left == 0 {
                false
            } else if near_left == 0 {
                true
            } else {
                rng.u64_bounded(dup_left + near_left) < dup_left
            };

            if choose_dup {
                heavy_dup_remaining[group_id] -= 1;
            } else {
                heavy_near_remaining[group_id] -= 1;
            }
            heavy_remaining -= 1;
            append_heavy_record(
                &mut block,
                &mut rng,
                &heavy_groups[group_id],
                choose_dup,
                args.heavy_payload_min,
                args.heavy_payload_max,
            );
        } else {
            let total_remaining = heavy_remaining + medium_remaining_total + random_remaining;
            let pick = rng.u64_bounded(total_remaining);

            if pick < heavy_remaining {
                let group_id = pick_weighted_heavy_group(
                    &mut rng,
                    &heavy_dup_remaining,
                    &heavy_near_remaining,
                    heavy_remaining,
                );
                let dup_left = heavy_dup_remaining[group_id];
                let near_left = heavy_near_remaining[group_id];
                let choose_dup = if dup_left == 0 {
                    false
                } else if near_left == 0 {
                    true
                } else {
                    rng.u64_bounded(dup_left + near_left) < dup_left
                };

                if choose_dup {
                    heavy_dup_remaining[group_id] -= 1;
                } else {
                    heavy_near_remaining[group_id] -= 1;
                }
                heavy_remaining -= 1;
                append_heavy_record(
                    &mut block,
                    &mut rng,
                    &heavy_groups[group_id],
                    choose_dup,
                    args.heavy_payload_min,
                    args.heavy_payload_max,
                );
            } else if pick < heavy_remaining + medium_remaining_total {
                let group_id = pick_live_medium_group(&mut rng, &medium_remaining);
                medium_remaining[group_id] -= 1;
                medium_remaining_total -= 1;
                append_medium_record(&mut block, &mut rng, &medium_groups[group_id]);
            } else {
                random_remaining -= 1;
                append_random_record(&mut block, &mut rng);
            }
        }

        rows_written += 1;
        if block.len() >= INDEX_STRIDE_BYTES {
            flush_block(
                &mut out,
                &mut idx,
                &mut block,
                &mut bytes_written,
                &mut index_entries,
            )?;
        }
    }

    flush_block(
        &mut out,
        &mut idx,
        &mut block,
        &mut bytes_written,
        &mut index_entries,
    )?;

    out.flush()?;
    idx.flush()?;

    if heavy_remaining != 0 || medium_remaining_total != 0 || random_remaining != 0 {
        return Err(io::Error::other(
            "generation ended before all row quotas were exhausted",
        ));
    }
    if heavy_dup_remaining.iter().any(|&v| v != 0) || heavy_near_remaining.iter().any(|&v| v != 0) {
        return Err(io::Error::other(
            "generation ended with heavy-group duplicate quotas still remaining",
        ));
    }
    if medium_remaining.iter().any(|&v| v != 0) {
        return Err(io::Error::other(
            "generation ended with medium-group quotas still remaining",
        ));
    }

    eprintln!(
        "generated PrefixSkew-HeavyPayload: rows={}, bytes={}, blocks={}, heavy={}, medium={}, random={}",
        rows_written, bytes_written, index_entries, heavy_rows, medium_rows, random_rows
    );

    Ok(())
}
