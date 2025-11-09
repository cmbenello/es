use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::diskio::aligned_reader::AlignedReader;
use crate::diskio::file::{SharedFd, file_size_fd};
use crate::{IoStatsTracker, SortInput};

/// Parallel KVBin reader for streams of [klen][key][vlen][val].
/// Uses a sidecar index with u64 offsets at record boundaries.
/// Accepted index filenames:
///   1) "<file>.idx"            (e.g., "lineitem.kvbin.idx")
///   2) same stem + ".idx"      (e.g., "lineitem.idx")
pub struct KvBinInputDirect {
    fd: Arc<SharedFd>,
    file_size: u64,
    checkpoints: Vec<u64>, // always includes 0 and file_size, sorted & deduped
}

impl KvBinInputDirect {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let fd = Arc::new(SharedFd::new_from_path(path).map_err(|e| e.to_string())?);
        let file_size = file_size_fd(fd.as_raw_fd()).map_err(|e| e.to_string())?;
        let mut checkpoints = Self::load_checkpoints(path, file_size);
        if checkpoints.len() <= 2 {
            if let Ok(mut synthetic) = Self::generate_synthetic_checkpoints(path, file_size) {
                checkpoints.extend(synthetic.drain(..));
                checkpoints.sort_unstable();
                checkpoints.dedup();
            }
        }
        Ok(Self {
            fd,
            file_size,
            checkpoints,
        })
    }

    /// Try both "<file>.idx" and "stem.idx". Parse little-endian u64 offsets.
    fn load_checkpoints(kvbin_path: &Path, file_size: u64) -> Vec<u64> {
        let mut candidates = Vec::new();

        // "<file>.idx" → e.g., "lineitem.kvbin.idx"
        candidates.push(PathBuf::from(format!("{}{}", kvbin_path.display(), ".idx")));

        // "stem.idx" → e.g., "lineitem.idx"
        if let (Some(stem), Some(parent)) = (kvbin_path.file_stem(), kvbin_path.parent()) {
            let alt = parent.join(PathBuf::from(stem).with_extension("idx"));
            candidates.push(alt);
        }

        let mut cps: Vec<u64> = vec![0];

        for p in candidates {
            if let Ok(f) = File::open(&p) {
                let mut rdr = BufReader::new(f);
                let mut buf = Vec::new();
                if rdr.read_to_end(&mut buf).is_ok() && buf.len() % 8 == 0 {
                    for chunk in buf.chunks_exact(8) {
                        let mut arr = [0u8; 8];
                        arr.copy_from_slice(chunk);
                        let off = u64::from_le_bytes(arr);
                        if off > 0 && off < file_size {
                            cps.push(off);
                        }
                    }
                    // first valid index wins
                    break;
                }
            }
        }

        cps.push(file_size);
        cps.sort_unstable();
        cps.dedup();
        cps
    }

    /// When no index is available, scan the file once to produce evenly spaced checkpoints.
    fn generate_synthetic_checkpoints(
        kvbin_path: &Path,
        file_size: u64,
    ) -> Result<Vec<u64>, String> {
        if file_size == 0 {
            return Ok(Vec::new());
        }

        const MIN_TARGET_BYTES: u64 = 1 * 1024 * 1024; // try to keep partitions >= 1 MiB
        const MAX_SYNTHETIC_PARTS: usize = 1024;

        let mut desired_parts = ((file_size + MIN_TARGET_BYTES - 1) / MIN_TARGET_BYTES) as usize;
        desired_parts = desired_parts.clamp(1, MAX_SYNTHETIC_PARTS);

        if desired_parts <= 1 {
            return Ok(Vec::new());
        }

        let target_bytes = ((file_size + desired_parts as u64 - 1) / desired_parts as u64).max(1);
        let mut next_cut = target_bytes;
        let mut checkpoints = Vec::with_capacity(desired_parts.saturating_sub(1));

        let file = File::open(kvbin_path).map_err(|e| format!("open kvbin for scan: {}", e))?;
        let mut reader = BufReader::with_capacity(1 << 20, file);
        let mut scratch = vec![0u8; 64 * 1024];
        let mut pos: u64 = 0;

        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {
                    pos += 4;
                    let klen = u32::from_le_bytes(len_buf) as u64;
                    Self::skip_bytes(&mut reader, klen, &mut scratch)
                        .map_err(|e| format!("scan kvbin key bytes: {}", e))?;
                    pos += klen;

                    reader
                        .read_exact(&mut len_buf)
                        .map_err(|e| format!("scan kvbin value length: {}", e))?;
                    pos += 4;
                    let vlen = u32::from_le_bytes(len_buf) as u64;
                    Self::skip_bytes(&mut reader, vlen, &mut scratch)
                        .map_err(|e| format!("scan kvbin value bytes: {}", e))?;
                    pos += vlen;

                    if pos >= next_cut && pos < file_size {
                        checkpoints.push(pos);
                        if checkpoints.len() >= desired_parts.saturating_sub(1) {
                            break;
                        }
                        next_cut += target_bytes;
                    }
                }
                Err(ref e) if e.kind() == ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(format!("scan kvbin checkpoints: {}", e)),
            }
        }

        Ok(checkpoints
            .into_iter()
            .filter(|&off| off > 0 && off < file_size)
            .collect())
    }

    fn skip_bytes<R: Read>(reader: &mut R, mut len: u64, scratch: &mut [u8]) -> io::Result<()> {
        while len > 0 {
            let chunk = len.min(scratch.len() as u64) as usize;
            reader.read_exact(&mut scratch[..chunk])?;
            len -= chunk as u64;
        }
        Ok(())
    }

    /// Group adjacent checkpoint blocks into ~equal byte ranges.
    /// Guarantees full coverage and no overlap.
    fn partition_ranges(&self, want: usize) -> Vec<(u64, u64)> {
        let cps = &self.checkpoints;
        if cps.len() < 2 || want <= 1 {
            return vec![(0, self.file_size)];
        }

        let blocks: Vec<(u64, u64)> = cps.windows(2).map(|w| (w[0], w[1])).collect();
        let parts = want.max(1).min(blocks.len());
        if parts == 1 {
            return vec![(0, self.file_size)];
        }

        let total: u64 = blocks.iter().map(|(s, e)| e - s).sum();
        let target = (total / parts as u64).max(1);

        let mut ranges = Vec::with_capacity(parts);
        let mut cur_start = blocks[0].0;
        let mut acc: u64 = 0;
        let mut made = 0usize;

        for (i, (s, e)) in blocks.iter().enumerate() {
            let sz = e - s;
            acc += sz;

            // force a cut if we must leave enough blocks to fill remaining parts
            let left_blocks = blocks.len().saturating_sub(i + 1);
            let left_parts = parts.saturating_sub(made + 1);
            let must_cut = left_blocks == left_parts;

            // cut when we reached target for this range (and still can make more ranges)
            if (acc >= target && (made + 1) < parts) || must_cut {
                ranges.push((cur_start, *e));
                cur_start = *e;
                acc = 0;
                made += 1;
            }
        }

        if ranges.is_empty() || ranges.last().unwrap().1 != self.file_size {
            ranges.push((cur_start, self.file_size));
        }

        ranges
    }
}

impl SortInput for KvBinInputDirect {
    fn create_parallel_scanners(
        &self,
        num_scanners: usize,
        io_tracker: Option<IoStatsTracker>,
    ) -> Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>> {
        let ranges = self.partition_ranges(num_scanners.max(1));
        let mut v: Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>> =
            Vec::with_capacity(ranges.len());

        for (start, end) in ranges {
            let rdr = AlignedReader::from_fd_with_start_position(
                self.fd.clone(),
                start,
                io_tracker.clone(),
            )
            .expect("KvBinScanner: failed to open reader");
            v.push(Box::new(KvBinScanner {
                rdr,
                pos: start,
                end,
                done: false,
            }));
        }
        v
    }
}

/// Iterates KVBin records within a [start, end) byte range.
/// Each record is [klen:u32][key:klen][vlen:u32][val:vlen].
struct KvBinScanner {
    rdr: AlignedReader,
    pos: u64, // absolute position in file
    end: u64, // exclusive
    done: bool,
}

impl KvBinScanner {
    #[inline]
    fn read_u32_le(&mut self) -> Option<u32> {
        if self.pos + 4 > self.end {
            self.done = true;
            return None;
        }
        let mut b = [0u8; 4];
        if self.rdr.read_exact(&mut b).is_ok() {
            self.pos += 4;
            Some(u32::from_le_bytes(b))
        } else {
            self.done = true;
            None
        }
    }

    #[inline]
    fn read_full(&mut self, len: usize) -> Option<Vec<u8>> {
        let need = len as u64;
        if self.pos + need > self.end {
            self.done = true;
            return None;
        }
        let mut buf = vec![0u8; len];
        if len == 0 {
            return Some(buf);
        }
        if self.rdr.read_exact(&mut buf).is_ok() {
            self.pos += need;
            Some(buf)
        } else {
            self.done = true;
            None
        }
    }
}

impl Iterator for KvBinScanner {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.pos >= self.end {
            self.done = true;
            return None;
        }

        let klen = match self.read_u32_le() {
            Some(n) => n as usize,
            None => return None,
        };
        let key = match self.read_full(klen) {
            Some(v) => v,
            None => return None,
        };
        let vlen = match self.read_u32_le() {
            Some(n) => n as usize,
            None => return None,
        };
        let val = match self.read_full(vlen) {
            Some(v) => v,
            None => return None,
        };

        Some((key, val))
    }
}

// ------------------------ optional helpers ------------------------

/// Scan and keep only records whose key starts with `prefix` (single scanner).
pub fn kvbin_scan_prefix(
    path: impl AsRef<Path>,
    prefix: &[u8],
) -> Result<impl Iterator<Item = (Vec<u8>, Vec<u8>)>, String> {
    let src = KvBinInputDirect::new(path)?;
    let mut its = src.create_parallel_scanners(1, None);
    let it = its.remove(0);
    Ok(it.filter(move |(k, _)| k.starts_with(prefix)))
}

/// Scan and keep only records whose key is in `[start, end)` (lexicographic, byte-wise).
pub fn kvbin_scan_range(
    path: impl AsRef<Path>,
    start: &[u8],
    end: &[u8],
) -> Result<impl Iterator<Item = (Vec<u8>, Vec<u8>)>, String> {
    let src = KvBinInputDirect::new(path)?;
    let mut its = src.create_parallel_scanners(1, None);
    let it = its.remove(0);
    Ok(it.filter(move |(k, _)| k.as_slice() >= start && k.as_slice() < end))
}
