use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::diskio::aligned_reader::AlignedReader;
use crate::diskio::file::{SharedFd, file_size_fd};
use crate::{IoStatsTracker, SortInput};

/// Parallel KVBin reader for streams of [klen][key][vlen][val].
/// Uses a sidecar index with u64 offsets at record boundaries.
pub struct KvBinInputDirect {
    fd: Arc<SharedFd>,
    file_size: u64,
    checkpoints: Vec<u64>,
}

impl KvBinInputDirect {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let fd = Arc::new(SharedFd::new_from_path(path).map_err(|e| e.to_string())?);
        let file_size = file_size_fd(fd.as_raw_fd()).map_err(|e| e.to_string())?;
        let checkpoints = Self::load_checkpoints(path, file_size);

        if checkpoints.len() <= 2 {
            panic!("No valid index found for '{}'", path.display());
        }

        Ok(Self { fd, file_size, checkpoints })
    }

    fn load_checkpoints(path: &Path, file_size: u64) -> Vec<u64> {
        let candidates = [
            PathBuf::from(format!("{}.idx", path.display())),
            path.parent()
                .and_then(|p| path.file_stem().map(|s| p.join(s).with_extension("idx")))
                .unwrap_or_default(),
        ];

        let mut cps = vec![0];

        for p in candidates.iter().filter(|p| p.as_os_str().len() > 0) {
            if let Ok(mut f) = File::open(p).map(BufReader::new) {
                let mut buf = Vec::new();
                if f.read_to_end(&mut buf).is_ok() && buf.len() % 8 == 0 {
                    cps.extend(
                        buf.chunks_exact(8)
                            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                            .filter(|&off| off > 0 && off < file_size),
                    );
                    break;
                }
            }
        }

        cps.push(file_size);
        cps.sort_unstable();
        cps.dedup();
        cps
    }

    fn partition_ranges(&self, want: usize) -> Vec<(u64, u64)> {
        let n_blocks = self.checkpoints.len().saturating_sub(1);

        if n_blocks == 0 || want <= 1 {
            return vec![(0, self.file_size)];
        }

        let parts = want.min(n_blocks);

        // Distribute blocks evenly among partitions
        let mut ranges = Vec::with_capacity(parts);
        let mut block_idx = 0;

        for i in 0..parts {
            let start = self.checkpoints[block_idx];
            // Distribute remaining blocks evenly among remaining partitions
            let remaining_blocks = n_blocks - block_idx;
            let remaining_parts = parts - i;
            let num_blocks = (remaining_blocks + remaining_parts - 1) / remaining_parts;
            block_idx += num_blocks;
            let end = self.checkpoints[block_idx];
            ranges.push((start, end));
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
        self.partition_ranges(num_scanners.max(1))
            .into_iter()
            .map(|(start, end)| {
                let rdr = AlignedReader::from_fd_with_start_position(
                    self.fd.clone(),
                    start,
                    io_tracker.clone(),
                )
                .expect("failed to open reader");
                Box::new(KvBinScanner { rdr, end, done: false })
                    as Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>
            })
            .collect()
    }
}

/// Iterates KVBin records within a byte range.
struct KvBinScanner {
    rdr: AlignedReader,
    end: u64,
    done: bool,
}

impl KvBinScanner {
    #[inline(always)]
    fn read_u32(&mut self) -> Option<u32> {
        let mut b = [0u8; 4];
        self.rdr.read_exact(&mut b).ok().map(|_| u32::from_le_bytes(b))
    }

    #[inline(always)]
    fn read_vec(&mut self, len: usize) -> Option<Vec<u8>> {
        if len == 0 {
            return Some(Vec::new());
        }
        let mut buf = Vec::with_capacity(len);
        unsafe { buf.set_len(len); }
        self.rdr.read_exact(&mut buf).ok().map(|_| buf)
    }
}

impl Iterator for KvBinScanner {
    type Item = (Vec<u8>, Vec<u8>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.rdr.position() >= self.end {
            self.done = true;
            return None;
        }

        let result = (|| {
            let klen = self.read_u32()? as usize;
            let key = self.read_vec(klen)?;
            let vlen = self.read_u32()? as usize;
            let val = self.read_vec(vlen)?;
            Some((key, val))
        })();

        if result.is_none() {
            self.done = true;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// Helper to create a KVBin file with given key-value pairs
    fn create_kvbin_file(dir: &Path, name: &str, records: &[(&[u8], &[u8])]) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();

        for (key, val) in records {
            file.write_all(&(key.len() as u32).to_le_bytes()).unwrap();
            file.write_all(key).unwrap();
            file.write_all(&(val.len() as u32).to_le_bytes()).unwrap();
            file.write_all(val).unwrap();
        }

        file.sync_all().unwrap();
        path
    }

    /// Helper to create an index file with given offsets
    fn create_index_file(dir: &Path, name: &str, offsets: &[u64]) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();

        for &offset in offsets {
            file.write_all(&offset.to_le_bytes()).unwrap();
        }

        file.sync_all().unwrap();
        path
    }

    /// Calculate byte offset after N records
    fn record_offset(records: &[(&[u8], &[u8])], n: usize) -> u64 {
        records[..n]
            .iter()
            .map(|(k, v)| 4 + k.len() + 4 + v.len())
            .sum::<usize>() as u64
    }

    #[test]
    fn test_basic_read_and_parallel_scanners() {
        let dir = TempDir::new().unwrap();
        let records: Vec<(&[u8], &[u8])> = vec![
            (b"a", b"1"),
            (b"b", b"2"),
            (b"c", b"3"),
            (b"d", b"4"),
            (b"e", b"5"),
            (b"f", b"6"),
        ];

        let kvbin_path = create_kvbin_file(dir.path(), "test.kvbin", &records);
        let file_size = std::fs::metadata(&kvbin_path).unwrap().len();

        let offsets: Vec<u64> = (1..=records.len())
            .map(|i| record_offset(&records, i))
            .filter(|&o| o < file_size)
            .collect();
        create_index_file(dir.path(), "test.kvbin.idx", &offsets);

        let input = KvBinInputDirect::new(&kvbin_path).unwrap();

        // Test single scanner
        let results: Vec<_> = input
            .create_parallel_scanners(1, None)
            .into_iter()
            .next()
            .unwrap()
            .collect();
        assert_eq!(results.len(), 6);
        assert_eq!(results[0], (b"a".to_vec(), b"1".to_vec()));
        assert_eq!(results[5], (b"f".to_vec(), b"6".to_vec()));

        // Test multiple scanners
        let scanners = input.create_parallel_scanners(3, None);
        assert!(scanners.len() >= 1 && scanners.len() <= 3);

        let mut all_results: Vec<_> = scanners
            .into_iter()
            .flat_map(|s| s.collect::<Vec<_>>())
            .collect();
        all_results.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(all_results.len(), 6);
    }

    #[test]
    fn test_edge_cases() {
        let dir = TempDir::new().unwrap();

        // Test empty keys/values and binary data
        let binary_key: Vec<u8> = (0..=255).collect();
        let binary_val: Vec<u8> = (0..=255).rev().collect();

        let records: Vec<(&[u8], &[u8])> = vec![
            (b"", b"empty_key"),
            (b"empty_val", b""),
            (&binary_key, &binary_val),
        ];

        let kvbin_path = create_kvbin_file(dir.path(), "test.kvbin", &records);
        let file_size = std::fs::metadata(&kvbin_path).unwrap().len();

        let offsets: Vec<u64> = (1..=records.len())
            .map(|i| record_offset(&records, i))
            .filter(|&o| o < file_size)
            .collect();
        create_index_file(dir.path(), "test.kvbin.idx", &offsets);

        let input = KvBinInputDirect::new(&kvbin_path).unwrap();
        let results: Vec<_> = input
            .create_parallel_scanners(1, None)
            .into_iter()
            .next()
            .unwrap()
            .collect();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (b"".to_vec(), b"empty_key".to_vec()));
        assert_eq!(results[1], (b"empty_val".to_vec(), b"".to_vec()));
        assert_eq!(results[2].0, binary_key);
        assert_eq!(results[2].1, binary_val);
    }

    #[test]
    fn test_large_records() {
        let dir = TempDir::new().unwrap();
        let large_key = vec![b'K'; 10000];
        let large_val = vec![b'V'; 50000];

        let records: Vec<(&[u8], &[u8])> = vec![
            (&large_key, &large_val),
            (b"small", b"record"),
        ];

        let kvbin_path = create_kvbin_file(dir.path(), "test.kvbin", &records);
        let file_size = std::fs::metadata(&kvbin_path).unwrap().len();

        let offsets: Vec<u64> = (1..=records.len())
            .map(|i| record_offset(&records, i))
            .filter(|&o| o < file_size)
            .collect();
        create_index_file(dir.path(), "test.kvbin.idx", &offsets);

        let input = KvBinInputDirect::new(&kvbin_path).unwrap();
        let results: Vec<_> = input
            .create_parallel_scanners(1, None)
            .into_iter()
            .next()
            .unwrap()
            .collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.len(), 10000);
        assert_eq!(results[0].1.len(), 50000);
    }

    #[test]
    fn test_stem_index_file() {
        let dir = TempDir::new().unwrap();
        let records: Vec<(&[u8], &[u8])> = vec![
            (b"key1", b"val1"),
            (b"key2", b"val2"),
        ];

        let kvbin_path = create_kvbin_file(dir.path(), "data.kvbin", &records);
        let file_size = std::fs::metadata(&kvbin_path).unwrap().len();

        // Create index with stem name (data.idx instead of data.kvbin.idx)
        let offsets: Vec<u64> = (1..records.len())
            .map(|i| record_offset(&records, i))
            .filter(|&o| o < file_size)
            .collect();
        create_index_file(dir.path(), "data.idx", &offsets);

        let input = KvBinInputDirect::new(&kvbin_path).unwrap();
        let results: Vec<_> = input
            .create_parallel_scanners(1, None)
            .into_iter()
            .next()
            .unwrap()
            .collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (b"key1".to_vec(), b"val1".to_vec()));
    }

    #[test]
    fn test_partition_ranges() {
        let dir = TempDir::new().unwrap();
        let records: Vec<(&[u8], &[u8])> = vec![
            (b"a", b"1"),
            (b"b", b"2"),
        ];

        let kvbin_path = create_kvbin_file(dir.path(), "test.kvbin", &records);
        let file_size = std::fs::metadata(&kvbin_path).unwrap().len();

        // Single checkpoint
        let offsets = vec![record_offset(&records, 1)];
        create_index_file(dir.path(), "test.kvbin.idx", &offsets);

        let input = KvBinInputDirect::new(&kvbin_path).unwrap();

        // Request 1 scanner - should get full range
        let ranges = input.partition_ranges(1);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], (0, file_size));

        // Request more scanners than blocks - should be limited
        let ranges = input.partition_ranges(10);
        assert!(ranges.len() <= 2);
        assert_eq!(ranges.first().unwrap().0, 0);
        assert_eq!(ranges.last().unwrap().1, file_size);
    }

    #[test]
    #[should_panic(expected = "No valid index found")]
    fn test_missing_index() {
        let dir = TempDir::new().unwrap();
        let records: Vec<(&[u8], &[u8])> = vec![(b"key", b"val")];
        let kvbin_path = create_kvbin_file(dir.path(), "test.kvbin", &records);
        let _ = KvBinInputDirect::new(&kvbin_path).unwrap();
    }

    #[test]
    fn test_sampled_offsets_distribution() {
        // Test sparse sampling and even distribution across scanners
        let dir = TempDir::new().unwrap();

        // Create 1000 records
        let records: Vec<(Vec<u8>, Vec<u8>)> = (0..1000)
            .map(|i| (format!("key{:04}", i).into_bytes(), format!("val{:04}", i).into_bytes()))
            .collect();
        let record_refs: Vec<(&[u8], &[u8])> =
            records.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();

        let kvbin_path = create_kvbin_file(dir.path(), "test.kvbin", &record_refs);
        let file_size = std::fs::metadata(&kvbin_path).unwrap().len();

        // Create sparse offsets every 50 records (20 checkpoints)
        let offsets: Vec<u64> = (1..=1000)
            .filter(|i| i % 50 == 0)
            .map(|i| record_offset(&record_refs, i))
            .filter(|&o| o < file_size)
            .collect();
        create_index_file(dir.path(), "test.kvbin.idx", &offsets);

        let input = KvBinInputDirect::new(&kvbin_path).unwrap();

        // Test with 8 scanners
        let num_scanners = 8;
        let scanners = input.create_parallel_scanners(num_scanners, None);

        let scanner_counts: Vec<usize> = scanners
            .into_iter()
            .map(|s| s.count())
            .collect();

        let total: usize = scanner_counts.iter().sum();
        assert_eq!(total, 1000, "Total records should be 1000");

        println!("Scanner distribution: {:?}", scanner_counts);

        // Calculate standard deviation to verify even distribution
        let mean = total as f64 / scanner_counts.len() as f64;
        let variance: f64 = scanner_counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / scanner_counts.len() as f64;
        let std_dev = variance.sqrt();

        println!("Mean: {:.1}, Std Dev: {:.1}", mean, std_dev);

        // Distribution should be reasonable (std dev < 50% of mean)
        assert!(
            std_dev < mean * 0.5,
            "Distribution too uneven: std_dev={:.1}, mean={:.1}",
            std_dev, mean
        );

        // Verify all records readable
        let input2 = KvBinInputDirect::new(&kvbin_path).unwrap();
        let all_results: Vec<_> = input2
            .create_parallel_scanners(num_scanners, None)
            .into_iter()
            .flat_map(|s| s.collect::<Vec<_>>())
            .collect();
        assert_eq!(all_results.len(), 1000);
    }
}
