// src/kvbin.rs
use std::fs;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::SortInput;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread; // Linux/Mac用: 位置指定書き込み

pub fn binary_file_name(
    input_path: &Path,
    key_columns: &[usize],
    value_columns: &[usize],
) -> Result<(PathBuf, PathBuf), String> {
    // Helper to stringify column indices like "0-2-5"
    let cols_to_str = |cols: &[usize]| {
        cols.iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("-")
    };

    let key_str = cols_to_str(key_columns);
    let value_str = cols_to_str(value_columns);

    let file_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "invalid input file name".to_string())?;

    // Use ./datasets directory and create it if it doesn't exist
    let datasets_dir = PathBuf::from("./datasets");
    fs::create_dir_all(&datasets_dir)
        .map_err(|e| format!("failed to create datasets directory: {}", e))?;

    // e.g. input: data.csv -> data.k-0-2.v-3-4.kvbin
    let data_file = format!("{file_stem}.k-{key_str}.v-{value_str}.kvbin");
    let idx_file = format!("{data_file}.idx");

    Ok((datasets_dir.join(data_file), datasets_dir.join(idx_file)))
}

// --- Configuration Constants ---

// Flush data buffer every 4MB.
// This size is chosen to maximize throughput on SSDs/HDDs by issuing large sequential writes.
const DATA_FLUSH_THRESHOLD: usize = 4 * 1024 * 1024;

// Flush index buffer every 64KB.
// Since we only write one index entry per 4MB data block, this buffer will fill up very slowly.
// This is extremely efficient (very few syscalls for the index file).
const IDX_FLUSH_THRESHOLD: usize = 64 * 1024;

/// Writes KV pairs to a binary file using multiple threads.
///
/// Format: [klen:u32][key][vlen:u32][val]
/// Index:  [offset:u64]
///         Points to the start of each 4MB data block.
///
/// # Architecture
/// - Block-based Indexing: Instead of indexing every N rows, we index the start of every flushed block.
/// - Lock-free file positioning: Uses `AtomicU64` to reserve file space.
pub fn create_kvbin_from_input(
    sort_input: Box<dyn SortInput>,
    data_path: impl AsRef<Path>,
    idx_path: impl AsRef<Path>,
    mut num_threads: usize,
) -> Result<(u64, u64), String> {
    // Auto-detect threads if 0 is passed
    if num_threads == 0 {
        num_threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(4).max(4))
            .unwrap_or(4);
    }

    println!(
        "[kvbin] Starting PARALLEL creation with {} threads...",
        num_threads
    );
    println!(
        "[kvbin] Indexing strategy: One checkpoint every ~{} MB block",
        DATA_FLUSH_THRESHOLD / 1024 / 1024
    );

    // 1. Open files (Shared via Arc)
    let data_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(data_path)
        .map_err(|e| format!("failed to open data file: {}", e))?;
    let data_file = Arc::new(data_file);

    let idx_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(idx_path)
        .map_err(|e| format!("failed to open idx file: {}", e))?;
    let idx_file = Arc::new(idx_file);

    // 2. Global Atomic Offsets
    let global_data_offset = Arc::new(AtomicU64::new(0));
    let global_idx_offset = Arc::new(AtomicU64::new(0));

    // Statistics
    let total_rows = Arc::new(AtomicU64::new(0));
    let total_checkpoints = Arc::new(AtomicU64::new(0));

    // 3. Create Scanners
    let scanners = sort_input.create_parallel_scanners(num_threads, None);

    // 4. Thread Execution
    thread::scope(|s| -> Result<(), String> {
        let mut handles = vec![];

        for scanner in scanners {
            // Clone Arcs for the thread
            let data_file = Arc::clone(&data_file);
            let idx_file = Arc::clone(&idx_file);
            let global_data_offset = Arc::clone(&global_data_offset);
            let global_idx_offset = Arc::clone(&global_idx_offset);
            let total_rows = Arc::clone(&total_rows);
            let total_checkpoints = Arc::clone(&total_checkpoints);

            let handle = s.spawn(move || -> Result<(), String> {
                // Thread-local buffers
                let mut data_buf: Vec<u8> = Vec::with_capacity(DATA_FLUSH_THRESHOLD + 4096);
                let mut idx_buf: Vec<u8> = Vec::with_capacity(IDX_FLUSH_THRESHOLD + 4096);

                let mut local_rows: u64 = 0;
                let mut local_checkpoints: u64 = 0;

                for (key, val) in scanner {
                    local_rows += 1;

                    // Serialize to buffer: [klen][key][vlen][val]
                    data_buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
                    data_buf.extend_from_slice(&key);
                    data_buf.extend_from_slice(&(val.len() as u32).to_le_bytes());
                    data_buf.extend_from_slice(&val);

                    // --- Flush Data Buffer Check ---
                    if data_buf.len() >= DATA_FLUSH_THRESHOLD {
                        let write_size = data_buf.len() as u64;
                        let my_data_offset =
                            global_data_offset.fetch_add(write_size, Ordering::Relaxed);
                        data_file
                            .write_all_at(&data_buf, my_data_offset)
                            .map_err(|e| format!("pwrite data failed: {}", e))?;
                        idx_buf.extend_from_slice(&my_data_offset.to_le_bytes());
                        local_checkpoints += 1;
                        data_buf.clear();
                    }

                    if idx_buf.len() >= IDX_FLUSH_THRESHOLD {
                        flush_index_buf(&idx_file, &global_idx_offset, &mut idx_buf)?;
                    }
                }

                // --- Final Cleanup ---

                // Flush remaining data
                if !data_buf.is_empty() {
                    let write_size = data_buf.len() as u64;
                    let my_data_offset =
                        global_data_offset.fetch_add(write_size, Ordering::Relaxed);
                    data_file
                        .write_all_at(&data_buf, my_data_offset)
                        .map_err(|e| format!("pwrite remaining data failed: {}", e))?;
                    idx_buf.extend_from_slice(&my_data_offset.to_le_bytes());
                    local_checkpoints += 1;
                }

                if !idx_buf.is_empty() {
                    flush_index_buf(&idx_file, &global_idx_offset, &mut idx_buf)?;
                }

                total_rows.fetch_add(local_rows, Ordering::Relaxed);
                total_checkpoints.fetch_add(local_checkpoints, Ordering::Relaxed);

                Ok(())
            });
            handles.push(handle);
        }

        let mut errors = Vec::new();
        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    eprintln!("[kvbin] Error in worker thread: {}", e);
                    errors.push(e);
                }
                Err(_) => {
                    let err = "worker thread panicked".to_string();
                    eprintln!("[kvbin] {}", err);
                    errors.push(err);
                }
            }
        }

        if !errors.is_empty() {
            return Err(format!(
                "kvbin creation failed: {} thread(s) encountered errors",
                errors.len()
            ));
        }

        Ok(())
    })?;

    println!("[kvbin] Flushing file buffers to disk...");
    data_file
        .sync_all()
        .map_err(|e| format!("sync data failed: {}", e))?;
    idx_file
        .sync_all()
        .map_err(|e| format!("sync idx failed: {}", e))?;

    println!("[kvbin] Evicting files from OS Page Cache...");

    unsafe {
        drop_cache(data_file.as_raw_fd());
        drop_cache(idx_file.as_raw_fd());
    }

    let final_rows = total_rows.load(Ordering::Relaxed);
    let final_checkpoints = total_checkpoints.load(Ordering::Relaxed);

    println!(
        "[kvbin] Completed. Rows: {}, Checkpoints (Blocks): {}",
        final_rows, final_checkpoints
    );

    println!("[kvbin] Sleeping 30 seconds to allow OS to settle disk writes...");
    std::thread::sleep(std::time::Duration::from_secs(30));

    Ok((final_rows, final_checkpoints))
}

/// Helper: Flushes the index buffer to the file using atomic reservation.
fn flush_index_buf(
    file: &File,
    global_offset: &AtomicU64,
    buf: &mut Vec<u8>,
) -> Result<(), String> {
    let size = buf.len() as u64;
    // Reserve space
    let offset = global_offset.fetch_add(size, Ordering::Relaxed);
    // Write at reserved position
    file.write_all_at(buf, offset)
        .map_err(|e| format!("pwrite index failed: {}", e))?;

    buf.clear();
    Ok(())
}

// OSごとの実装を分けるヘルパー関数
unsafe fn drop_cache(fd: i32) {
    // --- Linux の場合 ---
    #[cfg(target_os = "linux")]
    {
        let ret = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
        if ret != 0 {
            eprintln!(
                "[kvbin] Warning: posix_fadvise failed on Linux (err: {})",
                ret
            );
        }
    }

    // --- macOS の場合 ---
    #[cfg(target_os = "macos")]
    {
        // F_NOCACHE: このファイルに対するキャッシュを無効化する
        let ret = unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
        if ret == -1 {
            eprintln!("[kvbin] Warning: fcntl(F_NOCACHE) failed on macOS");
        }
    }
}
