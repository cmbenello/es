// src/kvbin.rs
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::SortInput;
use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::file::SharedFd;

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

/// Generic KVBin writer that takes any iterator producing (key, value) pairs.
/// Writes kvbin format: [klen:u32][key][vlen:u32][val] for each record.
/// Creates an index file (.idx) with offsets every `idx_every` records.
/// Returns (rows_written, checkpoints_written).
pub fn create_kvbin_from_input(
    sort_input: Box<dyn SortInput>,
    data_path: impl AsRef<Path>,
    idx_path: impl AsRef<Path>,
    idx_every: usize,
) -> Result<(u64, u64), String> {
    if idx_every == 0 {
        return Err("idx_every must be greater than 0".to_string());
    }

    println!("[kvbin] Starting kvbin creation...");
    println!("[kvbin] Data file: {:?}", data_path.as_ref());
    println!("[kvbin] Index file: {:?}", idx_path.as_ref());
    println!("[kvbin] Index checkpoint every {} records", idx_every);

    // Create output file with aligned writer
    let out_fd = Arc::new(
        SharedFd::new_from_path(data_path, false).map_err(|e| format!("open data file: {e}"))?,
    );
    let mut out =
        AlignedWriter::from_fd(out_fd).map_err(|e| format!("create aligned writer: {e}"))?;

    // Create index file with aligned writer
    let idx_fd = Arc::new(
        SharedFd::new_from_path(idx_path, false).map_err(|e| format!("open idx file: {e}"))?,
    );
    let mut idx =
        AlignedWriter::from_fd(idx_fd).map_err(|e| format!("create idx aligned writer: {e}"))?;

    let mut rows: u64 = 0;
    let mut index_points: u64 = 0;

    println!("[kvbin] Writing records...");
    let scanner = sort_input.create_parallel_scanners(1, None).remove(0);
    for (key, val) in scanner {
        // write [klen][key][vlen][val]
        let klen = key.len() as u32;
        let vlen = val.len() as u32;
        out.write_all(&klen.to_le_bytes())
            .map_err(|e| e.to_string())?;
        out.write_all(&key).map_err(|e| e.to_string())?;
        out.write_all(&vlen.to_le_bytes())
            .map_err(|e| e.to_string())?;
        out.write_all(&val).map_err(|e| e.to_string())?;

        rows += 1;

        // periodic indexing
        if rows % (idx_every as u64) == 0 {
            index_points += 1;
            let pos = out.position();
            idx.write_all(&pos.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }

        // Progress updates every 1M records
        if rows % 1_000_000 == 0 {
            println!(
                "[kvbin] Progress: {} million records written",
                rows / 1_000_000
            );
        }
    }

    println!(
        "[kvbin] Finished writing {} records with {} index checkpoints",
        rows, index_points
    );
    println!("[kvbin] Flushing data file...");
    out.flush().map_err(|e| e.to_string())?;
    println!("[kvbin] Flushing index file...");
    idx.flush().map_err(|e| e.to_string())?;

    let data_size = out.position();
    println!("[kvbin] Data file size: {} bytes", data_size);
    let index_size = idx.position();
    println!("[kvbin] Index file size: {} bytes", index_size);

    println!("[kvbin] Cooling down for 30 seconds...");
    std::thread::sleep(std::time::Duration::from_secs(30));
    println!("[kvbin] Done!");


    Ok((rows, index_points))
}
