// src/kvbin.rs
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use arrow::datatypes::{DataType, Schema};
use memchr::memchr;
use smallvec::SmallVec;

use crate::input_reader::csv_input_direct::CsvDirectConfig;
use crate::order_preserving_encoding::*;

// --------------------- fast parsers ---------------------

#[inline(always)]
fn fast_parse_i64(s: &[u8]) -> Result<i64, String> {
    if s.is_empty() {
        return Ok(0);
    }
    let mut i = 0usize;
    let mut neg = false;
    if s[0] == b'-' {
        neg = true;
        i = 1;
    }
    let mut v: i64 = 0;
    while i < s.len() {
        let d = s[i].wrapping_sub(b'0');
        if d > 9 {
            return Err(format!("bad i64 digit in {:?}", String::from_utf8_lossy(s)));
        }
        v = v * 10 + d as i64;
        i += 1;
    }
    Ok(if neg { -v } else { v })
}

#[inline(always)]
fn fast_parse_i32(s: &[u8]) -> Result<i32, String> {
    Ok(fast_parse_i64(s)? as i32)
}

#[inline(always)]
fn fast_parse_f64(s: &[u8]) -> Result<f64, String> {
    if s.is_empty() {
        return Ok(0.0);
    }
    let mut i = 0usize;
    let mut neg = false;
    if s[0] == b'-' {
        neg = true;
        i = 1;
    }
    let mut int: i64 = 0;
    while i < s.len() && s[i] != b'.' {
        let d = s[i].wrapping_sub(b'0');
        if d > 9 {
            return Err(format!("bad f64 int in {:?}", String::from_utf8_lossy(s)));
        }
        int = int * 10 + d as i64;
        i += 1;
    }
    let mut frac: i64 = 0;
    let mut flen = 0i32;
    if i < s.len() && s[i] == b'.' {
        i += 1;
        while i < s.len() {
            let d = s[i].wrapping_sub(b'0');
            if d > 9 {
                return Err(format!("bad f64 frac in {:?}", String::from_utf8_lossy(s)));
            }
            frac = frac * 10 + d as i64;
            flen += 1;
            i += 1;
        }
    }
    let mut v = int as f64;
    if flen > 0 {
        v += (frac as f64) / 10f64.powi(flen);
    }
    Ok(if neg { -v } else { v })
}

// days-from-civil (Howard Hinnant)
#[inline(always)]
fn days_from_civil(y: i32, m: i32, d: i32) -> i32 {
    let (y, m) = if m <= 2 { (y - 1, m + 12) } else { (y, m) };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m - 3) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe
}

#[inline(always)]
fn fast_parse_date32_yyyy_mm_dd(s: &[u8]) -> Result<i32, String> {
    // expects "YYYY-MM-DD"
    if s.len() < 10 {
        return Err("date too short".into());
    }
    let y = (s[0] - b'0') as i32 * 1000
        + (s[1] - b'0') as i32 * 100
        + (s[2] - b'0') as i32 * 10
        + (s[3] - b'0') as i32;
    let m = (s[5] - b'0') as i32 * 10 + (s[6] - b'0') as i32;
    let d = (s[8] - b'0') as i32 * 10 + (s[9] - b'0') as i32;
    Ok(days_from_civil(y, m, d) - days_from_civil(1970, 1, 1))
}

// --------------------- CSV split & type conversion ---------------------

#[inline(always)]
fn split_fields_bytes<'a>(row: &'a [u8], delim: u8) -> SmallVec<[&'a [u8]; 32]> {
    let mut out: SmallVec<[&[u8]; 32]> = SmallVec::new();

    // trim trailing CR/LF
    let mut end = row.len();
    while end > 0 && (row[end - 1] == b'\n' || row[end - 1] == b'\r') {
        end -= 1;
    }

    let mut s = 0usize;
    let mut i = 0usize;
    while i <= end {
        let e = if i == end {
            end
        } else if let Some(j) = memchr(delim, &row[i..end]) {
            i + j
        } else {
            end
        };
        out.push(&row[s..e]);
        if e == end {
            break;
        }
        i = e + 1;
        s = i;
    }
    out
}

#[inline(always)]
fn convert_bytes(s: &[u8], dt: &DataType) -> Result<Vec<u8>, String> {
    use DataType::*;
    match dt {
        Int64 => Ok(i64_to_order_preserving_bytes(fast_parse_i64(s)?).to_vec()),
        Int32 => Ok(i32_to_order_preserving_bytes(fast_parse_i32(s)?).to_vec()),
        Float64 => Ok(f64_to_order_preserving_bytes(fast_parse_f64(s)?).to_vec()),
        Float32 => Ok(f32_to_order_preserving_bytes(fast_parse_f64(s)? as f32).to_vec()),
        Date32 => Ok(i32_to_order_preserving_bytes(fast_parse_date32_yyyy_mm_dd(s)?).to_vec()),
        Utf8 | LargeUtf8 | Binary | LargeBinary => Ok(s.to_vec()),
        other => Err(format!("unsupported type: {:?}", other)),
    }
}

// --------------------- Public converters ---------------------

/// Convert CSV → KVBin using std::io::BufReader (robust line path).
/// Writes optional checkpoint index (.idx) every `idx_every` records (0 = none).
/// Returns (rows_written, checkpoints_written).
pub fn convert_csv_to_bin_with_index_bufio(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    cfg: CsvDirectConfig,
    idx_every: usize,
) -> Result<(u64, u64), String> {
    // input
    let f_in = File::open(&input).map_err(|e| format!("open input: {e}"))?;
    let mut rdr = BufReader::new(f_in);

    // output
    let f_out = File::create(&output).map_err(|e| format!("create out: {e}"))?;
    let mut out = BufWriter::new(f_out);

    // optional index file
    let mut idx_file = if idx_every > 0 {
        let idx_path = output.as_ref().with_extension("idx");
        Some(BufWriter::new(
            File::create(&idx_path).map_err(|e| format!("create idx: {e}"))?,
        ))
    } else {
        None
    };

    // separate handle to query current output length when checkpointing
    let mut out_len_handle = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&output)
        .map_err(|e| format!("reopen out for pos: {e}"))?;

    let schema: &Schema = &cfg.schema;

    // skip header if present
    let mut line = String::new();
    if cfg.has_headers {
        line.clear();
        rdr.read_line(&mut line)
            .map_err(|e| format!("read header: {e}"))?;
    }

    // bounds check
    let max_needed = cfg
        .key_columns
        .iter()
        .chain(cfg.value_columns.iter())
        .copied()
        .max()
        .unwrap_or(0);

    let mut rows: u64 = 0;
    let mut checkpoints: u64 = 0;

    loop {
        line.clear();
        let n = rdr
            .read_line(&mut line)
            .map_err(|e| format!("read line: {e}"))?;
        if n == 0 {
            break; // EOF
        }
        if line.trim().is_empty() {
            continue;
        }

        let fields = split_fields_bytes(line.as_bytes(), cfg.delimiter);
        if fields.len() <= max_needed {
            // malformed line, skip
            continue;
        }

        // build key
        let mut key = Vec::with_capacity(64);
        for (i, &col) in cfg.key_columns.iter().enumerate() {
            if i > 0 {
                key.push(0);
            }
            let dt = schema.field(col).data_type();
            key.extend(convert_bytes(fields[col], dt)?);
        }

        // build value
        let mut val = Vec::with_capacity(64);
        for (i, &col) in cfg.value_columns.iter().enumerate() {
            if i > 0 {
                val.push(0);
            }
            let dt = schema.field(col).data_type();
            val.extend(convert_bytes(fields[col], dt)?);
        }

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

        // periodic checkpoint
        if idx_every > 0 && rows % (idx_every as u64) == 0 {
            out.flush().map_err(|e| e.to_string())?;
            // current output length = seek to end
            let pos = out_len_handle
                .seek(SeekFrom::End(0))
                .map_err(|e| format!("seek out for pos: {e}"))?;
            if let Some(ref mut idx) = idx_file {
                idx.write_all(&pos.to_le_bytes())
                    .map_err(|e| e.to_string())?;
                checkpoints += 1;
            }
        }
    }

    out.flush().map_err(|e| e.to_string())?;
    if let Some(mut idx) = idx_file {
        idx.flush().map_err(|e| e.to_string())?;
    }

    Ok((rows, checkpoints))
}

/// Back-compat wrapper (no index).
pub fn convert_csv_to_bin(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    cfg: CsvDirectConfig,
) -> Result<(), String> {
    let (_rows, _ckpts) = convert_csv_to_bin_with_index_bufio(input, output, cfg, 0)?;
    Ok(())
}
