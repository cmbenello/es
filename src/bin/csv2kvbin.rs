use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use clap::Parser;

use es::CsvDirectConfig;
use es::kvbin::convert_csv_to_bin_with_index_bufio;

#[derive(Parser, Debug)]
#[command(name = "csv2kvbin", version, about = "CSV -> KVBin pre-binarizer")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,

    /// CSV delimiter. If omitted, we auto-detect (',' vs '|') from the first line.
    #[arg(long)]
    delim: Option<String>,

    /// CSV has headers. If omitted, we auto-detect from the first line.
    #[arg(long)]
    headers: Option<bool>,

    #[arg(long)]
    keys: String, // e.g. "10,0"
    #[arg(long)]
    vals: String, // e.g. "5,6"
    #[arg(long, default_value_t = 16)]
    threads: usize, // reserved
    #[arg(long = "buf-mb", default_value_t = 32)]
    buf_mb: usize, // reserved
    /// Write an index offset every N records into <output>.idx (0 = no index)
    #[arg(long = "idx-every", default_value_t = 100_000)]
    idx_every: usize,
}

fn parse_idx_list(s: &str) -> Vec<usize> {
    s.split(',')
        .filter(|t| !t.is_empty())
        .map(|t| t.parse().expect("bad index"))
        .collect()
}

fn lineitem_schema() -> Arc<Schema> {
    use DataType::*;
    Arc::new(Schema::new(vec![
        Field::new("l_orderkey", Int64, false),        // 0
        Field::new("l_partkey", Int64, false),         // 1
        Field::new("l_suppkey", Int64, false),         // 2
        Field::new("l_linenumber", Int32, false),      // 3
        Field::new("l_quantity", Float64, false),      // 4
        Field::new("l_extendedprice", Float64, false), // 5
        Field::new("l_discount", Float64, false),      // 6
        Field::new("l_tax", Float64, false),           // 7
        Field::new("l_returnflag", Utf8, false),       // 8
        Field::new("l_linestatus", Utf8, false),       // 9
        Field::new("l_shipdate", Date32, false),       // 10
        Field::new("l_commitdate", Date32, false),     // 11
        Field::new("l_receiptdate", Date32, false),    // 12
        Field::new("l_shipinstruct", Utf8, false),     // 13
        Field::new("l_shipmode", Utf8, false),         // 14
        Field::new("l_comment", Utf8, false),          // 15
    ]))
}

fn sniff_csv(input: &PathBuf) -> (u8, bool) {
    // Read a little from the start
    let mut f = File::open(input).expect("open input");
    let mut buf = [0u8; 4096];
    let n = f.read(&mut buf).unwrap_or(0);
    let s = std::str::from_utf8(&buf[..n]).unwrap_or("");

    // delimiter: pick the one with more occurrences on the first line
    let first_line = s.lines().next().unwrap_or(s);
    let commas = first_line.matches(',').count();
    let pipes = first_line.matches('|').count();
    let delim = if pipes > commas { b'|' } else { b',' };

    // headers: if first cell looks like "l_orderkey", assume headers
    let has_headers = first_line.contains("l_orderkey");

    (delim, has_headers)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("CWD = {}", std::env::current_dir()?.display());
    eprintln!("input = {}", args.input.display());
    eprintln!("output = {}", args.output.display());
    if !args.input.exists() {
        eprintln!("ERROR: input file does not exist");
        std::process::exit(2);
    }

    // Auto-detect when flags not provided
    let (sniff_delim, sniff_headers) = sniff_csv(&args.input);
    let delim_byte = args
        .delim
        .as_ref()
        .map(|d| d.as_bytes()[0])
        .unwrap_or(sniff_delim);
    let has_headers = args.headers.unwrap_or(sniff_headers);

    let mut cfg = CsvDirectConfig::new(lineitem_schema());
    cfg.delimiter = delim_byte;
    cfg.key_columns = parse_idx_list(&args.keys);
    cfg.value_columns = parse_idx_list(&args.vals);
    cfg.has_headers = has_headers;

    let (rows, checkpoints) =
        convert_csv_to_bin_with_index_bufio(&args.input, &args.output, cfg, args.idx_every)?;

    // print correct .idx path (no double suffix)
    let idx_path = format!("{}{}", args.output.display(), ".idx");
    eprintln!(
        "Done. Wrote {} records to {}. Checkpoints written: {} to {}",
        rows,
        args.output.display(),
        checkpoints,
        idx_path
    );

    if rows == 0 {
        eprintln!(
            "WARNING: wrote 0 rows. Likely delimiter/headers mismatch. \
             Try explicitly: --delim ',' --headers or --delim '|' --no-headers."
        );
    }

    Ok(())
}
