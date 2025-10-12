use crate::{SortOutput, order_preserving_encoding::decode_bytes};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

pub trait OutputVerifier {
    fn verify(&self, output: &dyn SortOutput) -> Result<(), Box<dyn std::error::Error>>;
}

/// Simple verifier that only checks sort order without detailed output
pub struct SimpleVerifier {
    pub print_sample: bool,
}

impl SimpleVerifier {
    pub fn new() -> Self {
        Self { print_sample: true }
    }

    pub fn new_quiet() -> Self {
        Self {
            print_sample: false,
        }
    }
}

impl Default for SimpleVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputVerifier for SimpleVerifier {
    fn verify(&self, output: &dyn SortOutput) -> Result<(), Box<dyn std::error::Error>> {
        let mut prev_key: Option<Vec<u8>> = None;
        let mut count = 0;

        let mut first_records = Vec::new();
        let mut last_records = Vec::new();

        for (key, _value) in output.iter() {
            if self.print_sample {
                if first_records.len() < 5 {
                    first_records.push(key.clone());
                }
                last_records.push(key.clone());
                if last_records.len() > 5 {
                    last_records.remove(0); // Keep only last 5
                }
            }

            if let Some(ref prev) = prev_key {
                if key < *prev {
                    eprintln!("ERROR: Sort order violation at record {}", count);
                    eprintln!("  Previous key: {:?}", prev);
                    eprintln!("  Current key: {:?}", key);
                    return Err("Sort order violation".into());
                }
            }
            prev_key = Some(key);
            count += 1;
        }

        println!("Verified {} records - all correctly sorted!", count);

        if self.print_sample {
            println!("\nFirst 5 records:");
            for (i, key) in first_records.iter().enumerate() {
                println!("  Record {}: Key = {:?}", i, key);
            }

            println!("\nLast 5 records:");
            for (i, key) in last_records.iter().enumerate() {
                println!("  Record {}: Key = {:?}", count - 5 + i, key);
            }
        }

        Ok(())
    }
}

/// Detailed CSV verifier that provides comprehensive statistics and key decoding
pub struct LineitemCsvVerifier {
    pub key_columns: Vec<usize>,
}

impl LineitemCsvVerifier {
    pub fn new(key_columns: Vec<usize>) -> Self {
        Self { key_columns }
    }

    fn get_lineitem_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("l_orderkey", DataType::Int64, false), // 0
            Field::new("l_partkey", DataType::Int64, false),  // 1
            Field::new("l_suppkey", DataType::Int64, false),  // 2
            Field::new("l_linenumber", DataType::Int32, false), // 3
            Field::new("l_quantity", DataType::Float64, false), // 4
            Field::new("l_extendedprice", DataType::Float64, false), // 5
            Field::new("l_discount", DataType::Float64, false), // 6
            Field::new("l_tax", DataType::Float64, false),    // 7
            Field::new("l_returnflag", DataType::Utf8, false), // 8
            Field::new("l_linestatus", DataType::Utf8, false), // 9
            Field::new("l_shipdate", DataType::Date32, false), // 10
            Field::new("l_commitdate", DataType::Date32, false), // 11
            Field::new("l_receiptdate", DataType::Date32, false), // 12
            Field::new("l_shipinstruct", DataType::Utf8, false), // 13
            Field::new("l_shipmode", DataType::Utf8, false),  // 14
            Field::new("l_comment", DataType::Utf8, false),   // 15
        ]))
    }

    fn get_column_names() -> &'static [&'static str] {
        &[
            "l_orderkey",      // 0
            "l_partkey",       // 1
            "l_suppkey",       // 2
            "l_linenumber",    // 3
            "l_quantity",      // 4
            "l_extendedprice", // 5
            "l_discount",      // 6
            "l_tax",           // 7
            "l_returnflag",    // 8
            "l_linestatus",    // 9
            "l_shipdate",      // 10
            "l_commitdate",    // 11
            "l_receiptdate",   // 12
            "l_shipinstruct",  // 13
            "l_shipmode",      // 14
            "l_comment",       // 15
        ]
    }

    fn decode_key(&self, key: &[u8]) -> String {
        let schema = Self::get_lineitem_schema();
        let column_names = Self::get_column_names();
        let mut parts = Vec::new();
        let mut offset = 0;

        for &col_idx in &self.key_columns {
            if offset >= key.len() {
                break;
            }

            let (field_type, expected_size) = if col_idx < schema.fields().len() {
                let field = schema.field(col_idx);
                let type_str = match field.data_type() {
                    DataType::Int64 => "Int64",
                    DataType::Int32 => "Int32",
                    DataType::Float64 => "Float64",
                    DataType::Date32 => "Date32",
                    DataType::Utf8 => "Utf8",
                    _ => "Unknown",
                };
                let size = match field.data_type() {
                    DataType::Int64 => 8,
                    DataType::Int32 => 4,
                    DataType::Float64 => 8,
                    DataType::Date32 => 4,
                    DataType::Utf8 => 0, // Variable length
                    _ => 0,
                };
                (type_str, size)
            } else {
                ("Unknown", 0)
            };

            if expected_size > 0 {
                let field_end = offset + expected_size;
                if field_end > key.len() {
                    break;
                }

                let field_bytes = &key[offset..field_end];
                let decoded = decode_bytes(field_bytes, field_type)
                    .unwrap_or_else(|_| format!("{:?}", field_bytes));

                if col_idx < column_names.len() {
                    parts.push(format!("{}={}", column_names[col_idx], decoded));
                } else {
                    parts.push(format!("col{}={}", col_idx, decoded));
                }

                offset = field_end;
                if offset < key.len() && key[offset] == 0 {
                    offset += 1;
                }
            } else {
                // Variable length string
                let field_end = key[offset..]
                    .iter()
                    .position(|&b| b == 0)
                    .map(|pos| offset + pos)
                    .unwrap_or(key.len());

                let field_bytes = &key[offset..field_end];
                let decoded = String::from_utf8_lossy(field_bytes);

                if col_idx < column_names.len() {
                    parts.push(format!("{}={}", column_names[col_idx], decoded));
                } else {
                    parts.push(format!("col{}={}", col_idx, decoded));
                }

                offset = field_end;
                if offset < key.len() && key[offset] == 0 {
                    offset += 1;
                }
            }
        }

        parts.join(", ")
    }
}

impl OutputVerifier for LineitemCsvVerifier {
    fn verify(&self, output: &dyn SortOutput) -> Result<(), Box<dyn std::error::Error>> {
        let mut prev_key: Option<Vec<u8>> = None;
        let mut count: usize = 0;

        // Store first and last 10 records with their sizes
        struct RecordInfo {
            decoded: String,
            key_size: usize,
            value_size: usize,
        }

        let mut first_records = Vec::new();
        let mut last_records = Vec::new();

        // Track total sizes for statistics
        let mut total_key_size = 0usize;
        let mut total_value_size = 0usize;

        for (key, value) in output.iter() {
            let key_size = key.len();
            let value_size = value.len();

            // Accumulate sizes
            total_key_size += key_size;
            total_value_size += value_size;

            // Store first 10 records
            if first_records.len() < 10 {
                first_records.push(RecordInfo {
                    decoded: self.decode_key(&key),
                    key_size,
                    value_size,
                });
            }

            // Keep last 10 records in a sliding window
            last_records.push(RecordInfo {
                decoded: self.decode_key(&key),
                key_size,
                value_size,
            });
            if last_records.len() > 10 {
                last_records.remove(0);
            }

            // Check sort order
            if let Some(ref prev) = prev_key {
                if key < *prev {
                    eprintln!("ERROR: Sort order violation at record {}", count);
                    eprintln!("  Previous key: {:?}", self.decode_key(prev));
                    eprintln!("  Current key: {:?}", self.decode_key(&key));
                    return Err("Sort order violation".into());
                }
            }
            prev_key = Some(key);
            count += 1;
        }

        println!("    Verified {} records - all correctly sorted!", count);

        // Print size statistics
        println!("\n    === SIZE STATISTICS ===");
        println!("    Total records: {}", count);
        println!(
            "    Average key size: {:.1} bytes",
            total_key_size as f64 / count as f64
        );
        println!(
            "    Average value size: {:.1} bytes",
            total_value_size as f64 / count as f64
        );
        println!(
            "    Total key bytes: {} ({:.2} MB)",
            total_key_size,
            total_key_size as f64 / (1024.0 * 1024.0)
        );
        println!(
            "    Total value bytes: {} ({:.2} MB)",
            total_value_size,
            total_value_size as f64 / (1024.0 * 1024.0)
        );
        println!(
            "    Total sort output: {} bytes ({:.2} MB)",
            total_key_size + total_value_size,
            (total_key_size + total_value_size) as f64 / (1024.0 * 1024.0)
        );

        // Print first 10 records with sizes
        println!(
            "\n    First {} records (with sizes):",
            first_records.len().min(10)
        );
        println!(
            "    {:>5} {:>10} {:>10}  {}",
            "#", "Key Size", "Val Size", "Key Values"
        );
        println!("    {}", "-".repeat(80));
        for (i, record) in first_records.iter().enumerate() {
            println!(
                "    {:>5} {:>10} {:>10}  {}",
                i, record.key_size, record.value_size, record.decoded
            );
        }

        // Print last 10 records with sizes
        println!(
            "\n    Last {} records (with sizes):",
            last_records.len().min(10)
        );
        println!(
            "    {:>5} {:>10} {:>10}  {}",
            "#", "Key Size", "Val Size", "Key Values"
        );
        println!("    {}", "-".repeat(80));
        let start_idx = count.saturating_sub(10);
        for (i, record) in last_records.iter().enumerate() {
            println!(
                "    {:>5} {:>10} {:>10}  {}",
                start_idx + i,
                record.key_size,
                record.value_size,
                record.decoded
            );
        }

        Ok(())
    }
}

/// Yellow Taxi CSV verifier that provides comprehensive statistics and key decoding
pub struct YellowTaxiCsvVerifier {
    pub key_columns: Vec<usize>,
}

impl YellowTaxiCsvVerifier {
    pub fn new(key_columns: Vec<usize>) -> Self {
        Self { key_columns }
    }

    fn get_yellow_taxi_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("trip_id", DataType::Int64, false),  // 0
            Field::new("vendor_id", DataType::Utf8, false), // 1
            Field::new("pickup_datetime", DataType::Utf8, false), // 2
            Field::new("dropoff_datetime", DataType::Utf8, false), // 3
            Field::new("store_and_fwd_flag", DataType::Int32, false), // 4
            Field::new("rate_code", DataType::Int32, false), // 5
            Field::new("pickup_longitude", DataType::Float64, false), // 6
            Field::new("pickup_latitude", DataType::Float64, false), // 7
            Field::new("dropoff_longitude", DataType::Float64, false), // 8
            Field::new("dropoff_latitude", DataType::Float64, false), // 9
            Field::new("payment_type", DataType::Int32, false), // 10
            Field::new("fare_amount", DataType::Float64, false), // 11
            Field::new("surcharge", DataType::Float64, false), // 12
            Field::new("mta_tax", DataType::Float64, false), // 13
            Field::new("tip_amount", DataType::Float64, false), // 14
            Field::new("tolls_amount", DataType::Float64, false), // 15
            Field::new("improvement_surcharge", DataType::Float64, false), // 16
            Field::new("total_amount", DataType::Float64, false), // 17
            Field::new("congestion_surcharge", DataType::Float64, false), // 18
            Field::new("airport_fee", DataType::Float64, false), // 19
            Field::new("payment_type_desc", DataType::Utf8, false), // 20
            Field::new("trip_type", DataType::Int32, false), // 21
            Field::new("pickup_geom", DataType::Utf8, false), // 22
            Field::new("dropoff_geom", DataType::Utf8, false), // 23
            Field::new("cab_type", DataType::Utf8, false),  // 24
            // Additional columns (25-50) - treat as Utf8 for flexibility
            Field::new("col_25", DataType::Utf8, true), // 25
            Field::new("col_26", DataType::Utf8, true), // 26
            Field::new("col_27", DataType::Utf8, true), // 27
            Field::new("col_28", DataType::Utf8, true), // 28
            Field::new("col_29", DataType::Utf8, true), // 29
            Field::new("col_30", DataType::Utf8, true), // 30
            Field::new("col_31", DataType::Utf8, true), // 31
            Field::new("col_32", DataType::Utf8, true), // 32
            Field::new("col_33", DataType::Utf8, true), // 33
            Field::new("pickup_borough", DataType::Utf8, true), // 34
            Field::new("col_35", DataType::Utf8, true), // 35
            Field::new("col_36", DataType::Utf8, true), // 36
            Field::new("col_37", DataType::Utf8, true), // 37
            Field::new("col_38", DataType::Utf8, true), // 38
            Field::new("pickup_zone", DataType::Utf8, true), // 39
            Field::new("col_40", DataType::Utf8, true), // 40
            Field::new("col_41", DataType::Utf8, true), // 41
            Field::new("col_42", DataType::Utf8, true), // 42
            Field::new("col_43", DataType::Utf8, true), // 43
            Field::new("dropoff_borough", DataType::Utf8, true), // 44
            Field::new("col_45", DataType::Utf8, true), // 45
            Field::new("col_46", DataType::Utf8, true), // 46
            Field::new("col_47", DataType::Utf8, true), // 47
            Field::new("col_48", DataType::Utf8, true), // 48
            Field::new("dropoff_zone", DataType::Utf8, true), // 49
            Field::new("col_50", DataType::Utf8, true), // 50
        ]))
    }

    fn get_column_names() -> &'static [&'static str] {
        &[
            "trip_id",               // 0
            "vendor_id",             // 1
            "pickup_datetime",       // 2
            "dropoff_datetime",      // 3
            "store_and_fwd_flag",    // 4
            "rate_code",             // 5
            "pickup_longitude",      // 6
            "pickup_latitude",       // 7
            "dropoff_longitude",     // 8
            "dropoff_latitude",      // 9
            "payment_type",          // 10
            "fare_amount",           // 11
            "surcharge",             // 12
            "mta_tax",               // 13
            "tip_amount",            // 14
            "tolls_amount",          // 15
            "improvement_surcharge", // 16
            "total_amount",          // 17
            "congestion_surcharge",  // 18
            "airport_fee",           // 19
            "payment_type_desc",     // 20
            "trip_type",             // 21
            "pickup_geom",           // 22
            "dropoff_geom",          // 23
            "cab_type",              // 24
            "col_25",                // 25
            "col_26",                // 26
            "col_27",                // 27
            "col_28",                // 28
            "col_29",                // 29
            "col_30",                // 30
            "col_31",                // 31
            "col_32",                // 32
            "col_33",                // 33
            "pickup_borough",        // 34
            "col_35",                // 35
            "col_36",                // 36
            "col_37",                // 37
            "col_38",                // 38
            "pickup_zone",           // 39
            "col_40",                // 40
            "col_41",                // 41
            "col_42",                // 42
            "col_43",                // 43
            "dropoff_borough",       // 44
            "col_45",                // 45
            "col_46",                // 46
            "col_47",                // 47
            "col_48",                // 48
            "dropoff_zone",          // 49
            "col_50",                // 50
        ]
    }

    fn decode_key(&self, key: &[u8]) -> String {
        let schema = Self::get_yellow_taxi_schema();
        let column_names = Self::get_column_names();
        let mut parts = Vec::new();
        let mut offset = 0;

        for &col_idx in &self.key_columns {
            if offset >= key.len() {
                break;
            }

            let (field_type, expected_size) = if col_idx < schema.fields().len() {
                let field = schema.field(col_idx);
                let type_str = match field.data_type() {
                    DataType::Int64 => "Int64",
                    DataType::Int32 => "Int32",
                    DataType::Float64 => "Float64",
                    DataType::Date32 => "Date32",
                    DataType::Utf8 => "Utf8",
                    _ => "Unknown",
                };
                let size = match field.data_type() {
                    DataType::Int64 => 8,
                    DataType::Int32 => 4,
                    DataType::Float64 => 8,
                    DataType::Date32 => 4,
                    DataType::Utf8 => 0, // Variable length
                    _ => 0,
                };
                (type_str, size)
            } else {
                ("Unknown", 0)
            };

            if expected_size > 0 {
                let field_end = offset + expected_size;
                if field_end > key.len() {
                    break;
                }

                let field_bytes = &key[offset..field_end];
                let decoded = decode_bytes(field_bytes, field_type)
                    .unwrap_or_else(|_| format!("{:?}", field_bytes));

                if col_idx < column_names.len() {
                    parts.push(format!("{}={}", column_names[col_idx], decoded));
                } else {
                    parts.push(format!("col{}={}", col_idx, decoded));
                }

                offset = field_end;
                if offset < key.len() && key[offset] == 0 {
                    offset += 1;
                }
            } else {
                // Variable length string
                let field_end = key[offset..]
                    .iter()
                    .position(|&b| b == 0)
                    .map(|pos| offset + pos)
                    .unwrap_or(key.len());

                let field_bytes = &key[offset..field_end];
                let decoded = String::from_utf8_lossy(field_bytes);

                if col_idx < column_names.len() {
                    parts.push(format!("{}={}", column_names[col_idx], decoded));
                } else {
                    parts.push(format!("col{}={}", col_idx, decoded));
                }

                offset = field_end;
                if offset < key.len() && key[offset] == 0 {
                    offset += 1;
                }
            }
        }

        parts.join(", ")
    }
}

impl OutputVerifier for YellowTaxiCsvVerifier {
    fn verify(&self, output: &dyn SortOutput) -> Result<(), Box<dyn std::error::Error>> {
        let mut prev_key: Option<Vec<u8>> = None;
        let mut count: usize = 0;

        // Store first and last 10 records with their sizes
        struct RecordInfo {
            decoded: String,
            key_size: usize,
            value_size: usize,
        }

        let mut first_records = Vec::new();
        let mut last_records = Vec::new();

        // Track total sizes for statistics
        let mut total_key_size = 0usize;
        let mut total_value_size = 0usize;

        for (key, value) in output.iter() {
            let key_size = key.len();
            let value_size = value.len();

            // Accumulate sizes
            total_key_size += key_size;
            total_value_size += value_size;

            // Store first 10 records
            if first_records.len() < 10 {
                first_records.push(RecordInfo {
                    decoded: self.decode_key(&key),
                    key_size,
                    value_size,
                });
            }

            // Keep last 10 records in a sliding window
            last_records.push(RecordInfo {
                decoded: self.decode_key(&key),
                key_size,
                value_size,
            });
            if last_records.len() > 10 {
                last_records.remove(0);
            }

            // Check sort order
            if let Some(ref prev) = prev_key {
                if key < *prev {
                    eprintln!("ERROR: Sort order violation at record {}", count);
                    eprintln!("  Previous key: {:?}", self.decode_key(prev));
                    eprintln!("  Current key: {:?}", self.decode_key(&key));
                    return Err("Sort order violation".into());
                }
            }
            prev_key = Some(key);
            count += 1;
        }

        println!("    Verified {} records - all correctly sorted!", count);

        // Print size statistics
        println!("\n    === SIZE STATISTICS ===");
        println!("    Total records: {}", count);
        println!(
            "    Average key size: {:.1} bytes",
            total_key_size as f64 / count as f64
        );
        println!(
            "    Average value size: {:.1} bytes",
            total_value_size as f64 / count as f64
        );
        println!(
            "    Total key bytes: {} ({:.2} MB)",
            total_key_size,
            total_key_size as f64 / (1024.0 * 1024.0)
        );
        println!(
            "    Total value bytes: {} ({:.2} MB)",
            total_value_size,
            total_value_size as f64 / (1024.0 * 1024.0)
        );
        println!(
            "    Total sort output: {} bytes ({:.2} MB)",
            total_key_size + total_value_size,
            (total_key_size + total_value_size) as f64 / (1024.0 * 1024.0)
        );

        // Print first 10 records with sizes
        println!(
            "\n    First {} records (with sizes):",
            first_records.len().min(10)
        );
        println!(
            "    {:>5} {:>10} {:>10}  {}",
            "#", "Key Size", "Val Size", "Key Values"
        );
        println!("    {}", "-".repeat(80));
        for (i, record) in first_records.iter().enumerate() {
            println!(
                "    {:>5} {:>10} {:>10}  {}",
                i, record.key_size, record.value_size, record.decoded
            );
        }

        // Print last 10 records with sizes
        println!(
            "\n    Last {} records (with sizes):",
            last_records.len().min(10)
        );
        println!(
            "    {:>5} {:>10} {:>10}  {}",
            "#", "Key Size", "Val Size", "Key Values"
        );
        println!("    {}", "-".repeat(80));
        let start_idx = count.saturating_sub(10);
        for (i, record) in last_records.iter().enumerate() {
            println!(
                "    {:>5} {:>10} {:>10}  {}",
                start_idx + i,
                record.key_size,
                record.value_size,
                record.decoded
            );
        }

        Ok(())
    }
}
