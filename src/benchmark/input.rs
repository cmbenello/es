use crate::{CsvDirectConfig, CsvInputDirect, GenSortInputDirect, SortInput};
use arrow::datatypes::{DataType, Field, Schema};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

pub trait BenchmarkInputProvider {
    fn create_sort_input(&self) -> Result<Box<dyn SortInput>, Box<dyn std::error::Error>>;
    fn estimate_data_size_mb(&self) -> Result<f64, Box<dyn std::error::Error>>;
    fn get_entry_count(&self) -> Option<usize>;
    fn get_description(&self) -> String;
}

pub struct GenSortInputProvider {
    pub path: PathBuf,
}

impl BenchmarkInputProvider for GenSortInputProvider {
    fn create_sort_input(&self) -> Result<Box<dyn SortInput>, Box<dyn std::error::Error>> {
        Ok(Box::new(GenSortInputDirect::new(&self.path)?))
    }

    fn estimate_data_size_mb(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let gensort_input = GenSortInputDirect::new(&self.path)?;
        let file_size = gensort_input.file_size()?;
        Ok(file_size as f64 / (1024.0 * 1024.0))
    }

    fn get_entry_count(&self) -> Option<usize> {
        if let Ok(gensort_input) = GenSortInputDirect::new(&self.path) {
            Some(gensort_input.len())
        } else {
            None
        }
    }

    fn get_description(&self) -> String {
        format!("GenSort file: {:?}", self.path)
    }
}

pub struct LineitemCsvInputProvider {
    pub path: PathBuf,
    pub key_columns: Vec<usize>,
    pub value_columns: Vec<usize>,
    pub delimiter: char,
    pub has_headers: bool,
}

impl LineitemCsvInputProvider {
    pub fn new(
        path: PathBuf,
        key_columns: Vec<usize>,
        value_columns: Vec<usize>,
        delimiter: char,
        has_headers: bool,
    ) -> Self {
        Self {
            path,
            key_columns,
            value_columns,
            delimiter,
            has_headers,
        }
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

    fn get_datatype_size(data_type: &DataType, field_name: &str) -> usize {
        match data_type {
            DataType::Int64
            | DataType::UInt64
            | DataType::Float64
            | DataType::Date64
            | DataType::Time64(_)
            | DataType::Timestamp(_, _) => 8,
            DataType::Int32
            | DataType::UInt32
            | DataType::Float32
            | DataType::Date32
            | DataType::Time32(_) => 4,
            DataType::Int16 | DataType::UInt16 => 2,
            DataType::Int8 | DataType::UInt8 | DataType::Boolean => 1,
            DataType::Utf8 | DataType::LargeUtf8 => match field_name {
                "l_returnflag" | "l_linestatus" => 1,
                "l_shipmode" => 7,
                "l_shipinstruct" => 25,
                "l_comment" => 44,
                _ => 20,
            },
            DataType::Binary | DataType::LargeBinary => 32,
            _ => 8,
        }
    }
}

impl BenchmarkInputProvider for LineitemCsvInputProvider {
    fn create_sort_input(&self) -> Result<Box<dyn SortInput>, Box<dyn std::error::Error>> {
        let schema = Self::get_lineitem_schema();

        let mut config = CsvDirectConfig::new(schema);
        config.delimiter = self.delimiter as u8;
        config.key_columns = self.key_columns.clone();
        config.value_columns = self.value_columns.clone();
        config.has_headers = self.has_headers;

        Ok(Box::new(CsvInputDirect::new(
            self.path.to_str().unwrap(),
            config,
        )?))
    }

    fn estimate_data_size_mb(&self) -> Result<f64, Box<dyn std::error::Error>> {
        const SAMPLE_SIZE: usize = 1000;
        let schema = Self::get_lineitem_schema();
        let all_columns = [&self.key_columns[..], &self.value_columns[..]].concat();

        let mut line_count = 0;
        let mut total_selected_bytes = 0;
        let mut total_line_bytes = 0;

        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);

        for (i, line) in reader.lines().enumerate() {
            let line = line?;

            if self.has_headers && i == 0 {
                continue;
            }

            if line_count >= SAMPLE_SIZE {
                break;
            }

            total_line_bytes += line.len() + 1;

            let fields: Vec<&str> = line.split(self.delimiter).collect();

            for &col_idx in &all_columns {
                if col_idx < fields.len() {
                    let size = if col_idx < schema.fields().len() {
                        let field_type = schema.field(col_idx).data_type();
                        match field_type {
                            DataType::Utf8 => fields[col_idx].len() + 2,
                            _ => {
                                Self::get_datatype_size(field_type, schema.field(col_idx).name())
                                    + 1
                            }
                        }
                    } else {
                        fields[col_idx].len() + 2
                    };
                    total_selected_bytes += size;
                }
            }

            line_count += 1;
        }

        if line_count > 0 {
            let file_metadata = std::fs::metadata(&self.path)?;
            let file_size = file_metadata.len();

            let avg_line_size = total_line_bytes as f64 / line_count as f64;
            let estimated_total_rows = (file_size as f64 / avg_line_size) as usize;

            let avg_selected_bytes_per_row = total_selected_bytes as f64 / line_count as f64;

            let estimated_mb =
                (estimated_total_rows as f64 * avg_selected_bytes_per_row) / (1024.0 * 1024.0);

            Ok(estimated_mb)
        } else {
            let bytes_per_row: usize = all_columns
                .iter()
                .map(|&col_idx| {
                    if col_idx < schema.fields().len() {
                        let field = schema.field(col_idx);
                        Self::get_datatype_size(field.data_type(), field.name())
                    } else {
                        8
                    }
                })
                .sum::<usize>()
                + 4;

            let file_metadata = std::fs::metadata(&self.path)?;
            let file_size = file_metadata.len();
            let estimated_rows = (file_size as f64 / (1024.0 * 1024.0 * 1024.0)) * 6_000_000.0;
            Ok((estimated_rows * bytes_per_row as f64) / (1024.0 * 1024.0))
        }
    }

    fn get_entry_count(&self) -> Option<usize> {
        // For CSV files, entry count is determined during processing
        None
    }

    fn get_description(&self) -> String {
        format!(
            "CSV file: {:?}, key_columns: {:?}, value_columns: {:?}, delimiter: '{}', headers: {}",
            self.path, self.key_columns, self.value_columns, self.delimiter, self.has_headers
        )
    }
}

pub struct YellowTaxiCsvInputProvider {
    pub path: PathBuf,
    pub key_columns: Vec<usize>,
    pub value_columns: Vec<usize>,
    pub delimiter: char,
    pub has_headers: bool,
}

impl YellowTaxiCsvInputProvider {
    pub fn new(
        path: PathBuf,
        key_columns: Vec<usize>,
        value_columns: Vec<usize>,
        delimiter: char,
        has_headers: bool,
    ) -> Self {
        Self {
            path,
            key_columns,
            value_columns,
            delimiter,
            has_headers,
        }
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

    fn get_datatype_size(data_type: &DataType, field_name: &str) -> usize {
        match data_type {
            DataType::Int64
            | DataType::UInt64
            | DataType::Float64
            | DataType::Date64
            | DataType::Time64(_)
            | DataType::Timestamp(_, _) => 8,
            DataType::Int32
            | DataType::UInt32
            | DataType::Float32
            | DataType::Date32
            | DataType::Time32(_) => 4,
            DataType::Int16 | DataType::UInt16 => 2,
            DataType::Int8 | DataType::UInt8 | DataType::Boolean => 1,
            DataType::Utf8 | DataType::LargeUtf8 => match field_name {
                "vendor_id" => 3,        // VTS, CMT, DDS
                "pickup_datetime" => 19, // 2012-08-31 22:00:00
                "dropoff_datetime" => 19,
                "payment_type_desc" => 3,                   // CSH, CRE, etc
                "cab_type" => 6,                            // yellow, green
                "pickup_geom" | "dropoff_geom" => 60,       // WKT geometry strings
                "pickup_borough" | "dropoff_borough" => 10, // Manhattan, Queens, etc
                "pickup_zone" | "dropoff_zone" => 30,
                _ => 10,
            },
            DataType::Binary | DataType::LargeBinary => 32,
            _ => 8,
        }
    }
}

impl BenchmarkInputProvider for YellowTaxiCsvInputProvider {
    fn create_sort_input(&self) -> Result<Box<dyn SortInput>, Box<dyn std::error::Error>> {
        let schema = Self::get_yellow_taxi_schema();

        let mut config = CsvDirectConfig::new(schema);
        config.delimiter = self.delimiter as u8;
        config.key_columns = self.key_columns.clone();
        config.value_columns = self.value_columns.clone();
        config.has_headers = self.has_headers;

        Ok(Box::new(CsvInputDirect::new(
            self.path.to_str().unwrap(),
            config,
        )?))
    }

    fn estimate_data_size_mb(&self) -> Result<f64, Box<dyn std::error::Error>> {
        const SAMPLE_SIZE: usize = 1000;
        let schema = Self::get_yellow_taxi_schema();
        let all_columns = [&self.key_columns[..], &self.value_columns[..]].concat();

        let mut line_count = 0;
        let mut total_selected_bytes = 0;
        let mut total_line_bytes = 0;

        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);

        for (i, line) in reader.lines().enumerate() {
            let line = line?;

            if self.has_headers && i == 0 {
                continue;
            }

            if line_count >= SAMPLE_SIZE {
                break;
            }

            total_line_bytes += line.len() + 1;

            let fields: Vec<&str> = line.split(self.delimiter).collect();

            for &col_idx in &all_columns {
                if col_idx < fields.len() {
                    let size = if col_idx < schema.fields().len() {
                        let field_type = schema.field(col_idx).data_type();
                        match field_type {
                            DataType::Utf8 => fields[col_idx].len() + 2,
                            _ => {
                                Self::get_datatype_size(field_type, schema.field(col_idx).name())
                                    + 1
                            }
                        }
                    } else {
                        fields[col_idx].len() + 2
                    };
                    total_selected_bytes += size;
                }
            }

            line_count += 1;
        }

        if line_count > 0 {
            let file_metadata = std::fs::metadata(&self.path)?;
            let file_size = file_metadata.len();

            let avg_line_size = total_line_bytes as f64 / line_count as f64;
            let estimated_total_rows = (file_size as f64 / avg_line_size) as usize;

            let avg_selected_bytes_per_row = total_selected_bytes as f64 / line_count as f64;

            let estimated_mb =
                (estimated_total_rows as f64 * avg_selected_bytes_per_row) / (1024.0 * 1024.0);

            Ok(estimated_mb)
        } else {
            let bytes_per_row: usize = all_columns
                .iter()
                .map(|&col_idx| {
                    if col_idx < schema.fields().len() {
                        let field = schema.field(col_idx);
                        Self::get_datatype_size(field.data_type(), field.name())
                    } else {
                        8
                    }
                })
                .sum::<usize>()
                + 4;

            let file_metadata = std::fs::metadata(&self.path)?;
            let file_size = file_metadata.len();
            // Estimate ~20M rows per 8GB for yellow taxi data
            let estimated_rows = (file_size as f64 / (1024.0 * 1024.0 * 1024.0)) * 2_500_000.0;
            Ok((estimated_rows * bytes_per_row as f64) / (1024.0 * 1024.0))
        }
    }

    fn get_entry_count(&self) -> Option<usize> {
        // For CSV files, entry count is determined during processing
        None
    }

    fn get_description(&self) -> String {
        format!(
            "Yellow Taxi CSV file: {:?}, key_columns: {:?}, value_columns: {:?}, delimiter: '{}', headers: {}",
            self.path, self.key_columns, self.value_columns, self.delimiter, self.has_headers
        )
    }
}
