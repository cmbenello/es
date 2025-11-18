use std::cmp::Ordering;

use crate::ovc::offset_value_coding_u64::OVCU64;

pub type ReplacementScanner = Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>;

/// Sink that receives events from the replacement selection driver.
pub trait ReplacementSelectionSink {
    /// Called before the driver emits records for a new run.
    fn start_run(&mut self);
    /// Called for every sorted record that should be appended to the current run.
    fn push_record(&mut self, key: &[u8], value: &[u8]);
    /// Called after the driver finishes producing a run.
    fn finish_run(&mut self);
}

/// Timing information produced by the replacement selection driver.
#[derive(Default)]
pub struct ReplacementSelectionStats {
    pub records_emitted: usize,
}

mod heap;
pub use heap::run_replacement_selection;

pub(crate) fn ensure_entry_fits(entry_size: usize, limit: usize) {
    if entry_size > limit {
        panic!("Failed to append data to sort buffer, it should have space");
    }
}

pub(crate) fn next_record(
    pending_record: &mut Option<(Vec<u8>, Vec<u8>)>,
    scanner: &mut dyn Iterator<Item = (Vec<u8>, Vec<u8>)>,
) -> Option<(Vec<u8>, Vec<u8>)> {
    if let Some(record) = pending_record.take() {
        return Some(record);
    }

    scanner.next()
}

pub(crate) fn compute_ovc_delta(prev_key: Option<&[u8]>, key: &[u8]) -> OVCU64 {
    const CHUNK_SIZE: usize = 6;

    let Some(prev) = prev_key else {
        return OVCU64::initial_value();
    };

    let min_len = prev.len().min(key.len());
    for i in 0..min_len {
        match key[i].cmp(&prev[i]) {
            Ordering::Less => {
                panic!("Input must be non-decreasing for OVC encoding");
            }
            Ordering::Greater => {
                let aligned_offset = (i / CHUNK_SIZE) * CHUNK_SIZE;
                let chunk_end = (aligned_offset + CHUNK_SIZE).min(key.len());
                return OVCU64::normal_value(&key[aligned_offset..chunk_end], aligned_offset);
            }
            Ordering::Equal => continue,
        }
    }

    match key.len().cmp(&prev.len()) {
        Ordering::Greater => {
            let aligned_offset = (prev.len() / CHUNK_SIZE) * CHUNK_SIZE;
            let chunk_end = (aligned_offset + CHUNK_SIZE).min(key.len());
            OVCU64::normal_value(&key[aligned_offset..chunk_end], aligned_offset)
        }
        Ordering::Equal => OVCU64::duplicate_value(),
        Ordering::Less => {
            panic!("Input must be non-decreasing for OVC encoding");
        }
    }
}
