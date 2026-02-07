use crate::ovc::offset_value_coding_32::OVCU32;
use crate::sketch::Sketch;
use crate::sort::core::engine::RunSummary;

pub trait RunSink: Send {
    type MergeableRun: RunSummary + Send + Sync + 'static;

    fn start_run(&mut self);
    fn push_record(&mut self, key: &[u8], value: &[u8]);
    /// Push a record that carries an OVC delta; default falls back to plain KV.
    fn push_record_with_ovc(&mut self, ovc: OVCU32, key: &[u8], value: &[u8]) {
        let _ = ovc;
        self.push_record(key, value);
    }
    fn finish_run(&mut self);

    fn finalize(self) -> Vec<Self::MergeableRun>;
}

/// A RunSink that discards all records (for benchmarking final merge without I/O)
pub struct DiscardRunSink {
    record_count: usize,
}

impl DiscardRunSink {
    pub fn new() -> Self {
        Self { record_count: 0 }
    }
}

/// Empty run marker for discarded output
pub struct EmptyRun {
    entries: usize,
}

impl RunSummary for EmptyRun {
    fn total_entries(&self) -> usize {
        self.entries
    }

    fn total_bytes(&self) -> usize {
        0
    }
}

impl RunSink for DiscardRunSink {
    type MergeableRun = EmptyRun;

    fn start_run(&mut self) {
        // No-op
    }

    fn push_record(&mut self, _key: &[u8], _value: &[u8]) {
        self.record_count += 1;
    }

    fn finish_run(&mut self) {
        // No-op
    }

    fn finalize(self) -> Vec<Self::MergeableRun> {
        let run = EmptyRun {
            entries: self.record_count,
        };
        // Return empty sketch as we don't track key distribution in discard mode
        vec![run]
    }
}
