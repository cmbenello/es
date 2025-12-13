use crate::ovc::offset_value_coding::OVCU64;
use crate::sketch::Sketch;
use crate::sort::core::engine::RunSummary;

pub trait RunSink: Send {
    type MergeableRun: RunSummary + Send + Sync + 'static;

    fn start_run(&mut self);
    fn push_record(&mut self, key: &[u8], value: &[u8]);
    /// Push a record that carries an OVC delta; default falls back to plain KV.
    fn push_record_with_ovc(&mut self, ovc: OVCU64, key: &[u8], value: &[u8]) {
        let _ = ovc;
        self.push_record(key, value);
    }
    fn finish_run(&mut self);

    fn finalize(self) -> (Vec<Self::MergeableRun>, Sketch<Vec<u8>>);
}
