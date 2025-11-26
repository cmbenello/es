use crate::sketch::Sketch;
use crate::sort::core::engine::RunSummary;

pub trait RunSink: Send {
    type MergeableRun: RunSummary + Send + Sync + 'static;

    fn start_run(&mut self);
    fn push_record(&mut self, key: &[u8], value: &[u8]);
    fn finish_run(&mut self);

    fn finalize(self) -> (Vec<Self::MergeableRun>, Sketch<Vec<u8>>);
}
