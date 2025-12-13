pub type ReplacementScanner = Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>;

/// Timing information produced by the replacement selection driver.
#[derive(Default)]
pub struct ReplacementSelectionStats {
    pub records_emitted: usize,
}

/// Handler used by replacement-selection backends to stream output and notify run boundaries.
pub trait RunEmitter<T> {
    fn emit(&mut self, record: T);
    fn on_run_switch(&mut self);
}

mod heap;
mod tol;
mod tol_ovc;
pub use heap::run_replacement_selection;
pub use tol::run_replacement_selection_tol;
pub use tol_ovc::run_replacement_selection_ovc;

#[inline(always)]
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

pub trait RecordSize {
    fn size(&self) -> usize;
}

// Example impl for byte vectors
impl<T: AsRef<[u8]>> RecordSize for T {
    fn size(&self) -> usize {
        self.as_ref().len()
    }
}
