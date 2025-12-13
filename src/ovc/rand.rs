use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};
use std::cell::RefCell;

// Thread-local `SmallRng` state.
thread_local! {
    pub static THREAD_RNG_KEY: RefCell<SmallRng> = RefCell::new(SmallRng::from_os_rng());
}

pub fn set_seed(seed: u64) {
    THREAD_RNG_KEY.with(|rng_cell| {
        *rng_cell.borrow_mut() = SmallRng::seed_from_u64(seed);
    });
}

/// A handle to the thread-local `SmallRng`—similar to `rand::ThreadRng`.
#[derive(Debug, Clone)]
pub struct SmallThreadRng;

impl RngCore for SmallThreadRng {
    fn next_u32(&mut self) -> u32 {
        THREAD_RNG_KEY.with(|rng_cell| rng_cell.borrow_mut().next_u32())
    }

    fn next_u64(&mut self) -> u64 {
        THREAD_RNG_KEY.with(|rng_cell| rng_cell.borrow_mut().next_u64())
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        THREAD_RNG_KEY.with(|rng_cell| rng_cell.borrow_mut().fill_bytes(dest))
    }
}

pub fn small_thread_rng() -> SmallThreadRng {
    SmallThreadRng
}
