use es::rand::small_thread_rng;
use rand::prelude::*;
use std::time::Instant;

// Simple Tree of Losers simulation
// We use a flat array where index i's parent is i/2.
// This is cache-optimized layout, so if THIS is slow, pointer-based is worse.
struct TournamentTree {
    tree: Vec<i64>,
    size: usize,
}

impl TournamentTree {
    fn new(size: usize) -> Self {
        // Initialize with dummy data
        let mut tree = vec![0; size * 2];
        let mut rng = small_thread_rng();
        for i in size..2 * size {
            tree[i] = rng.random::<i64>();
        }
        // Build initial tree (simplified)
        for i in (1..size).rev() {
            tree[i] = std::cmp::min(tree[2 * i], tree[2 * i + 1]);
        }
        Self { tree, size }
    }

    // Simulate the critical path: Replace winner & Re-tournament
    #[inline(always)]
    fn push_pop(&mut self, new_val: i64) -> i64 {
        // Winner is at index 1. We replace the leaf that generated it.
        // For simulation, we just pick a random leaf index to replace
        // to mimic the unpredictability of real replacement selection.
        // In real RS, we would track the index of the winner.
        let leaf_idx = self.size + (new_val as usize % self.size);

        self.tree[leaf_idx] = new_val;
        let mut idx = leaf_idx;

        // The "Pointer Chasing" Loop
        while idx > 1 {
            idx /= 2;
            // Comparison (Memory Access)
            let left = self.tree[2 * idx];
            let right = self.tree[2 * idx + 1];
            self.tree[idx] = if left < right { left } else { right };
        }
        return self.tree[1];
    }
}

fn main() {
    println!("HeapSize_KB,Ops_Per_Sec,Latency_ns");

    // Test sizes from 32KB to 512MB
    let sizes = vec![
        4 * 1024,         // 32 KB (L1 Cache fits)
        32 * 1024,        // 256 KB (L2 Cache fits)
        512 * 1024,       // 4 MB (L3 Cache fits)
        4 * 1024 * 1024,  // 32 MB (L3 miss likely)
        64 * 1024 * 1024, // 512 MB (Main RAM only)
    ];

    for &count in &sizes {
        // Each i64 is 8 bytes.
        let mem_size_kb = (count * 8) / 1024;
        let mut tree = TournamentTree::new(count);

        // Warmup
        for i in 0..100_000 {
            tree.push_pop(i);
        }

        // Benchmark
        let iterations = 10_000_000;
        let start = Instant::now();
        for i in 0..iterations {
            // Use a simple predictable generator to avoid benchmarking the RNG
            tree.push_pop(i);
        }
        let duration = start.elapsed();

        let nanos_per_op = duration.as_nanos() as f64 / iterations as f64;
        let ops_per_sec = iterations as f64 / duration.as_secs_f64();

        println!("{},{:.0},{:.2}", mem_size_kb, ops_per_sec, nanos_per_op);
    }
}

// ### 3. How to Run and Plot
// Run this command in your terminal:
// ```bash
// cargo run --release --example cache_bench > cache_results.csv
