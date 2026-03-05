use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use crate::helpers::{vol_read, vol_write, xorshift64};

pub const BENCH_PASSES: u32 = 3;
pub const CHASE_REGION_ELEMS: usize = 8 * 1024 * 1024; // 64 MB worth of u64s
pub const CHASE_STEPS: usize = 10_000_000;

pub const STRIDES: &[(usize, &str)] = &[
    (8, "64 B"),
    (32, "256 B"),
    (128, "1 KB"),
    (512, "4 KB"),
    (8192, "64 KB"),
    (32768, "256 KB"),
    (131072, "1 MB"),
];

pub fn bench_seq_write(region: &mut [u64]) -> f64 {
    let size_bytes = region.len() * size_of::<u64>();
    let mut best = f64::MAX;

    for _ in 0..BENCH_PASSES {
        let start = Instant::now();
        for (i, elem) in region.iter_mut().enumerate() {
            vol_write(elem, i as u64);
        }
        best = best.min(start.elapsed().as_secs_f64());
    }

    size_bytes as f64 / best / 1e9
}

pub fn bench_seq_read(region: &[u64]) -> f64 {
    let size_bytes = region.len() * size_of::<u64>();
    let mut best = f64::MAX;

    for _ in 0..BENCH_PASSES {
        let start = Instant::now();
        let mut sum = 0u64;
        for elem in region.iter() {
            sum = sum.wrapping_add(vol_read(elem));
        }
        black_box(sum);
        best = best.min(start.elapsed().as_secs_f64());
    }

    size_bytes as f64 / best / 1e9
}

pub fn bench_copy(region: &mut [u64]) -> f64 {
    let half = region.len() / 2;
    let total_bytes = half * size_of::<u64>() * 2; // read + write
    let mut best = f64::MAX;

    for _ in 0..BENCH_PASSES {
        let (src, dst) = region.split_at_mut(half);
        let start = Instant::now();
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            vol_write(d, vol_read(s));
        }
        best = best.min(start.elapsed().as_secs_f64());
    }

    total_bytes as f64 / best / 1e9
}

pub fn bench_random_latency(region: &mut [u64]) -> (f64, usize) {
    let chase_len = region.len().min(CHASE_REGION_ELEMS);
    let chase_region = &mut region[..chase_len];
    let chase_mb = chase_len * size_of::<u64>() / (1024 * 1024);

    // Build a random Hamiltonian cycle (Sattolo's algorithm)
    for (i, elem) in chase_region.iter_mut().enumerate() {
        *elem = i as u64;
    }
    let mut rng = 0x12345678u64;
    for i in (1..chase_len).rev() {
        let j = (xorshift64(&mut rng) as usize) % i;
        chase_region.swap(i, j);
    }

    // Warmup
    let mut idx = 0usize;
    for _ in 0..CHASE_STEPS / 10 {
        idx = chase_region[idx] as usize;
    }
    black_box(idx);

    // Measure
    idx = 0;
    let start = Instant::now();
    for _ in 0..CHASE_STEPS {
        idx = chase_region[idx] as usize;
    }
    black_box(idx);
    let elapsed = start.elapsed();

    let ns_per_access = elapsed.as_nanos() as f64 / CHASE_STEPS as f64;
    (ns_per_access, chase_mb)
}

pub fn bench_stride_read(region: &[u64], stride_elems: usize) -> f64 {
    let accesses = region.len() / stride_elems;
    let bytes_touched = accesses * size_of::<u64>();
    let mut best = f64::MAX;

    let passes = (BENCH_PASSES as usize).max(1_000_000 / accesses.max(1));

    for _ in 0..passes {
        let start = Instant::now();
        let mut sum = 0u64;
        let mut i = 0;
        while i < region.len() {
            sum = sum.wrapping_add(vol_read(&region[i]));
            i += stride_elems;
        }
        black_box(sum);
        best = best.min(start.elapsed().as_secs_f64());
    }

    bytes_touched as f64 / best / 1e9
}

/// Multi-threaded sequential read bandwidth.
/// Partitions `region` among `num_threads` threads, each doing sequential volatile reads.
/// Returns aggregate GB/s.
pub fn bench_mt_seq_read(region: &[u64], num_threads: u32) -> f64 {
    if num_threads == 0 {
        return 0.0;
    }
    let total_elems = region.len();
    let size_bytes = total_elems * size_of::<u64>();
    let mut best = f64::MAX;
    let region_ptr = region.as_ptr() as usize;

    for _ in 0..BENCH_PASSES {
        let barrier = Arc::new(Barrier::new(num_threads as usize + 1));
        let mut handles = Vec::with_capacity(num_threads as usize);
        let chunk_size = total_elems / num_threads as usize;

        for t in 0..num_threads as usize {
            let bar = Arc::clone(&barrier);
            let start_offset = t * chunk_size;
            let count = if t == (num_threads as usize - 1) {
                total_elems - start_offset
            } else {
                chunk_size
            };
            let ptr_val = region_ptr;

            handles.push(thread::spawn(move || {
                let base = ptr_val as *const u64;
                bar.wait();
                let mut sum = 0u64;
                for i in 0..count {
                    sum = sum.wrapping_add(unsafe { std::ptr::read_volatile(base.add(start_offset + i)) });
                }
                black_box(sum);
            }));
        }

        barrier.wait();
        let start = Instant::now();
        for h in handles {
            let _ = h.join();
        }
        best = best.min(start.elapsed().as_secs_f64());
    }

    size_bytes as f64 / best / 1e9
}

/// Loaded latency: measure pointer-chase latency while bandwidth threads
/// saturate memory with sequential reads.
/// Returns ns per access.
pub fn bench_loaded_latency(region: &mut [u64], num_threads: u32) -> f64 {
    let chase_len = region.len().min(CHASE_REGION_ELEMS);
    if chase_len < 2 {
        return 0.0;
    }

    // Build Sattolo pointer chase on the first chase_len elements
    let chase_region = &mut region[..chase_len];
    for (i, elem) in chase_region.iter_mut().enumerate() {
        *elem = i as u64;
    }
    let mut rng = 0x12345678u64;
    for i in (1..chase_len).rev() {
        let j = (xorshift64(&mut rng) as usize) % i;
        chase_region.swap(i, j);
    }

    // Warmup chase
    let mut idx = 0usize;
    for _ in 0..CHASE_STEPS / 10 {
        idx = region[idx] as usize;
    }
    black_box(idx);

    // Spawn bandwidth-saturating threads on the remainder if possible
    let stop = Arc::new(AtomicBool::new(false));
    let mut bw_handles = Vec::new();

    let remainder_start = chase_len;
    let remainder_len = region.len() - chase_len;

    if num_threads > 1 && remainder_len > 0 {
        let bw_threads = (num_threads - 1) as usize;
        let region_ptr = region.as_ptr() as usize;
        let chunk_size = remainder_len / bw_threads;

        for t in 0..bw_threads {
            let stop_flag = Arc::clone(&stop);
            let ptr_val = region_ptr;
            let start_off = remainder_start + t * chunk_size;
            let count = if t == bw_threads - 1 {
                remainder_len - t * chunk_size
            } else {
                chunk_size
            };

            bw_handles.push(thread::spawn(move || {
                let base = ptr_val as *const u64;
                while !stop_flag.load(Ordering::Relaxed) {
                    let mut sum = 0u64;
                    for i in 0..count {
                        sum = sum.wrapping_add(unsafe { std::ptr::read_volatile(base.add(start_off + i)) });
                    }
                    black_box(sum);
                }
            }));
        }

        // Let bandwidth threads ramp up
        thread::sleep(std::time::Duration::from_millis(5));
    }

    // Measure pointer chase under load
    idx = 0;
    let start = Instant::now();
    for _ in 0..CHASE_STEPS {
        idx = region[idx] as usize;
    }
    black_box(idx);
    let elapsed = start.elapsed();

    // Stop bandwidth threads
    stop.store(true, Ordering::Relaxed);
    for h in bw_handles {
        let _ = h.join();
    }

    elapsed.as_nanos() as f64 / CHASE_STEPS as f64
}
