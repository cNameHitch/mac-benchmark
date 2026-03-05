use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use bench_core::xorshift64;

pub const CPU_BENCH_PASSES: u32 = 5;
pub const CPU_BENCH_ITERS: u64 = 100_000_000;
pub const CHASE_STEPS: usize = 5_000_000;

pub const CACHE_LADDER_SIZES: &[(usize, &str)] = &[
    (32 * 1024, "32 KB"),
    (64 * 1024, "64 KB"),
    (128 * 1024, "128 KB"),
    (256 * 1024, "256 KB"),
    (512 * 1024, "512 KB"),
    (1024 * 1024, "1 MB"),
    (2 * 1024 * 1024, "2 MB"),
    (4 * 1024 * 1024, "4 MB"),
    (8 * 1024 * 1024, "8 MB"),
    (16 * 1024 * 1024, "16 MB"),
    (32 * 1024 * 1024, "32 MB"),
    (64 * 1024 * 1024, "64 MB"),
];

// ---------------------------------------------------------------------------
// Integer throughput
// ---------------------------------------------------------------------------

/// Dependent integer chain: x = x.wrapping_mul(A).wrapping_add(B)
/// Measures single-core integer latency (IPC=1).
pub fn bench_int_dependent() -> f64 {
    let mut best = f64::MAX;
    let a: u64 = 6364136223846793005;
    let b: u64 = 1442695040888963407;

    for _ in 0..CPU_BENCH_PASSES {
        let mut x: u64 = 0xDEADBEEFCAFEBABE;
        let start = Instant::now();
        for _ in 0..CPU_BENCH_ITERS {
            x = x.wrapping_mul(a).wrapping_add(b);
        }
        black_box(x);
        best = best.min(start.elapsed().as_secs_f64());
    }

    CPU_BENCH_ITERS as f64 / best / 1e9
}

/// 4 independent integer chains — measures ALU width / ILP.
pub fn bench_int_independent() -> f64 {
    let mut best = f64::MAX;
    let a: u64 = 6364136223846793005;
    let b: u64 = 1442695040888963407;

    for _ in 0..CPU_BENCH_PASSES {
        let mut x0: u64 = 0xDEADBEEF00000001;
        let mut x1: u64 = 0xDEADBEEF00000002;
        let mut x2: u64 = 0xDEADBEEF00000003;
        let mut x3: u64 = 0xDEADBEEF00000004;
        let iters = CPU_BENCH_ITERS / 4;
        let start = Instant::now();
        for _ in 0..iters {
            x0 = x0.wrapping_mul(a).wrapping_add(b);
            x1 = x1.wrapping_mul(a).wrapping_add(b);
            x2 = x2.wrapping_mul(a).wrapping_add(b);
            x3 = x3.wrapping_mul(a).wrapping_add(b);
        }
        black_box(x0.wrapping_add(x1).wrapping_add(x2).wrapping_add(x3));
        best = best.min(start.elapsed().as_secs_f64());
    }

    CPU_BENCH_ITERS as f64 / best / 1e9
}

// ---------------------------------------------------------------------------
// Floating-point throughput
// ---------------------------------------------------------------------------

/// Dependent FP64 FMA chain: x = x.mul_add(a, b)
pub fn bench_fp_dependent() -> f64 {
    let mut best = f64::MAX;
    let a: f64 = 1.0000001;
    let b: f64 = 0.0000001;

    for _ in 0..CPU_BENCH_PASSES {
        let mut x: f64 = 1.0;
        let start = Instant::now();
        for _ in 0..CPU_BENCH_ITERS {
            x = x.mul_add(a, b);
        }
        black_box(x);
        best = best.min(start.elapsed().as_secs_f64());
    }

    CPU_BENCH_ITERS as f64 / best / 1e9
}

/// 4 independent FP64 FMA chains.
pub fn bench_fp_independent() -> f64 {
    let mut best = f64::MAX;
    let a: f64 = 1.0000001;
    let b: f64 = 0.0000001;

    for _ in 0..CPU_BENCH_PASSES {
        let mut x0: f64 = 1.0;
        let mut x1: f64 = 2.0;
        let mut x2: f64 = 3.0;
        let mut x3: f64 = 4.0;
        let iters = CPU_BENCH_ITERS / 4;
        let start = Instant::now();
        for _ in 0..iters {
            x0 = x0.mul_add(a, b);
            x1 = x1.mul_add(a, b);
            x2 = x2.mul_add(a, b);
            x3 = x3.mul_add(a, b);
        }
        black_box(x0 + x1 + x2 + x3);
        best = best.min(start.elapsed().as_secs_f64());
    }

    CPU_BENCH_ITERS as f64 / best / 1e9
}

// ---------------------------------------------------------------------------
// FP32 throughput
// ---------------------------------------------------------------------------

/// Dependent FP32 FMA chain: x = x.mul_add(a, b)
pub fn bench_fp32_dependent() -> f64 {
    let mut best = f64::MAX;
    let a: f32 = 1.0000001;
    let b: f32 = 0.0000001;

    for _ in 0..CPU_BENCH_PASSES {
        let mut x: f32 = 1.0;
        let start = Instant::now();
        for _ in 0..CPU_BENCH_ITERS {
            x = x.mul_add(a, b);
        }
        black_box(x);
        best = best.min(start.elapsed().as_secs_f64());
    }

    CPU_BENCH_ITERS as f64 / best / 1e9
}

/// 4 independent FP32 FMA chains.
pub fn bench_fp32_independent() -> f64 {
    let mut best = f64::MAX;
    let a: f32 = 1.0000001;
    let b: f32 = 0.0000001;

    for _ in 0..CPU_BENCH_PASSES {
        let mut x0: f32 = 1.0;
        let mut x1: f32 = 2.0;
        let mut x2: f32 = 3.0;
        let mut x3: f32 = 4.0;
        let iters = CPU_BENCH_ITERS / 4;
        let start = Instant::now();
        for _ in 0..iters {
            x0 = x0.mul_add(a, b);
            x1 = x1.mul_add(a, b);
            x2 = x2.mul_add(a, b);
            x3 = x3.mul_add(a, b);
        }
        black_box(x0 + x1 + x2 + x3);
        best = best.min(start.elapsed().as_secs_f64());
    }

    CPU_BENCH_ITERS as f64 / best / 1e9
}

// ---------------------------------------------------------------------------
// Cache latency ladder (pointer chase at various working set sizes)
// ---------------------------------------------------------------------------

/// Run pointer-chase at the given working set size (in bytes).
/// Returns ns per access.
pub fn bench_cache_latency(size_bytes: usize) -> f64 {
    let elem_count = size_bytes / size_of::<usize>();
    if elem_count < 2 {
        return 0.0;
    }

    let mut arena: Vec<usize> = (0..elem_count).collect();
    let mut rng = 0x12345678u64;
    for i in (1..elem_count).rev() {
        let j = (xorshift64(&mut rng) as usize) % i;
        arena.swap(i, j);
    }

    // Warmup
    let mut idx = 0usize;
    for _ in 0..CHASE_STEPS / 10 {
        idx = arena[idx];
    }
    black_box(idx);

    // Measure
    idx = 0;
    let start = Instant::now();
    for _ in 0..CHASE_STEPS {
        idx = arena[idx];
    }
    black_box(idx);
    let elapsed = start.elapsed();

    elapsed.as_nanos() as f64 / CHASE_STEPS as f64
}

// ---------------------------------------------------------------------------
// Bandwidth ladder (sequential read at various working set sizes)
// ---------------------------------------------------------------------------

/// Sequential read throughput at the given working set size (in bytes).
/// Returns GB/s.
pub fn bench_cache_bandwidth(size_bytes: usize) -> f64 {
    let elem_count = size_bytes / size_of::<u64>();
    if elem_count == 0 {
        return 0.0;
    }

    let arena: Vec<u64> = (0..elem_count as u64).collect();
    let passes = (CPU_BENCH_PASSES as usize).max(3);
    let mut best = f64::MAX;

    let reps = (10_000_000usize / elem_count).max(1);

    for _ in 0..passes {
        let start = Instant::now();
        let mut sum = 0u64;
        for _ in 0..reps {
            for val in arena.iter() {
                sum = sum.wrapping_add(*val);
            }
        }
        black_box(sum);
        best = best.min(start.elapsed().as_secs_f64());
    }

    let total_bytes = elem_count * reps * size_of::<u64>();
    total_bytes as f64 / best / 1e9
}

// ---------------------------------------------------------------------------
// Multi-threaded CPU benchmarks
// ---------------------------------------------------------------------------

const MT_PASSES: u32 = 3;

/// Multi-threaded integer benchmark. Returns aggregate Gops/s.
pub fn bench_mt_int(num_threads: u32) -> f64 {
    let mut best = f64::MAX;
    let a: u64 = 6364136223846793005;
    let b: u64 = 1442695040888963407;
    let iters = CPU_BENCH_ITERS / 4;

    for _ in 0..MT_PASSES {
        let barrier = Arc::new(Barrier::new(num_threads as usize + 1));
        let mut handles = Vec::with_capacity(num_threads as usize);

        for t in 0..num_threads {
            let bar = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                let mut x0: u64 = 0xDEADBEEF00000001_u64.wrapping_add(t as u64);
                let mut x1: u64 = 0xDEADBEEF00000002_u64.wrapping_add(t as u64);
                let mut x2: u64 = 0xDEADBEEF00000003_u64.wrapping_add(t as u64);
                let mut x3: u64 = 0xDEADBEEF00000004_u64.wrapping_add(t as u64);
                bar.wait();
                for _ in 0..iters {
                    x0 = x0.wrapping_mul(a).wrapping_add(b);
                    x1 = x1.wrapping_mul(a).wrapping_add(b);
                    x2 = x2.wrapping_mul(a).wrapping_add(b);
                    x3 = x3.wrapping_mul(a).wrapping_add(b);
                }
                black_box(x0.wrapping_add(x1).wrapping_add(x2).wrapping_add(x3));
            }));
        }

        barrier.wait();
        let start = Instant::now();
        for h in handles {
            let _ = h.join();
        }
        let elapsed = start.elapsed().as_secs_f64();
        best = best.min(elapsed);
    }

    (CPU_BENCH_ITERS as f64 * num_threads as f64) / best / 1e9
}

/// Multi-threaded FP64 FMA benchmark. Returns aggregate Gflops/s.
pub fn bench_mt_fp(num_threads: u32) -> f64 {
    let mut best = f64::MAX;
    let a: f64 = 1.0000001;
    let b: f64 = 0.0000001;
    let iters = CPU_BENCH_ITERS / 4;

    for _ in 0..MT_PASSES {
        let barrier = Arc::new(Barrier::new(num_threads as usize + 1));
        let mut handles = Vec::with_capacity(num_threads as usize);

        for t in 0..num_threads {
            let bar = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                let mut x0: f64 = 1.0 + t as f64 * 0.01;
                let mut x1: f64 = 2.0 + t as f64 * 0.01;
                let mut x2: f64 = 3.0 + t as f64 * 0.01;
                let mut x3: f64 = 4.0 + t as f64 * 0.01;
                bar.wait();
                for _ in 0..iters {
                    x0 = x0.mul_add(a, b);
                    x1 = x1.mul_add(a, b);
                    x2 = x2.mul_add(a, b);
                    x3 = x3.mul_add(a, b);
                }
                black_box(x0 + x1 + x2 + x3);
            }));
        }

        barrier.wait();
        let start = Instant::now();
        for h in handles {
            let _ = h.join();
        }
        let elapsed = start.elapsed().as_secs_f64();
        best = best.min(elapsed);
    }

    (CPU_BENCH_ITERS as f64 * num_threads as f64) / best / 1e9
}

// ---------------------------------------------------------------------------
// NEON SIMD benchmarks (aarch64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use std::arch::aarch64::*;
    use std::hint::black_box;
    use std::time::Instant;
    use super::{CPU_BENCH_PASSES, CPU_BENCH_ITERS};

    /// 4 independent vfmaq_f32 chains (4 lanes each = 16 FP32 ops/iter).
    /// Returns Gflops/s.
    pub fn bench_neon_fp32() -> f64 {
        let mut best = f64::MAX;
        let iters = CPU_BENCH_ITERS / 4;

        for _ in 0..CPU_BENCH_PASSES {
            unsafe {
                let a = vdupq_n_f32(1.0000001);
                let b = vdupq_n_f32(0.0000001);
                let mut c0 = vdupq_n_f32(1.0);
                let mut c1 = vdupq_n_f32(2.0);
                let mut c2 = vdupq_n_f32(3.0);
                let mut c3 = vdupq_n_f32(4.0);

                let start = Instant::now();
                for _ in 0..iters {
                    c0 = vfmaq_f32(b, c0, a);
                    c1 = vfmaq_f32(b, c1, a);
                    c2 = vfmaq_f32(b, c2, a);
                    c3 = vfmaq_f32(b, c3, a);
                }
                black_box(vaddq_f32(vaddq_f32(c0, c1), vaddq_f32(c2, c3)));
                best = best.min(start.elapsed().as_secs_f64());
            }
        }

        (iters as f64 * 4.0 * 4.0) / best / 1e9
    }

    /// 4 independent vmulq_u32 + vaddq_u32 chains.
    /// Returns Gops/s.
    pub fn bench_neon_int() -> f64 {
        let mut best = f64::MAX;
        let iters = CPU_BENCH_ITERS / 4;

        for _ in 0..CPU_BENCH_PASSES {
            unsafe {
                let a = vdupq_n_u32(3);
                let b = vdupq_n_u32(7);
                let mut c0 = vdupq_n_u32(1);
                let mut c1 = vdupq_n_u32(2);
                let mut c2 = vdupq_n_u32(3);
                let mut c3 = vdupq_n_u32(4);

                let start = Instant::now();
                for _ in 0..iters {
                    c0 = vaddq_u32(vmulq_u32(c0, a), b);
                    c1 = vaddq_u32(vmulq_u32(c1, a), b);
                    c2 = vaddq_u32(vmulq_u32(c2, a), b);
                    c3 = vaddq_u32(vmulq_u32(c3, a), b);
                }
                black_box(vaddq_u32(vaddq_u32(c0, c1), vaddq_u32(c2, c3)));
                best = best.min(start.elapsed().as_secs_f64());
            }
        }

        (iters as f64 * 4.0 * 4.0) / best / 1e9
    }
}

// ---------------------------------------------------------------------------
// P-core / E-core isolation (macOS only)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub mod core_isolation {
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    const QOS_CLASS_USER_INITIATED: u32 = 0x19;
    const QOS_CLASS_BACKGROUND: u32 = 0x09;

    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: libc::c_int) -> libc::c_int;
    }

    fn set_qos(qos_class: u32) {
        unsafe { pthread_set_qos_class_self_np(qos_class, 0); }
    }

    fn restore_qos() {
        set_qos(QOS_CLASS_USER_INITIATED);
    }

    pub fn bench_pcore_int() -> f64 {
        set_qos(QOS_CLASS_USER_INTERACTIVE);
        let v = super::bench_int_independent();
        restore_qos();
        v
    }

    pub fn bench_pcore_fp() -> f64 {
        set_qos(QOS_CLASS_USER_INTERACTIVE);
        let v = super::bench_fp_independent();
        restore_qos();
        v
    }

    pub fn bench_ecore_int() -> f64 {
        set_qos(QOS_CLASS_BACKGROUND);
        let v = super::bench_int_independent();
        restore_qos();
        v
    }

    pub fn bench_ecore_fp() -> f64 {
        set_qos(QOS_CLASS_BACKGROUND);
        let v = super::bench_fp_independent();
        restore_qos();
        v
    }
}
