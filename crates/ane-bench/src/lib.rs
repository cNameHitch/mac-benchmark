#![cfg(target_os = "macos")]

//! Apple Neural Engine / AMX benchmark suite.
//!
//! Benchmarks LLM-relevant matrix operations using the Accelerate framework,
//! which leverages Apple Silicon's AMX (Apple Matrix coprocessor) blocks.
//! This is the same compute path used by llama.cpp, MLX, and other local
//! LLM inference engines on macOS.

use std::time::Instant;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transA: i32,
        transB: i32,
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        A: *const f32,
        lda: i32,
        B: *const f32,
        ldb: i32,
        beta: f32,
        C: *mut f32,
        ldc: i32,
    );

    fn cblas_sgemv(
        order: i32,
        trans: i32,
        M: i32,
        N: i32,
        alpha: f32,
        A: *const f32,
        lda: i32,
        X: *const f32,
        incX: i32,
        beta: f32,
        Y: *mut f32,
        incY: i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

const BENCH_PASSES: u32 = 5;

// LLM-relevant dimensions
const SGEMM_N: usize = 4096;
const PREFILL_M: usize = 512;
const PREFILL_K: usize = 4096;
const PREFILL_N: usize = 4096;

/// SGEMM 4096x4096: Square matrix multiply via Accelerate (AMX).
/// Returns TFLOPS.
pub fn bench_sgemm() -> f64 {
    let m = SGEMM_N;
    let n = SGEMM_N;
    let k = SGEMM_N;

    let a = vec![0.001f32; m * k];
    let b = vec![0.001f32; k * n];
    let mut c = vec![0.0f32; m * n];

    // Warmup
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }

    let mut best = f64::MAX;
    for _ in 0..BENCH_PASSES {
        let start = Instant::now();
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                m as i32, n as i32, k as i32,
                1.0, a.as_ptr(), k as i32,
                b.as_ptr(), n as i32,
                0.0, c.as_mut_ptr(), n as i32,
            );
        }
        best = best.min(start.elapsed().as_secs_f64());
    }

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / best / 1e12
}

/// SGEMV 4096x4096: Matrix-vector multiply (simulates single-token decode).
/// Returns GFLOPS.
pub fn bench_gemv() -> f64 {
    let m = SGEMM_N;
    let n = SGEMM_N;

    let a = vec![0.001f32; m * n];
    let x = vec![0.001f32; n];
    let mut y = vec![0.0f32; m];

    // Warmup
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
            m as i32, n as i32,
            1.0, a.as_ptr(), n as i32,
            x.as_ptr(), 1,
            0.0, y.as_mut_ptr(), 1,
        );
    }

    let mut best = f64::MAX;
    for _ in 0..BENCH_PASSES {
        let start = Instant::now();
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
                m as i32, n as i32,
                1.0, a.as_ptr(), n as i32,
                x.as_ptr(), 1,
                0.0, y.as_mut_ptr(), 1,
            );
        }
        best = best.min(start.elapsed().as_secs_f64());
    }

    let flops = 2.0 * m as f64 * n as f64;
    flops / best / 1e9
}

/// Prefill SGEMM 512x4096 * 4096x4096: Simulates batch prefill in transformers.
/// Returns TFLOPS.
pub fn bench_prefill() -> f64 {
    let m = PREFILL_M;
    let k = PREFILL_K;
    let n = PREFILL_N;

    let a = vec![0.001f32; m * k];
    let b = vec![0.001f32; k * n];
    let mut c = vec![0.0f32; m * n];

    // Warmup
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }

    let mut best = f64::MAX;
    for _ in 0..BENCH_PASSES {
        let start = Instant::now();
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                m as i32, n as i32, k as i32,
                1.0, a.as_ptr(), k as i32,
                b.as_ptr(), n as i32,
                0.0, c.as_mut_ptr(), n as i32,
            );
        }
        best = best.min(start.elapsed().as_secs_f64());
    }

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / best / 1e12
}

/// Estimate LLM decode tokens/sec for a 7B parameter model.
///
/// A 7B model requires ~14 GFLOPS per token (2 * params for matrix ops).
/// Uses GEMV throughput since decode is memory-bound (batch=1).
/// Returns estimated tokens/sec.
pub fn estimate_token_throughput() -> f64 {
    let m = SGEMM_N;
    let n = SGEMM_N;

    let a = vec![0.001f32; m * n];
    let x = vec![0.001f32; n];
    let mut y = vec![0.0f32; m];

    // Warmup
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
            m as i32, n as i32,
            1.0, a.as_ptr(), n as i32,
            x.as_ptr(), 1,
            0.0, y.as_mut_ptr(), 1,
        );
    }

    // Measure GEMV throughput
    let mut best = f64::MAX;
    for _ in 0..BENCH_PASSES {
        let start = Instant::now();
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
                m as i32, n as i32,
                1.0, a.as_ptr(), n as i32,
                x.as_ptr(), 1,
                0.0, y.as_mut_ptr(), 1,
            );
        }
        best = best.min(start.elapsed().as_secs_f64());
    }

    let gflops = 2.0 * m as f64 * n as f64 / best / 1e9;

    // A 7B model has ~64 linear layers of ~4096x4096 equivalent.
    // Total FLOPs per token ~ 2 * 7e9 = 14 GFLOPS.
    // Decode is memory-bound, so tok/s ~ GEMV_GFLOPS * (layers_benchmarked / total_layers).
    // More precisely: tok/s = measured single-layer GFLOPS / (14000 GFLOPS-per-token / 64 layers)
    // = measured GFLOPS / 218.75 GFLOPS-per-layer
    let layers: f64 = 64.0;
    let time_per_layer = (2.0 * m as f64 * n as f64) / (gflops * 1e9);
    let time_per_token = time_per_layer * layers;

    1.0 / time_per_token
}
