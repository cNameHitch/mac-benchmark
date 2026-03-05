#![cfg(target_os = "macos")]

use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

const THROUGHPUT_THREADS: u64 = 1_000_000;
const THROUGHPUT_ITERS: u64 = 1024;
const THROUGHPUT_CHAINS: u64 = 4;

const BW_BUFFER_BYTES: u64 = 64 * 1024 * 1024; // 64 MB
const BW_FLOAT4_COUNT: u64 = BW_BUFFER_BYTES / 16;
const BW_THREADS: u64 = 65536;

const ALLOC_TEST_BYTES: u64 = 16 * 1024 * 1024; // 16 MB
const ALLOC_ITERS: u32 = 100;

const MATMUL_N: u64 = 1024;
const MATMUL_TILE: u64 = 16;

const BENCH_PASSES: u32 = 3;

pub struct GpuContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    fp32_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fp16_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    int32_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    buf_read_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    buf_write_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl GpuContext {
    pub fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "No Metal device found".to_string())?;

        let queue = device
            .newCommandQueue()
            .ok_or_else(|| "Failed to create command queue".to_string())?;

        let throughput_src = include_str!("shaders/throughput.metal");
        let bandwidth_src = include_str!("shaders/bandwidth.metal");
        let matmul_src = include_str!("shaders/matmul.metal");

        let throughput_lib = compile_library(&device, throughput_src, "throughput")?;
        let bandwidth_lib = compile_library(&device, bandwidth_src, "bandwidth")?;
        let matmul_lib = compile_library(&device, matmul_src, "matmul")?;

        let fp32_pipeline = make_pipeline(&device, &throughput_lib, "fp32_throughput")?;
        let fp16_pipeline = make_pipeline(&device, &throughput_lib, "fp16_throughput")?;
        let int32_pipeline = make_pipeline(&device, &throughput_lib, "int32_throughput")?;
        let buf_read_pipeline = make_pipeline(&device, &bandwidth_lib, "buffer_read")?;
        let buf_write_pipeline = make_pipeline(&device, &bandwidth_lib, "buffer_write")?;
        let matmul_pipeline = make_pipeline(&device, &matmul_lib, "matmul")?;

        Ok(Self {
            device,
            queue,
            fp32_pipeline,
            fp16_pipeline,
            int32_pipeline,
            buf_read_pipeline,
            buf_write_pipeline,
            matmul_pipeline,
        })
    }

    pub fn bench_fp32_throughput(&self) -> f64 {
        self.throughput_bench(&self.fp32_pipeline, THROUGHPUT_THREADS, 4) // 4 bytes per float
    }

    pub fn bench_fp16_throughput(&self) -> f64 {
        self.throughput_bench(&self.fp16_pipeline, THROUGHPUT_THREADS, 2) // 2 bytes per half
    }

    pub fn bench_int32_throughput(&self) -> f64 {
        self.throughput_bench(&self.int32_pipeline, THROUGHPUT_THREADS, 4)
    }

    fn throughput_bench(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        num_threads: u64,
        elem_size: u64,
    ) -> f64 {
        let out_buf = match self.device.newBufferWithLength_options(
            (num_threads * elem_size) as usize,
            MTLResourceOptions::StorageModeShared,
        ) {
            Some(b) => b,
            None => return 0.0,
        };

        let mut best = f64::MAX;
        for _ in 0..BENCH_PASSES {
            let elapsed = self.dispatch_1d(pipeline, &[&out_buf], num_threads);
            best = best.min(elapsed);
        }

        // Each thread does CHAINS * ITERS FMA ops. Each FMA = 2 flops (mul + add).
        let total_ops = num_threads * THROUGHPUT_CHAINS * THROUGHPUT_ITERS * 2;
        total_ops as f64 / best / 1e12 // Tflops/s (or Tops/s for int)
    }

    pub fn bench_buffer_read(&self) -> f64 {
        let src_buf = match self.device.newBufferWithLength_options(
            BW_BUFFER_BYTES as usize,
            MTLResourceOptions::StorageModeShared,
        ) {
            Some(b) => b,
            None => return 0.0,
        };
        let out_buf = match self.device.newBufferWithLength_options(
            (BW_THREADS * 4) as usize,
            MTLResourceOptions::StorageModeShared,
        ) {
            Some(b) => b,
            None => return 0.0,
        };
        let count_buf = match self.make_u32_buffer(BW_FLOAT4_COUNT as u32) {
            Some(b) => b,
            None => return 0.0,
        };

        let mut best = f64::MAX;
        for _ in 0..BENCH_PASSES {
            let elapsed = self.dispatch_1d(
                &self.buf_read_pipeline,
                &[&src_buf, &out_buf, &count_buf],
                BW_THREADS,
            );
            best = best.min(elapsed);
        }

        BW_BUFFER_BYTES as f64 / best / 1e9 // GB/s
    }

    pub fn bench_buffer_write(&self) -> f64 {
        let dst_buf = match self.device.newBufferWithLength_options(
            BW_BUFFER_BYTES as usize,
            MTLResourceOptions::StorageModeShared,
        ) {
            Some(b) => b,
            None => return 0.0,
        };
        let count_buf = match self.make_u32_buffer(BW_FLOAT4_COUNT as u32) {
            Some(b) => b,
            None => return 0.0,
        };

        let mut best = f64::MAX;
        for _ in 0..BENCH_PASSES {
            let elapsed = self.dispatch_1d(
                &self.buf_write_pipeline,
                &[&dst_buf, &count_buf],
                BW_THREADS,
            );
            best = best.min(elapsed);
        }

        BW_BUFFER_BYTES as f64 / best / 1e9 // GB/s
    }

    pub fn bench_buffer_alloc(&self) -> f64 {
        let mut times = Vec::with_capacity(ALLOC_ITERS as usize);
        for _ in 0..ALLOC_ITERS {
            let start = Instant::now();
            let _buf = self.device.newBufferWithLength_options(
                ALLOC_TEST_BYTES as usize,
                MTLResourceOptions::StorageModeShared,
            );
            times.push(start.elapsed().as_secs_f64() * 1e6); // microseconds
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2] // median in microseconds
    }

    pub fn bench_matmul(&self) -> f64 {
        let n = MATMUL_N;
        let elems = (n * n) as usize;
        let bytes = elems * 4; // float

        let a_buf = match self.device.newBufferWithLength_options(
            bytes,
            MTLResourceOptions::StorageModeShared,
        ) {
            Some(b) => b,
            None => return 0.0,
        };
        let b_buf = match self.device.newBufferWithLength_options(
            bytes,
            MTLResourceOptions::StorageModeShared,
        ) {
            Some(b) => b,
            None => return 0.0,
        };
        let c_buf = match self.device.newBufferWithLength_options(
            bytes,
            MTLResourceOptions::StorageModeShared,
        ) {
            Some(b) => b,
            None => return 0.0,
        };

        // Initialize A and B with some data
        unsafe {
            let a_ptr = a_buf.contents().as_ptr() as *mut f32;
            let b_ptr = b_buf.contents().as_ptr() as *mut f32;
            for i in 0..elems {
                *a_ptr.add(i) = (i % 17) as f32 * 0.01;
                *b_ptr.add(i) = (i % 13) as f32 * 0.01;
            }
        }

        let n_buf = match self.make_u32_buffer(n as u32) {
            Some(b) => b,
            None => return 0.0,
        };

        let mut best = f64::MAX;
        for _ in 0..BENCH_PASSES {
            let cmd_buf = match self.queue.commandBuffer() {
                Some(b) => b,
                None => return 0.0,
            };

            let encoder = match cmd_buf.computeCommandEncoder() {
                Some(e) => e,
                None => return 0.0,
            };

            encoder.setComputePipelineState(&self.matmul_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&a_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&c_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&n_buf), 0, 3);
            }

            let grid = MTLSize { width: n as usize, height: n as usize, depth: 1 };
            let threadgroup = MTLSize { width: MATMUL_TILE as usize, height: MATMUL_TILE as usize, depth: 1 };

            encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
            encoder.endEncoding();

            let start = Instant::now();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            let elapsed = start.elapsed().as_secs_f64();
            best = best.min(elapsed);
        }

        // 2 * N^3 flops for matrix multiply
        let flops = 2.0 * (n as f64).powi(3);
        flops / best / 1e12 // Tflops/s
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn dispatch_1d(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buffers: &[&ProtocolObject<dyn MTLBuffer>],
        num_threads: u64,
    ) -> f64 {
        let cmd_buf = match self.queue.commandBuffer() {
            Some(b) => b,
            None => return f64::MAX,
        };

        let encoder = match cmd_buf.computeCommandEncoder() {
            Some(e) => e,
            None => return f64::MAX,
        };

        encoder.setComputePipelineState(pipeline);
        for (i, buf) in buffers.iter().enumerate() {
            unsafe { encoder.setBuffer_offset_atIndex(Some(*buf), 0, i); }
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as u64;
        let threadgroup_size = max_threads.min(num_threads);
        let grid = MTLSize { width: num_threads as usize, height: 1, depth: 1 };
        let threadgroup = MTLSize { width: threadgroup_size as usize, height: 1, depth: 1 };

        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
        encoder.endEncoding();

        let start = Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        start.elapsed().as_secs_f64()
    }

    fn make_u32_buffer(
        &self,
        value: u32,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let buf = self.device.newBufferWithLength_options(
            4,
            MTLResourceOptions::StorageModeShared,
        )?;
        unsafe {
            let ptr = buf.contents().as_ptr() as *mut u32;
            *ptr = value;
        }
        Some(buf)
    }
}

fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    let src = NSString::from_str(source);
    device.newLibraryWithSource_options_error(&src, None)
        .map_err(|e| format!("Failed to compile {name} shader: {}", e.localizedDescription()))
}

fn make_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    function_name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
    let name = NSString::from_str(function_name);
    let func = library.newFunctionWithName(&name)
        .ok_or_else(|| format!("Function '{function_name}' not found in shader"))?;
    device.newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| format!("Failed to create pipeline for '{function_name}': {}", e.localizedDescription()))
}
