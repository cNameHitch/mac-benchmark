# Glossary

Terms used throughout mac-benchmark, including all metrics seen in log output.

---

## Run Modes

| Term | Description |
|---|---|
| **Test** | Runs 10 memory correctness tests (pattern writes followed by verification reads) on the allocated region. |
| **Bench** | Runs memory performance benchmarks: sequential write/read, copy, random read latency, and stride throughput. |
| **CPU** | Runs single-threaded CPU throughput benchmarks (integer, FP64, FP32, NEON) plus cache hierarchy profiling (latency ladder and bandwidth ladder). |
| **MT CPU** | Multi-threaded CPU benchmark that saturates all cores with integer and floating-point workloads. |
| **All** | Runs Test + Bench + CPU in a single pass. |
| **Stress** | Continuously repeats the 10 correctness tests in a loop until stopped or duration expires. |
| **Full Stress** | Comprehensive stress mode that runs every cycle: correctness tests, all memory benchmarks, all CPU benchmarks, cache latency ladder, and cache bandwidth ladder. Tracks min/max/avg statistics and regression detection across passes. `--dashboard` is an alias. |

---

## Memory Benchmarks

| Term | Log Label | Description |
|---|---|---|
| **Seq Write** | `Seq Write` / `Sequential write` | Throughput (GB/s) of writing every element in the allocated region sequentially using volatile writes. Best of 3 passes. |
| **Seq Read** | `Seq Read` / `Sequential read` | Throughput (GB/s) of reading every element sequentially using volatile reads. Best of 3 passes. |
| **Copy** | `Copy` / `Copy (read+write)` | Throughput (GB/s) of reading from one half of the region and writing to the other half. Counts both read and write bytes. Best of 3 passes. |
| **Latency** | `Latency` / `Random read latency` | Average nanoseconds per access for a pointer-chase (random traversal) over a 64 MB working set built with Sattolo's algorithm. Measures main-memory random access latency. |
| **Stride Read** | `64 B stride` through `1 MB stride` | Throughput (GB/s) when reading elements at a fixed stride across the region. Seven stride sizes are tested: 64 B, 256 B, 1 KB, 4 KB, 64 KB, 256 KB, and 1 MB. Smaller strides are more cache-friendly; larger strides stress the TLB and prefetcher. |
| **MT Read** | `MT Read (Nc)` / `MT sequential read (Nc)` | Aggregate sequential read throughput (GB/s) with the region partitioned across N threads reading in parallel. Measures total memory bandwidth the system can deliver. |
| **Loaded Lat** | `Loaded Lat` / `Loaded latency` | Pointer-chase latency (ns) measured while background threads simultaneously saturate memory bandwidth with sequential reads. Shows how latency degrades under contention. |

---

## CPU Benchmarks

| Term | Log Label | Description |
|---|---|---|
| **Int dep** | `Int dep` / `Integer (dependent)` | Throughput (Gops/s) of a single dependent integer chain: `x = x * A + B` (wrapping). Measures single-chain integer latency — only one operation can be in flight at a time (IPC = 1). |
| **Int indep** | `Int indep` / `Integer (4 independent)` | Throughput (Gops/s) of 4 independent integer multiply-add chains running simultaneously. Measures the ALU width / instruction-level parallelism (ILP) the core can exploit. |
| **FP64 dep** | `FP64 dep` / `FP64 (dependent FMA)` | Throughput (Gflops/s) of a single dependent 64-bit floating-point FMA chain: `x = fma(x, a, b)`. Measures FP pipeline latency. |
| **FP64 indep** | `FP64 indep` / `FP64 (4 independent FMA)` | Throughput (Gflops/s) of 4 independent FP64 FMA chains. Measures FP execution width / ILP. |
| **FP32 dep** | `FP32 dep` / `FP32 (dependent FMA)` | Same as FP64 dep but using 32-bit single-precision floats. |
| **FP32 indep** | `FP32 indep` / `FP32 (4 independent FMA)` | Same as FP64 indep but using 32-bit single-precision floats. |
| **NEON FP32** | `NEON FP32` / `NEON FP32 FMA` | Throughput (Gflops/s) of 4 independent NEON `vfmaq_f32` chains (4 lanes each = 16 FP32 ops per iteration). Measures SIMD floating-point throughput on AArch64. |
| **NEON Int** | `NEON Int` / `NEON Int mul` | Throughput (Gops/s) of 4 independent NEON `vmulq_u32` + `vaddq_u32` chains (4 lanes each). Measures SIMD integer throughput. |
| **P-core Int** | `P-core Int` / `P-core Integer` | Integer throughput (Gops/s) measured while pinned to a performance core via macOS QoS class `USER_INTERACTIVE`. |
| **P-core FP** | `P-core FP` / `P-core FP64` | FP64 throughput (Gflops/s) measured while pinned to a performance core. |
| **E-core Int** | `E-core Int` / `E-core Integer` | Integer throughput (Gops/s) measured while pinned to an efficiency core via macOS QoS class `BACKGROUND`. |
| **E-core FP** | `E-core FP` / `E-core FP64` | FP64 throughput (Gflops/s) measured while pinned to an efficiency core. |
| **MT Int** | `MT Int (Nc)` / `Integer multi-thread (Nc)` | Aggregate integer throughput (Gops/s) with N threads each running the 4-chain independent integer workload. |
| **MT FP** | `MT FP (Nc)` / `FP64 multi-thread (Nc)` | Aggregate FP64 throughput (Gflops/s) with N threads each running the 4-chain independent FMA workload. |

---

## Cache Hierarchy Profiling

| Term | Log Label | Description |
|---|---|---|
| **Latency Ladder** | `Latency ladder` / `Latency <size>` | Pointer-chase latency (ns per access) at 12 working set sizes from 32 KB to 64 MB. Reveals L1, L2, L3, and main memory latency boundaries. Built using Sattolo's algorithm to create a random Hamiltonian cycle. |
| **BW Ladder** | `BW ladder` / `Bandwidth <size>` | Sequential read throughput (GB/s) at the same 12 working set sizes. Shows how bandwidth drops as the working set exceeds each cache level. |
| **Cache Ladder Sizes** | — | The 12 tested sizes: 32 KB, 64 KB, 128 KB, 256 KB, 512 KB, 1 MB, 2 MB, 4 MB, 8 MB, 16 MB, 32 MB, 64 MB. |

---

## Memory Correctness Tests

| Term | Description |
|---|---|
| **Solid Bits** | Writes a uniform pattern (all 0x00 or all 0xFF) to every location, then verifies every location matches. |
| **Checkerboard** | Writes alternating-bit patterns (0xAA… or 0x55…) to detect coupling faults between adjacent bits. |
| **Walking Ones** | Writes a single 1-bit shifted through all 64 bit positions, verifying after each. Detects stuck-at-0 faults. |
| **Walking Zeros** | Writes all-ones with a single 0-bit shifted through each position. Detects stuck-at-1 faults. |
| **Address-as-Value** | Writes each location's index as its value, then reads back. Detects address-decode faults. |
| **March C-** | A classic March test algorithm. Writes all zeros, then walks forward flipping 0→1, walks forward flipping 1→0, walks backward flipping 0→1, walks backward flipping 1→0, and does a final all-zero check. Detects stuck-at, transition, and coupling faults. |
| **Random Fill** | Fills memory with a PRNG sequence (xorshift64) seeded with a known value, then replays the same seed to verify. Two seeds are tested (0xDEAD, 0xBEEF). |
| **Passes** | Number of complete test cycles (each cycle runs all 10 tests). |
| **Passed** | Total individual test invocations that succeeded across all passes. |
| **Failed** | Total individual test invocations where read-back did not match the expected value. |

---

## Stress Test Statistics

| Term | Log Label | Description |
|---|---|---|
| **MIN** | `MIN` | The lowest value recorded for a metric across all passes. For throughput metrics, this is the worst result. |
| **MAX** | `MAX` | The highest value recorded. For throughput metrics, this is the best result. |
| **AVG** | `AVG` | The arithmetic mean of all recorded values for a metric. |
| **Regression** | — | Detected when a metric degrades by more than 15% from its best value (throughput drops below 85% of max, or latency exceeds 115% of min). |

---

## Units

| Unit | Meaning |
|---|---|
| **GB/s** | Gigabytes per second (10^9 bytes/s). Used for all throughput/bandwidth metrics. |
| **Gops/s** | Giga-operations per second (10^9 ops/s). Used for integer throughput metrics. |
| **Gflops/s** | Giga-floating-point-operations per second (10^9 flops/s). Used for FP throughput metrics. |
| **ns** | Nanoseconds. Used for all latency metrics. |
| **MB** | Megabytes. Used for the memory region size parameter. |

---

## System Info Fields

| Term | Description |
|---|---|
| **Chip** | The Apple silicon chip name (e.g., "Apple M3 Max"), detected via `system_profiler SPHardwareDataType`. |
| **Model ID** | The Mac hardware model identifier, read from `sysctl hw.model`. |
| **P-cores** | Performance cores — the high-clock, wide-execution cores in Apple's big.LITTLE design. Count read from `sysctl hw.perflevel0.logicalcpu`. |
| **E-cores** | Efficiency cores — the low-power cores. Count read from `sysctl hw.perflevel1.logicalcpu`. |
| **Total Cores** | Total logical CPU count from `sysctl hw.ncpu`. |
| **Memory** | Total physical RAM in GB, from `sysctl hw.memsize`. |
| **Threads** | Number of threads used for multi-threaded benchmarks. Defaults to total core count; configurable with `--threads` / `-T`. |

---

## Log File Fields

| Term | Description |
|---|---|
| **Date** | Timestamp when the run started. |
| **Mode** | Which run mode was used (e.g., "All"). |
| **Size** | The memory region size in MB. |
| **Threads** | Thread count used for MT benchmarks. |
| **Elapsed** | Total wall-clock time for the run. |

---

## Implementation Details

| Term | Description |
|---|---|
| **Volatile Read/Write** | Uses `ptr::read_volatile` / `ptr::write_volatile` to prevent the compiler from optimizing away memory accesses. Essential for accurate memory benchmarking and correctness testing. |
| **Sattolo's Algorithm** | A shuffle algorithm that produces a random Hamiltonian cycle (single cycle visiting all elements). Used to build pointer-chase chains for latency measurement — ensures every element is visited exactly once in a random order. |
| **Pointer Chase** | A latency measurement technique where each array element contains the index of the next element to visit. Forces serial dependent loads, preventing the CPU from overlapping or prefetching accesses. |
| **xorshift64** | A fast PRNG used for shuffle randomness and random-fill test patterns. |
| **black_box** | Rust's `std::hint::black_box` — prevents the compiler from optimizing away computed values, ensuring benchmark work is not elided. |
| **FMA** | Fused Multiply-Add. A single instruction that computes `a * b + c`. Used in FP benchmarks via Rust's `f64::mul_add` / `f32::mul_add` and NEON's `vfmaq_f32`. |
| **NEON** | ARM's SIMD instruction set (Advanced SIMD) available on AArch64. Used for vectorized FP32 and integer benchmarks with 128-bit registers (4 × f32 or 4 × u32 lanes). |
| **ILP** | Instruction-Level Parallelism. The ability of a CPU core to execute multiple independent instructions simultaneously. Measured by comparing dependent (1 chain) vs. independent (4 chain) throughput. |
| **QoS Class** | macOS Quality of Service classes used to influence thread scheduling. `USER_INTERACTIVE` (0x21) targets P-cores; `BACKGROUND` (0x09) targets E-cores. Used for P-core/E-core isolation benchmarks. |
| **BENCH_PASSES** | Number of times each benchmark is repeated to find the best (minimum-time) result. Default: 3 for memory, 5 for CPU. |
| **CHASE_STEPS** | Number of pointer-chase iterations per latency measurement. 10M for memory latency, 5M for cache latency ladder. |
| **Barrier** | A `std::sync::Barrier` used to synchronize thread start times in multi-threaded benchmarks, ensuring all threads begin work simultaneously. |
