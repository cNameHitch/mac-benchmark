use std::fs;
use std::io::{stdout, Write};
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;
use ratatui::widgets::*;

use bench_core::{format_duration, install_signal_handler, RUNNING};
use mem_bench::bench::*;
use mem_bench::tests::run_test_pass;

// ---------------------------------------------------------------------------
// Run mode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
pub enum RunMode {
    Test,
    Bench,
    Cpu,
    MtCpu,
    Gpu,
    All,
    Stress,
    FullStress,
}

impl RunMode {
    const ALL_MODES: [RunMode; 8] = [
        RunMode::Test,
        RunMode::Bench,
        RunMode::Cpu,
        RunMode::MtCpu,
        RunMode::Gpu,
        RunMode::All,
        RunMode::Stress,
        RunMode::FullStress,
    ];

    fn label(self) -> &'static str {
        match self {
            RunMode::Test => "Test",
            RunMode::Bench => "Bench",
            RunMode::Cpu => "CPU",
            RunMode::MtCpu => "MT CPU",
            RunMode::Gpu => "GPU",
            RunMode::All => "All",
            RunMode::Stress => "Stress",
            RunMode::FullStress => "Full Stress",
        }
    }

    fn description(self) -> &'static str {
        match self {
            RunMode::Test => "Continuous memory correctness tests",
            RunMode::Bench => "Continuous memory bandwidth & latency benchmarks",
            RunMode::Cpu => "Continuous CPU throughput + cache hierarchy profiling",
            RunMode::MtCpu => "Multi-threaded CPU benchmark (saturates all cores)",
            RunMode::Gpu => "GPU compute throughput, memory bandwidth, and matrix multiply (Metal)",
            RunMode::All => "Continuous tests + memory + CPU + GPU benchmarks",
            RunMode::Stress => "Continuous correctness stress test",
            RunMode::FullStress => "Full stress: tests + all benchmarks every cycle",
        }
    }
}

// ---------------------------------------------------------------------------
// Transitions & screens
// ---------------------------------------------------------------------------

enum Transition {
    Stay,
    ToRunning(RunMode, usize, Option<Duration>, u32),
    ToSummary,
    ToMenu,
    Exit,
}

enum Screen {
    Menu(MenuState),
    Running(RunningState),
    Summary(SummaryState),
}

// ---------------------------------------------------------------------------
// Menu state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum MenuField {
    ModeList,
    SizeMb,
    Duration,
    Threads,
}

struct MenuState {
    chip_name: String,
    model_name: String,
    total_cores: u32,
    p_cores: u32,
    e_cores: u32,
    gpu_cores: u32,
    metal_version: String,
    os_version: String,
    os_build: String,
    uptime_secs: u64,
    dynamic: mac_sysinfo::DynamicInfo,
    last_dynamic_poll: Instant,
    selected_mode: usize,
    size_input: String,
    duration_input: String,
    threads_input: String,
    active_field: MenuField,
}

impl MenuState {
    fn new(info: &mac_sysinfo::SystemInfo, size_mb: usize, duration: Option<Duration>, threads: Option<u32>) -> Self {
        let dynamic = mac_sysinfo::poll_dynamic();
        Self {
            chip_name: info.chip_name.clone(),
            model_name: info.model_name.clone(),
            total_cores: info.total_cores,
            p_cores: info.p_cores,
            e_cores: info.e_cores,
            gpu_cores: info.gpu_cores,
            metal_version: info.metal_version.clone(),
            os_version: info.os_version.clone(),
            os_build: info.os_build.clone(),
            uptime_secs: info.uptime_secs,
            dynamic,
            last_dynamic_poll: Instant::now(),
            selected_mode: 0,
            size_input: size_mb.to_string(),
            duration_input: duration.map(|d| (d.as_secs() / 60).to_string()).unwrap_or_default(),
            threads_input: threads.unwrap_or(info.total_cores).to_string(),
            active_field: MenuField::ModeList,
        }
    }

    fn parsed_size_mb(&self) -> usize {
        self.size_input.parse().unwrap_or(256)
    }

    fn parsed_duration(&self) -> Option<Duration> {
        if self.duration_input.is_empty() {
            None
        } else {
            self.duration_input.parse::<u64>().ok().map(|m| Duration::from_secs(m * 60))
        }
    }

    fn parsed_threads(&self) -> u32 {
        self.threads_input.parse().unwrap_or(self.total_cores)
    }

    fn selected_run_mode(&self) -> RunMode {
        RunMode::ALL_MODES[self.selected_mode]
    }
}

// ---------------------------------------------------------------------------
// Worker messages
// ---------------------------------------------------------------------------

enum WorkerMsg {
    BenchResult(BenchMetric),
    StressPassResult {
        pass_num: u64,
        passed: u32,
        failed: u32,
        errors: Vec<String>,
        duration: Duration,
    },
    StressTestProgress(String),
    GpuStatus(String),
    Done,
}

enum BenchMetric {
    SeqWrite(f64),
    SeqRead(f64),
    Copy(f64),
    Latency { ns: f64, working_set_mb: usize },
    Stride { label: String, gbps: f64 },
    IntDependent(f64),
    IntIndependent(f64),
    FpDependent(f64),
    FpIndependent(f64),
    MtInt(f64),
    MtFp(f64),
    CacheLatency { size_label: String, ns: f64 },
    CacheBandwidth { size_label: String, gbps: f64 },
    Fp32Dependent(f64),
    Fp32Independent(f64),
    NeonFp32(f64),
    NeonInt(f64),
    PcoreInt(f64),
    PcoreFp(f64),
    EcoreInt(f64),
    EcoreFp(f64),
    MtSeqRead(f64),
    LoadedLatency(f64),
    GpuFp32(f64),
    GpuFp16(f64),
    GpuInt32(f64),
    GpuBufRead(f64),
    GpuBufWrite(f64),
    GpuBufAlloc(f64),
    GpuMatmul(f64),
}

// ---------------------------------------------------------------------------
// MetricStats (moved from main.rs)
// ---------------------------------------------------------------------------

struct MetricStats {
    min: f64,
    max: f64,
    sum: f64,
    last: f64,
    count: u64,
    higher_is_better: bool,
}

impl MetricStats {
    fn new(higher_is_better: bool) -> Self {
        Self {
            min: f64::MAX,
            max: f64::MIN,
            sum: 0.0,
            last: 0.0,
            count: 0,
            higher_is_better,
        }
    }

    fn record(&mut self, value: f64) -> bool {
        let regression = if self.count > 0 {
            let best = self.best();
            if self.higher_is_better {
                value < best * 0.85
            } else {
                value > best * 1.15
            }
        } else {
            false
        };
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
        self.last = value;
        self.count += 1;
        regression
    }

    fn avg(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum / self.count as f64 }
    }

    fn best(&self) -> f64 {
        if self.higher_is_better { self.max } else { self.min }
    }
}

struct FullStressStats {
    seq_write: MetricStats,
    seq_read: MetricStats,
    copy: MetricStats,
    random_latency: MetricStats,
    strides: Vec<(String, MetricStats)>,
    int_dep: MetricStats,
    int_indep: MetricStats,
    fp_dep: MetricStats,
    fp_indep: MetricStats,
    fp32_dep: MetricStats,
    fp32_indep: MetricStats,
    neon_fp32: MetricStats,
    neon_int: MetricStats,
    pcore_int: MetricStats,
    pcore_fp: MetricStats,
    ecore_int: MetricStats,
    ecore_fp: MetricStats,
    mt_int: MetricStats,
    mt_fp: MetricStats,
    mt_seq_read: MetricStats,
    loaded_latency: MetricStats,
    cache_latencies: Vec<(String, MetricStats)>,
    cache_bandwidths: Vec<(String, MetricStats)>,
    gpu_fp32: MetricStats,
    gpu_fp16: MetricStats,
    gpu_int32: MetricStats,
    gpu_buf_read: MetricStats,
    gpu_buf_write: MetricStats,
    gpu_buf_alloc: MetricStats,
    gpu_matmul: MetricStats,
    total_passed: u64,
    total_failed: u64,
}

impl FullStressStats {
    fn new(region_len: usize, max_bytes: usize) -> Self {
        let strides = STRIDES
            .iter()
            .filter(|&&(stride_elems, _)| stride_elems < region_len)
            .map(|&(_, label)| (label.to_string(), MetricStats::new(true)))
            .collect();

        let cache_latencies = cpu_bench::CACHE_LADDER_SIZES
            .iter()
            .filter(|&&(size, _)| size <= max_bytes)
            .map(|&(_, label)| (label.to_string(), MetricStats::new(false)))
            .collect();

        let cache_bandwidths = cpu_bench::CACHE_LADDER_SIZES
            .iter()
            .filter(|&&(size, _)| size <= max_bytes)
            .map(|&(_, label)| (label.to_string(), MetricStats::new(true)))
            .collect();

        Self {
            seq_write: MetricStats::new(true),
            seq_read: MetricStats::new(true),
            copy: MetricStats::new(true),
            random_latency: MetricStats::new(false),
            strides,
            int_dep: MetricStats::new(true),
            int_indep: MetricStats::new(true),
            fp_dep: MetricStats::new(true),
            fp_indep: MetricStats::new(true),
            fp32_dep: MetricStats::new(true),
            fp32_indep: MetricStats::new(true),
            neon_fp32: MetricStats::new(true),
            neon_int: MetricStats::new(true),
            pcore_int: MetricStats::new(true),
            pcore_fp: MetricStats::new(true),
            ecore_int: MetricStats::new(true),
            ecore_fp: MetricStats::new(true),
            mt_int: MetricStats::new(true),
            mt_fp: MetricStats::new(true),
            mt_seq_read: MetricStats::new(true),
            loaded_latency: MetricStats::new(false),
            cache_latencies,
            cache_bandwidths,
            gpu_fp32: MetricStats::new(true),
            gpu_fp16: MetricStats::new(true),
            gpu_int32: MetricStats::new(true),
            gpu_buf_read: MetricStats::new(true),
            gpu_buf_write: MetricStats::new(true),
            gpu_buf_alloc: MetricStats::new(false), // lower is better
            gpu_matmul: MetricStats::new(true),
            total_passed: 0,
            total_failed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Dashboard state
// ---------------------------------------------------------------------------

const SPARKLINE_LEN: usize = 64;

struct DashboardState {
    mode: RunMode,
    size_mb: usize,
    num_threads: u32,
    start: Instant,
    chip_name: String,
    gpu_cores: u32,
    metal_version: String,
    os_version: String,
    dynamic: mac_sysinfo::DynamicInfo,
    last_dynamic_poll: Instant,

    // Bench results
    seq_write_gbps: Option<f64>,
    seq_read_gbps: Option<f64>,
    copy_gbps: Option<f64>,
    latency_ns: Option<f64>,
    latency_ws_mb: Option<usize>,
    strides: Vec<(String, f64)>,

    // CPU bench results
    int_dependent: Option<f64>,
    int_independent: Option<f64>,
    fp_dependent: Option<f64>,
    fp_independent: Option<f64>,
    fp32_dependent: Option<f64>,
    fp32_independent: Option<f64>,
    neon_fp32: Option<f64>,
    neon_int: Option<f64>,
    pcore_int: Option<f64>,
    pcore_fp: Option<f64>,
    ecore_int: Option<f64>,
    ecore_fp: Option<f64>,
    mt_int_gops: Option<f64>,
    mt_fp_gflops: Option<f64>,
    mt_seq_read_gbps: Option<f64>,
    loaded_latency_ns: Option<f64>,
    cache_latencies: Vec<(String, f64)>,
    cache_bandwidths: Vec<(String, f64)>,

    // GPU bench results
    gpu_fp32_tflops: Option<f64>,
    gpu_fp16_tflops: Option<f64>,
    gpu_int32_tops: Option<f64>,
    gpu_buf_read_gbps: Option<f64>,
    gpu_buf_write_gbps: Option<f64>,
    gpu_buf_alloc_us: Option<f64>,
    gpu_matmul_tflops: Option<f64>,
    gpu_status: Option<String>,

    // Sparkline history - bandwidth
    write_history: Vec<u64>,
    read_history: Vec<u64>,
    copy_history: Vec<u64>,
    latency_history: Vec<u64>,
    mt_seq_read_history: Vec<u64>,
    loaded_latency_history: Vec<u64>,

    // Sparkline history - CPU throughput
    int_dep_history: Vec<u64>,
    int_indep_history: Vec<u64>,
    fp_dep_history: Vec<u64>,
    fp_indep_history: Vec<u64>,
    fp32_dep_history: Vec<u64>,
    fp32_indep_history: Vec<u64>,
    neon_fp32_history: Vec<u64>,
    neon_int_history: Vec<u64>,
    pcore_int_history: Vec<u64>,
    pcore_fp_history: Vec<u64>,
    ecore_int_history: Vec<u64>,
    ecore_fp_history: Vec<u64>,
    mt_int_history: Vec<u64>,
    mt_fp_history: Vec<u64>,

    // Sparkline history - GPU
    gpu_fp32_history: Vec<u64>,
    gpu_fp16_history: Vec<u64>,
    gpu_buf_read_history: Vec<u64>,
    gpu_matmul_history: Vec<u64>,

    // Stress state
    stress_pass: u64,
    stress_total_passed: u64,
    stress_total_failed: u64,
    stress_current_test: String,
    stress_recent_errors: Vec<String>,
    pass_history: Vec<PassRecord>,

    // Metric stats (all modes)
    metric_stats: Option<FullStressStats>,

    // Scroll offset for panels that overflow
    scroll_offset: u16,
}

struct PassRecord {
    pass_num: u64,
    passed: u32,
    failed: u32,
    duration: Duration,
}

impl DashboardState {
    fn new(mode: RunMode, size_mb: usize, num_threads: u32, info: &mac_sysinfo::SystemInfo) -> Self {
        let count = size_mb * 1024 * 1024 / size_of::<u64>();
        let max_bytes = size_mb * 1024 * 1024;
        let metric_stats = Some(FullStressStats::new(count, max_bytes));

        Self {
            mode,
            size_mb,
            num_threads,
            start: Instant::now(),
            chip_name: info.chip_name.clone(),
            gpu_cores: info.gpu_cores,
            metal_version: info.metal_version.clone(),
            os_version: info.os_version.clone(),
            dynamic: mac_sysinfo::poll_dynamic(),
            last_dynamic_poll: Instant::now(),
            seq_write_gbps: None,
            seq_read_gbps: None,
            copy_gbps: None,
            latency_ns: None,
            latency_ws_mb: None,
            strides: Vec::new(),
            int_dependent: None,
            int_independent: None,
            fp_dependent: None,
            fp_independent: None,
            fp32_dependent: None,
            fp32_independent: None,
            neon_fp32: None,
            neon_int: None,
            pcore_int: None,
            pcore_fp: None,
            ecore_int: None,
            ecore_fp: None,
            mt_int_gops: None,
            mt_fp_gflops: None,
            mt_seq_read_gbps: None,
            loaded_latency_ns: None,
            cache_latencies: Vec::new(),
            cache_bandwidths: Vec::new(),
            gpu_fp32_tflops: None,
            gpu_fp16_tflops: None,
            gpu_int32_tops: None,
            gpu_buf_read_gbps: None,
            gpu_buf_write_gbps: None,
            gpu_buf_alloc_us: None,
            gpu_matmul_tflops: None,
            gpu_status: None,
            write_history: Vec::new(),
            read_history: Vec::new(),
            copy_history: Vec::new(),
            latency_history: Vec::new(),
            mt_seq_read_history: Vec::new(),
            loaded_latency_history: Vec::new(),
            int_dep_history: Vec::new(),
            int_indep_history: Vec::new(),
            fp_dep_history: Vec::new(),
            fp_indep_history: Vec::new(),
            fp32_dep_history: Vec::new(),
            fp32_indep_history: Vec::new(),
            neon_fp32_history: Vec::new(),
            neon_int_history: Vec::new(),
            pcore_int_history: Vec::new(),
            pcore_fp_history: Vec::new(),
            ecore_int_history: Vec::new(),
            ecore_fp_history: Vec::new(),
            mt_int_history: Vec::new(),
            mt_fp_history: Vec::new(),
            gpu_fp32_history: Vec::new(),
            gpu_fp16_history: Vec::new(),
            gpu_buf_read_history: Vec::new(),
            gpu_matmul_history: Vec::new(),
            stress_pass: 0,
            stress_total_passed: 0,
            stress_total_failed: 0,
            stress_current_test: String::new(),
            stress_recent_errors: Vec::new(),
            pass_history: Vec::new(),
            metric_stats,
            scroll_offset: 0,
        }
    }

    fn apply(&mut self, msg: WorkerMsg) {
        match msg {
            WorkerMsg::BenchResult(metric) => {
                match metric {
                    BenchMetric::SeqWrite(v) => {
                        self.seq_write_gbps = Some(v);
                        push_sparkline(&mut self.write_history, v);
                        if let Some(ref mut s) = self.metric_stats {
                            s.seq_write.record(v);
                        }
                    }
                    BenchMetric::SeqRead(v) => {
                        self.seq_read_gbps = Some(v);
                        push_sparkline(&mut self.read_history, v);
                        if let Some(ref mut s) = self.metric_stats {
                            s.seq_read.record(v);
                        }
                    }
                    BenchMetric::Copy(v) => {
                        self.copy_gbps = Some(v);
                        push_sparkline(&mut self.copy_history, v);
                        if let Some(ref mut s) = self.metric_stats {
                            s.copy.record(v);
                        }
                    }
                    BenchMetric::Latency { ns, working_set_mb } => {
                        self.latency_ns = Some(ns);
                        self.latency_ws_mb = Some(working_set_mb);
                        push_sparkline(&mut self.latency_history, ns);
                        if let Some(ref mut s) = self.metric_stats {
                            s.random_latency.record(ns);
                        }
                    }
                    BenchMetric::Stride { label, gbps } => {
                        if let Some(entry) = self.strides.iter_mut().find(|(l, _)| *l == label) {
                            entry.1 = gbps;
                        } else {
                            self.strides.push((label.clone(), gbps));
                        }
                        if let Some(ref mut s) = self.metric_stats {
                            if let Some(entry) = s.strides.iter_mut().find(|(l, _)| *l == label) {
                                entry.1.record(gbps);
                            }
                        }
                    }
                    BenchMetric::IntDependent(v) => {
                        self.int_dependent = Some(v);
                        push_sparkline(&mut self.int_dep_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.int_dep.record(v); }
                    }
                    BenchMetric::IntIndependent(v) => {
                        self.int_independent = Some(v);
                        push_sparkline(&mut self.int_indep_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.int_indep.record(v); }
                    }
                    BenchMetric::FpDependent(v) => {
                        self.fp_dependent = Some(v);
                        push_sparkline(&mut self.fp_dep_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.fp_dep.record(v); }
                    }
                    BenchMetric::FpIndependent(v) => {
                        self.fp_independent = Some(v);
                        push_sparkline(&mut self.fp_indep_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.fp_indep.record(v); }
                    }
                    BenchMetric::MtInt(v) => {
                        self.mt_int_gops = Some(v);
                        push_sparkline(&mut self.mt_int_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.mt_int.record(v); }
                    }
                    BenchMetric::MtFp(v) => {
                        self.mt_fp_gflops = Some(v);
                        push_sparkline(&mut self.mt_fp_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.mt_fp.record(v); }
                    }
                    BenchMetric::CacheLatency { size_label, ns } => {
                        if let Some(entry) = self.cache_latencies.iter_mut().find(|(l, _)| *l == size_label) {
                            entry.1 = ns;
                        } else {
                            self.cache_latencies.push((size_label.clone(), ns));
                        }
                        if let Some(ref mut s) = self.metric_stats {
                            if let Some(entry) = s.cache_latencies.iter_mut().find(|(l, _)| *l == size_label) {
                                entry.1.record(ns);
                            }
                        }
                    }
                    BenchMetric::CacheBandwidth { size_label, gbps } => {
                        if let Some(entry) = self.cache_bandwidths.iter_mut().find(|(l, _)| *l == size_label) {
                            entry.1 = gbps;
                        } else {
                            self.cache_bandwidths.push((size_label.clone(), gbps));
                        }
                        if let Some(ref mut s) = self.metric_stats {
                            if let Some(entry) = s.cache_bandwidths.iter_mut().find(|(l, _)| *l == size_label) {
                                entry.1.record(gbps);
                            }
                        }
                    }
                    BenchMetric::Fp32Dependent(v) => {
                        self.fp32_dependent = Some(v);
                        push_sparkline(&mut self.fp32_dep_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.fp32_dep.record(v); }
                    }
                    BenchMetric::Fp32Independent(v) => {
                        self.fp32_independent = Some(v);
                        push_sparkline(&mut self.fp32_indep_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.fp32_indep.record(v); }
                    }
                    BenchMetric::NeonFp32(v) => {
                        self.neon_fp32 = Some(v);
                        push_sparkline(&mut self.neon_fp32_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.neon_fp32.record(v); }
                    }
                    BenchMetric::NeonInt(v) => {
                        self.neon_int = Some(v);
                        push_sparkline(&mut self.neon_int_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.neon_int.record(v); }
                    }
                    BenchMetric::PcoreInt(v) => {
                        self.pcore_int = Some(v);
                        push_sparkline(&mut self.pcore_int_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.pcore_int.record(v); }
                    }
                    BenchMetric::PcoreFp(v) => {
                        self.pcore_fp = Some(v);
                        push_sparkline(&mut self.pcore_fp_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.pcore_fp.record(v); }
                    }
                    BenchMetric::EcoreInt(v) => {
                        self.ecore_int = Some(v);
                        push_sparkline(&mut self.ecore_int_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.ecore_int.record(v); }
                    }
                    BenchMetric::EcoreFp(v) => {
                        self.ecore_fp = Some(v);
                        push_sparkline(&mut self.ecore_fp_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.ecore_fp.record(v); }
                    }
                    BenchMetric::MtSeqRead(v) => {
                        self.mt_seq_read_gbps = Some(v);
                        push_sparkline(&mut self.mt_seq_read_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.mt_seq_read.record(v); }
                    }
                    BenchMetric::LoadedLatency(v) => {
                        self.loaded_latency_ns = Some(v);
                        push_sparkline(&mut self.loaded_latency_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.loaded_latency.record(v); }
                    }
                    BenchMetric::GpuFp32(v) => {
                        self.gpu_fp32_tflops = Some(v);
                        push_sparkline(&mut self.gpu_fp32_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.gpu_fp32.record(v); }
                    }
                    BenchMetric::GpuFp16(v) => {
                        self.gpu_fp16_tflops = Some(v);
                        push_sparkline(&mut self.gpu_fp16_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.gpu_fp16.record(v); }
                    }
                    BenchMetric::GpuInt32(v) => {
                        self.gpu_int32_tops = Some(v);
                        if let Some(ref mut s) = self.metric_stats { s.gpu_int32.record(v); }
                    }
                    BenchMetric::GpuBufRead(v) => {
                        self.gpu_buf_read_gbps = Some(v);
                        push_sparkline(&mut self.gpu_buf_read_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.gpu_buf_read.record(v); }
                    }
                    BenchMetric::GpuBufWrite(v) => {
                        self.gpu_buf_write_gbps = Some(v);
                        if let Some(ref mut s) = self.metric_stats { s.gpu_buf_write.record(v); }
                    }
                    BenchMetric::GpuBufAlloc(v) => {
                        self.gpu_buf_alloc_us = Some(v);
                        if let Some(ref mut s) = self.metric_stats { s.gpu_buf_alloc.record(v); }
                    }
                    BenchMetric::GpuMatmul(v) => {
                        self.gpu_matmul_tflops = Some(v);
                        push_sparkline(&mut self.gpu_matmul_history, v);
                        if let Some(ref mut s) = self.metric_stats { s.gpu_matmul.record(v); }
                    }
                }
            }
            WorkerMsg::StressPassResult { pass_num, passed, failed, errors, duration } => {
                self.stress_pass = pass_num;
                self.stress_total_passed += passed as u64;
                self.stress_total_failed += failed as u64;
                self.stress_current_test.clear();

                if let Some(ref mut s) = self.metric_stats {
                    s.total_passed += passed as u64;
                    s.total_failed += failed as u64;
                }

                for err in &errors {
                    self.stress_recent_errors.push(err.clone());
                    if self.stress_recent_errors.len() > 10 {
                        self.stress_recent_errors.remove(0);
                    }
                }

                self.pass_history.insert(0, PassRecord {
                    pass_num,
                    passed,
                    failed,
                    duration,
                });
                if self.pass_history.len() > 50 {
                    self.pass_history.pop();
                }
            }
            WorkerMsg::StressTestProgress(name) => {
                self.stress_current_test = name;
            }
            WorkerMsg::GpuStatus(msg) => {
                self.gpu_status = Some(msg);
            }
            WorkerMsg::Done => {}
        }
    }
}

fn push_sparkline(history: &mut Vec<u64>, value: f64) {
    history.push((value * 100.0) as u64);
    if history.len() > SPARKLINE_LEN {
        history.remove(0);
    }
}

// ---------------------------------------------------------------------------
// Running state (owns worker thread + channel + dashboard)
// ---------------------------------------------------------------------------

struct RunningState {
    mode: RunMode,
    size_mb: usize,
    duration: Option<Duration>,
    dashboard: DashboardState,
    rx: mpsc::Receiver<WorkerMsg>,
    worker: Option<thread::JoinHandle<()>>,
    done: bool,
    run_start: Instant,
}

// ---------------------------------------------------------------------------
// Summary state
// ---------------------------------------------------------------------------

struct SummaryState {
    mode: RunMode,
    size_mb: usize,
    dashboard: DashboardState,
    selected_action: SummaryAction,
}

#[derive(Clone, Copy, PartialEq)]
enum SummaryAction {
    BackToMenu,
    Quit,
}

// ---------------------------------------------------------------------------
// Worker sub-functions
// ---------------------------------------------------------------------------

fn is_running() -> bool {
    RUNNING.load(Ordering::Relaxed)
}

fn worker_run_mem_bench(region: &mut [u64], num_threads: u32, tx: &mpsc::Sender<WorkerMsg>) {
    if !is_running() { return; }
    let v = bench_seq_write(region);
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::SeqWrite(v)));

    if !is_running() { return; }
    let v = bench_seq_read(region);
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::SeqRead(v)));

    if !is_running() { return; }
    let v = bench_copy(region);
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::Copy(v)));

    if !is_running() { return; }
    let (ns, ws_mb) = bench_random_latency(region);
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::Latency { ns, working_set_mb: ws_mb }));

    for &(stride_elems, label) in STRIDES {
        if !is_running() { return; }
        if stride_elems >= region.len() { continue; }
        let gbps = bench_stride_read(region, stride_elems);
        let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::Stride {
            label: label.to_string(),
            gbps,
        }));
    }

    if !is_running() { return; }
    let v = bench_mt_seq_read(region, num_threads);
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::MtSeqRead(v)));

    if !is_running() { return; }
    let v = bench_loaded_latency(region, num_threads);
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::LoadedLatency(v)));
}

fn worker_run_cpu_bench(max_bytes: usize, num_threads: u32, tx: &mpsc::Sender<WorkerMsg>) {
    if !is_running() { return; }
    let v = cpu_bench::bench_int_dependent();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::IntDependent(v)));

    if !is_running() { return; }
    let v = cpu_bench::bench_int_independent();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::IntIndependent(v)));

    if !is_running() { return; }
    let v = cpu_bench::bench_fp_dependent();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::FpDependent(v)));

    if !is_running() { return; }
    let v = cpu_bench::bench_fp_independent();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::FpIndependent(v)));

    if !is_running() { return; }
    let v = cpu_bench::bench_mt_int(num_threads);
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::MtInt(v)));

    if !is_running() { return; }
    let v = cpu_bench::bench_mt_fp(num_threads);
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::MtFp(v)));

    for &(size, label) in cpu_bench::CACHE_LADDER_SIZES {
        if !is_running() { return; }
        if size > max_bytes { break; }
        let ns = cpu_bench::bench_cache_latency(size);
        let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::CacheLatency {
            size_label: label.to_string(),
            ns,
        }));
    }

    for &(size, label) in cpu_bench::CACHE_LADDER_SIZES {
        if !is_running() { return; }
        if size > max_bytes { break; }
        let gbps = cpu_bench::bench_cache_bandwidth(size);
        let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::CacheBandwidth {
            size_label: label.to_string(),
            gbps,
        }));
    }

    // FP32 benchmarks
    if !is_running() { return; }
    let v = cpu_bench::bench_fp32_dependent();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::Fp32Dependent(v)));

    if !is_running() { return; }
    let v = cpu_bench::bench_fp32_independent();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::Fp32Independent(v)));

    // NEON SIMD benchmarks
    #[cfg(target_arch = "aarch64")]
    {
        if !is_running() { return; }
        let v = cpu_bench::neon::bench_neon_fp32();
        let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::NeonFp32(v)));

        if !is_running() { return; }
        let v = cpu_bench::neon::bench_neon_int();
        let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::NeonInt(v)));
    }

    // P-core / E-core isolation
    #[cfg(target_os = "macos")]
    {
        if !is_running() { return; }
        let v = cpu_bench::core_isolation::bench_pcore_int();
        let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::PcoreInt(v)));

        if !is_running() { return; }
        let v = cpu_bench::core_isolation::bench_pcore_fp();
        let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::PcoreFp(v)));

        if !is_running() { return; }
        let v = cpu_bench::core_isolation::bench_ecore_int();
        let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::EcoreInt(v)));

        if !is_running() { return; }
        let v = cpu_bench::core_isolation::bench_ecore_fp();
        let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::EcoreFp(v)));
    }
}

#[cfg(target_os = "macos")]
fn worker_run_gpu_bench(tx: &mpsc::Sender<WorkerMsg>) {
    let ctx = match gpu_bench::GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(WorkerMsg::GpuStatus(e));
            return;
        }
    };

    if !is_running() { return; }
    let v = ctx.bench_fp32_throughput();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::GpuFp32(v)));

    if !is_running() { return; }
    let v = ctx.bench_fp16_throughput();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::GpuFp16(v)));

    if !is_running() { return; }
    let v = ctx.bench_int32_throughput();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::GpuInt32(v)));

    if !is_running() { return; }
    let v = ctx.bench_buffer_read();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::GpuBufRead(v)));

    if !is_running() { return; }
    let v = ctx.bench_buffer_write();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::GpuBufWrite(v)));

    if !is_running() { return; }
    let v = ctx.bench_buffer_alloc();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::GpuBufAlloc(v)));

    if !is_running() { return; }
    let v = ctx.bench_matmul();
    let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::GpuMatmul(v)));
}

fn worker_run_stress_loop(region: &mut [u64], tx: &mpsc::Sender<WorkerMsg>) {
    let mut pass_num = 0u64;
    while is_running() {
        pass_num += 1;
        let _ = tx.send(WorkerMsg::StressTestProgress("Running tests...".to_string()));
        let pass_start = Instant::now();
        let (passed, failed, errors) = run_test_pass(region);
        if !is_running() { break; }
        let _ = tx.send(WorkerMsg::StressPassResult {
            pass_num, passed, failed, errors, duration: pass_start.elapsed(),
        });
    }
}

fn worker_run_full_stress_loop(region: &mut [u64], size_mb: usize, num_threads: u32, tx: &mpsc::Sender<WorkerMsg>) {
    let mut pass_num = 0u64;
    let max_bytes = size_mb * 1024 * 1024;

    while is_running() {
        pass_num += 1;
        let _ = tx.send(WorkerMsg::StressTestProgress("Running tests...".to_string()));
        let pass_start = Instant::now();
        let (passed, failed, errors) = run_test_pass(region);
        if !is_running() { break; }
        let _ = tx.send(WorkerMsg::StressPassResult {
            pass_num, passed, failed, errors, duration: pass_start.elapsed(),
        });

        worker_run_mem_bench(region, num_threads, tx);
        if !is_running() { break; }

        worker_run_cpu_bench(max_bytes, num_threads, tx);
        if !is_running() { break; }

        #[cfg(target_os = "macos")]
        worker_run_gpu_bench(tx);
        if !is_running() { break; }
    }
}

fn worker_loop(mode: RunMode, mut region: Vec<u64>, size_mb: usize, num_threads: u32, tx: mpsc::Sender<WorkerMsg>) {
    let max_bytes = size_mb * 1024 * 1024;
    match mode {
        RunMode::Test => {
            // Loop: run_test_pass each cycle, send StressPassResult
            let mut pass_num = 0u64;
            while is_running() {
                pass_num += 1;
                let _ = tx.send(WorkerMsg::StressTestProgress("Running tests...".to_string()));
                let pass_start = Instant::now();
                let (passed, failed, errors) = run_test_pass(&mut region);
                if !is_running() { break; }
                let _ = tx.send(WorkerMsg::StressPassResult {
                    pass_num, passed, failed, errors, duration: pass_start.elapsed(),
                });
            }
        }
        RunMode::Bench => {
            while is_running() {
                worker_run_mem_bench(&mut region, num_threads, &tx);
            }
        }
        RunMode::Cpu => {
            while is_running() {
                worker_run_cpu_bench(max_bytes, num_threads, &tx);
            }
        }
        RunMode::MtCpu => {
            while is_running() {
                if !is_running() { break; }
                let v = cpu_bench::bench_mt_int(num_threads);
                let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::MtInt(v)));

                if !is_running() { break; }
                let v = cpu_bench::bench_mt_fp(num_threads);
                let _ = tx.send(WorkerMsg::BenchResult(BenchMetric::MtFp(v)));
            }
        }
        RunMode::Gpu => {
            while is_running() {
                #[cfg(target_os = "macos")]
                worker_run_gpu_bench(&tx);
                if !is_running() { break; }
            }
        }
        RunMode::All => {
            // Loop: tests → StressPassResult → mem bench → cpu bench
            let mut pass_num = 0u64;
            while is_running() {
                pass_num += 1;
                let _ = tx.send(WorkerMsg::StressTestProgress("Running tests...".to_string()));
                let pass_start = Instant::now();
                let (passed, failed, errors) = run_test_pass(&mut region);
                if !is_running() { break; }
                let _ = tx.send(WorkerMsg::StressPassResult {
                    pass_num, passed, failed, errors, duration: pass_start.elapsed(),
                });

                worker_run_mem_bench(&mut region, num_threads, &tx);
                if !is_running() { break; }

                worker_run_cpu_bench(max_bytes, num_threads, &tx);
                if !is_running() { break; }

                #[cfg(target_os = "macos")]
                worker_run_gpu_bench(&tx);
                if !is_running() { break; }
            }
        }
        RunMode::Stress => {
            worker_run_stress_loop(&mut region, &tx);
        }
        RunMode::FullStress => {
            worker_run_full_stress_loop(&mut region, size_mb, num_threads, &tx);
        }
    }
    let _ = tx.send(WorkerMsg::Done);
}

// ---------------------------------------------------------------------------
// start_running helper
// ---------------------------------------------------------------------------

fn start_running(mode: RunMode, size_mb: usize, duration: Option<Duration>, num_threads: u32, info: &mac_sysinfo::SystemInfo) -> RunningState {
    let count = size_mb * 1024 * 1024 / size_of::<u64>();
    let region = vec![0u64; count];
    let (tx, rx) = mpsc::channel();
    let dashboard = DashboardState::new(mode, size_mb, num_threads, info);

    let worker = thread::spawn(move || {
        worker_loop(mode, region, size_mb, num_threads, tx);
    });

    RunningState {
        mode,
        size_mb,
        duration,
        dashboard,
        rx,
        worker: Some(worker),
        done: false,
        run_start: Instant::now(),
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run(mode: Option<RunMode>, size_mb: usize, duration: Option<Duration>, threads: Option<u32>) {
    install_signal_handler();

    let info = mac_sysinfo::detect();
    let thread_count = threads.unwrap_or(info.total_cores);

    // Set up panic hook to restore terminal
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = terminal::disable_raw_mode();
        let _ = stdout().execute(LeaveAlternateScreen);
        original_hook(panic_info);
    }));

    // Enter TUI
    terminal::enable_raw_mode().expect("failed to enable raw mode");
    stdout().execute(EnterAlternateScreen).expect("failed to enter alternate screen");
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend).expect("failed to create terminal");

    let mut screen = match mode {
        None => Screen::Menu(MenuState::new(&info, size_mb, duration, threads)),
        Some(m) => {
            RUNNING.store(true, Ordering::Relaxed);
            Screen::Running(start_running(m, size_mb, duration, thread_count, &info))
        }
    };

    loop {
        let transition = match &mut screen {
            Screen::Menu(menu) => {
                if menu.last_dynamic_poll.elapsed() >= Duration::from_secs(3) {
                    menu.dynamic = mac_sysinfo::poll_dynamic();
                    menu.last_dynamic_poll = Instant::now();
                }
                let _ = terminal.draw(|frame| draw_menu(frame, menu));
                handle_menu_input(menu)
            }
            Screen::Running(rs) => {
                // Check duration limit
                if let Some(d) = rs.duration {
                    if rs.run_start.elapsed() >= d {
                        RUNNING.store(false, Ordering::Relaxed);
                    }
                }

                // Drain messages
                while let Ok(msg) = rs.rx.try_recv() {
                    if matches!(msg, WorkerMsg::Done) {
                        rs.done = true;
                        rs.dashboard.apply(msg);
                        break;
                    }
                    rs.dashboard.apply(msg);
                }

                // Poll dynamic system info every 3 seconds
                if rs.dashboard.last_dynamic_poll.elapsed() >= Duration::from_secs(3) {
                    rs.dashboard.dynamic = mac_sysinfo::poll_dynamic();
                    rs.dashboard.last_dynamic_poll = Instant::now();
                }

                if rs.done {
                    Transition::ToSummary
                } else {
                    let _ = terminal.draw(|frame| draw_running(frame, &rs.dashboard));
                    handle_running_input(&mut rs.dashboard)
                }
            }
            Screen::Summary(ss) => {
                let _ = terminal.draw(|frame| draw_summary(frame, ss));
                handle_summary_input(ss)
            }
        };

        screen = match transition {
            Transition::Stay => screen,
            Transition::ToRunning(mode, sz, dur, threads) => {
                RUNNING.store(true, Ordering::Relaxed);
                Screen::Running(start_running(mode, sz, dur, threads, &info))
            }
            Transition::ToSummary => {
                // Join the worker thread
                if let Screen::Running(mut rs) = screen {
                    RUNNING.store(false, Ordering::Relaxed);
                    // Drain remaining messages
                    while let Ok(msg) = rs.rx.try_recv() {
                        rs.dashboard.apply(msg);
                    }
                    if let Some(handle) = rs.worker.take() {
                        let _ = handle.join();
                    }
                    save_log(&rs.dashboard, rs.mode, rs.size_mb);
                    Screen::Summary(SummaryState {
                        mode: rs.mode,
                        size_mb: rs.size_mb,
                        dashboard: rs.dashboard,
                        selected_action: SummaryAction::BackToMenu,
                    })
                } else {
                    screen
                }
            }
            Transition::ToMenu => {
                RUNNING.store(true, Ordering::Relaxed);
                Screen::Menu(MenuState::new(&info, size_mb, duration, threads))
            }
            Transition::Exit => break,
        };
    }

    // Cleanup
    terminal::disable_raw_mode().expect("failed to disable raw mode");
    stdout().execute(LeaveAlternateScreen).expect("failed to leave alternate screen");
}

// ---------------------------------------------------------------------------
// Input handling
// ---------------------------------------------------------------------------

fn handle_menu_input(menu: &mut MenuState) -> Transition {
    if event::poll(Duration::from_millis(100)).unwrap_or(false) {
        if let Ok(Event::Key(key)) = event::read() {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return Transition::Exit,
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return Transition::Exit;
                }
                KeyCode::Tab | KeyCode::BackTab => {
                    menu.active_field = match menu.active_field {
                        MenuField::ModeList => MenuField::SizeMb,
                        MenuField::SizeMb => MenuField::Duration,
                        MenuField::Duration => MenuField::Threads,
                        MenuField::Threads => MenuField::ModeList,
                    };
                }
                KeyCode::Up => {
                    if menu.active_field == MenuField::ModeList && menu.selected_mode > 0 {
                        menu.selected_mode -= 1;
                    }
                }
                KeyCode::Down => {
                    if menu.active_field == MenuField::ModeList
                        && menu.selected_mode < RunMode::ALL_MODES.len() - 1
                    {
                        menu.selected_mode += 1;
                    }
                }
                KeyCode::Char(c) if c.is_ascii_digit() => {
                    match menu.active_field {
                        MenuField::SizeMb => menu.size_input.push(c),
                        MenuField::Duration => menu.duration_input.push(c),
                        MenuField::Threads => menu.threads_input.push(c),
                        _ => {}
                    }
                }
                KeyCode::Backspace => {
                    match menu.active_field {
                        MenuField::SizeMb => { menu.size_input.pop(); }
                        MenuField::Duration => { menu.duration_input.pop(); }
                        MenuField::Threads => { menu.threads_input.pop(); }
                        _ => {}
                    }
                }
                KeyCode::Enter => {
                    let mode = menu.selected_run_mode();
                    let sz = menu.parsed_size_mb();
                    let dur = menu.parsed_duration();
                    let threads = menu.parsed_threads();
                    return Transition::ToRunning(mode, sz, dur, threads);
                }
                _ => {}
            }
        }
    }
    Transition::Stay
}

fn handle_running_input(dashboard: &mut DashboardState) -> Transition {
    if event::poll(Duration::from_millis(100)).unwrap_or(false) {
        if let Ok(Event::Key(key)) = event::read() {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => {
                    RUNNING.store(false, Ordering::Relaxed);
                    return Transition::ToSummary;
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    RUNNING.store(false, Ordering::Relaxed);
                    return Transition::ToSummary;
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    dashboard.scroll_offset = dashboard.scroll_offset.saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    dashboard.scroll_offset = dashboard.scroll_offset.saturating_add(1);
                }
                KeyCode::PageUp => {
                    dashboard.scroll_offset = dashboard.scroll_offset.saturating_sub(10);
                }
                KeyCode::PageDown => {
                    dashboard.scroll_offset = dashboard.scroll_offset.saturating_add(10);
                }
                KeyCode::Home => {
                    dashboard.scroll_offset = 0;
                }
                _ => {}
            }
        }
    }
    Transition::Stay
}

fn handle_summary_input(ss: &mut SummaryState) -> Transition {
    if event::poll(Duration::from_millis(100)).unwrap_or(false) {
        if let Ok(Event::Key(key)) = event::read() {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return Transition::Exit,
                KeyCode::Char('b') | KeyCode::Enter => {
                    if ss.selected_action == SummaryAction::BackToMenu {
                        return Transition::ToMenu;
                    } else {
                        return Transition::Exit;
                    }
                }
                KeyCode::Tab | KeyCode::BackTab => {
                    ss.selected_action = match ss.selected_action {
                        SummaryAction::BackToMenu => SummaryAction::Quit,
                        SummaryAction::Quit => SummaryAction::BackToMenu,
                    };
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    ss.dashboard.scroll_offset = ss.dashboard.scroll_offset.saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    ss.dashboard.scroll_offset = ss.dashboard.scroll_offset.saturating_add(1);
                }
                KeyCode::PageUp => {
                    ss.dashboard.scroll_offset = ss.dashboard.scroll_offset.saturating_sub(10);
                }
                KeyCode::PageDown => {
                    ss.dashboard.scroll_offset = ss.dashboard.scroll_offset.saturating_add(10);
                }
                KeyCode::Home => {
                    ss.dashboard.scroll_offset = 0;
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return Transition::Exit;
                }
                _ => {}
            }
        }
    }
    Transition::Stay
}

// ---------------------------------------------------------------------------
// Draw: Menu
// ---------------------------------------------------------------------------

fn thermal_color(state: mac_sysinfo::ThermalState) -> Color {
    match state {
        mac_sysinfo::ThermalState::Nominal => Color::Green,
        mac_sysinfo::ThermalState::Fair => Color::Yellow,
        mac_sysinfo::ThermalState::Serious => Color::Red,
        mac_sysinfo::ThermalState::Critical => Color::LightRed,
        mac_sysinfo::ThermalState::Unknown => Color::DarkGray,
    }
}

fn battery_color(percent: u32) -> Color {
    match percent {
        0..=10 => Color::Red,
        11..=25 => Color::Yellow,
        _ => Color::Green,
    }
}

fn draw_system_bar(frame: &mut Frame, state: &DashboardState, area: Rect) {
    let d = &state.dynamic;

    let mut spans: Vec<Span> = vec![
        Span::styled(" Thermal: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            d.thermal_state.label(),
            Style::default().fg(thermal_color(d.thermal_state)),
        ),
        Span::styled(
            format!("  |  RAM: {:.1}/{:.0} GB", d.memory_used_gb, d.memory_total_gb),
            Style::default().fg(Color::DarkGray),
        ),
    ];

    if let Some(pct) = d.battery_percent {
        let status = if d.battery_charging == Some(true) {
            " [charging]"
        } else if d.battery_on_ac == Some(true) {
            " [AC]"
        } else {
            ""
        };
        spans.push(Span::styled("  |  Batt: ", Style::default().fg(Color::DarkGray)));
        spans.push(Span::styled(
            format!("{}%{}", pct, status),
            Style::default().fg(battery_color(pct)),
        ));
        if let Some(temp) = d.battery_temp_c {
            spans.push(Span::styled(
                format!(" {:.1}C", temp),
                Style::default().fg(Color::DarkGray),
            ));
        }
    }

    if state.gpu_cores > 0 {
        spans.push(Span::styled(
            format!("  |  GPU: {} cores", state.gpu_cores),
            Style::default().fg(Color::DarkGray),
        ));
    }

    let bar = Paragraph::new(Line::from(spans))
        .block(Block::bordered().border_style(Style::default().fg(Color::DarkGray)));
    frame.render_widget(bar, area);
}

fn draw_menu(frame: &mut Frame, menu: &MenuState) {
    let area = frame.area();

    let outer = Layout::vertical([
        Constraint::Length(3),  // title
        Constraint::Length(7),  // system info
        Constraint::Min(10),   // mode list
        Constraint::Length(3),  // size input
        Constraint::Length(3),  // duration input
        Constraint::Length(3),  // threads input
        Constraint::Length(3),  // instructions
    ])
    .split(area);

    // Title
    let title_block = Block::bordered()
        .title(" Mac Benchmark ")
        .border_style(Style::default().fg(Color::Cyan));
    frame.render_widget(title_block, outer[0]);

    // System info panel
    let cores_str = if menu.p_cores > 0 && menu.e_cores > 0 {
        format!("{} ({}P + {}E)", menu.total_cores, menu.p_cores, menu.e_cores)
    } else {
        format!("{}", menu.total_cores)
    };

    let gpu_str = if menu.gpu_cores > 0 {
        format!("{} cores ({})", menu.gpu_cores, menu.metal_version)
    } else {
        "N/A".to_string()
    };

    let mem_usage = format!(
        "{:.1}/{:.0} GB used",
        menu.dynamic.memory_used_gb, menu.dynamic.memory_total_gb
    );

    let mut line2_parts: Vec<Span> = vec![
        Span::styled("  Thermal: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            menu.dynamic.thermal_state.label(),
            Style::default().fg(thermal_color(menu.dynamic.thermal_state)),
        ),
    ];

    if let Some(pct) = menu.dynamic.battery_percent {
        let charging = if menu.dynamic.battery_charging == Some(true) {
            " [charging]"
        } else if menu.dynamic.battery_on_ac == Some(true) {
            " [AC]"
        } else {
            ""
        };
        line2_parts.push(Span::styled("  |  Battery: ", Style::default().fg(Color::DarkGray)));
        line2_parts.push(Span::styled(
            format!("{}%{}", pct, charging),
            Style::default().fg(battery_color(pct)),
        ));
        if let Some(temp) = menu.dynamic.battery_temp_c {
            line2_parts.push(Span::styled(
                format!("  {:.1}C", temp),
                Style::default().fg(Color::DarkGray),
            ));
        }
        if let Some(cycles) = menu.dynamic.battery_cycle_count {
            line2_parts.push(Span::styled(
                format!("  {} cycles", cycles),
                Style::default().fg(Color::DarkGray),
            ));
        }
    }

    let os_str = if !menu.os_version.is_empty() {
        format!("macOS {} ({})", menu.os_version, menu.os_build)
    } else {
        String::new()
    };

    let uptime_str = if menu.uptime_secs > 0 {
        format!("up {}", mac_sysinfo::format_uptime(menu.uptime_secs))
    } else {
        String::new()
    };

    let sys_lines = vec![
        Line::from(vec![
            Span::styled("  Chip: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&menu.chip_name, Style::default().fg(Color::White)),
            Span::styled("  |  CPU: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&cores_str, Style::default().fg(Color::White)),
            Span::styled("  |  GPU: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&gpu_str, Style::default().fg(Color::White)),
            Span::styled("  |  RAM: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&mem_usage, Style::default().fg(Color::White)),
        ]),
        Line::from(line2_parts),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(&os_str, Style::default().fg(Color::DarkGray)),
            Span::styled("  |  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{} ({})", menu.model_name, menu.chip_name),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled("  |  ", Style::default().fg(Color::DarkGray)),
            Span::styled(&uptime_str, Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let sys_block = Block::bordered()
        .title(" System ")
        .border_style(Style::default().fg(Color::DarkGray));
    let sys_para = Paragraph::new(sys_lines).block(sys_block);
    frame.render_widget(sys_para, outer[1]);

    // Mode list
    let mode_border_style = if menu.active_field == MenuField::ModeList {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let mode_block = Block::bordered()
        .title(" Select Mode (Up/Down, Enter to start) ")
        .border_style(mode_border_style);
    let inner_mode = mode_block.inner(outer[2]);
    frame.render_widget(mode_block, outer[2]);

    let items: Vec<ListItem> = RunMode::ALL_MODES
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let marker = if i == menu.selected_mode { "> " } else { "  " };
            let style = if i == menu.selected_mode {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!("{}{:<14}", marker, m.label()), style),
                Span::styled(m.description(), Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect();
    let list = List::new(items);
    frame.render_widget(list, inner_mode);

    // Size input
    let size_border_style = if menu.active_field == MenuField::SizeMb {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let cursor = if menu.active_field == MenuField::SizeMb { "_" } else { "" };
    let size_para = Paragraph::new(format!("{}{}", menu.size_input, cursor))
        .block(Block::bordered().title(" Size (MB) ").border_style(size_border_style));
    frame.render_widget(size_para, outer[3]);

    // Duration input
    let dur_border_style = if menu.active_field == MenuField::Duration {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let dur_cursor = if menu.active_field == MenuField::Duration { "_" } else { "" };
    let dur_display = if menu.duration_input.is_empty() && menu.active_field != MenuField::Duration {
        "unlimited".to_string()
    } else {
        format!("{}{}", menu.duration_input, dur_cursor)
    };
    let dur_para = Paragraph::new(dur_display)
        .block(Block::bordered().title(" Duration (minutes, empty=unlimited) ").border_style(dur_border_style));
    frame.render_widget(dur_para, outer[4]);

    // Threads input
    let threads_border_style = if menu.active_field == MenuField::Threads {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let threads_cursor = if menu.active_field == MenuField::Threads { "_" } else { "" };
    let threads_display = format!("{}{}", menu.threads_input, threads_cursor);
    let threads_title = format!(" Threads (MT benchmarks, max: {}) ", menu.total_cores);
    let threads_para = Paragraph::new(threads_display)
        .block(Block::bordered().title(threads_title).border_style(threads_border_style));
    frame.render_widget(threads_para, outer[5]);

    // Instructions
    let instructions = Paragraph::new(" Tab: switch field | Up/Down: select mode | Enter: start | q: quit ")
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::bordered().border_style(Style::default().fg(Color::DarkGray)));
    frame.render_widget(instructions, outer[6]);
}

// ---------------------------------------------------------------------------
// Draw: Running (mode-adaptive)
// ---------------------------------------------------------------------------

fn draw_running(frame: &mut Frame, state: &DashboardState) {
    match state.mode {
        RunMode::Test => draw_running_test(frame, state),
        RunMode::Bench => draw_running_bench(frame, state),
        RunMode::Cpu => draw_running_cpu(frame, state),
        RunMode::MtCpu => draw_running_mt_cpu(frame, state),
        RunMode::Gpu => draw_running_gpu(frame, state),
        RunMode::All => draw_running_all(frame, state),
        RunMode::Stress => draw_running_stress(frame, state),
        RunMode::FullStress => draw_running_full_stress(frame, state),
    }
}

fn running_title_bar(frame: &mut Frame, state: &DashboardState, area: Rect) {
    let elapsed = format_duration(state.start.elapsed());
    let title = if state.chip_name.is_empty() || state.chip_name == "Unknown" {
        format!(
            " {} | {} MB | Elapsed: {} | j/k: scroll | q: stop ",
            state.mode.label(), state.size_mb, elapsed
        )
    } else {
        format!(
            " {} | {} | {} MB | Elapsed: {} | j/k: scroll | q: stop ",
            state.mode.label(), state.chip_name, state.size_mb, elapsed
        )
    };
    let title_block = Block::bordered()
        .title(title)
        .border_style(Style::default().fg(Color::Cyan));
    frame.render_widget(title_block, area);
}

// Test: title + stress panel + pass history (same layout as Stress)
fn draw_running_test(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();
    let outer = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(8),
        Constraint::Min(6),
    ]).split(area);

    running_title_bar(frame, state, outer[0]);
    draw_system_bar(frame, state, outer[1]);
    draw_stress_panel(frame, state, outer[2]);
    draw_pass_history(frame, state, outer[3]);
}

// Bench: title + bench panel + bandwidth/latency sparklines
fn draw_running_bench(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();
    let outer = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(5),
        Constraint::Length(8),
    ]).split(area);

    running_title_bar(frame, state, outer[0]);
    draw_system_bar(frame, state, outer[1]);
    draw_bench_panel(frame, state, outer[2]);

    let sparkline_panels = Layout::horizontal([
        Constraint::Percentage(75),
        Constraint::Percentage(25),
    ]).split(outer[3]);

    draw_bandwidth_sparklines(frame, state, sparkline_panels[0]);
    draw_latency_sparkline(frame, state, sparkline_panels[1]);
}

// CPU: title + CPU panel + CPU throughput sparklines
fn draw_running_cpu(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();
    let outer = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(5),
        Constraint::Length(14),
    ]).split(area);

    running_title_bar(frame, state, outer[0]);
    draw_system_bar(frame, state, outer[1]);
    draw_cpu_panel(frame, state, outer[2]);
    draw_cpu_sparklines(frame, state, outer[3]);
}

// MT CPU: title + MT panel + MT sparklines
fn draw_running_mt_cpu(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();
    let outer = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(5),
        Constraint::Length(6),
    ]).split(area);

    running_title_bar(frame, state, outer[0]);
    draw_system_bar(frame, state, outer[1]);
    draw_mt_cpu_panel(frame, state, outer[2]);
    draw_mt_cpu_sparklines(frame, state, outer[3]);
}

// GPU: title + GPU panel + GPU sparklines
fn draw_running_gpu(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();
    let outer = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(5),
        Constraint::Length(8),
    ]).split(area);

    running_title_bar(frame, state, outer[0]);
    draw_system_bar(frame, state, outer[1]);
    draw_gpu_panel(frame, state, outer[2]);
    draw_gpu_sparklines(frame, state, outer[3]);
}

// All: title + panels (stress/bench/cpu/gpu) + bandwidth sparklines + pass history
fn draw_running_all(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();

    let outer = Layout::vertical([
        Constraint::Length(3),  // title
        Constraint::Length(3),  // system bar
        Constraint::Min(12),   // panels
        Constraint::Length(7), // sparklines
        Constraint::Min(6),    // pass history
    ]).split(area);

    running_title_bar(frame, state, outer[0]);
    draw_system_bar(frame, state, outer[1]);

    let top_panels = Layout::horizontal([
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
    ]).split(outer[2]);

    draw_bench_panel(frame, state, top_panels[0]);
    draw_cpu_panel(frame, state, top_panels[1]);
    draw_gpu_panel(frame, state, top_panels[2]);
    draw_stress_panel(frame, state, top_panels[3]);

    let sparkline_panels = Layout::horizontal([
        Constraint::Percentage(75),
        Constraint::Percentage(25),
    ]).split(outer[3]);

    draw_bandwidth_sparklines(frame, state, sparkline_panels[0]);
    draw_latency_sparkline(frame, state, sparkline_panels[1]);

    draw_pass_history(frame, state, outer[4]);
}

fn draw_running_stress(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();
    let outer = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(8),
        Constraint::Min(6),
    ]).split(area);

    running_title_bar(frame, state, outer[0]);
    draw_system_bar(frame, state, outer[1]);
    draw_stress_panel(frame, state, outer[2]);
    draw_pass_history(frame, state, outer[3]);
}

fn draw_running_full_stress(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();

    let outer = Layout::vertical([
        Constraint::Length(3),  // title
        Constraint::Length(3),  // system bar
        Constraint::Min(12),   // panels
        Constraint::Length(7), // sparklines
        Constraint::Min(6),    // pass history
    ]).split(area);

    running_title_bar(frame, state, outer[0]);
    draw_system_bar(frame, state, outer[1]);

    let top_panels = Layout::horizontal([
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
    ]).split(outer[2]);

    draw_bench_panel(frame, state, top_panels[0]);
    draw_cpu_panel(frame, state, top_panels[1]);
    draw_gpu_panel(frame, state, top_panels[2]);
    draw_stress_panel(frame, state, top_panels[3]);

    let sparkline_panels = Layout::horizontal([
        Constraint::Percentage(75),
        Constraint::Percentage(25),
    ]).split(outer[3]);

    draw_bandwidth_sparklines(frame, state, sparkline_panels[0]);
    draw_latency_sparkline(frame, state, sparkline_panels[1]);

    draw_pass_history(frame, state, outer[4]);
}

// ---------------------------------------------------------------------------
// Log saving
// ---------------------------------------------------------------------------

fn local_timestamp() -> (String, String) {
    unsafe {
        let mut t: libc::time_t = 0;
        libc::time(&mut t);
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        let file_ts = format!(
            "{:04}{:02}{:02}_{:02}{:02}{:02}",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec,
        );
        let display_ts = format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec,
        );
        (file_ts, display_ts)
    }
}

fn save_log(dashboard: &DashboardState, mode: RunMode, size_mb: usize) {
    let (file_ts, display_ts) = local_timestamp();
    let mode_snake = mode.label().to_lowercase().replace(' ', "_");
    let dir = "logs";
    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("Failed to create logs directory: {e}");
        return;
    }
    let filename = format!("{dir}/{mode_snake}_{size_mb}MB_{file_ts}.log");
    let file = match fs::File::create(&filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create log file {filename}: {e}");
            return;
        }
    };
    let mut w = std::io::BufWriter::new(file);
    if let Err(e) = write_log_content(&mut w, dashboard, mode, size_mb, &display_ts) {
        eprintln!("Failed to write log file {filename}: {e}");
    }
}

fn write_log_content(
    w: &mut impl Write,
    ds: &DashboardState,
    mode: RunMode,
    size_mb: usize,
    display_ts: &str,
) -> std::io::Result<()> {
    writeln!(w, "===============================================")?;
    writeln!(w, "Mac Benchmark Results")?;
    writeln!(w, "===============================================")?;
    writeln!(w, "Date:       {display_ts}")?;
    writeln!(w, "Mode:       {}", mode.label())?;
    writeln!(w, "Chip:       {}", ds.chip_name)?;
    if ds.gpu_cores > 0 {
        writeln!(w, "GPU:        {} cores ({})", ds.gpu_cores, ds.metal_version)?;
    }
    writeln!(w, "Size:       {size_mb} MB")?;
    writeln!(w, "Threads:    {}", ds.num_threads)?;
    if !ds.os_version.is_empty() {
        writeln!(w, "macOS:      {}", ds.os_version)?;
    }
    writeln!(w, "Elapsed:    {}", format_duration(ds.start.elapsed()))?;
    writeln!(w, "Thermal:    {}", ds.dynamic.thermal_state.label())?;
    writeln!(w, "RAM usage:  {:.1}/{:.0} GB", ds.dynamic.memory_used_gb, ds.dynamic.memory_total_gb)?;
    if let Some(pct) = ds.dynamic.battery_percent {
        let status = if ds.dynamic.battery_charging == Some(true) { " (charging)" }
            else if ds.dynamic.battery_on_ac == Some(true) { " (AC)" }
            else { " (battery)" };
        writeln!(w, "Battery:    {}%{}", pct, status)?;
    }

    // Memory benchmarks
    if ds.seq_write_gbps.is_some() {
        writeln!(w)?;
        writeln!(w, "--- Memory Benchmarks ---")?;
        if let Some(v) = ds.seq_write_gbps { writeln!(w, "Seq Write:       {v:.2} GB/s")?; }
        if let Some(v) = ds.seq_read_gbps  { writeln!(w, "Seq Read:        {v:.2} GB/s")?; }
        if let Some(v) = ds.copy_gbps      { writeln!(w, "Copy:            {v:.2} GB/s")?; }
        if let Some(v) = ds.latency_ns     { writeln!(w, "Latency:         {v:.1} ns")?; }
        for (label, gbps) in &ds.strides {
            writeln!(w, "{label:>6} stride:  {gbps:.2} GB/s")?;
        }
        if let Some(v) = ds.mt_seq_read_gbps {
            writeln!(w, "MT Read ({:>2}c):   {v:.2} GB/s", ds.num_threads)?;
        }
        if let Some(v) = ds.loaded_latency_ns {
            writeln!(w, "Loaded Lat:      {v:.1} ns")?;
        }
    }

    // CPU benchmarks
    if ds.int_dependent.is_some() {
        writeln!(w)?;
        writeln!(w, "--- CPU Benchmarks ---")?;
        if let Some(v) = ds.int_dependent    { writeln!(w, "Int dep:         {v:.2} Gops/s")?; }
        if let Some(v) = ds.int_independent  { writeln!(w, "Int indep:       {v:.2} Gops/s")?; }
        if let Some(v) = ds.fp_dependent     { writeln!(w, "FP64 dep:        {v:.2} Gflops/s")?; }
        if let Some(v) = ds.fp_independent   { writeln!(w, "FP64 indep:      {v:.2} Gflops/s")?; }
        if let Some(v) = ds.fp32_dependent   { writeln!(w, "FP32 dep:        {v:.2} Gflops/s")?; }
        if let Some(v) = ds.fp32_independent { writeln!(w, "FP32 indep:      {v:.2} Gflops/s")?; }
        if let Some(v) = ds.neon_fp32        { writeln!(w, "NEON FP32:       {v:.2} Gflops/s")?; }
        if let Some(v) = ds.neon_int         { writeln!(w, "NEON Int:        {v:.2} Gops/s")?; }
        if let Some(v) = ds.pcore_int        { writeln!(w, "P-core Int:      {v:.2} Gops/s")?; }
        if let Some(v) = ds.pcore_fp         { writeln!(w, "P-core FP:       {v:.2} Gflops/s")?; }
        if let Some(v) = ds.ecore_int        { writeln!(w, "E-core Int:      {v:.2} Gops/s")?; }
        if let Some(v) = ds.ecore_fp         { writeln!(w, "E-core FP:       {v:.2} Gflops/s")?; }
        if let Some(v) = ds.mt_int_gops {
            writeln!(w, "MT Int ({:>2}c):    {v:.2} Gops/s", ds.num_threads)?;
        }
        if let Some(v) = ds.mt_fp_gflops {
            writeln!(w, "MT FP  ({:>2}c):    {v:.2} Gflops/s", ds.num_threads)?;
        }
        if !ds.cache_latencies.is_empty() {
            writeln!(w, "Latency ladder:")?;
            for (label, ns) in &ds.cache_latencies {
                writeln!(w, "  {label:>6}       {ns:.1} ns")?;
            }
        }
        if !ds.cache_bandwidths.is_empty() {
            writeln!(w, "BW ladder:")?;
            for (label, gbps) in &ds.cache_bandwidths {
                writeln!(w, "  {label:>6}       {gbps:.1} GB/s")?;
            }
        }
    }

    // GPU benchmarks
    if ds.gpu_fp32_tflops.is_some() {
        writeln!(w)?;
        writeln!(w, "--- GPU Benchmarks (Metal) ---")?;
        if let Some(v) = ds.gpu_fp32_tflops   { writeln!(w, "FP32:            {v:.3} Tflops/s")?; }
        if let Some(v) = ds.gpu_fp16_tflops   { writeln!(w, "FP16:            {v:.3} Tflops/s")?; }
        if let Some(v) = ds.gpu_int32_tops    { writeln!(w, "Int32:           {v:.3} Tops/s")?; }
        if let Some(v) = ds.gpu_buf_read_gbps { writeln!(w, "Buf Read:        {v:.2} GB/s")?; }
        if let Some(v) = ds.gpu_buf_write_gbps{ writeln!(w, "Buf Write:       {v:.2} GB/s")?; }
        if let Some(v) = ds.gpu_buf_alloc_us  { writeln!(w, "Buf Alloc:       {v:.1} us")?; }
        if let Some(v) = ds.gpu_matmul_tflops { writeln!(w, "MatMul:          {v:.3} Tflops/s")?; }
    }

    // Stress test
    if ds.stress_pass > 0 {
        writeln!(w)?;
        writeln!(w, "--- Stress Test ---")?;
        writeln!(w, "Passes:   {}", ds.stress_pass)?;
        writeln!(w, "Passed:   {}", ds.stress_total_passed)?;
        writeln!(w, "Failed:   {}", ds.stress_total_failed)?;
    }

    // Statistics table
    if let Some(ref stats) = ds.metric_stats {
        if stats.seq_write.count > 0 {
            writeln!(w)?;
            writeln!(w, "--- Statistics (min/max/avg) ---")?;
            writeln!(w, "{:<40} {:>10} {:>10} {:>10}", "METRIC", "MIN", "MAX", "AVG")?;
            writeln!(w, "{}", "-".repeat(74))?;

            fn stat_line(w: &mut impl Write, name: &str, unit: &str, s: &MetricStats) -> std::io::Result<()> {
                if s.count == 0 { return Ok(()); }
                if unit == "ns" {
                    writeln!(w, "{:<40} {:>9.1} {:>9.1} {:>9.1} {unit}", name, s.min, s.max, s.avg())
                } else {
                    writeln!(w, "{:<40} {:>9.2} {:>9.2} {:>9.2} {unit}", name, s.min, s.max, s.avg())
                }
            }

            stat_line(w, "Sequential write", "GB/s", &stats.seq_write)?;
            stat_line(w, "Sequential read", "GB/s", &stats.seq_read)?;
            stat_line(w, "Copy (read+write)", "GB/s", &stats.copy)?;
            for (label, s) in &stats.strides {
                stat_line(w, &format!("{label} stride"), "GB/s", s)?;
            }
            stat_line(w, "Integer (dependent)", "Gops/s", &stats.int_dep)?;
            stat_line(w, "Integer (4 independent)", "Gops/s", &stats.int_indep)?;
            stat_line(w, "FP64 (dependent FMA)", "Gflops/s", &stats.fp_dep)?;
            stat_line(w, "FP64 (4 independent FMA)", "Gflops/s", &stats.fp_indep)?;
            stat_line(w, "FP32 (dependent FMA)", "Gflops/s", &stats.fp32_dep)?;
            stat_line(w, "FP32 (4 independent FMA)", "Gflops/s", &stats.fp32_indep)?;
            stat_line(w, "NEON FP32 FMA", "Gflops/s", &stats.neon_fp32)?;
            stat_line(w, "NEON Int mul", "Gops/s", &stats.neon_int)?;
            stat_line(w, &format!("Integer multi-thread ({}c)", ds.num_threads), "Gops/s", &stats.mt_int)?;
            stat_line(w, &format!("FP64 multi-thread ({}c)", ds.num_threads), "Gflops/s", &stats.mt_fp)?;
            stat_line(w, "P-core Integer", "Gops/s", &stats.pcore_int)?;
            stat_line(w, "P-core FP64", "Gflops/s", &stats.pcore_fp)?;
            stat_line(w, "E-core Integer", "Gops/s", &stats.ecore_int)?;
            stat_line(w, "E-core FP64", "Gflops/s", &stats.ecore_fp)?;
            stat_line(w, "Random read latency", "ns", &stats.random_latency)?;
            stat_line(w, &format!("MT sequential read ({}c)", ds.num_threads), "GB/s", &stats.mt_seq_read)?;
            stat_line(w, "Loaded latency", "ns", &stats.loaded_latency)?;
            for (label, s) in &stats.cache_latencies {
                stat_line(w, &format!("Latency {label}"), "ns", s)?;
            }
            for (label, s) in &stats.cache_bandwidths {
                stat_line(w, &format!("Bandwidth {label}"), "GB/s", s)?;
            }
            stat_line(w, "GPU FP32 throughput", "Tflops/s", &stats.gpu_fp32)?;
            stat_line(w, "GPU FP16 throughput", "Tflops/s", &stats.gpu_fp16)?;
            stat_line(w, "GPU Int32 throughput", "Tops/s", &stats.gpu_int32)?;
            stat_line(w, "GPU buffer read", "GB/s", &stats.gpu_buf_read)?;
            stat_line(w, "GPU buffer write", "GB/s", &stats.gpu_buf_write)?;
            stat_line(w, "GPU buffer alloc", "us", &stats.gpu_buf_alloc)?;
            stat_line(w, "GPU matmul", "Tflops/s", &stats.gpu_matmul)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Draw: Summary
// ---------------------------------------------------------------------------

fn draw_summary(frame: &mut Frame, ss: &SummaryState) {
    let area = frame.area();

    let outer = Layout::vertical([
        Constraint::Length(3),  // title
        Constraint::Min(5),    // results body
        Constraint::Length(3), // action buttons
    ]).split(area);

    // Title bar - green for "complete"
    let title = format!(" Run Complete - {} | {} MB ", ss.mode.label(), ss.size_mb);
    let title_block = Block::bordered()
        .title(title)
        .border_style(Style::default().fg(Color::Green));
    frame.render_widget(title_block, outer[0]);

    // Results body
    let border_style = Style::default().fg(Color::Yellow);

    let mut lines: Vec<Line> = Vec::new();
    let val_style = Style::default().fg(Color::Cyan);
    let hdr_style = Style::default().fg(Color::White).add_modifier(Modifier::BOLD);

    let ds = &ss.dashboard;

    // Stress results
    if ds.stress_pass > 0 {
        lines.push(Line::from(Span::styled("  Stress Test:", hdr_style)));
        lines.push(Line::from(vec![
            Span::raw("    Passes:  "),
            Span::styled(format!("{}", ds.stress_pass), val_style),
        ]));
        lines.push(Line::from(vec![
            Span::raw("    Passed:  "),
            Span::styled(format!("{}", ds.stress_total_passed), Style::default().fg(Color::Green)),
            Span::raw("  Failed: "),
            Span::styled(format!("{}", ds.stress_total_failed), if ds.stress_total_failed > 0 {
                Style::default().fg(Color::Red)
            } else {
                Style::default().fg(Color::Green)
            }),
        ]));
        lines.push(Line::raw(""));
    }

    // Bench results
    if ds.seq_write_gbps.is_some() {
        lines.push(Line::from(Span::styled("  Memory Benchmarks:", hdr_style)));
        if let Some(v) = ds.seq_write_gbps {
            lines.push(Line::from(vec![
                Span::raw("    Seq Write:  "),
                Span::styled(format!("{:.2} GB/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.seq_read_gbps {
            lines.push(Line::from(vec![
                Span::raw("    Seq Read:   "),
                Span::styled(format!("{:.2} GB/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.copy_gbps {
            lines.push(Line::from(vec![
                Span::raw("    Copy:       "),
                Span::styled(format!("{:.2} GB/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.latency_ns {
            lines.push(Line::from(vec![
                Span::raw("    Latency:    "),
                Span::styled(format!("{:.1} ns", v), val_style),
            ]));
        }
        for (label, gbps) in &ds.strides {
            lines.push(Line::from(vec![
                Span::raw(format!("    {:>6} stride: ", label)),
                Span::styled(format!("{:.2} GB/s", gbps), val_style),
            ]));
        }
        if let Some(v) = ds.mt_seq_read_gbps {
            lines.push(Line::from(vec![
                Span::raw(format!("    MT Read ({:>2}c):", ds.num_threads)),
                Span::styled(format!("{:.2} GB/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.loaded_latency_ns {
            lines.push(Line::from(vec![
                Span::raw("    Loaded Lat: "),
                Span::styled(format!("{:.1} ns", v), val_style),
            ]));
        }
        lines.push(Line::raw(""));
    }

    // CPU results
    if ds.int_dependent.is_some() {
        lines.push(Line::from(Span::styled("  CPU Benchmarks:", hdr_style)));
        if let Some(v) = ds.int_dependent {
            lines.push(Line::from(vec![
                Span::raw("    Int dep:    "),
                Span::styled(format!("{:.2} Gops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.int_independent {
            lines.push(Line::from(vec![
                Span::raw("    Int indep:  "),
                Span::styled(format!("{:.2} Gops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.fp_dependent {
            lines.push(Line::from(vec![
                Span::raw("    FP64 dep:   "),
                Span::styled(format!("{:.2} Gflops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.fp_independent {
            lines.push(Line::from(vec![
                Span::raw("    FP64 indep: "),
                Span::styled(format!("{:.2} Gflops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.fp32_dependent {
            lines.push(Line::from(vec![
                Span::raw("    FP32 dep:   "),
                Span::styled(format!("{:.2} Gflops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.fp32_independent {
            lines.push(Line::from(vec![
                Span::raw("    FP32 indep: "),
                Span::styled(format!("{:.2} Gflops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.neon_fp32 {
            lines.push(Line::from(vec![
                Span::raw("    NEON FP32:  "),
                Span::styled(format!("{:.2} Gflops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.neon_int {
            lines.push(Line::from(vec![
                Span::raw("    NEON Int:   "),
                Span::styled(format!("{:.2} Gops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.pcore_int {
            lines.push(Line::from(vec![
                Span::raw("    P-core Int: "),
                Span::styled(format!("{:.2} Gops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.pcore_fp {
            lines.push(Line::from(vec![
                Span::raw("    P-core FP:  "),
                Span::styled(format!("{:.2} Gflops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.ecore_int {
            lines.push(Line::from(vec![
                Span::raw("    E-core Int: "),
                Span::styled(format!("{:.2} Gops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.ecore_fp {
            lines.push(Line::from(vec![
                Span::raw("    E-core FP:  "),
                Span::styled(format!("{:.2} Gflops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.mt_int_gops {
            lines.push(Line::from(vec![
                Span::raw(format!("    MT Int ({:>2}c):", ds.num_threads)),
                Span::styled(format!("{:.2} Gops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.mt_fp_gflops {
            lines.push(Line::from(vec![
                Span::raw(format!("    MT FP  ({:>2}c):", ds.num_threads)),
                Span::styled(format!("{:.2} Gflops/s", v), val_style),
            ]));
        }
        if !ds.cache_latencies.is_empty() {
            lines.push(Line::raw("    Latency ladder:"));
            for (label, ns) in &ds.cache_latencies {
                lines.push(Line::from(vec![
                    Span::raw(format!("      {:>6}  ", label)),
                    Span::styled(format!("{:.1} ns", ns), val_style),
                ]));
            }
        }
        if !ds.cache_bandwidths.is_empty() {
            lines.push(Line::raw("    BW ladder:"));
            for (label, gbps) in &ds.cache_bandwidths {
                lines.push(Line::from(vec![
                    Span::raw(format!("      {:>6}  ", label)),
                    Span::styled(format!("{:.1} GB/s", gbps), val_style),
                ]));
            }
        }
        lines.push(Line::raw(""));
    }

    // GPU results
    if ds.gpu_fp32_tflops.is_some() {
        lines.push(Line::from(Span::styled("  GPU Benchmarks (Metal):", hdr_style)));
        if let Some(v) = ds.gpu_fp32_tflops {
            lines.push(Line::from(vec![
                Span::raw("    FP32:       "),
                Span::styled(format!("{:.3} Tflops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.gpu_fp16_tflops {
            lines.push(Line::from(vec![
                Span::raw("    FP16:       "),
                Span::styled(format!("{:.3} Tflops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.gpu_int32_tops {
            lines.push(Line::from(vec![
                Span::raw("    Int32:      "),
                Span::styled(format!("{:.3} Tops/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.gpu_buf_read_gbps {
            lines.push(Line::from(vec![
                Span::raw("    Buf Read:   "),
                Span::styled(format!("{:.2} GB/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.gpu_buf_write_gbps {
            lines.push(Line::from(vec![
                Span::raw("    Buf Write:  "),
                Span::styled(format!("{:.2} GB/s", v), val_style),
            ]));
        }
        if let Some(v) = ds.gpu_buf_alloc_us {
            lines.push(Line::from(vec![
                Span::raw("    Buf Alloc:  "),
                Span::styled(format!("{:.1} us", v), val_style),
            ]));
        }
        if let Some(v) = ds.gpu_matmul_tflops {
            lines.push(Line::from(vec![
                Span::raw("    MatMul:     "),
                Span::styled(format!("{:.3} Tflops/s", v), val_style),
            ]));
        }
        lines.push(Line::raw(""));
    }

    // Min/max/avg table (now available for all modes)
    if let Some(ref stats) = ds.metric_stats {
        if stats.seq_write.count > 0 {
            lines.push(Line::from(Span::styled(
                format!("  {:<40} {:>10} {:>10} {:>10}", "METRIC", "MIN", "MAX", "AVG"),
                hdr_style,
            )));
            lines.push(Line::styled(format!("  {}", "-".repeat(72)), Style::default().fg(Color::DarkGray)));

            fn stat_line<'a>(name: &str, unit: &str, s: &MetricStats) -> Line<'a> {
                if s.count == 0 { return Line::raw(""); }
                if unit == "ns" {
                    Line::styled(
                        format!("  {:<40} {:>9.1} {:>9.1} {:>9.1} {}", name, s.min, s.max, s.avg(), unit),
                        Style::default().fg(Color::Cyan),
                    )
                } else {
                    Line::styled(
                        format!("  {:<40} {:>9.2} {:>9.2} {:>9.2} {}", name, s.min, s.max, s.avg(), unit),
                        Style::default().fg(Color::Cyan),
                    )
                }
            }

            lines.push(stat_line("Sequential write", "GB/s", &stats.seq_write));
            lines.push(stat_line("Sequential read", "GB/s", &stats.seq_read));
            lines.push(stat_line("Copy (read+write)", "GB/s", &stats.copy));
            for (label, s) in &stats.strides {
                lines.push(stat_line(&format!("{} stride", label), "GB/s", s));
            }
            lines.push(Line::styled(format!("  {}", "-".repeat(72)), Style::default().fg(Color::DarkGray)));
            lines.push(stat_line("Integer (dependent)", "Gops/s", &stats.int_dep));
            lines.push(stat_line("Integer (4 independent)", "Gops/s", &stats.int_indep));
            lines.push(stat_line("FP64 (dependent FMA)", "Gflops/s", &stats.fp_dep));
            lines.push(stat_line("FP64 (4 independent FMA)", "Gflops/s", &stats.fp_indep));
            lines.push(stat_line("FP32 (dependent FMA)", "Gflops/s", &stats.fp32_dep));
            lines.push(stat_line("FP32 (4 independent FMA)", "Gflops/s", &stats.fp32_indep));
            lines.push(stat_line("NEON FP32 FMA", "Gflops/s", &stats.neon_fp32));
            lines.push(stat_line("NEON Int mul", "Gops/s", &stats.neon_int));
            lines.push(stat_line(
                &format!("Integer multi-thread ({}c)", ds.num_threads), "Gops/s", &stats.mt_int,
            ));
            lines.push(stat_line(
                &format!("FP64 multi-thread ({}c)", ds.num_threads), "Gflops/s", &stats.mt_fp,
            ));
            lines.push(stat_line("P-core Integer", "Gops/s", &stats.pcore_int));
            lines.push(stat_line("P-core FP64", "Gflops/s", &stats.pcore_fp));
            lines.push(stat_line("E-core Integer", "Gops/s", &stats.ecore_int));
            lines.push(stat_line("E-core FP64", "Gflops/s", &stats.ecore_fp));
            lines.push(Line::styled(format!("  {}", "-".repeat(72)), Style::default().fg(Color::DarkGray)));
            lines.push(stat_line("Random read latency", "ns", &stats.random_latency));
            lines.push(stat_line(
                &format!("MT sequential read ({}c)", ds.num_threads), "GB/s", &stats.mt_seq_read,
            ));
            lines.push(stat_line("Loaded latency", "ns", &stats.loaded_latency));
            lines.push(Line::styled(format!("  {}", "-".repeat(72)), Style::default().fg(Color::DarkGray)));
            for (label, s) in &stats.cache_latencies {
                lines.push(stat_line(&format!("Latency {}", label), "ns", s));
            }
            lines.push(Line::styled(format!("  {}", "-".repeat(72)), Style::default().fg(Color::DarkGray)));
            for (label, s) in &stats.cache_bandwidths {
                lines.push(stat_line(&format!("Bandwidth {}", label), "GB/s", s));
            }
            if stats.gpu_fp32.count > 0 {
                lines.push(Line::styled(format!("  {}", "-".repeat(72)), Style::default().fg(Color::DarkGray)));
                lines.push(stat_line("GPU FP32 throughput", "Tflops/s", &stats.gpu_fp32));
                lines.push(stat_line("GPU FP16 throughput", "Tflops/s", &stats.gpu_fp16));
                lines.push(stat_line("GPU Int32 throughput", "Tops/s", &stats.gpu_int32));
                lines.push(stat_line("GPU buffer read", "GB/s", &stats.gpu_buf_read));
                lines.push(stat_line("GPU buffer write", "GB/s", &stats.gpu_buf_write));
                lines.push(stat_line("GPU buffer alloc", "us", &stats.gpu_buf_alloc));
                lines.push(stat_line("GPU matmul", "Tflops/s", &stats.gpu_matmul));
            }
        }
    }

    let elapsed = format_duration(ds.start.elapsed());
    lines.push(Line::raw(""));
    lines.push(Line::from(vec![
        Span::raw("  Total time: "),
        Span::styled(elapsed, val_style),
    ]));

    let content_height = lines.len() as u16;
    let inner_height = outer[1].height.saturating_sub(2);
    let max_scroll = content_height.saturating_sub(inner_height);
    let scroll = ss.dashboard.scroll_offset.min(max_scroll);
    let overflows = content_height > inner_height;

    let results_title = if overflows {
        format!(" RESULTS [{}/{}] (j/k scroll) ", scroll + 1, max_scroll + 1)
    } else {
        " RESULTS ".to_string()
    };
    let block = Block::bordered()
        .title(results_title)
        .border_style(border_style);

    let para = Paragraph::new(lines).block(block).scroll((scroll, 0));
    frame.render_widget(para, outer[1]);

    // Action buttons
    let back_style = if ss.selected_action == SummaryAction::BackToMenu {
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let quit_style = if ss.selected_action == SummaryAction::Quit {
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let actions = Line::from(vec![
        Span::raw("  "),
        Span::styled(" [B]ack to Menu ", back_style),
        Span::raw("    "),
        Span::styled(" [Q]uit ", quit_style),
        Span::raw("    Tab: switch  |  Enter: select"),
    ]);
    let action_block = Block::bordered()
        .border_style(Style::default().fg(Color::DarkGray));
    let action_para = Paragraph::new(actions).block(action_block);
    frame.render_widget(action_para, outer[2]);
}

// ---------------------------------------------------------------------------
// Reusable panel drawing
// ---------------------------------------------------------------------------

fn draw_bench_panel(frame: &mut Frame, state: &DashboardState, area: Rect) {
    let border_style = Style::default().fg(Color::Yellow);
    let scroll = state.scroll_offset;

    let mut lines = Vec::new();
    let val_style = Style::default().fg(Color::Cyan);

    if let Some(v) = state.seq_write_gbps {
        lines.push(Line::from(vec![
            Span::raw("  Seq Write:  "),
            Span::styled(format!("{:>8.2} GB/s", v), val_style),
        ]));
    }
    if let Some(v) = state.seq_read_gbps {
        lines.push(Line::from(vec![
            Span::raw("  Seq Read:   "),
            Span::styled(format!("{:>8.2} GB/s", v), val_style),
        ]));
    }
    if let Some(v) = state.copy_gbps {
        lines.push(Line::from(vec![
            Span::raw("  Copy:       "),
            Span::styled(format!("{:>8.2} GB/s", v), val_style),
        ]));
    }
    if let Some(v) = state.latency_ns {
        lines.push(Line::from(vec![
            Span::raw("  Latency:    "),
            Span::styled(format!("{:>8.1} ns", v), val_style),
        ]));
    }

    if !state.strides.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::raw("  Stride throughput:"));
        for (label, gbps) in &state.strides {
            lines.push(Line::from(vec![
                Span::raw(format!("    {:>6}  ", label)),
                Span::styled(format!("{:>8.2} GB/s", gbps), val_style),
            ]));
        }
    }

    if let Some(v) = state.mt_seq_read_gbps {
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::raw(format!("  MT Read ({:>2}c):", state.num_threads)),
            Span::styled(format!("{:>8.2} GB/s", v), val_style),
        ]));
    }
    if let Some(v) = state.loaded_latency_ns {
        let loaded_line = if let Some(unloaded) = state.latency_ns {
            Line::from(vec![
                Span::raw("  Loaded Lat:  "),
                Span::styled(format!("{:>7.1} ns", v), val_style),
                Span::styled(format!("  (unloaded: {:.1} ns)", unloaded), Style::default().fg(Color::DarkGray)),
            ])
        } else {
            Line::from(vec![
                Span::raw("  Loaded Lat:  "),
                Span::styled(format!("{:>7.1} ns", v), val_style),
            ])
        };
        lines.push(loaded_line);
    }

    if lines.is_empty() {
        lines.push(Line::styled("  Waiting for results...", Style::default().fg(Color::DarkGray)));
    }

    let content_height = lines.len() as u16;
    let inner_height = area.height.saturating_sub(2);
    let max_scroll = content_height.saturating_sub(inner_height);
    let effective_scroll = scroll.min(max_scroll);
    let overflows = content_height > inner_height;

    let title = if overflows {
        format!(" MEMORY BENCH [{}/{}] ", effective_scroll + 1, max_scroll + 1)
    } else {
        " MEMORY BENCH ".to_string()
    };
    let block = Block::bordered()
        .title(title)
        .border_style(border_style);

    let paragraph = Paragraph::new(lines).block(block).scroll((effective_scroll, 0));
    frame.render_widget(paragraph, area);
}

fn draw_cpu_panel(frame: &mut Frame, state: &DashboardState, area: Rect) {
    draw_cpu_panel_inner(frame, state, area, state.scroll_offset);
}

fn draw_cpu_panel_inner(frame: &mut Frame, state: &DashboardState, area: Rect, scroll: u16) {
    let border_style = Style::default().fg(Color::Yellow);

    let mut lines = Vec::new();
    let val_style = Style::default().fg(Color::Cyan);
    let hdr_style = Style::default().fg(Color::White).add_modifier(Modifier::BOLD);

    if let Some(v) = state.int_dependent {
        lines.push(Line::from(vec![
            Span::raw("  Int dep:    "),
            Span::styled(format!("{:>8.2} Gops/s", v), val_style),
        ]));
    }
    if let Some(v) = state.int_independent {
        lines.push(Line::from(vec![
            Span::raw("  Int indep:  "),
            Span::styled(format!("{:>8.2} Gops/s", v), val_style),
        ]));
    }
    if let Some(v) = state.fp_dependent {
        lines.push(Line::from(vec![
            Span::raw("  FP64 dep:   "),
            Span::styled(format!("{:>8.2} Gflops/s", v), val_style),
        ]));
    }
    if let Some(v) = state.fp_independent {
        lines.push(Line::from(vec![
            Span::raw("  FP64 indep: "),
            Span::styled(format!("{:>8.2} Gflops/s", v), val_style),
        ]));
    }

    // FP32 results
    if let Some(v) = state.fp32_dependent {
        lines.push(Line::from(vec![
            Span::raw("  FP32 dep:   "),
            Span::styled(format!("{:>8.2} Gflops/s", v), val_style),
        ]));
    }
    if let Some(v) = state.fp32_independent {
        lines.push(Line::from(vec![
            Span::raw("  FP32 indep: "),
            Span::styled(format!("{:>8.2} Gflops/s", v), val_style),
        ]));
    }

    // NEON results
    if state.neon_fp32.is_some() || state.neon_int.is_some() {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled("  NEON SIMD:", hdr_style)));
        if let Some(v) = state.neon_fp32 {
            lines.push(Line::from(vec![
                Span::raw("    FP32 FMA: "),
                Span::styled(format!("{:>8.2} Gflops/s", v), val_style),
            ]));
        }
        if let Some(v) = state.neon_int {
            lines.push(Line::from(vec![
                Span::raw("    Int mul:  "),
                Span::styled(format!("{:>8.2} Gops/s", v), val_style),
            ]));
        }
    }

    // Core isolation results
    if state.pcore_int.is_some() || state.ecore_int.is_some() {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled("  Core isolation:", hdr_style)));
        if let Some(v) = state.pcore_int {
            lines.push(Line::from(vec![
                Span::raw("    P-core Int: "),
                Span::styled(format!("{:>6.2} Gops/s", v), val_style),
            ]));
        }
        if let Some(v) = state.pcore_fp {
            lines.push(Line::from(vec![
                Span::raw("    P-core FP:  "),
                Span::styled(format!("{:>6.2} Gflops/s", v), val_style),
            ]));
        }
        if let Some(v) = state.ecore_int {
            lines.push(Line::from(vec![
                Span::raw("    E-core Int: "),
                Span::styled(format!("{:>6.2} Gops/s", v), val_style),
            ]));
        }
        if let Some(v) = state.ecore_fp {
            lines.push(Line::from(vec![
                Span::raw("    E-core FP:  "),
                Span::styled(format!("{:>6.2} Gflops/s", v), val_style),
            ]));
        }
    }

    if state.mt_int_gops.is_some() || state.mt_fp_gflops.is_some() {
        lines.push(Line::raw(""));
        if let Some(v) = state.mt_int_gops {
            lines.push(Line::from(vec![
                Span::raw(format!("  MT Int ({:>2}c):", state.num_threads)),
                Span::styled(format!("{:>8.2} Gops/s", v), val_style),
            ]));
        }
        if let Some(v) = state.mt_fp_gflops {
            lines.push(Line::from(vec![
                Span::raw(format!("  MT FP  ({:>2}c):", state.num_threads)),
                Span::styled(format!("{:>8.2} Gflops/s", v), val_style),
            ]));
        }
    }

    if !state.cache_latencies.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled("  Latency ladder:", hdr_style)));
        for (label, ns) in &state.cache_latencies {
            lines.push(Line::from(vec![
                Span::raw(format!("    {:>6}  ", label)),
                Span::styled(format!("{:>8.1} ns", ns), val_style),
            ]));
        }
    }

    if !state.cache_bandwidths.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled("  BW ladder:", hdr_style)));
        for (label, gbps) in &state.cache_bandwidths {
            lines.push(Line::from(vec![
                Span::raw(format!("    {:>6}  ", label)),
                Span::styled(format!("{:>8.1} GB/s", gbps), val_style),
            ]));
        }
    }

    if lines.is_empty() {
        lines.push(Line::styled("  Waiting for results...", Style::default().fg(Color::DarkGray)));
    }

    let content_height = lines.len() as u16;
    let inner_height = area.height.saturating_sub(2); // border top+bottom
    let max_scroll = content_height.saturating_sub(inner_height);
    let effective_scroll = scroll.min(max_scroll);
    let overflows = content_height > inner_height;

    let title = if overflows {
        format!(" CPU BENCH [{}/{}] (j/k scroll) ", effective_scroll + 1, max_scroll + 1)
    } else {
        " CPU BENCH ".to_string()
    };
    let block = Block::bordered()
        .title(title)
        .border_style(border_style);

    let paragraph = Paragraph::new(lines).block(block).scroll((effective_scroll, 0));
    frame.render_widget(paragraph, area);
}

fn draw_gpu_panel(frame: &mut Frame, state: &DashboardState, area: Rect) {
    let border_style = Style::default().fg(Color::Yellow);
    let scroll = state.scroll_offset;

    let mut lines = Vec::new();
    let val_style = Style::default().fg(Color::Cyan);
    let hdr_style = Style::default().fg(Color::White).add_modifier(Modifier::BOLD);

    if let Some(ref msg) = state.gpu_status {
        lines.push(Line::styled(format!("  {}", msg), Style::default().fg(Color::Red)));
    }

    if state.gpu_fp32_tflops.is_some() || state.gpu_fp16_tflops.is_some() || state.gpu_int32_tops.is_some() {
        lines.push(Line::from(Span::styled("  Compute:", hdr_style)));
        if let Some(v) = state.gpu_fp32_tflops {
            lines.push(Line::from(vec![
                Span::raw("    FP32:   "),
                Span::styled(format!("{:>8.3} Tflops", v), val_style),
            ]));
        }
        if let Some(v) = state.gpu_fp16_tflops {
            lines.push(Line::from(vec![
                Span::raw("    FP16:   "),
                Span::styled(format!("{:>8.3} Tflops", v), val_style),
            ]));
        }
        if let Some(v) = state.gpu_int32_tops {
            lines.push(Line::from(vec![
                Span::raw("    Int32:  "),
                Span::styled(format!("{:>8.3} Tops", v), val_style),
            ]));
        }
    }

    if state.gpu_buf_read_gbps.is_some() || state.gpu_buf_write_gbps.is_some() {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled("  Memory BW:", hdr_style)));
        if let Some(v) = state.gpu_buf_read_gbps {
            lines.push(Line::from(vec![
                Span::raw("    Read:   "),
                Span::styled(format!("{:>8.2} GB/s", v), val_style),
            ]));
        }
        if let Some(v) = state.gpu_buf_write_gbps {
            lines.push(Line::from(vec![
                Span::raw("    Write:  "),
                Span::styled(format!("{:>8.2} GB/s", v), val_style),
            ]));
        }
    }

    if let Some(v) = state.gpu_buf_alloc_us {
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::raw("  Buf Alloc:  "),
            Span::styled(format!("{:>6.1} us", v), val_style),
        ]));
    }

    if let Some(v) = state.gpu_matmul_tflops {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled("  Workload:", hdr_style)));
        lines.push(Line::from(vec![
            Span::raw("    MatMul: "),
            Span::styled(format!("{:>8.3} Tflops", v), val_style),
        ]));
    }

    if lines.is_empty() {
        lines.push(Line::styled("  Waiting for results...", Style::default().fg(Color::DarkGray)));
    }

    let content_height = lines.len() as u16;
    let inner_height = area.height.saturating_sub(2);
    let max_scroll = content_height.saturating_sub(inner_height);
    let effective_scroll = scroll.min(max_scroll);
    let overflows = content_height > inner_height;

    let title = if overflows {
        format!(" GPU BENCH [{}/{}] ", effective_scroll + 1, max_scroll + 1)
    } else {
        " GPU BENCH (Metal) ".to_string()
    };
    let block = Block::bordered()
        .title(title)
        .border_style(border_style);

    let paragraph = Paragraph::new(lines).block(block).scroll((effective_scroll, 0));
    frame.render_widget(paragraph, area);
}

fn draw_gpu_sparklines(frame: &mut Frame, state: &DashboardState, area: Rect) {
    let block = Block::bordered()
        .title(" GPU THROUGHPUT HISTORY ")
        .border_style(Style::default().fg(Color::Yellow));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut sparkline_rows: Vec<(&str, &[u64], Color)> = Vec::new();

    if !state.gpu_fp32_history.is_empty() {
        sparkline_rows.push(("FP32:       ", &state.gpu_fp32_history, Color::Green));
    }
    if !state.gpu_fp16_history.is_empty() {
        sparkline_rows.push(("FP16:       ", &state.gpu_fp16_history, Color::Blue));
    }
    if !state.gpu_buf_read_history.is_empty() {
        sparkline_rows.push(("Buf Read:   ", &state.gpu_buf_read_history, Color::Magenta));
    }
    if !state.gpu_matmul_history.is_empty() {
        sparkline_rows.push(("MatMul:     ", &state.gpu_matmul_history, Color::Cyan));
    }

    let constraints: Vec<Constraint> = sparkline_rows.iter().map(|_| Constraint::Length(1)).collect();
    let rows = Layout::vertical(constraints).split(inner);

    let label_width = 12u16;

    for (i, (label, data, color)) in sparkline_rows.iter().enumerate() {
        if i >= rows.len() { break; }
        let label_area = Rect { width: label_width, ..rows[i] };
        let spark_area = Rect { x: rows[i].x + label_width, width: rows[i].width.saturating_sub(label_width), ..rows[i] };
        frame.render_widget(Span::styled(*label, Style::default().fg(*color)), label_area);
        frame.render_widget(
            Sparkline::default().data(*data).style(Style::default().fg(*color)),
            spark_area,
        );
    }
}

fn draw_mt_cpu_panel(frame: &mut Frame, state: &DashboardState, area: Rect) {
    let block = Block::bordered()
        .title(format!(" MT CPU BENCH ({} threads) ", state.num_threads))
        .border_style(Style::default().fg(Color::Yellow));

    let mut lines = Vec::new();
    let val_style = Style::default().fg(Color::Cyan);

    if let Some(v) = state.mt_int_gops {
        lines.push(Line::from(vec![
            Span::raw(format!("  MT Int ({:>2}c):", state.num_threads)),
            Span::styled(format!("{:>8.2} Gops/s", v), val_style),
        ]));
    }
    if let Some(v) = state.mt_fp_gflops {
        lines.push(Line::from(vec![
            Span::raw(format!("  MT FP  ({:>2}c):", state.num_threads)),
            Span::styled(format!("{:>8.2} Gflops/s", v), val_style),
        ]));
    }

    if lines.is_empty() {
        lines.push(Line::styled("  Waiting for results...", Style::default().fg(Color::DarkGray)));
    }

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}

fn draw_mt_cpu_sparklines(frame: &mut Frame, state: &DashboardState, area: Rect) {
    let block = Block::bordered()
        .title(" MT CPU THROUGHPUT HISTORY ")
        .border_style(Style::default().fg(Color::Yellow));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(inner);

    let label_width = 12u16;

    if !state.mt_int_history.is_empty() {
        let label_area = Rect { width: label_width, ..rows[0] };
        let spark_area = Rect { x: rows[0].x + label_width, width: rows[0].width.saturating_sub(label_width), ..rows[0] };
        frame.render_widget(Span::styled("MT Int:     ", Style::default().fg(Color::Cyan)), label_area);
        frame.render_widget(
            Sparkline::default().data(&state.mt_int_history).style(Style::default().fg(Color::Cyan)),
            spark_area,
        );
    }

    if !state.mt_fp_history.is_empty() {
        let label_area = Rect { width: label_width, ..rows[1] };
        let spark_area = Rect { x: rows[1].x + label_width, width: rows[1].width.saturating_sub(label_width), ..rows[1] };
        frame.render_widget(Span::styled("MT FP:      ", Style::default().fg(Color::LightRed)), label_area);
        frame.render_widget(
            Sparkline::default().data(&state.mt_fp_history).style(Style::default().fg(Color::LightRed)),
            spark_area,
        );
    }
}

fn draw_stress_panel(frame: &mut Frame, state: &DashboardState, area: Rect) {
    let block = Block::bordered()
        .title(" STRESS TEST ")
        .border_style(Style::default().fg(Color::Yellow));

    let mut lines = Vec::new();
    let val_style = Style::default().fg(Color::Cyan);

    lines.push(Line::from(vec![
        Span::raw("  Pass:    "),
        Span::styled(format!("{}", state.stress_pass), val_style),
    ]));

    let pass_style = Style::default().fg(Color::Green);
    let fail_style = Style::default().fg(if state.stress_total_failed > 0 { Color::Red } else { Color::Green });

    lines.push(Line::from(vec![
        Span::raw("  Passed:  "),
        Span::styled(format!("{}", state.stress_total_passed), pass_style),
        Span::raw("  Failed: "),
        Span::styled(format!("{}", state.stress_total_failed), fail_style),
    ]));

    if !state.stress_current_test.is_empty() {
        lines.push(Line::from(vec![
            Span::raw("  Current: "),
            Span::styled(&*state.stress_current_test, Style::default().fg(Color::White)),
        ]));
    }

    if let Some(last) = state.pass_history.first() {
        lines.push(Line::from(vec![
            Span::raw("  Pass Time: "),
            Span::styled(format!("{:.1?}", last.duration), val_style),
        ]));
    }

    lines.push(Line::raw(""));
    lines.push(Line::raw("  Recent Errors:"));
    if state.stress_recent_errors.is_empty() {
        lines.push(Line::styled("  (none)", Style::default().fg(Color::DarkGray)));
    } else {
        for err in state.stress_recent_errors.iter().rev().take(5) {
            lines.push(Line::styled(format!("  {}", err), Style::default().fg(Color::Red)));
        }
    }

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}

fn draw_bandwidth_sparklines(frame: &mut Frame, state: &DashboardState, area: Rect) {
    let block = Block::bordered()
        .title(" BANDWIDTH HISTORY (GB/s) ")
        .border_style(Style::default().fg(Color::Yellow));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(inner);

    let label_width = 7u16;

    if !state.write_history.is_empty() {
        let label_area = Rect { width: label_width, ..rows[0] };
        let spark_area = Rect { x: rows[0].x + label_width, width: rows[0].width.saturating_sub(label_width), ..rows[0] };
        frame.render_widget(Span::styled("Write: ", Style::default().fg(Color::Green)), label_area);
        frame.render_widget(
            Sparkline::default().data(&state.write_history).style(Style::default().fg(Color::Green)),
            spark_area,
        );
    }

    if !state.read_history.is_empty() {
        let label_area = Rect { width: label_width, ..rows[1] };
        let spark_area = Rect { x: rows[1].x + label_width, width: rows[1].width.saturating_sub(label_width), ..rows[1] };
        frame.render_widget(Span::styled("Read:  ", Style::default().fg(Color::Blue)), label_area);
        frame.render_widget(
            Sparkline::default().data(&state.read_history).style(Style::default().fg(Color::Blue)),
            spark_area,
        );
    }

    if !state.copy_history.is_empty() {
        let label_area = Rect { width: label_width, ..rows[2] };
        let spark_area = Rect { x: rows[2].x + label_width, width: rows[2].width.saturating_sub(label_width), ..rows[2] };
        frame.render_widget(Span::styled("Copy:  ", Style::default().fg(Color::Magenta)), label_area);
        frame.render_widget(
            Sparkline::default().data(&state.copy_history).style(Style::default().fg(Color::Magenta)),
            spark_area,
        );
    }

    if !state.mt_seq_read_history.is_empty() {
        let label_area = Rect { width: label_width, ..rows[3] };
        let spark_area = Rect { x: rows[3].x + label_width, width: rows[3].width.saturating_sub(label_width), ..rows[3] };
        frame.render_widget(Span::styled("MT Rd: ", Style::default().fg(Color::Cyan)), label_area);
        frame.render_widget(
            Sparkline::default().data(&state.mt_seq_read_history).style(Style::default().fg(Color::Cyan)),
            spark_area,
        );
    }
}

fn draw_latency_sparkline(frame: &mut Frame, state: &DashboardState, area: Rect) {
    let block = Block::bordered()
        .title(" LATENCY (ns) ")
        .border_style(Style::default().fg(Color::Yellow));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if !state.latency_history.is_empty() {
        frame.render_widget(
            Sparkline::default()
                .data(&state.latency_history)
                .style(Style::default().fg(Color::Yellow)),
            inner,
        );
    }
}

fn draw_cpu_sparklines(frame: &mut Frame, state: &DashboardState, area: Rect) {
    let block = Block::bordered()
        .title(" CPU THROUGHPUT HISTORY ")
        .border_style(Style::default().fg(Color::Yellow));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Build dynamic list of non-empty sparkline rows
    let mut sparkline_rows: Vec<(&str, &[u64], Color)> = Vec::new();

    if !state.int_dep_history.is_empty() {
        sparkline_rows.push(("Int dep:    ", &state.int_dep_history, Color::Green));
    }
    if !state.int_indep_history.is_empty() {
        sparkline_rows.push(("Int indep:  ", &state.int_indep_history, Color::Blue));
    }
    if !state.fp_dep_history.is_empty() {
        sparkline_rows.push(("FP64 dep:   ", &state.fp_dep_history, Color::Magenta));
    }
    if !state.fp_indep_history.is_empty() {
        sparkline_rows.push(("FP64 indep: ", &state.fp_indep_history, Color::Yellow));
    }
    if !state.fp32_dep_history.is_empty() {
        sparkline_rows.push(("FP32 dep:   ", &state.fp32_dep_history, Color::LightGreen));
    }
    if !state.fp32_indep_history.is_empty() {
        sparkline_rows.push(("FP32 indep: ", &state.fp32_indep_history, Color::LightBlue));
    }
    if !state.neon_fp32_history.is_empty() {
        sparkline_rows.push(("NEON FP32:  ", &state.neon_fp32_history, Color::LightMagenta));
    }
    if !state.neon_int_history.is_empty() {
        sparkline_rows.push(("NEON Int:   ", &state.neon_int_history, Color::LightYellow));
    }
    if !state.mt_int_history.is_empty() {
        sparkline_rows.push(("MT Int:     ", &state.mt_int_history, Color::Cyan));
    }
    if !state.mt_fp_history.is_empty() {
        sparkline_rows.push(("MT FP:      ", &state.mt_fp_history, Color::LightRed));
    }

    let constraints: Vec<Constraint> = sparkline_rows.iter().map(|_| Constraint::Length(1)).collect();
    let rows = Layout::vertical(constraints).split(inner);

    let label_width = 12u16;

    for (i, (label, data, color)) in sparkline_rows.iter().enumerate() {
        if i >= rows.len() { break; }
        let label_area = Rect { width: label_width, ..rows[i] };
        let spark_area = Rect { x: rows[i].x + label_width, width: rows[i].width.saturating_sub(label_width), ..rows[i] };
        frame.render_widget(Span::styled(*label, Style::default().fg(*color)), label_area);
        frame.render_widget(
            Sparkline::default().data(*data).style(Style::default().fg(*color)),
            spark_area,
        );
    }
}

fn draw_pass_history(frame: &mut Frame, state: &DashboardState, area: Rect) {
    let header = Row::new(vec!["Pass", "Passed", "Failed", "Duration", "Status"])
        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = state
        .pass_history
        .iter()
        .take(area.height.saturating_sub(4) as usize)
        .map(|p| {
            let status_style = if p.failed == 0 {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Red)
            };
            let status = if p.failed == 0 { "PASS" } else { "FAIL" };
            Row::new(vec![
                Cell::from(format!("{}", p.pass_num)),
                Cell::from(format!("{}", p.passed)).style(Style::default().fg(Color::Green)),
                Cell::from(format!("{}", p.failed)).style(if p.failed > 0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default()
                }),
                Cell::from(format!("{:.1?}", p.duration)),
                Cell::from(status).style(status_style),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(12),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(
        Block::bordered()
            .title(" PASS HISTORY ")
            .border_style(Style::default().fg(Color::Yellow)),
    );

    frame.render_widget(table, area);
}
