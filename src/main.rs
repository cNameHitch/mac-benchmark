use std::env;
use std::time::Duration;

mod helpers;
mod tests;
mod bench;
mod cpu_bench;
mod sysinfo;
mod dashboard;

use dashboard::RunMode;

const DEFAULT_SIZE_MB: usize = 256;

fn main() {
    let (mode, size_mb, duration_minutes, threads) = parse_config();

    match mode {
        Some(InternalMode::Info) => {
            let info = sysinfo::detect();
            info.print();
        }
        Some(InternalMode::Run(run_mode)) => {
            let duration = duration_minutes.map(|m| Duration::from_secs(m * 60));
            dashboard::run(Some(run_mode), size_mb, duration, threads);
        }
        None => {
            let duration = duration_minutes.map(|m| Duration::from_secs(m * 60));
            dashboard::run(None, size_mb, duration, threads);
        }
    }
}

// Internal mode for parsing only — Info goes to stdout, everything else to TUI
enum InternalMode {
    Info,
    Run(RunMode),
}

fn parse_config() -> (Option<InternalMode>, usize, Option<u64>, Option<u32>) {
    let args: Vec<String> = env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("Usage: {} [OPTIONS] [SIZE_MB]", args[0]);
        eprintln!();
        eprintln!("Launches an interactive TUI by default. Use flags to skip the menu.");
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  SIZE_MB            Memory size in megabytes (default: {DEFAULT_SIZE_MB})");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --test             Run memory correctness tests");
        eprintln!("  --bench            Run memory performance benchmarks");
        eprintln!("  --cpu              Run CPU benchmarks + cache hierarchy profiling");
        eprintln!("  --mt-cpu           Run multi-threaded CPU benchmarks (all cores)");
        eprintln!("  --all              Run tests + memory benchmarks + CPU benchmarks");
        eprintln!("  --stress           Continuous correctness stress test");
        eprintln!("  --full-stress      Full stress: correctness + all benchmarks each cycle");
        eprintln!("  --dashboard        Alias for --full-stress (interactive TUI)");
        eprintln!("  --info             Print system info (chip, cores, memory) and exit");
        eprintln!("  --duration MINS    Time limit in minutes (stress/full-stress)");
        eprintln!("  --threads N, -T N  Number of threads for MT benchmarks (default: all cores)");
        eprintln!("  -h, --help         Show this help");
        std::process::exit(0);
    }

    let mut mode: Option<InternalMode> = None;
    let mut size_mb = DEFAULT_SIZE_MB;
    let mut duration_minutes = None;
    let mut threads: Option<u32> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--test" | "-t" => mode = Some(InternalMode::Run(RunMode::Test)),
            "--bench" | "-b" => mode = Some(InternalMode::Run(RunMode::Bench)),
            "--cpu" => mode = Some(InternalMode::Run(RunMode::Cpu)),
            "--mt-cpu" => mode = Some(InternalMode::Run(RunMode::MtCpu)),
            "--all" | "-a" => mode = Some(InternalMode::Run(RunMode::All)),
            "--stress" | "-s" => mode = Some(InternalMode::Run(RunMode::Stress)),
            "--full-stress" | "-F" | "--dashboard" => mode = Some(InternalMode::Run(RunMode::FullStress)),
            "--info" => mode = Some(InternalMode::Info),
            "--duration" | "-d" => {
                i += 1;
                duration_minutes = Some(
                    args.get(i)
                        .and_then(|a| a.parse::<u64>().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Error: --duration requires a value in minutes");
                            std::process::exit(1);
                        }),
                );
            }
            "--threads" | "-T" => {
                i += 1;
                threads = Some(
                    args.get(i)
                        .and_then(|a| a.parse::<u32>().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Error: --threads requires a positive integer");
                            std::process::exit(1);
                        }),
                );
            }
            _ => {
                size_mb = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Error: '{}' is not a valid argument", args[i]);
                    eprintln!("Run with --help for usage.");
                    std::process::exit(1);
                });
            }
        }
        i += 1;
    }

    (mode, size_mb, duration_minutes, threads)
}
