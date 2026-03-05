# mac-benchmark

A memory and CPU testing, benchmarking, and stress testing tool written in Rust. Designed for comparing performance across Apple M-series Macs (M1, M2, M3, M4 and their Pro/Max/Ultra variants).

## Usage

```
mac-benchmark [OPTIONS] [SIZE_MB]
```

`SIZE_MB` sets the memory region / max working set size in megabytes (default: 256).

By default, launching without any mode flag opens an **interactive TUI menu** where you can select a mode, configure size and duration, and start a run. After a run completes, a summary screen is shown with options to return to the menu or quit.

CLI flags act as **shortcuts** that skip the menu and go directly to the running screen.

## Modes

### Interactive Menu (default)

```
cargo run --release
```

Launches the TUI with a menu for selecting mode, size, and duration. Navigation:
- **Up/Down** — select mode
- **Tab** — cycle between mode list, size input, and duration input
- **Digits/Backspace** — edit size or duration fields
- **Enter** — start the selected mode
- **q** — quit

### Memory Tests

```
cargo run --release -- --test 512
```

Runs 10 correctness tests (solid bits, checkerboard, walking ones/zeros, address-as-value, March C-, random fill) on a 512 MB region.

### Memory Benchmarks

```
cargo run --release -- --bench 512
```

Measures sequential write/read, copy, random read latency, and stride read throughput across 7 stride sizes.

### CPU Benchmarks

```
cargo run --release -- --cpu 64
```

Runs CPU and cache hierarchy benchmarks with a max working set of 64 MB:

- **Integer throughput** — dependent chain and 4 independent chains
- **FP64 throughput** — dependent FMA chain and 4 independent FMA chains
- **Cache latency ladder** — pointer-chase at 12 working set sizes (32 KB to 64 MB)
- **Bandwidth ladder** — sequential read throughput at the same sizes

### All (Tests + Benchmarks)

```
cargo run --release -- --all 512
```

Runs memory tests, memory benchmarks, and CPU benchmarks.

### Stress Test

```
cargo run --release -- --stress 512
cargo run --release -- --stress 512 --duration 10
```

Continuously repeats all 10 correctness tests in a loop until stopped with `q` or duration expires.

### Full Stress Test

```
cargo run --release -- --full-stress 64
cargo run --release -- --full-stress 64 --duration 5
```

Comprehensive stress test that runs every cycle:

1. **Correctness** — all 10 memory tests
2. **Memory benchmarks** — sequential write/read, copy, random latency, stride throughput
3. **CPU benchmarks** — integer/FP dependent + independent chains
4. **Cache latency ladder** — pointer chase at each fitting working set size
5. **Cache bandwidth ladder** — sequential read at each fitting working set size

Includes sparkline history, pass history table, and regression tracking (min/max/avg for every metric). `--dashboard` is an alias for `--full-stress`.

### System Info

```
cargo run --release -- --info
```

Prints chip name, model ID, core counts (P + E), and memory size to stdout and exits (no TUI).

## TUI Flow

All modes except `--info` and `--help` run inside the TUI:

```
Launch → Menu (or Running if CLI flag given) → Summary → Menu or Quit
```

- **Running screen** adapts its layout to the active mode
- **Summary screen** shows all results and offers `[B]ack to Menu` or `[Q]uit`
- The app never auto-closes; the summary stays up until you choose

## Options

| Flag | Short | Description |
|---|---|---|
| `--test` | `-t` | Memory correctness tests |
| `--bench` | `-b` | Memory performance benchmarks |
| `--cpu` | | CPU benchmarks + cache hierarchy profiling |
| `--all` | `-a` | Tests + memory + CPU benchmarks |
| `--stress` | `-s` | Continuous correctness stress test |
| `--full-stress` | `-F` | Full stress: correctness + all benchmarks per cycle |
| `--dashboard` | | Alias for `--full-stress` |
| `--info` | | Print system info and exit (stdout) |
| `--duration MINS` | `-d MINS` | Time limit in minutes |
| `--help` | `-h` | Show help (stdout) |

## Examples

```sh
# Interactive menu (default)
cargo run --release

# Show system info (stdout, no TUI)
cargo run --release -- --info

# Skip menu, run tests on 512 MB → summary screen
cargo run --release -- --test 512

# Skip menu, memory benchmarks on 1 GB → summary screen
cargo run --release -- --bench 1024

# Skip menu, full stress 64 MB for 5 minutes → summary screen
cargo run --release -- --full-stress 64 --duration 5

# Skip menu, stress test 512 MB for 10 minutes
cargo run --release -- --stress 512 --duration 10
```
