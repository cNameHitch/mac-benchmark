use std::process::Command;

pub struct SystemInfo {
    pub chip_name: String,
    pub model_id: String,
    pub model_name: String,
    pub model_number: String,
    pub p_cores: u32,
    pub e_cores: u32,
    pub total_cores: u32,
    pub gpu_cores: u32,
    pub metal_version: String,
    pub memory_gb: u64,
    pub os_version: String,
    pub os_build: String,
    pub serial_suffix: String,
    pub uptime_secs: u64,
}

#[derive(Clone)]
pub struct DynamicInfo {
    pub battery_percent: Option<u32>,
    pub battery_charging: Option<bool>,
    pub battery_on_ac: Option<bool>,
    pub battery_cycle_count: Option<u32>,
    pub battery_temp_c: Option<f64>,
    pub memory_used_gb: f64,
    pub memory_total_gb: f64,
    pub thermal_state: ThermalState,
}

#[derive(Clone, Copy, PartialEq)]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
    Unknown,
}

impl ThermalState {
    pub fn label(self) -> &'static str {
        match self {
            ThermalState::Nominal => "Nominal",
            ThermalState::Fair => "Fair",
            ThermalState::Serious => "Serious",
            ThermalState::Critical => "Critical",
            ThermalState::Unknown => "Unknown",
        }
    }
}

fn sysctl_string(key: &str) -> Option<String> {
    Command::new("sysctl")
        .args(["-n", key])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .filter(|s| !s.is_empty())
}

fn sysctl_u64(key: &str) -> Option<u64> {
    sysctl_string(key).and_then(|s| s.parse().ok())
}

fn sysctl_u32(key: &str) -> Option<u32> {
    sysctl_string(key).and_then(|s| s.parse().ok())
}

fn run_cmd(cmd: &str, args: &[&str]) -> Option<String> {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .filter(|s| !s.is_empty())
}

fn parse_hw_data(text: &str, key: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(key) {
            return Some(rest.trim().to_string());
        }
    }
    None
}

fn detect_hardware_info() -> (String, String, String, u32, String) {
    let mut chip_name = "Unknown".to_string();
    let mut model_name = String::new();
    let mut model_number = String::new();
    let mut gpu_cores = 0u32;
    let mut metal_version = String::new();

    if let Some(text) = run_cmd("system_profiler", &["SPHardwareDataType"]) {
        if let Some(v) = parse_hw_data(&text, "Chip:") {
            chip_name = v;
        }
        if let Some(v) = parse_hw_data(&text, "Model Name:") {
            model_name = v;
        }
        if let Some(v) = parse_hw_data(&text, "Model Number:") {
            model_number = v;
        }
    }

    if let Some(text) = run_cmd("system_profiler", &["SPDisplaysDataType"]) {
        if let Some(v) = parse_hw_data(&text, "Total Number of Cores:") {
            if let Ok(n) = v.parse::<u32>() {
                gpu_cores = n;
            }
        }
        if let Some(v) = parse_hw_data(&text, "Metal Support:") {
            metal_version = v;
        }
    }

    (chip_name, model_name, model_number, gpu_cores, metal_version)
}

fn detect_uptime() -> u64 {
    // sysctl kern.boottime returns: { sec = 1234567890, usec = 0 }
    if let Some(s) = sysctl_string("kern.boottime") {
        if let Some(start) = s.find("sec = ") {
            let rest = &s[start + 6..];
            if let Some(end) = rest.find(',') {
                if let Ok(boot_sec) = rest[..end].trim().parse::<u64>() {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    return now.saturating_sub(boot_sec);
                }
            }
        }
    }
    0
}

pub fn detect() -> SystemInfo {
    let (chip_name, model_name, model_number, gpu_cores, metal_version) = detect_hardware_info();
    let model_id = sysctl_string("hw.model").unwrap_or_else(|| "Unknown".to_string());
    let total_cores = sysctl_u32("hw.ncpu").unwrap_or(0);
    let p_cores = sysctl_u32("hw.perflevel0.logicalcpu").unwrap_or(0);
    let e_cores = sysctl_u32("hw.perflevel1.logicalcpu").unwrap_or(0);
    let memory_bytes = sysctl_u64("hw.memsize").unwrap_or(0);
    let memory_gb = memory_bytes / (1024 * 1024 * 1024);

    let os_version = run_cmd("sw_vers", &["-productVersion"]).unwrap_or_default();
    let os_build = run_cmd("sw_vers", &["-buildVersion"]).unwrap_or_default();

    // Last 4 chars of serial for identification without exposing full serial
    let serial_suffix = run_cmd("system_profiler", &["SPHardwareDataType"])
        .and_then(|text| parse_hw_data(&text, "Serial Number (system):"))
        .map(|s| {
            let len = s.len();
            if len > 4 { format!("...{}", &s[len - 4..]) } else { s }
        })
        .unwrap_or_default();

    let uptime_secs = detect_uptime();

    SystemInfo {
        chip_name,
        model_id,
        model_name,
        model_number,
        p_cores,
        e_cores,
        total_cores,
        gpu_cores,
        metal_version,
        memory_gb,
        os_version,
        os_build,
        serial_suffix,
        uptime_secs,
    }
}

/// Poll dynamic system info (battery, memory, thermal).
/// This is lightweight enough to call every few seconds.
pub fn poll_dynamic() -> DynamicInfo {
    let (battery_percent, battery_charging, battery_on_ac, battery_cycle_count, battery_temp_c) = poll_battery();
    let (memory_used_gb, memory_total_gb) = poll_memory();
    let thermal_state = poll_thermal();

    DynamicInfo {
        battery_percent,
        battery_charging,
        battery_on_ac,
        battery_cycle_count,
        battery_temp_c,
        memory_used_gb,
        memory_total_gb,
        thermal_state,
    }
}

fn poll_battery() -> (Option<u32>, Option<bool>, Option<bool>, Option<u32>, Option<f64>) {
    let output = match Command::new("ioreg")
        .args(["-r", "-n", "AppleSmartBattery", "-d", "1"])
        .output()
    {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => return (None, None, None, None, None),
    };

    if output.is_empty() || !output.contains("AppleSmartBattery") {
        return (None, None, None, None, None);
    }

    let percent = extract_ioreg_int(&output, "\"CurrentCapacity\"").map(|v| v as u32);
    let is_charging = extract_ioreg_bool(&output, "\"IsCharging\"");
    let on_ac = extract_ioreg_bool(&output, "\"ExternalConnected\"");
    let cycles = extract_ioreg_int(&output, "\"CycleCount\"").map(|v| v as u32);
    // Temperature is in centi-degrees Celsius (e.g., 3020 = 30.20 C)
    let temp = extract_ioreg_int(&output, "\"Temperature\"").map(|v| v as f64 / 100.0);

    (percent, is_charging, on_ac, cycles, temp)
}

fn poll_memory() -> (f64, f64) {
    let total_bytes = sysctl_u64("hw.memsize").unwrap_or(0);
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    // Parse vm_stat output to get used memory
    let vm_output = match Command::new("vm_stat").output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => return (0.0, total_gb),
    };

    // Get page size from first line
    let page_size: f64 = vm_output
        .lines()
        .next()
        .and_then(|line| {
            line.split("page size of ")
                .nth(1)
                .and_then(|s| s.split(' ').next())
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(16384.0);

    let parse_pages = |key: &str| -> u64 {
        vm_output
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| {
                l.split(':')
                    .nth(1)
                    .map(|s| s.trim().trim_end_matches('.'))
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(0)
    };

    let active = parse_pages("Pages active");
    let wired = parse_pages("Pages wired down");
    let speculative = parse_pages("Pages speculative");
    let compressed = parse_pages("Pages occupied by compressor");

    let used_bytes = (active + wired + speculative + compressed) as f64 * page_size;
    let used_gb = used_bytes / (1024.0 * 1024.0 * 1024.0);

    (used_gb, total_gb)
}

fn poll_thermal() -> ThermalState {
    // Use pmset -g therm to detect thermal warnings
    let output = match Command::new("pmset").args(["-g", "therm"]).output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => return ThermalState::Unknown,
    };

    // If any thermal/performance warning is recorded, it's not nominal
    if output.contains("CPU_Speed_Limit") {
        // Check if speed limit is below 100
        for line in output.lines() {
            if line.contains("CPU_Speed_Limit") {
                if let Some(val) = line.split('=').nth(1) {
                    if let Ok(limit) = val.trim().parse::<u32>() {
                        return match limit {
                            90..=99 => ThermalState::Fair,
                            70..=89 => ThermalState::Serious,
                            0..=69 => ThermalState::Critical,
                            _ => ThermalState::Nominal,
                        };
                    }
                }
            }
        }
    }

    if output.contains("No thermal warning level has been recorded")
        && output.contains("No performance warning level has been recorded")
    {
        return ThermalState::Nominal;
    }

    if output.contains("thermal warning level") && !output.contains("No thermal warning") {
        return ThermalState::Fair;
    }

    ThermalState::Nominal
}

fn extract_ioreg_int(text: &str, key: &str) -> Option<i64> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.contains(key) {
            if let Some(eq_pos) = trimmed.find('=') {
                let val_str = trimmed[eq_pos + 1..].trim();
                // Handle values that might end with other data
                let val_str = val_str.split_whitespace().next().unwrap_or(val_str);
                if let Ok(v) = val_str.parse::<i64>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn extract_ioreg_bool(text: &str, key: &str) -> Option<bool> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.contains(key) {
            if trimmed.contains("Yes") {
                return Some(true);
            } else if trimmed.contains("No") {
                return Some(false);
            }
        }
    }
    None
}

impl SystemInfo {
    pub fn print(&self) {
        println!("=== System Info ===");
        println!("  Model   : {} ({})", self.model_name, self.model_id);
        println!("  Chip    : {}", self.chip_name);
        if self.p_cores > 0 && self.e_cores > 0 {
            println!(
                "  Cores   : {} ({}P + {}E)",
                self.total_cores, self.p_cores, self.e_cores
            );
        } else {
            println!("  Cores   : {}", self.total_cores);
        }
        if self.gpu_cores > 0 {
            println!("  GPU     : {} cores ({})", self.gpu_cores, self.metal_version);
        }
        println!("  Memory  : {} GB", self.memory_gb);
        if !self.os_version.is_empty() {
            println!("  macOS   : {} ({})", self.os_version, self.os_build);
        }
        if self.uptime_secs > 0 {
            println!("  Uptime  : {}", format_uptime(self.uptime_secs));
        }
    }
}

pub fn format_uptime(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m", minutes)
    }
}
