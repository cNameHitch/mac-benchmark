use std::process::Command;

pub struct SystemInfo {
    pub chip_name: String,
    pub model_id: String,
    pub p_cores: u32,
    pub e_cores: u32,
    pub total_cores: u32,
    pub memory_gb: u64,
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

fn detect_chip_name() -> String {
    // Try system_profiler for the human-readable chip name
    if let Ok(output) = Command::new("system_profiler")
        .arg("SPHardwareDataType")
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            for line in text.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("Chip:") {
                    return trimmed.strip_prefix("Chip:").unwrap().trim().to_string();
                }
            }
        }
    }
    "Unknown".to_string()
}

pub fn detect() -> SystemInfo {
    let chip_name = detect_chip_name();
    let model_id = sysctl_string("hw.model").unwrap_or_else(|| "Unknown".to_string());
    let total_cores = sysctl_u32("hw.ncpu").unwrap_or(0);
    let p_cores = sysctl_u32("hw.perflevel0.logicalcpu").unwrap_or(0);
    let e_cores = sysctl_u32("hw.perflevel1.logicalcpu").unwrap_or(0);
    let memory_bytes = sysctl_u64("hw.memsize").unwrap_or(0);
    let memory_gb = memory_bytes / (1024 * 1024 * 1024);

    SystemInfo {
        chip_name,
        model_id,
        p_cores,
        e_cores,
        total_cores,
        memory_gb,
    }
}

impl SystemInfo {
    pub fn print(&self) {
        println!("=== System Info ===");
        println!("  Chip    : {}", self.chip_name);
        println!("  Model   : {}", self.model_id);
        if self.p_cores > 0 && self.e_cores > 0 {
            println!(
                "  Cores   : {} ({}P + {}E)",
                self.total_cores, self.p_cores, self.e_cores
            );
        } else {
            println!("  Cores   : {}", self.total_cores);
        }
        println!("  Memory  : {} GB", self.memory_gb);
    }
}
