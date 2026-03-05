use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub type TestError = (usize, u64, u64);

pub static RUNNING: AtomicBool = AtomicBool::new(true);

unsafe extern "C" fn on_sigint(_sig: i32) {
    RUNNING.store(false, Ordering::Relaxed);
}

pub fn install_signal_handler() {
    unsafe extern "C" {
        fn signal(sig: i32, handler: unsafe extern "C" fn(i32));
    }
    unsafe { signal(2, on_sigint) }; // SIGINT = 2
}

pub fn vol_write(ptr: &mut u64, val: u64) {
    unsafe { ptr::write_volatile(ptr, val) };
}

pub fn vol_read(ptr: &u64) -> u64 {
    unsafe { ptr::read_volatile(ptr) }
}

pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

pub fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h{:02}m{:02}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    }
}
