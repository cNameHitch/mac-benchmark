use std::sync::atomic::Ordering;

use crate::helpers::{vol_read, vol_write, xorshift64, TestError, RUNNING};

pub fn pattern_test(region: &mut [u64], pattern: u64) -> Result<(), TestError> {
    for elem in region.iter_mut() {
        vol_write(elem, pattern);
    }
    for (i, elem) in region.iter().enumerate() {
        let val = vol_read(elem);
        if val != pattern {
            return Err((i, pattern, val));
        }
    }
    Ok(())
}

pub fn walking_ones_test(region: &mut [u64]) -> Result<(), TestError> {
    for bit in 0..64 {
        pattern_test(region, 1u64 << bit)?;
    }
    Ok(())
}

pub fn walking_zeros_test(region: &mut [u64]) -> Result<(), TestError> {
    for bit in 0..64 {
        pattern_test(region, !(1u64 << bit))?;
    }
    Ok(())
}

pub fn address_test(region: &mut [u64]) -> Result<(), TestError> {
    for (i, elem) in region.iter_mut().enumerate() {
        vol_write(elem, i as u64);
    }
    for (i, elem) in region.iter().enumerate() {
        let expected = i as u64;
        let val = vol_read(elem);
        if val != expected {
            return Err((i, expected, val));
        }
    }
    Ok(())
}

pub fn march_test(region: &mut [u64]) -> Result<(), TestError> {
    let zero: u64 = 0;
    let one: u64 = !0;

    for elem in region.iter_mut() {
        vol_write(elem, zero);
    }
    for (i, elem) in region.iter_mut().enumerate() {
        let val = vol_read(elem);
        if val != zero { return Err((i, zero, val)); }
        vol_write(elem, one);
    }
    for (i, elem) in region.iter_mut().enumerate() {
        let val = vol_read(elem);
        if val != one { return Err((i, one, val)); }
        vol_write(elem, zero);
    }
    for (i, elem) in region.iter_mut().enumerate().rev() {
        let val = vol_read(elem);
        if val != zero { return Err((i, zero, val)); }
        vol_write(elem, one);
    }
    for (i, elem) in region.iter_mut().enumerate().rev() {
        let val = vol_read(elem);
        if val != one { return Err((i, one, val)); }
        vol_write(elem, zero);
    }
    for (i, elem) in region.iter().enumerate() {
        let val = vol_read(elem);
        if val != zero { return Err((i, zero, val)); }
    }
    Ok(())
}

pub fn random_test(region: &mut [u64], seed: u64) -> Result<(), TestError> {
    let mut state = seed;
    for elem in region.iter_mut() {
        vol_write(elem, xorshift64(&mut state));
    }
    state = seed;
    for (i, elem) in region.iter().enumerate() {
        let expected = xorshift64(&mut state);
        let val = vol_read(elem);
        if val != expected {
            return Err((i, expected, val));
        }
    }
    Ok(())
}

/// Run all correctness tests in a pass, returning (passed, failed, error messages).
pub fn run_test_pass(region: &mut [u64]) -> (u32, u32, Vec<String>) {
    let mut passed = 0u32;
    let mut failed = 0u32;
    let mut errors = Vec::new();

    macro_rules! check {
        ($name:expr, $result:expr) => {
            if !RUNNING.load(Ordering::Relaxed) {
                return (passed, failed, errors);
            }
            match $result {
                Ok(()) => passed += 1,
                Err((offset, expected, actual)) => {
                    failed += 1;
                    errors.push(format!(
                        "{}: offset={} expected={:#018x} actual={:#018x}",
                        $name, offset, expected, actual
                    ));
                }
            }
        };
    }

    check!("Solid bits (0x00)", pattern_test(region, 0x0000000000000000));
    check!("Solid bits (0xFF)", pattern_test(region, 0xFFFFFFFFFFFFFFFF));
    check!("Checkerboard (0xAA)", pattern_test(region, 0xAAAAAAAAAAAAAAAA));
    check!("Checkerboard (0x55)", pattern_test(region, 0x5555555555555555));
    check!("Walking ones", walking_ones_test(region));
    check!("Walking zeros", walking_zeros_test(region));
    check!("Address-as-value", address_test(region));
    check!("March C-", march_test(region));
    check!("Random fill (seed=0xDEAD)", random_test(region, 0xDEAD));
    check!("Random fill (seed=0xBEEF)", random_test(region, 0xBEEF));

    (passed, failed, errors)
}
