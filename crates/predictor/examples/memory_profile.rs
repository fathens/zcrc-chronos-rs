//! Memory profiling for Predictor with different pool_threads.
//!
//! Measures RSS (Resident Set Size) before and after predictions to estimate
//! per-prediction memory consumption across different thread pool sizes.
//!
//! Run: `cargo run -p predictor --example memory_profile --release`

use chrono::{NaiveDate, TimeDelta};
use predictor::{BigDecimal, PredictionInput, Predictor};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

/// Get current process RSS in bytes.
#[cfg(target_os = "macos")]
fn get_rss_bytes() -> u64 {
    use std::mem;

    extern "C" {
        fn mach_task_self() -> u32;
    }

    unsafe {
        let mut info = mem::MaybeUninit::<libc::mach_task_basic_info_data_t>::uninit();
        let mut count = (mem::size_of::<libc::mach_task_basic_info_data_t>()
            / mem::size_of::<libc::natural_t>())
            as libc::mach_msg_type_number_t;
        let ret = libc::task_info(
            mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast(),
            &mut count,
        );
        if ret != libc::KERN_SUCCESS {
            eprintln!("task_info failed: {ret}");
            return 0;
        }
        let info = info.assume_init();
        info.resident_size
    }
}

#[cfg(target_os = "linux")]
fn get_rss_bytes() -> u64 {
    use std::fs;
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(val) = line.strip_prefix("VmRSS:") {
            let kb: u64 = val
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
            return kb * 1024;
        }
    }
    0
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn get_rss_bytes() -> u64 {
    eprintln!("RSS measurement not supported on this platform");
    0
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn make_input(n: usize) -> PredictionInput {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    let data = (0..n)
        .map(|i| {
            let ts = base + chrono::Duration::minutes(15 * i as i64);
            let val = 100.0
                + 1.5 * i as f64
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / 96.0).sin()
                + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 672.0).sin();
            let decimal = BigDecimal::from_str(&format!("{val:.8}")).unwrap();
            (ts, decimal)
        })
        .collect();

    PredictionInput {
        data,
        horizon: TimeDelta::hours(24),
    }
}

fn run_sequential_test(pool_threads: usize, input: &PredictionInput, num_predictions: usize) {
    let predictor = Predictor::new(pool_threads).unwrap();

    let rss_before = get_rss_bytes();
    let start = Instant::now();

    let mut peak_rss = rss_before;
    for i in 0..num_predictions {
        let result = predictor.predict(input);
        let rss_now = get_rss_bytes();
        peak_rss = peak_rss.max(rss_now);
        match result {
            Ok(f) => println!(
                "  prediction {}: {} forecast points, RSS: {}",
                i + 1,
                f.forecast_values.len(),
                format_bytes(rss_now)
            ),
            Err(e) => println!("  prediction {} failed: {e}", i + 1),
        }
    }

    let elapsed = start.elapsed();
    let rss_after = get_rss_bytes();

    println!("  --- Summary ---");
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  RSS before: {}", format_bytes(rss_before));
    println!("  RSS after:  {}", format_bytes(rss_after));
    println!("  Peak RSS:   {}", format_bytes(peak_rss));
    println!(
        "  RSS delta:  {}",
        format_bytes(peak_rss.saturating_sub(rss_before))
    );
}

fn run_concurrent_test(pool_threads: usize, input: &PredictionInput, num_threads: usize) {
    let predictor = Arc::new(Predictor::new(pool_threads).unwrap());

    let rss_before = get_rss_bytes();
    let start = Instant::now();

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let predictor = predictor.clone();
            let input = input.clone();
            std::thread::spawn(move || predictor.predict(&input))
        })
        .collect();

    for (i, h) in handles.into_iter().enumerate() {
        match h.join().unwrap() {
            Ok(f) => println!(
                "  thread {}: {} forecast points",
                i,
                f.forecast_values.len()
            ),
            Err(e) => println!("  thread {i} failed: {e}"),
        }
    }

    let elapsed = start.elapsed();
    let rss_after = get_rss_bytes();
    let peak_rss = get_rss_bytes(); // approximate: measured after join

    println!("  --- Summary ---");
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  RSS before: {}", format_bytes(rss_before));
    println!("  RSS after:  {}", format_bytes(rss_after));
    println!(
        "  RSS delta:  {}",
        format_bytes(peak_rss.saturating_sub(rss_before))
    );
}

fn main() {
    let data_size = 2880;
    let num_predictions = 5;
    let num_concurrent = 4;

    println!("=== Memory Profile: Predictor ===");
    println!("Data size: {data_size} points (simulating 30 days @ 15min intervals)");
    println!("Initial RSS: {}", format_bytes(get_rss_bytes()));
    println!();

    let input = make_input(data_size);

    // Sequential tests
    for pool_threads in [1, 2, 3] {
        println!("--- Sequential: pool_threads={pool_threads}, {num_predictions} predictions ---");
        run_sequential_test(pool_threads, &input, num_predictions);
        println!();
    }

    // Concurrent tests
    for pool_threads in [1, 2, 3] {
        println!("--- Concurrent: pool_threads={pool_threads}, {num_concurrent} threads ---");
        run_concurrent_test(pool_threads, &input, num_concurrent);
        println!();
    }

    println!("Final RSS: {}", format_bytes(get_rss_bytes()));
}
