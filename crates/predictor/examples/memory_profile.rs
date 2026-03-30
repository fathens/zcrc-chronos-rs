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
            // Use 22 significant digits to match production data (avg 22, max 33)
            let decimal = BigDecimal::from_str(&format!("{val:.20}")).unwrap();
            (ts, decimal)
        })
        .collect();

    PredictionInput {
        data,
        horizon: TimeDelta::hours(24),
    }
}

/// Run total_predictions predictions across num_threads threads concurrently.
/// Mimics production: buffer_unordered(num_threads) processing total_predictions tokens.
fn run_concurrent_test(
    pool_threads: usize,
    input: &PredictionInput,
    num_threads: usize,
    total_predictions: usize,
) {
    let predictor = Arc::new(Predictor::new(pool_threads).unwrap());
    let work_per_thread = total_predictions.div_ceil(num_threads);

    let rss_before = get_rss_bytes();
    let start = Instant::now();

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let predictor = predictor.clone();
            let input = input.clone();
            let count = if t < num_threads - 1 {
                work_per_thread
            } else {
                total_predictions.saturating_sub(work_per_thread * t)
            };
            std::thread::spawn(move || {
                let mut ok = 0usize;
                let mut fail = 0usize;
                for _ in 0..count {
                    match predictor.predict(&input) {
                        Ok(_) => ok += 1,
                        Err(_) => fail += 1,
                    }
                }
                (ok, fail)
            })
        })
        .collect();

    let mut total_ok = 0;
    let mut total_fail = 0;
    for (i, h) in handles.into_iter().enumerate() {
        let (ok, fail) = h.join().unwrap();
        println!("  thread {i}: {ok} ok, {fail} failed");
        total_ok += ok;
        total_fail += fail;
    }

    let elapsed = start.elapsed();
    let rss_after = get_rss_bytes();

    println!("  --- Summary ---");
    println!("  Predictions: {total_ok} ok, {total_fail} failed");
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  RSS before: {}", format_bytes(rss_before));
    println!("  RSS after:  {}", format_bytes(rss_after));
    println!(
        "  RSS delta:  {}",
        format_bytes(rss_after.saturating_sub(rss_before))
    );
}

fn main() {
    let data_size = 2880;
    let num_predictions = 293; // Match production token count
    let num_concurrent = 4;

    println!("=== Memory Profile: Predictor ===");
    println!("Data size: {data_size} points (simulating 30 days @ 15min intervals)");
    println!("Initial RSS: {}", format_bytes(get_rss_bytes()));
    println!();

    let input = make_input(data_size);

    // Concurrent tests: 4 threads processing 293 predictions (matching production)
    for pool_threads in [1, 2, 3] {
        println!(
            "--- pool_threads={pool_threads}, {num_concurrent} threads, {num_predictions} predictions ---"
        );
        run_concurrent_test(pool_threads, &input, num_concurrent, num_predictions);
        println!();
    }

    println!("Final RSS: {}", format_bytes(get_rss_bytes()));
}
