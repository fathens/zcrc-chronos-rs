//! Real-world data accuracy tests.
//!
//! These tests use actual price data patterns (31 days each) to validate
//! prediction accuracy. They are marked `#[ignore]` for manual execution.
//!
//! Run with: cargo test -p predictor --test real_data_test -- --ignored --nocapture

use chrono::{NaiveDateTime, TimeDelta};
use common::BigDecimal;
use num_traits::FromPrimitive;
use predictor::{predict, PredictionInput};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

/// Structure matching the real data JSON files.
#[derive(Deserialize)]
struct RealDataFile {
    description: String,
    #[allow(dead_code)]
    base_token: String,
    #[allow(dead_code)]
    quote_token: String,
    #[allow(dead_code)]
    decimals: u32,
    data: Vec<DataPoint>,
}

#[derive(Deserialize)]
struct DataPoint {
    timestamp: String,
    #[allow(dead_code)]
    rate: String,
    price: String,
}

/// Get the path to the tests/real directory.
fn real_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/real")
}

/// Find all data files for a pattern (e.g., "uptrend" → ["uptrend-01.json", "uptrend-02.json", ...]).
fn find_pattern_files(pattern: &str) -> Vec<String> {
    let dir = real_data_dir();
    let prefix = format!("{}-", pattern);

    let mut files: Vec<String> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Failed to read directory {}: {}", dir.display(), e))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix) && name.ends_with(".json") {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    files.sort(); // Ensure consistent ordering: -01, -02, -03, ...
    assert!(
        !files.is_empty(),
        "No data files found for pattern '{}' (expected {}-*.json)",
        pattern,
        pattern
    );
    files
}

/// Load and parse a real data JSON file.
fn load_real_data(filename: &str) -> RealDataFile {
    let path = real_data_dir().join(filename);
    let content = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("Failed to read {}: {}", path.display(), e);
    });
    serde_json::from_str(&content).unwrap_or_else(|e| {
        panic!("Failed to parse {}: {}", path.display(), e);
    })
}

/// Compute MAPE (Mean Absolute Percentage Error).
fn compute_mape(forecast: f64, actual: f64) -> f64 {
    ((forecast - actual) / actual).abs() * 100.0
}

/// Test result containing forecast, actual value, and MAPE.
struct TestResult {
    forecast: f64,
    actual: f64,
    mape: f64,
    description: String,
    data_points_used: usize,
}

/// Run a real data test for the given file.
///
/// Strategy:
/// 1. Load the data (31 days of price data)
/// 2. Use the first 30 days as training data (downsampled to ~720 points)
/// 3. Find the data point closest to 24 hours after training end
/// 4. Run prediction with horizon=1
/// 5. Compare forecast vs actual
///
/// Note: Data is downsampled to ~720 points for faster test execution.
/// The normalize module now handles large time gaps correctly, so full data
/// can be used if needed.
fn run_real_data_test(filename: &str) -> TestResult {
    let file_data = load_real_data(filename);

    // Parse all timestamps and prices
    let parsed: Vec<(NaiveDateTime, f64)> = file_data
        .data
        .iter()
        .map(|dp| {
            let ts = NaiveDateTime::parse_from_str(&dp.timestamp, "%Y-%m-%dT%H:%M:%S%.f")
                .unwrap_or_else(|e| panic!("Failed to parse timestamp '{}': {}", dp.timestamp, e));
            let price: f64 = dp.price.parse().unwrap_or_else(|e| {
                panic!("Failed to parse price '{}': {}", dp.price, e);
            });
            (ts, price)
        })
        .collect();

    assert!(
        parsed.len() >= 100,
        "Need at least 100 data points, got {}",
        parsed.len()
    );

    // Find the cutoff: 30 days from the start
    let start_time = parsed[0].0;
    let cutoff_time = start_time + chrono::Duration::days(30);

    // Split data: training (first 30 days) and find actual value (~24h after cutoff)
    type DataRef<'a> = Vec<&'a (NaiveDateTime, f64)>;
    let (train_data, future_data): (DataRef<'_>, DataRef<'_>) =
        parsed.iter().partition(|(ts, _)| *ts < cutoff_time);

    assert!(
        !train_data.is_empty(),
        "No training data before cutoff time"
    );
    assert!(
        !future_data.is_empty(),
        "No future data after cutoff time for validation"
    );

    // Find the data point closest to 24 hours after the last training point
    let last_train_time = train_data.last().unwrap().0;
    let target_time = last_train_time + chrono::Duration::hours(24);

    let actual_point = future_data
        .iter()
        .min_by_key(|(ts, _)| (*ts - target_time).num_seconds().abs())
        .expect("No future data point found");

    let actual_value = actual_point.1;

    // Downsample if too many data points (keep ~720 points = 30 days * 24 hours)
    // This is required because the prediction pipeline fails with 2000+ points
    let max_points = 720;
    let step = if train_data.len() > max_points {
        train_data.len() / max_points
    } else {
        1
    };

    let sampled_train: Vec<&(NaiveDateTime, f64)> =
        train_data.iter().step_by(step).cloned().collect();

    // Prepare training data as BTreeMap
    let data: BTreeMap<NaiveDateTime, BigDecimal> = sampled_train
        .iter()
        .map(|(ts, price)| (*ts, BigDecimal::from_f64(*price).unwrap()))
        .collect();

    // Debug info
    let prices: Vec<f64> = sampled_train.iter().map(|(_, p)| *p).collect();
    let min_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "Training data: {} points (sampled from {}), price range: {:.6e} - {:.6e}",
        sampled_train.len(),
        train_data.len(),
        min_price,
        max_price,
    );

    // Calculate median interval to determine horizon
    let timestamps: Vec<_> = sampled_train.iter().map(|(ts, _)| *ts).collect();
    let median_interval = if timestamps.len() >= 2 {
        let mut intervals: Vec<i64> = timestamps
            .windows(2)
            .map(|w| (w[1] - w[0]).num_seconds())
            .collect();
        intervals.sort();
        intervals[intervals.len() / 2]
    } else {
        3600 // Default to 1 hour
    };

    // Run prediction with horizon = 1 step (using the median interval)
    let input = PredictionInput {
        data,
        horizon: TimeDelta::seconds(median_interval),
    };

    let result = match predict(&input) {
        Ok(r) => r,
        Err(e) => {
            panic!("Prediction failed: {:?}", e);
        }
    };

    let forecast_value: f64 = result.forecast_values[0]
        .to_string()
        .parse()
        .expect("Failed to parse forecast value");

    let mape = compute_mape(forecast_value, actual_value);

    TestResult {
        forecast: forecast_value,
        actual: actual_value,
        mape,
        description: file_data.description,
        data_points_used: train_data.len(),
    }
}

/// Macro to generate individual test functions for each pattern.
/// Tests all files matching the pattern (e.g., "uptrend" tests uptrend-01.json, uptrend-02.json, ...).
macro_rules! real_data_test {
    ($name:ident, $pattern:expr, $threshold:expr) => {
        #[test]
        #[ignore]
        fn $name() {
            let files = find_pattern_files($pattern);
            let mut all_passed = true;

            println!();
            println!("Pattern: {} ({} files)", $pattern, files.len());
            println!("{}", "-".repeat(60));

            for file in &files {
                let result = run_real_data_test(file);
                let passed = result.mape < $threshold;
                let status = if passed { "PASS" } else { "FAIL" };

                println!("  {} ({})", file, result.description);
                println!(
                    "    {} points, MAPE={:.2}% (threshold={:.0}%) [{}]",
                    result.data_points_used, result.mape, $threshold, status
                );

                if !passed {
                    all_passed = false;
                }
            }

            println!();
            assert!(
                all_passed,
                "One or more files in pattern '{}' exceeded MAPE threshold {:.0}%",
                $pattern, $threshold
            );
        }
    };
}

// Generate tests for each pattern with appropriate thresholds
// Thresholds set to ~2.5x current MAPE, rounded to nearest 5%
// Each pattern may have multiple data files (-01, -02, etc.)
real_data_test!(test_uptrend_accuracy, "uptrend", 5.0);
real_data_test!(test_downtrend_accuracy, "downtrend", 15.0);
real_data_test!(test_low_volatility_accuracy, "low_volatility", 10.0);
real_data_test!(test_range_accuracy, "range", 5.0);
real_data_test!(test_gradual_change_accuracy, "gradual_change", 25.0);
real_data_test!(test_double_bottom_accuracy, "double_bottom", 5.0);
real_data_test!(test_high_volatility_accuracy, "high_volatility", 50.0);
real_data_test!(test_spike_up_accuracy, "spike_up", 5.0);
real_data_test!(test_spike_down_accuracy, "spike_down", 15.0);
real_data_test!(test_v_recovery_accuracy, "v_recovery", 30.0);

/// Run all patterns and produce a summary report.
#[test]
#[ignore]
fn test_all_patterns_summary() {
    let patterns = [
        ("uptrend", 5.0),
        ("downtrend", 15.0),
        ("low_volatility", 10.0),
        ("range", 5.0),
        ("gradual_change", 25.0),
        ("double_bottom", 5.0),
        ("high_volatility", 50.0),
        ("spike_up", 5.0),
        ("spike_down", 15.0),
        ("v_recovery", 30.0),
    ];

    println!();
    println!("=== Real Data Prediction Accuracy Summary ===");
    println!();
    println!(
        "{:<25} {:>12} {:>12} {:>10} {:>10} {:>8}",
        "File", "Forecast", "Actual", "MAPE%", "Threshold", "Status"
    );
    println!("{}", "-".repeat(85));

    let mut passed = 0;
    let mut failed = 0;
    let mut total_mape = 0.0;
    let mut total_files = 0;

    for (pattern, threshold) in &patterns {
        let files = find_pattern_files(pattern);

        for file in &files {
            let result = run_real_data_test(file);
            let status = if result.mape < *threshold {
                passed += 1;
                "PASS"
            } else {
                failed += 1;
                "FAIL"
            };
            total_mape += result.mape;
            total_files += 1;

            let name = file.trim_end_matches(".json");
            println!(
                "{:<25} {:>12.4e} {:>12.4e} {:>10.2} {:>10.0} {:>8}",
                name, result.forecast, result.actual, result.mape, threshold, status
            );
        }
    }

    println!("{}", "-".repeat(85));
    println!(
        "Total: {} passed, {} failed, Average MAPE: {:.2}%",
        passed,
        failed,
        total_mape / total_files as f64
    );
    println!();

    assert_eq!(
        failed, 0,
        "{} out of {} files failed their MAPE thresholds",
        failed, total_files
    );
}
