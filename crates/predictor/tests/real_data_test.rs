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

/// Compute MASE (Mean Absolute Scaled Error).
/// MASE < 1.0 means the forecast is better than naive prediction (using last training value).
fn compute_mase(forecast: f64, actual: f64, last_train: f64) -> f64 {
    let mae = (forecast - actual).abs();
    let naive_mae = (last_train - actual).abs();
    if naive_mae < 1e-15 {
        return f64::INFINITY;
    }
    mae / naive_mae
}

/// Check if the forecast direction matches the actual direction.
/// Direction is relative to the last training value.
fn check_direction(forecast: f64, actual: f64, last_train: f64) -> bool {
    let forecast_direction = forecast - last_train;
    let actual_direction = actual - last_train;
    forecast_direction.signum() == actual_direction.signum()
}

/// Get forecast value at target timestamp, with linear interpolation if needed.
fn get_forecast_at_timestamp(
    forecast_values: &BTreeMap<NaiveDateTime, BigDecimal>,
    target_ts: NaiveDateTime,
) -> Option<f64> {
    // Exact match check
    if let Some(val) = forecast_values.get(&target_ts) {
        return val.to_string().parse().ok();
    }

    // Get values before and after target timestamp
    let before: Option<(&NaiveDateTime, &BigDecimal)> =
        forecast_values.range(..target_ts).next_back();
    let after: Option<(&NaiveDateTime, &BigDecimal)> = forecast_values.range(target_ts..).next();

    match (before, after) {
        (Some((ts1, v1)), Some((ts2, v2))) => {
            // Linear interpolation
            let v1: f64 = v1.to_string().parse().ok()?;
            let v2: f64 = v2.to_string().parse().ok()?;
            let total_secs = (*ts2 - *ts1).num_seconds() as f64;
            let elapsed_secs = (target_ts - *ts1).num_seconds() as f64;
            let ratio = elapsed_secs / total_secs;
            Some(v1 + ratio * (v2 - v1))
        }
        (Some((_, v)), None) | (None, Some((_, v))) => {
            // Out of range: use nearest value
            Some(v.to_string().parse().ok()?)
        }
        (None, None) => None,
    }
}

/// Test result containing forecast, actual value, and accuracy metrics.
struct TestResult {
    forecast: f64,
    actual: f64,
    #[allow(dead_code)]
    last_train: f64,
    mape: f64,
    mase: f64,
    direction_correct: bool,
    #[allow(dead_code)]
    description: String,
    #[allow(dead_code)]
    data_points_used: usize,
}

/// Run a real data test for the given file.
///
/// Strategy:
/// 1. Load all data
/// 2. Use all data except the last 24 hours for training
/// 3. Predict 24 hours ahead
/// 4. Compare forecast vs the last data point
///
/// Note: Data is downsampled to ~720 points for faster test execution.
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

    // Use the last data point as actual value
    let actual_point = parsed.last().unwrap();
    let actual_value = actual_point.1;

    // Cutoff: 24 hours before the last data point
    let cutoff_time = actual_point.0 - chrono::Duration::hours(24);

    // Split data: training (before cutoff)
    let train_data: Vec<&(NaiveDateTime, f64)> =
        parsed.iter().filter(|(ts, _)| *ts < cutoff_time).collect();

    assert!(
        train_data.len() >= 50,
        "Need at least 50 training data points, got {}",
        train_data.len()
    );

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

    // Get the last training value (for MASE and direction calculations)
    let last_train_value = sampled_train.last().unwrap().1;

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

    // Run prediction with horizon = 24 hours to cover the actual data point
    let input = PredictionInput {
        data,
        horizon: TimeDelta::hours(24),
    };

    let result = match predict(&input) {
        Ok(r) => r,
        Err(e) => {
            panic!("Prediction failed: {:?}", e);
        }
    };

    // Get forecast value at the actual data point's timestamp (with interpolation)
    let forecast_value = get_forecast_at_timestamp(&result.forecast_values, actual_point.0)
        .expect("Could not get forecast at target timestamp");

    let mape = compute_mape(forecast_value, actual_value);
    let mase = compute_mase(forecast_value, actual_value, last_train_value);
    let direction_correct = check_direction(forecast_value, actual_value, last_train_value);

    TestResult {
        forecast: forecast_value,
        actual: actual_value,
        last_train: last_train_value,
        mape,
        mase,
        direction_correct,
        description: file_data.description,
        data_points_used: train_data.len(),
    }
}

/// Macro to generate individual test functions for each pattern.
/// Tests all files matching the pattern (e.g., "uptrend" tests uptrend-01.json, uptrend-02.json, ...).
///
/// Parameters:
/// - $name: Test function name
/// - $pattern: Pattern name (e.g., "uptrend")
/// - $mape_threshold: Maximum allowed MAPE percentage
/// - $mase_threshold: Maximum allowed MASE (inf values always pass)
/// - $min_dir_correct: Minimum number of files that must have correct direction
macro_rules! real_data_test {
    ($name:ident, $pattern:expr, $mape_threshold:expr, $mase_threshold:expr, $min_dir_correct:expr) => {
        #[test]
        #[ignore]
        fn $name() {
            let files = find_pattern_files($pattern);
            let mut all_passed = true;
            let mut direction_correct_count = 0usize;
            let mut failure_reasons: Vec<String> = Vec::new();

            println!();
            println!("Pattern: {} ({} files)", $pattern, files.len());
            println!("{}", "-".repeat(90));
            println!(
                "{:<20} {:>8} {:>8} {:>5} {:>12} {:>12} {:>8}",
                "File", "MAPE%", "MASE", "Dir", "MAPE Thr", "MASE Thr", "Status"
            );
            println!("{}", "-".repeat(90));

            for file in &files {
                let result = run_real_data_test(file);
                let mape_ok = result.mape < $mape_threshold;
                let mase_ok = !result.mase.is_finite() || result.mase < $mase_threshold;
                let passed = mape_ok && mase_ok;
                let status = if passed { "PASS" } else { "FAIL" };
                let dir_symbol = if result.direction_correct { "Y" } else { "N" };

                if result.direction_correct {
                    direction_correct_count += 1;
                }

                let name = file.trim_end_matches(".json");
                let mase_str = if result.mase.is_finite() {
                    format!("{:.2}", result.mase)
                } else {
                    "-".to_string()
                };
                println!(
                    "{:<20} {:>8.2} {:>8} {:>5} {:>12.0}% {:>12.1} {:>8}",
                    name,
                    result.mape,
                    mase_str,
                    dir_symbol,
                    $mape_threshold,
                    $mase_threshold,
                    status
                );

                if !passed {
                    all_passed = false;
                    if !mape_ok {
                        failure_reasons.push(format!(
                            "{}: MAPE {:.2}% > {:.0}%",
                            name, result.mape, $mape_threshold
                        ));
                    }
                    if !mase_ok {
                        failure_reasons.push(format!(
                            "{}: MASE {:.2} > {:.1}",
                            name, result.mase, $mase_threshold
                        ));
                    }
                }
            }

            // Check direction accuracy threshold
            #[allow(unused_comparisons)]
            let dir_ok = direction_correct_count >= $min_dir_correct;
            println!();
            println!(
                "Direction: {}/{} correct (minimum: {})",
                direction_correct_count,
                files.len(),
                $min_dir_correct
            );

            if !dir_ok {
                failure_reasons.push(format!(
                    "Direction accuracy: {}/{} < {} required",
                    direction_correct_count,
                    files.len(),
                    $min_dir_correct
                ));
            }

            println!();
            assert!(
                all_passed && dir_ok,
                "Pattern '{}' failed:\n  {}",
                $pattern,
                failure_reasons.join("\n  ")
            );
        }
    };
}

// Generate tests for each pattern with appropriate thresholds
// Parameters: (name, pattern, mape_threshold, mase_threshold, min_direction_correct)
// Each pattern may have multiple data files (-01, -02, etc.)
//
// Thresholds:
// - MAPE: Maximum allowed percentage error
// - MASE: Maximum allowed scaled error (inf values always pass)
// - Direction: Minimum number of files with correct direction prediction
real_data_test!(test_uptrend_accuracy, "uptrend", 5.0, 2.0, 3);
real_data_test!(test_downtrend_accuracy, "downtrend", 15.0, 2.0, 3);
real_data_test!(test_low_volatility_accuracy, "low_volatility", 10.0, 2.0, 4);
real_data_test!(test_range_accuracy, "range", 5.0, 2.0, 2);
real_data_test!(test_gradual_change_accuracy, "gradual_change", 25.0, 2.0, 2);
real_data_test!(test_double_bottom_accuracy, "double_bottom", 5.0, 2.0, 2);
real_data_test!(
    test_high_volatility_accuracy,
    "high_volatility",
    30.0,
    15.0,
    0
);
real_data_test!(test_spike_up_accuracy, "spike_up", 5.0, 2.0, 4);
real_data_test!(test_spike_down_accuracy, "spike_down", 1500.0, 2.0, 0);
real_data_test!(test_v_recovery_accuracy, "v_recovery", 30.0, 2.0, 3);

/// Run all patterns and produce a summary report.
#[test]
#[ignore]
fn test_all_patterns_summary() {
    // (pattern, mape_threshold, mase_threshold, min_direction_correct)
    let patterns: [(&str, f64, f64, usize); 10] = [
        ("uptrend", 5.0, 2.0, 3),
        ("downtrend", 15.0, 2.0, 3),
        ("low_volatility", 10.0, 2.0, 4),
        ("range", 5.0, 2.0, 2),
        ("gradual_change", 25.0, 2.0, 2),
        ("double_bottom", 5.0, 2.0, 2),
        ("high_volatility", 30.0, 15.0, 0),
        ("spike_up", 5.0, 2.0, 4),
        ("spike_down", 1500.0, 2.0, 0),
        ("v_recovery", 30.0, 2.0, 3),
    ];

    println!();
    println!("=== Real Data Prediction Accuracy Summary ===");
    println!();
    println!(
        "{:<20} {:>12} {:>12} {:>8} {:>8} {:>5} {:>8} {:>8} {:>8}",
        "File", "Forecast", "Actual", "MAPE%", "MASE", "Dir", "MAPE Th", "MASE Th", "Status"
    );
    println!("{}", "-".repeat(105));

    let mut passed = 0;
    let mut failed = 0;
    let mut total_mape = 0.0;
    let mut total_mase = 0.0;
    let mut finite_mase_count = 0;
    let mut total_direction_correct = 0;
    let mut total_files = 0;
    let mut pattern_failures: Vec<String> = Vec::new();

    for (pattern, mape_threshold, mase_threshold, min_dir_correct) in &patterns {
        let files = find_pattern_files(pattern);
        let mut pattern_dir_correct = 0usize;

        for file in &files {
            let result = run_real_data_test(file);
            let mape_ok = result.mape < *mape_threshold;
            let mase_ok = !result.mase.is_finite() || result.mase < *mase_threshold;
            let file_passed = mape_ok && mase_ok;

            if file_passed {
                passed += 1;
            } else {
                failed += 1;
            }
            total_mape += result.mape;
            if result.mase.is_finite() {
                total_mase += result.mase;
                finite_mase_count += 1;
            }
            if result.direction_correct {
                total_direction_correct += 1;
                pattern_dir_correct += 1;
            }
            total_files += 1;

            let name = file.trim_end_matches(".json");
            let dir_symbol = if result.direction_correct { "Y" } else { "N" };
            let mase_str = if result.mase.is_finite() {
                format!("{:.2}", result.mase)
            } else {
                "-".to_string()
            };
            let status = if file_passed { "PASS" } else { "FAIL" };
            println!(
                "{:<20} {:>12.4e} {:>12.4e} {:>8.2} {:>8} {:>5} {:>8.0}% {:>8.1} {:>8}",
                name,
                result.forecast,
                result.actual,
                result.mape,
                mase_str,
                dir_symbol,
                mape_threshold,
                mase_threshold,
                status
            );
        }

        // Check direction threshold for this pattern
        if pattern_dir_correct < *min_dir_correct {
            pattern_failures.push(format!(
                "{}: Direction {}/{} < {} required",
                pattern,
                pattern_dir_correct,
                files.len(),
                min_dir_correct
            ));
        }
    }

    println!("{}", "-".repeat(105));
    let direction_accuracy = total_direction_correct as f64 / total_files as f64 * 100.0;
    let avg_mase = if finite_mase_count > 0 {
        total_mase / finite_mase_count as f64
    } else {
        0.0
    };
    println!(
        "Total: {} passed, {} failed | Avg MAPE: {:.2}% | Avg MASE: {:.2} | Direction: {:.0}% ({}/{})",
        passed,
        failed,
        total_mape / total_files as f64,
        avg_mase,
        direction_accuracy,
        total_direction_correct,
        total_files
    );
    println!();

    let all_passed = failed == 0 && pattern_failures.is_empty();
    assert!(
        all_passed,
        "Test failures:\n  - {} file(s) failed MAPE/MASE thresholds\n  - {}",
        failed,
        if pattern_failures.is_empty() {
            "All direction thresholds met".to_string()
        } else {
            pattern_failures.join("\n  - ")
        }
    );
}
