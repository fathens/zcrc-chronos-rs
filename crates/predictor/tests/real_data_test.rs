//! Real-world data accuracy tests.
//!
//! These tests use actual price data patterns (31 days each) to validate
//! prediction accuracy. They are marked `#[ignore]` for manual execution.
//!
//! Run with: cargo test -p predictor --test real_data_test -- --ignored --nocapture

use chrono::NaiveDateTime;
use common::BigDecimal;
use num_traits::FromPrimitive;
use predictor::{predict, PredictionInput};
use serde::Deserialize;
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
    let data = load_real_data(filename);

    // Parse all timestamps and prices
    let parsed: Vec<(NaiveDateTime, f64)> = data
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

    // Prepare training data
    let timestamps: Vec<NaiveDateTime> = sampled_train.iter().map(|(ts, _)| *ts).collect();
    let values: Vec<BigDecimal> = sampled_train
        .iter()
        .map(|(_, price)| BigDecimal::from_f64(*price).unwrap())
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

    // Run prediction with horizon=1 (one time step ahead)
    let input = PredictionInput {
        timestamps,
        values,
        horizon: 1,
        time_budget_secs: Some(120.0),
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
        description: data.description,
        data_points_used: train_data.len(),
    }
}

/// Macro to generate individual test functions for each pattern.
macro_rules! real_data_test {
    ($name:ident, $file:expr, $threshold:expr) => {
        #[test]
        #[ignore]
        fn $name() {
            let result = run_real_data_test($file);
            println!();
            println!("Pattern: {}", $file);
            println!("Description: {}", result.description);
            println!("Data points used: {}", result.data_points_used);
            println!(
                "Forecast: {:.6e}, Actual: {:.6e}",
                result.forecast, result.actual
            );
            println!("MAPE: {:.2}%", result.mape);
            println!();
            assert!(
                result.mape < $threshold,
                "MAPE {:.2}% exceeds threshold {:.2}% for {}",
                result.mape,
                $threshold,
                $file
            );
        }
    };
}

// Generate tests for each pattern with appropriate thresholds
// Stable patterns: tighter thresholds
real_data_test!(test_uptrend_accuracy, "uptrend.json", 50.0);
real_data_test!(test_downtrend_accuracy, "downtrend.json", 50.0);
real_data_test!(test_low_volatility_accuracy, "low_volatility.json", 30.0);
real_data_test!(test_range_accuracy, "range.json", 30.0);
real_data_test!(test_gradual_change_accuracy, "gradual_change.json", 50.0);
real_data_test!(test_double_bottom_accuracy, "double_bottom.json", 50.0);

// Volatile patterns: looser thresholds
real_data_test!(test_high_volatility_accuracy, "high_volatility.json", 100.0);
real_data_test!(test_spike_up_accuracy, "spike_up.json", 100.0);
real_data_test!(test_spike_down_accuracy, "spike_down.json", 100.0);
real_data_test!(test_v_recovery_accuracy, "v_recovery.json", 100.0);

/// Run all patterns and produce a summary report.
#[test]
#[ignore]
fn test_all_patterns_summary() {
    let patterns = [
        ("uptrend.json", 50.0),
        ("downtrend.json", 50.0),
        ("low_volatility.json", 30.0),
        ("range.json", 30.0),
        ("gradual_change.json", 50.0),
        ("double_bottom.json", 50.0),
        ("high_volatility.json", 100.0),
        ("spike_up.json", 100.0),
        ("spike_down.json", 100.0),
        ("v_recovery.json", 100.0),
    ];

    println!();
    println!("=== Real Data Prediction Accuracy Summary ===");
    println!();
    println!(
        "{:<20} {:>12} {:>12} {:>10} {:>10} {:>8}",
        "Pattern", "Forecast", "Actual", "MAPE%", "Threshold", "Status"
    );
    println!("{}", "-".repeat(80));

    let mut passed = 0;
    let mut failed = 0;
    let mut total_mape = 0.0;

    for (filename, threshold) in &patterns {
        let result = run_real_data_test(filename);
        let status = if result.mape < *threshold {
            passed += 1;
            "PASS"
        } else {
            failed += 1;
            "FAIL"
        };
        total_mape += result.mape;

        let name = filename.trim_end_matches(".json");
        println!(
            "{:<20} {:>12.4e} {:>12.4e} {:>10.2} {:>10.0} {:>8}",
            name, result.forecast, result.actual, result.mape, threshold, status
        );
    }

    println!("{}", "-".repeat(80));
    println!(
        "Total: {} passed, {} failed, Average MAPE: {:.2}%",
        passed,
        failed,
        total_mape / patterns.len() as f64
    );
    println!();

    assert_eq!(
        failed,
        0,
        "{} out of {} patterns failed their MAPE thresholds",
        failed,
        patterns.len()
    );
}
