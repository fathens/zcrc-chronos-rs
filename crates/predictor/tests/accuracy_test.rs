//! Accuracy integration tests: verify the full prediction pipeline
//! produces forecasts within acceptable MASE thresholds.

use chrono::NaiveDate;
use chrono::NaiveDateTime;
use chrono::TimeDelta;
use common::BigDecimal;
use num_traits::FromPrimitive;
use std::collections::BTreeMap;

fn make_data(values: &[f64]) -> BTreeMap<NaiveDateTime, BigDecimal> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let ts = base + TimeDelta::hours(i as i64);
            let val = BigDecimal::from_f64(v).unwrap();
            (ts, val)
        })
        .collect()
}

/// Compute MASE for the pipeline result against known test values.
fn compute_mase(forecast: &[f64], actual: &[f64], train: &[f64], season: usize) -> f64 {
    let season = season.max(1);
    let mae: f64 = forecast
        .iter()
        .zip(actual)
        .map(|(f, a)| (f - a).abs())
        .sum::<f64>()
        / forecast.len() as f64;

    if train.len() <= season {
        return f64::INFINITY;
    }

    let naive_mae: f64 = train
        .iter()
        .skip(season)
        .zip(train.iter())
        .map(|(curr, prev)| (curr - prev).abs())
        .sum::<f64>()
        / (train.len() - season) as f64;

    if naive_mae < 1e-15 {
        return f64::INFINITY;
    }
    mae / naive_mae
}

fn run_pipeline(train_values: &[f64], horizon: usize) -> Vec<f64> {
    use predictor::{predict, PredictionInput};

    let input = PredictionInput {
        data: make_data(train_values),
        horizon: TimeDelta::hours(horizon as i64),
    };

    let result = predict(&input).expect("pipeline should succeed");
    result
        .forecast_values
        .values()
        .map(|v| v.to_string().parse::<f64>().unwrap())
        .collect()
}

#[test]
fn test_pure_trend_accuracy() {
    let n = 200;
    let horizon = 20;
    let full: Vec<f64> = (0..n + horizon).map(|i| 100.0 + 2.5 * i as f64).collect();
    let train = &full[..n];
    let actual = &full[n..];

    let forecast = run_pipeline(train, horizon);
    let mase = compute_mase(&forecast, actual, train, 1);
    assert!(mase < 0.5, "Pure trend MASE = {mase:.3}, expected < 0.5");
}

#[test]
fn test_trend_plus_seasonal_accuracy() {
    let n = 200;
    let horizon = 12;
    let period = 12;
    // Add slight noise to prevent degenerate MASE denominator
    let mut state: u64 = 77;
    let full: Vec<f64> = (0..n + horizon)
        .map(|i| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
            100.0
                + 1.5 * i as f64
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                + noise * 2.0
        })
        .collect();
    let train = &full[..n];
    let actual = &full[n..];

    let forecast = run_pipeline(train, horizon);
    let mase = compute_mase(&forecast, actual, train, period);
    // Trend + seasonal with noise: ETS handles well, but ensemble may vary.
    assert!(
        mase < 2.0,
        "Trend+seasonal MASE = {mase:.3}, expected < 2.0"
    );
}

#[test]
fn test_pure_seasonal_accuracy() {
    let n = 200;
    let horizon = 12;
    let period = 12;
    // Add moderate noise so MASE denominator is non-degenerate
    let mut state: u64 = 55;
    let full: Vec<f64> = (0..n + horizon)
        .map(|i| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
            500.0
                + 50.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                + noise * 8.0
        })
        .collect();
    let train = &full[..n];
    let actual = &full[n..];

    let forecast = run_pipeline(train, horizon);
    let mase = compute_mase(&forecast, actual, train, period);
    // Pure seasonal with noise: MASE depends on noise level and period detection.
    // Threshold reflects realistic ensemble performance with filtering.
    assert!(mase < 7.0, "Pure seasonal MASE = {mase:.3}, expected < 7.0");
}

#[test]
fn test_multi_seasonal_accuracy() {
    let n = 200;
    let horizon = 12;
    let period = 12; // primary period for MASE scaling
    let mut state: u64 = 33;
    let full: Vec<f64> = (0..n + horizon)
        .map(|i| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
            500.0
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
                + 15.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin()
                + noise * 5.0
        })
        .collect();
    let train = &full[..n];
    let actual = &full[n..];

    let forecast = run_pipeline(train, horizon);
    let mase = compute_mase(&forecast, actual, train, period);
    assert!(
        mase < 3.0,
        "Multi-seasonal MASE = {mase:.3}, expected < 3.0"
    );
}

#[test]
fn test_stationary_noise_accuracy() {
    // Stationary noise: predictions should be near the mean.
    // MASE threshold is relaxed since noise is inherently unpredictable.
    let n = 200;
    let horizon = 10;
    // Deterministic pseudo-random noise
    let mut state: u64 = 42;
    let full: Vec<f64> = (0..n + horizon)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let frac = ((state >> 33) as f64) / (u32::MAX as f64);
            100.0 + (frac * 2.0 - 1.0) * 10.0
        })
        .collect();
    let train = &full[..n];
    let actual = &full[n..];

    let forecast = run_pipeline(train, horizon);
    let mase = compute_mase(&forecast, actual, train, 1);
    assert!(
        mase < 1.2,
        "Stationary noise MASE = {mase:.3}, expected < 1.2"
    );
}
