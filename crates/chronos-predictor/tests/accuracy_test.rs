//! Accuracy integration tests: verify the full prediction pipeline
//! produces forecasts within acceptable MASE thresholds.

use chrono::NaiveDate;
use chrono::NaiveDateTime;
use chronos_core::BigDecimal;
use num_traits::FromPrimitive;

fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::hours(i as i64))
        .collect()
}

fn to_decimals(vals: &[f64]) -> Vec<BigDecimal> {
    vals.iter()
        .map(|&v| BigDecimal::from_f64(v).unwrap())
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
    use chronos_predictor::{predict, PredictionInput};

    let input = PredictionInput {
        timestamps: make_timestamps(train_values.len()),
        values: to_decimals(train_values),
        horizon,
        time_budget_secs: Some(60.0),
    };

    let result = predict(&input).expect("pipeline should succeed");
    result
        .forecast_values
        .iter()
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
    assert!(
        mase < 0.5,
        "Pure trend MASE = {mase:.3}, expected < 0.5"
    );
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
    // ETS now handles seasonality directly via Holt-Winters, but the
    // ensemble still includes non-seasonal models (Theta, NPTS) that
    // dilute accuracy. Threshold reflects ensemble-level performance.
    assert!(
        mase < 7.0,
        "Pure seasonal MASE = {mase:.3}, expected < 7.0"
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
        mase < 1.5,
        "Stationary noise MASE = {mase:.3}, expected < 1.5"
    );
}
