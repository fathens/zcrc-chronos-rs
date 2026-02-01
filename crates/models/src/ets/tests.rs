use super::*;
use chrono::NaiveDate;
use common::ForecastModel;

fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::hours(i as i64))
        .collect()
}

/// Deterministic LCG noise in [-amplitude, amplitude].
fn lcg_noise(seed: u64, n: usize, amplitude: f64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let frac = ((state >> 33) as f64) / (u32::MAX as f64);
            (frac * 2.0 - 1.0) * amplitude
        })
        .collect()
}

/// MASE: MAE / seasonal-naive MAE.
fn compute_mase(forecast: &[f64], actual: &[f64], train: &[f64], season: usize) -> f64 {
    let mae: f64 = forecast
        .iter()
        .zip(actual)
        .map(|(f, a)| (f - a).abs())
        .sum::<f64>()
        / forecast.len() as f64;

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

#[test]
fn test_ets_linear_trend() {
    let mut model = EtsModel::new(None);
    let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 2.0).collect();
    let ts = make_timestamps(50);
    let output = model.fit_predict(&values, &ts, 5).unwrap();
    assert_eq!(output.mean.len(), 5);
    // Should predict increasing values
    assert!(output.mean[0] > values.last().copied().unwrap_or(0.0) - 10.0);
    assert!(output.lower_quantile.is_some());
    assert!(output.upper_quantile.is_some());
}

#[test]
fn test_ets_insufficient_data() {
    let mut model = EtsModel::new(None);
    let result = model.fit_predict(&[1.0, 2.0], &make_timestamps(2), 3);
    assert!(result.is_err());
}

#[test]
fn test_ets_constant_series() {
    let mut model = EtsModel::new(None);
    let values = vec![100.0; 30];
    let ts = make_timestamps(30);
    let output = model.fit_predict(&values, &ts, 5).unwrap();
    assert_eq!(output.mean.len(), 5);
    // Constant series → predictions should be near 100
    for v in &output.mean {
        assert!((*v - 100.0).abs() < 10.0, "Expected ~100, got {}", v);
    }
}

#[test]
fn test_ets_seasonal_additive() {
    let n = 120;
    let m = 12;
    let values: Vec<f64> = (0..n)
        .map(|i| 500.0 + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / m as f64).sin())
        .collect();
    let ts = make_timestamps(n);
    let mut model = EtsModel::new(Some(m));
    let output = model.fit_predict(&values, &ts, 12).unwrap();
    assert_eq!(output.mean.len(), 12);
    assert!(output.lower_quantile.is_some());
    assert!(output.upper_quantile.is_some());
}

#[test]
fn test_ets_seasonal_with_trend() {
    let n = 120;
    let m = 12;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            100.0 + 2.0 * i as f64 + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / m as f64).sin()
        })
        .collect();
    let ts = make_timestamps(n);
    let mut model = EtsModel::new(Some(m));
    let output = model.fit_predict(&values, &ts, 12).unwrap();
    assert_eq!(output.mean.len(), 12);
    // Should forecast upward trend
    let last_train = values.last().copied().unwrap();
    let forecast_mean: f64 = output.mean.iter().sum::<f64>() / output.mean.len() as f64;
    assert!(
        forecast_mean > last_train - 50.0,
        "Forecast mean ({:.1}) should be near or above last value ({:.1})",
        forecast_mean,
        last_train
    );
}

#[test]
fn test_ets_seasonal_insufficient_cycles_falls_back_to_nonseasonal() {
    let m = 12;
    let values = vec![1.0; 20]; // Less than 2*12=24
    let ts = make_timestamps(20);
    let mut model = EtsModel::new(Some(m));
    // Should fall back to non-seasonal model instead of failing
    let result = model.fit_predict(&values, &ts, 5);
    assert!(
        result.is_ok(),
        "Expected fallback to non-seasonal, got error"
    );
    let output = result.unwrap();
    assert_eq!(output.mean.len(), 5);
}

// -----------------------------------------------------------------------
// MASE accuracy tests: ETS with Holt-Winters should beat seasonal naive
// -----------------------------------------------------------------------

#[test]
fn test_ets_seasonal_mase_additive() {
    let m = 12;
    let n = 200;
    let horizon = 12;
    let noise = lcg_noise(77, n + horizon, 8.0);
    let full: Vec<f64> = (0..n + horizon)
        .map(|i| 500.0 + 50.0 * (2.0 * std::f64::consts::PI * i as f64 / m as f64).sin() + noise[i])
        .collect();
    let train = &full[..n];
    let actual = &full[n..];
    let ts = make_timestamps(n);

    let mut model = EtsModel::new(Some(m));
    let output = model.fit_predict(train, &ts, horizon).unwrap();
    let mase = compute_mase(&output.mean, actual, train, m);
    assert!(
        mase < 1.0,
        "Additive seasonal MASE = {mase:.3}, expected < 1.0 (beat seasonal naive)"
    );
}

#[test]
fn test_ets_seasonal_mase_trend_plus_seasonal() {
    let m = 12;
    let n = 200;
    let horizon = 12;
    let noise = lcg_noise(88, n + horizon, 5.0);
    let full: Vec<f64> = (0..n + horizon)
        .map(|i| {
            100.0
                + 1.5 * i as f64
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / m as f64).sin()
                + noise[i]
        })
        .collect();
    let train = &full[..n];
    let actual = &full[n..];
    let ts = make_timestamps(n);

    let mut model = EtsModel::new(Some(m));
    let output = model.fit_predict(train, &ts, horizon).unwrap();
    let mase = compute_mase(&output.mean, actual, train, m);
    assert!(
        mase < 1.0,
        "Trend+seasonal MASE = {mase:.3}, expected < 1.0 (beat seasonal naive)"
    );
}

#[test]
fn test_ets_seasonal_mase_multiplicative() {
    let m = 12;
    let n = 200;
    let horizon = 12;
    let noise = lcg_noise(99, n + horizon, 5.0);
    let full: Vec<f64> = (0..n + horizon)
        .map(|i| {
            let base = 500.0 + 0.5 * i as f64;
            let ratio = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * i as f64 / m as f64).sin();
            base * ratio + noise[i]
        })
        .collect();
    let train = &full[..n];
    let actual = &full[n..];
    let ts = make_timestamps(n);

    let mut model = EtsModel::new(Some(m));
    let output = model.fit_predict(train, &ts, horizon).unwrap();
    let mase = compute_mase(&output.mean, actual, train, m);
    assert!(
        mase < 1.5,
        "Multiplicative seasonal MASE = {mase:.3}, expected < 1.5"
    );
}
