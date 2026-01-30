/// Compute Mean Absolute Error.
pub fn mae(forecast: &[f64], actual: &[f64]) -> f64 {
    assert_eq!(forecast.len(), actual.len());
    if forecast.is_empty() {
        return 0.0;
    }
    forecast
        .iter()
        .zip(actual)
        .map(|(f, a)| (f - a).abs())
        .sum::<f64>()
        / forecast.len() as f64
}

/// Compute Mean Absolute Scaled Error.
///
/// MASE < 1 means the forecast is better than the seasonal naive baseline.
///
/// * `train_values` – the historical data used for fitting
/// * `season` – seasonal period (1 for non-seasonal)
pub fn mase(forecast: &[f64], actual: &[f64], train_values: &[f64], season: usize) -> f64 {
    assert_eq!(forecast.len(), actual.len());
    let season = season.max(1);

    if train_values.len() <= season {
        let abs_actual: f64 = actual.iter().map(|a| a.abs()).sum();
        let mean_abs = abs_actual / actual.len().max(1) as f64;
        return if mean_abs > 1e-15 {
            mae(forecast, actual) / mean_abs
        } else {
            f64::INFINITY
        };
    }

    let naive_errors: Vec<f64> = train_values
        .iter()
        .skip(season)
        .zip(train_values.iter())
        .map(|(curr, prev)| (curr - prev).abs())
        .collect();

    let naive_mae = naive_errors.iter().sum::<f64>() / naive_errors.len().max(1) as f64;

    if naive_mae < 1e-15 {
        return f64::INFINITY;
    }

    mae(forecast, actual) / naive_mae
}
