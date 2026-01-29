/// A complete set of forecast accuracy metrics.
#[derive(Debug, Clone)]
pub struct MetricSet {
    /// Mean Absolute Error
    pub mae: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error (0–∞)
    pub mape: f64,
    /// Symmetric MAPE (0–200%)
    pub smape: f64,
    /// Mean Absolute Scaled Error (primary metric; scale-independent, M4 standard)
    pub mase: f64,
    /// Weighted Absolute Percentage Error
    pub wape: f64,
}

/// Compute a full set of accuracy metrics.
///
/// # Arguments
/// * `forecast` – predicted values
/// * `actual` – ground-truth values (same length as `forecast`)
/// * `train_values` – historical values used for fitting (needed for MASE denominator)
/// * `season` – seasonal period for MASE naive scaling (1 = non-seasonal)
///
/// # Panics
/// Panics if `forecast` and `actual` have different lengths or are empty.
pub fn compute_metrics(
    forecast: &[f64],
    actual: &[f64],
    train_values: &[f64],
    season: usize,
) -> MetricSet {
    assert_eq!(
        forecast.len(),
        actual.len(),
        "forecast and actual must have the same length"
    );
    assert!(!forecast.is_empty(), "forecast must not be empty");

    let n = forecast.len() as f64;

    // MAE
    let mae = forecast
        .iter()
        .zip(actual)
        .map(|(f, a)| (f - a).abs())
        .sum::<f64>()
        / n;

    // RMSE
    let rmse = (forecast
        .iter()
        .zip(actual)
        .map(|(f, a)| (f - a).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    // MAPE (skip zeros in actual to avoid Inf)
    let (mape_sum, mape_count) = forecast
        .iter()
        .zip(actual)
        .fold((0.0, 0u64), |(s, c), (f, a)| {
            if a.abs() > 1e-15 {
                (s + ((f - a) / a).abs(), c + 1)
            } else {
                (s, c)
            }
        });
    let mape = if mape_count > 0 {
        mape_sum / mape_count as f64 * 100.0
    } else {
        f64::INFINITY
    };

    // sMAPE (0–200%)
    let (smape_sum, smape_count) =
        forecast
            .iter()
            .zip(actual)
            .fold((0.0, 0u64), |(s, c), (f, a)| {
                let denom = f.abs() + a.abs();
                if denom > 1e-15 {
                    (s + 2.0 * (f - a).abs() / denom, c + 1)
                } else {
                    (s, c)
                }
            });
    let smape = if smape_count > 0 {
        smape_sum / smape_count as f64 * 100.0
    } else {
        0.0
    };

    // MASE
    let mase = compute_mase(forecast, actual, train_values, season);

    // WAPE (= total |error| / total |actual|)
    let abs_error_sum: f64 = forecast
        .iter()
        .zip(actual)
        .map(|(f, a)| (f - a).abs())
        .sum();
    let abs_actual_sum: f64 = actual.iter().map(|a| a.abs()).sum();
    let wape = if abs_actual_sum > 1e-15 {
        abs_error_sum / abs_actual_sum * 100.0
    } else {
        f64::INFINITY
    };

    MetricSet {
        mae,
        rmse,
        mape,
        smape,
        mase,
        wape,
    }
}

/// Compute MASE: MAE of forecast / MAE of seasonal-naive in-sample.
///
/// Denominator = mean |y_t - y_{t-season}| over the training set.
/// A MASE < 1 means the forecast beats the seasonal naive benchmark.
fn compute_mase(forecast: &[f64], actual: &[f64], train_values: &[f64], season: usize) -> f64 {
    let season = season.max(1);

    // In-sample naive forecast error
    if train_values.len() <= season {
        // Not enough training data to compute naive error – fall back to simple scale
        let abs_actual: f64 = actual.iter().map(|a| a.abs()).sum();
        let mean_abs = abs_actual / actual.len() as f64;
        return if mean_abs > 1e-15 {
            let mae: f64 = forecast
                .iter()
                .zip(actual)
                .map(|(f, a)| (f - a).abs())
                .sum::<f64>()
                / actual.len() as f64;
            mae / mean_abs
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

    let naive_mae = naive_errors.iter().sum::<f64>() / naive_errors.len() as f64;

    // If naive MAE is effectively zero (pure seasonal/constant data),
    // MASE is undefined. Use a reasonable floor to avoid explosion.
    // 1e-6 is chosen because smaller naive errors indicate near-perfect
    // seasonal patterns where MASE comparison is meaningless.
    if naive_mae < 1e-6 {
        // Near-perfect seasonal/constant series – cannot scale meaningfully
        return f64::INFINITY;
    }

    let mae: f64 = forecast
        .iter()
        .zip(actual)
        .map(|(f, a)| (f - a).abs())
        .sum::<f64>()
        / forecast.len() as f64;

    mae / naive_mae
}

/// Lightweight MAE computation (for use in trainer evaluation).
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

/// Lightweight MASE computation (for use in trainer evaluation).
pub fn mase(forecast: &[f64], actual: &[f64], train_values: &[f64], season: usize) -> f64 {
    compute_mase(forecast, actual, train_values, season)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_forecast() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let train = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let m = compute_metrics(&actual, &actual, &train, 1);
        assert_eq!(m.mae, 0.0);
        assert_eq!(m.rmse, 0.0);
        assert_eq!(m.smape, 0.0);
        assert_eq!(m.mase, 0.0);
    }

    #[test]
    fn test_known_values() {
        let forecast = vec![2.0, 4.0, 6.0];
        let actual = vec![1.0, 3.0, 5.0];
        let train = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let m = compute_metrics(&forecast, &actual, &train, 1);
        // MAE = (1+1+1)/3 = 1.0
        assert!((m.mae - 1.0).abs() < 1e-10);
        // RMSE = sqrt((1+1+1)/3) = 1.0
        assert!((m.rmse - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mase_above_one_means_worse_than_naive() {
        // Train: 0, 10, 0, 10, 0, 10 (period=2, naive_mae=10)
        let train = vec![0.0, 10.0, 0.0, 10.0, 0.0, 10.0];
        let actual = vec![0.0, 10.0];
        // Forecast is far off
        let forecast = vec![20.0, -10.0];
        let m = compute_metrics(&forecast, &actual, &train, 2);
        assert!(m.mase > 1.0, "MASE should be > 1 for bad forecast");
    }

    #[test]
    fn test_smape_bounded() {
        let a = vec![100.0];
        let f1 = vec![110.0];
        let f2 = vec![90.0];
        let train = vec![95.0, 100.0, 105.0, 100.0];
        let m1 = compute_metrics(&f1, &a, &train, 1);
        let m2 = compute_metrics(&f2, &a, &train, 1);
        // sMAPE is bounded 0–200% and both overpredict/underpredict are penalised
        assert!(m1.smape > 0.0 && m1.smape <= 200.0);
        assert!(m2.smape > 0.0 && m2.smape <= 200.0);
        // Both should be similar magnitude for equal absolute error
        assert!((m1.smape - m2.smape).abs() < 2.0);
    }
}
