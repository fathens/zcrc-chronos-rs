use augurs::prelude::*;
use chrono::NaiveDateTime;
use chronos_core::{ChronosError, ForecastModel, ForecastOutput, ModelCategory, Result};
use tracing::debug;

/// Theta model: decomposes the series into two "theta lines" and
/// combines ETS(A,A,N) on the modified series with a linear trend.
///
/// Simplified implementation using augurs ETS as the base.
/// The standard Theta method applies theta=0 (linear) and theta=2 (amplified).
pub struct ThetaModel;

impl ThetaModel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ThetaModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ForecastModel for ThetaModel {
    fn name(&self) -> &str {
        "Theta"
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Fast
    }

    fn fit_predict(
        &mut self,
        values: &[f64],
        _timestamps: &[NaiveDateTime],
        horizon: usize,
    ) -> Result<ForecastOutput> {
        if values.len() < 3 {
            return Err(ChronosError::InsufficientData(
                "Theta requires at least 3 data points".into(),
            ));
        }

        let n = values.len();
        debug!(data_length = n, horizon = horizon, "Theta model fitting");

        // Step 1: Compute second differences (theta line decomposition)
        // theta_line(0) = linear trend (no curvature)
        // theta_line(2) = 2x curvature amplification

        // Linear trend component (theta=0): simple linear regression
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let (slope, intercept) = simple_linreg(&x, values);

        // Theta=2 line: amplify curvature by modifying differences
        let theta2_values: Vec<f64> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let linear = slope * i as f64 + intercept;
                2.0 * v - linear
            })
            .collect();

        // Step 2: Fit ETS(A,N,N) on theta=2 line.
        // Fall back to naive (last value) if ETS cannot fit (e.g. constant data).
        let ets_points: Option<Vec<f64>> = (|| -> Option<Vec<f64>> {
            let auto = augurs::ets::AutoETS::new(1, "ZZN").ok()?;
            let fitted = auto.fit(&theta2_values).ok()?;
            let forecast = fitted.predict(horizon, None).ok()?;
            Some(forecast.point)
        })();

        let theta2_forecast = ets_points.unwrap_or_else(|| {
            // Fallback: repeat last theta2 value
            let last = *theta2_values.last().unwrap();
            vec![last; horizon]
        });

        // Step 3: Combine theta=0 (linear extrapolation) and theta=2 (ETS forecast)
        let mean: Vec<f64> = (0..horizon)
            .map(|h| {
                let linear = slope * (n + h) as f64 + intercept;
                (linear + theta2_forecast[h]) / 2.0
            })
            .collect();

        let lower: Option<Vec<f64>> = None;
        let upper: Option<Vec<f64>> = None;

        Ok(ForecastOutput {
            mean,
            lower_quantile: lower,
            upper_quantile: upper,
            model_name: "Theta".into(),
        })
    }
}

fn simple_linreg(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|a| a * a).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return (0.0, sum_y / n);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
        let base = NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        (0..n)
            .map(|i| base + chrono::Duration::hours(i as i64))
            .collect()
    }

    #[test]
    fn test_theta_linear_trend() {
        let mut model = ThetaModel::new();
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 3.0).collect();
        let ts = make_timestamps(50);
        let output = model.fit_predict(&values, &ts, 5).unwrap();
        assert_eq!(output.mean.len(), 5);
        // Theta on linear data should extrapolate the trend
        assert!(output.mean[0] > 150.0);
    }

    #[test]
    fn test_theta_insufficient_data() {
        let mut model = ThetaModel::new();
        let result = model.fit_predict(&[1.0, 2.0], &make_timestamps(2), 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_theta_constant_series() {
        let mut model = ThetaModel::new();
        let values = vec![50.0; 30];
        let ts = make_timestamps(30);
        let output = model.fit_predict(&values, &ts, 5).unwrap();
        assert_eq!(output.mean.len(), 5);
        for v in &output.mean {
            assert!((*v - 50.0).abs() < 10.0, "Expected ~50, got {}", v);
        }
    }
}
