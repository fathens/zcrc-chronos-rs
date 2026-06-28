use augurs::prelude::*;
use chrono::NaiveDateTime;
use common::{ChronosError, ForecastModel, ForecastOutput, ModelCategory, Result};
use tracing::debug;

use crate::ets::ETS_SPEC;

/// Damping coefficient for the theta=0 (linear) component of
/// [`ThetaModel`].
///
/// The classical Theta line uses an undamped linear extrapolation
/// `slope × (n + h)` for the theta-0 component. On low-SNR crypto data
/// this produces over-confident multi-day forecasts that the production
/// `top_decile_decomposition` diagnostic identified as the dominant
/// "high-confidence wrong-direction" tail (Theta accounted for 47/108
/// of the TOP decile pullers at h = 168 h on 1084 NEAR-token snapshots).
///
/// Replacing the linear extrapolation with a damped trend bounds the
/// total deviation: the cumulative trend `slope × Σᵢ₌₁ʰ⁺¹ φⁱ` is
/// monotone in `h` but asymptotes at `slope × φ / (1 − φ)` as
/// `h → ∞`, so the model never extrapolates more than `φ / (1 − φ)`
/// steps' worth of slope no matter how long the horizon. With
/// `φ = 0.97` the asymptote is ~32 steps of slope; at `h = 168` the
/// damped sum is ~32, versus 168 for the undamped extrapolation
/// (≈ 5× attenuation on a one-week-ahead forecast).
///
/// `φ` is intentionally a `pub(crate)` constant so the production team
/// can grid-search it via a patched build until `predict_sweep`
/// `decile_spread` at `h = 168` crosses zero.
pub(crate) const THETA_DAMPING_PHI: f64 = 0.97;

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
            let auto = augurs::ets::AutoETS::new(1, ETS_SPEC).ok()?;
            let fitted = auto.fit(&theta2_values).ok()?;
            let forecast = fitted.predict(horizon, None).ok()?;
            Some(forecast.point)
        })();

        let theta2_forecast = ets_points.unwrap_or_else(|| {
            // Fallback: repeat last theta2 value
            let last = *theta2_values.last().unwrap();
            vec![last; horizon]
        });

        // Step 3: Combine theta=0 (damped linear extrapolation) and
        // theta=2 (ETS forecast). The damped trend bounds the cumulative
        // extrapolation at slope × φ / (1 − φ) regardless of horizon —
        // see THETA_DAMPING_PHI for the rationale.
        let phi = THETA_DAMPING_PHI;
        let linear_last_fit = slope * (n as f64 - 1.0) + intercept;
        let mean: Vec<f64> = (0..horizon)
            .map(|h| {
                // Σᵢ₌₁ʰ⁺¹ φⁱ = φ × (1 − φʰ⁺¹) / (1 − φ) for φ ≠ 1, else h+1.
                let h_plus_1 = (h + 1) as f64;
                let damped_steps = if (phi - 1.0).abs() < 1e-12 {
                    h_plus_1
                } else {
                    phi * (1.0 - phi.powi((h + 1) as i32)) / (1.0 - phi)
                };
                let linear = linear_last_fit + slope * damped_steps;
                (linear + theta2_forecast[h]) / 2.0
            })
            .collect();

        // Step 4: Residual-based prediction intervals (80% level)
        // Compute in-sample fitted values and residuals
        let in_sample_fitted: Vec<f64> = (0..n)
            .map(|i| {
                let linear = slope * i as f64 + intercept;
                (linear + theta2_values[i]) / 2.0
            })
            .collect();

        let residuals: Vec<f64> = values
            .iter()
            .zip(in_sample_fitted.iter())
            .map(|(actual, fitted)| actual - fitted)
            .collect();

        let residual_std = {
            let mean_r = residuals.iter().sum::<f64>() / n as f64;
            let var = residuals.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>()
                / (n as f64 - 1.0).max(1.0);
            var.sqrt()
        };

        // z_{0.90} = 1.2816 for 80% prediction interval (10th–90th percentile)
        let z = 1.2816;
        let lower: Vec<f64> = (0..horizon)
            .map(|h| mean[h] - z * residual_std * ((h + 1) as f64).sqrt())
            .collect();
        let upper: Vec<f64> = (0..horizon)
            .map(|h| mean[h] + z * residual_std * ((h + 1) as f64).sqrt())
            .collect();

        Ok(ForecastOutput {
            mean,
            lower_quantile: Some(lower),
            upper_quantile: Some(upper),
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
    fn test_theta_damped_trend_asymptotes_at_long_horizon() {
        // On a clean linear trend the theta=0 component would, without
        // damping, predict slope × h additional movement at step h. The
        // damped formula bounds the cumulative trend at
        // slope × φ / (1 − φ). For φ = 0.97 the asymptote is ~32 steps
        // of slope, so a 200-step-ahead forecast must NOT continue the
        // trend linearly.
        let mut model = ThetaModel::new();
        // Clean linear trend with no noise: slope = 1.
        let values: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let ts = make_timestamps(50);
        // The theta-2 component on a perfectly linear series will inherit
        // the same trend, so the blended forecast still grows. The damped
        // theta=0 contribution alone should asymptote, so the gap between
        // two distant horizon points stays bounded.
        let output = model.fit_predict(&values, &ts, 200).unwrap();
        assert_eq!(output.mean.len(), 200);
        // The theta-0 contribution to mean[199] over what an undamped
        // extrapolation would give: undamped_slope × 200 = 200, damped
        // sum (asymptotic) ≈ 32. The blend halves both contributions, so
        // mean[199] should be at least slope × 200 / 2 = 100 below the
        // undamped Theta extrapolation. We assert the looser bound that
        // the forecast does not run away linearly.
        let last_train = *values.last().unwrap(); // 149
        let undamped_extrapolation = last_train + 200.0;
        assert!(
            output.mean[199] < undamped_extrapolation,
            "damped theta forecast {} must not exceed undamped extrapolation {}",
            output.mean[199],
            undamped_extrapolation
        );
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
