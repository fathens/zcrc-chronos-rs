use augurs::mstl::MSTLModel;
use augurs::prelude::*;
use chrono::NaiveDateTime;
use chronos_core::{ChronosError, ForecastModel, ForecastOutput, ModelCategory, Result};
use tracing::debug;

/// MSTL (Multiple Seasonal-Trend decomposition using Loess) model.
///
/// Decomposes the series into trend + multiple seasonal components + remainder,
/// then forecasts each component (trend via AutoETS, seasonal via repetition).
pub struct MstlEtsModel {
    periods: Vec<usize>,
}

impl MstlEtsModel {
    /// Create a new MSTL model with the given seasonal periods.
    ///
    /// If `periods` is empty or `None`, a single period of 1 is used
    /// (effectively non-seasonal MSTL, which degrades to ETS on trend).
    pub fn new(periods: Option<Vec<usize>>) -> Self {
        let periods = periods
            .unwrap_or_default()
            .into_iter()
            .filter(|&p| p > 1)
            .collect::<Vec<_>>();
        Self { periods }
    }
}

impl ForecastModel for MstlEtsModel {
    fn name(&self) -> &str {
        "MSTL"
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
                "MSTL requires at least 3 data points".into(),
            ));
        }

        // Periods must be < n/2 for valid STL decomposition
        let n = values.len();
        let valid_periods: Vec<usize> = self
            .periods
            .iter()
            .copied()
            .filter(|&p| p > 1 && p < n / 2)
            .collect();

        if valid_periods.is_empty() {
            // No valid seasonal periods – fall back to plain ETS
            debug!(
                data_length = n,
                "MSTL: no valid periods, falling back to ETS"
            );
            let trend_model = augurs::ets::AutoETS::new(1, "ZZN")
                .map_err(|e| ChronosError::ModelError(format!("MSTL ETS init: {e}")))?;
            let fitted = trend_model
                .fit(values)
                .map_err(|e| ChronosError::ModelError(format!("MSTL ETS fit: {e}")))?;
            let forecast = fitted
                .predict(horizon, 0.80)
                .map_err(|e| ChronosError::ModelError(format!("MSTL ETS predict: {e}")))?;

            return Ok(ForecastOutput {
                mean: forecast.point,
                lower_quantile: forecast.intervals.as_ref().map(|iv| iv.lower.clone()),
                upper_quantile: forecast.intervals.as_ref().map(|iv| iv.upper.clone()),
                model_name: "MSTL".into(),
            });
        }

        debug!(
            periods = ?valid_periods,
            data_length = n,
            horizon = horizon,
            "MSTL fitting"
        );

        // Use AutoETS as the trend model for MSTL decomposition
        let trend_model = augurs::ets::AutoETS::new(1, "ZZN")
            .map_err(|e| ChronosError::ModelError(format!("MSTL ETS init: {e}")))?
            .into_trend_model();

        let mstl = MSTLModel::new(valid_periods, trend_model);

        let fitted = mstl
            .fit(values)
            .map_err(|e| ChronosError::ModelError(format!("MSTL fit: {e}")))?;

        let forecast = fitted
            .predict(horizon, 0.80)
            .map_err(|e| ChronosError::ModelError(format!("MSTL predict: {e}")))?;

        Ok(ForecastOutput {
            mean: forecast.point,
            lower_quantile: forecast.intervals.as_ref().map(|iv| iv.lower.clone()),
            upper_quantile: forecast.intervals.as_ref().map(|iv| iv.upper.clone()),
            model_name: "MSTL".into(),
        })
    }
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
    fn test_mstl_seasonal_data() {
        let n = 120;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                500.0
                    + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
                    + 1.0 * i as f64
            })
            .collect();
        let ts = make_timestamps(n);
        let mut model = MstlEtsModel::new(Some(vec![12]));
        let output = model.fit_predict(&values, &ts, 12).unwrap();
        assert_eq!(output.mean.len(), 12);
        assert!(output.lower_quantile.is_some());
        assert!(output.upper_quantile.is_some());
    }

    #[test]
    fn test_mstl_no_valid_periods_falls_back() {
        let n = 30;
        let values: Vec<f64> = (0..n).map(|i| 100.0 + 2.0 * i as f64).collect();
        let ts = make_timestamps(n);
        // Period 50 > n/2, so invalid → falls back to plain ETS
        let mut model = MstlEtsModel::new(Some(vec![50]));
        let output = model.fit_predict(&values, &ts, 5).unwrap();
        assert_eq!(output.mean.len(), 5);
    }

    #[test]
    fn test_mstl_empty_periods() {
        let n = 50;
        let values: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let ts = make_timestamps(n);
        let mut model = MstlEtsModel::new(None);
        let output = model.fit_predict(&values, &ts, 5).unwrap();
        assert_eq!(output.mean.len(), 5);
    }

    #[test]
    fn test_mstl_insufficient_data() {
        let mut model = MstlEtsModel::new(Some(vec![12]));
        let result = model.fit_predict(&[1.0, 2.0], &make_timestamps(2), 3);
        assert!(result.is_err());
    }
}
