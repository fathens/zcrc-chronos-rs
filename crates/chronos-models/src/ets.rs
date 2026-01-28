use augurs::prelude::*;
use chrono::NaiveDateTime;
use chronos_core::{ChronosError, ForecastModel, ForecastOutput, ModelCategory, Result};
use tracing::debug;

/// ETS model wrapper around augurs AutoETS.
///
/// Uses AutoETS with "ZZN" spec (auto error, auto trend, no seasonality)
/// for non-seasonal data, or "ZZZ" when a season_length is provided.
pub struct EtsModel {
    season_length: Option<usize>,
}

impl EtsModel {
    pub fn new(season_length: Option<usize>) -> Self {
        Self { season_length }
    }
}

impl ForecastModel for EtsModel {
    fn name(&self) -> &str {
        "ETS"
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
                "ETS requires at least 3 data points".into(),
            ));
        }

        let (season_len, spec) = match self.season_length {
            Some(s) if s > 1 => (s, "ZZZ"),
            _ => (1, "ZZN"),
        };

        debug!(
            season_length = season_len,
            spec = spec,
            horizon = horizon,
            data_length = values.len(),
            "ETS fitting"
        );

        let auto = augurs::ets::AutoETS::new(season_len, spec)
            .map_err(|e| ChronosError::ModelError(format!("ETS init: {e}")))?;

        let fitted = auto
            .fit(values)
            .map_err(|e| ChronosError::ModelError(format!("ETS fit: {e}")))?;

        // Predict with 90% confidence interval (matching Python's 0.1/0.9 quantiles)
        let forecast = fitted
            .predict(horizon, 0.80)
            .map_err(|e| ChronosError::ModelError(format!("ETS predict: {e}")))?;

        let lower = forecast.intervals.as_ref().map(|iv| iv.lower.clone());
        let upper = forecast.intervals.as_ref().map(|iv| iv.upper.clone());

        Ok(ForecastOutput {
            mean: forecast.point,
            lower_quantile: lower,
            upper_quantile: upper,
            model_name: "ETS".into(),
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
            assert!(
                (*v - 100.0).abs() < 10.0,
                "Expected ~100, got {}",
                v
            );
        }
    }
}
