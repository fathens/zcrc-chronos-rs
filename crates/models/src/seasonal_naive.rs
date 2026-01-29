use chrono::NaiveDateTime;
use common::{ForecastModel, ForecastOutput, ModelCategory, Result, ChronosError};
use tracing::debug;

/// SeasonalNaive model: repeats the last seasonal cycle as forecast.
///
/// If no season_length is provided, defaults to the data length
/// (equivalent to a naive "repeat last values" approach).
pub struct SeasonalNaiveModel {
    season_length: Option<usize>,
}

impl SeasonalNaiveModel {
    pub fn new(season_length: Option<usize>) -> Self {
        Self { season_length }
    }
}

impl ForecastModel for SeasonalNaiveModel {
    fn name(&self) -> &str {
        "SeasonalNaive"
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
        if values.is_empty() {
            return Err(ChronosError::InsufficientData(
                "SeasonalNaive requires at least 1 data point".into(),
            ));
        }

        let n = values.len();
        let period = self.season_length.unwrap_or(n).min(n);

        debug!(period = period, horizon = horizon, "SeasonalNaive forecasting");

        // Take the last `period` values and cycle them
        let last_cycle = &values[n.saturating_sub(period)..];
        let mean: Vec<f64> = (0..horizon)
            .map(|i| last_cycle[i % last_cycle.len()])
            .collect();

        Ok(ForecastOutput {
            mean,
            lower_quantile: None,
            upper_quantile: None,
            model_name: "SeasonalNaive".into(),
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
    fn test_seasonal_naive_repeats_cycle() {
        let mut model = SeasonalNaiveModel::new(Some(3));
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let ts = make_timestamps(6);
        let output = model.fit_predict(&values, &ts, 6).unwrap();
        // Last 3 values: [40, 50, 60], repeated twice
        assert_eq!(output.mean, vec![40.0, 50.0, 60.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn test_seasonal_naive_no_period() {
        let mut model = SeasonalNaiveModel::new(None);
        let values = vec![1.0, 2.0, 3.0];
        let ts = make_timestamps(3);
        let output = model.fit_predict(&values, &ts, 6).unwrap();
        // Period = data length (3), so repeats [1, 2, 3]
        assert_eq!(output.mean, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_seasonal_naive_single_point() {
        let mut model = SeasonalNaiveModel::new(Some(5));
        let values = vec![42.0];
        let ts = make_timestamps(1);
        let output = model.fit_predict(&values, &ts, 3).unwrap();
        assert_eq!(output.mean, vec![42.0, 42.0, 42.0]);
    }

    #[test]
    fn test_seasonal_naive_empty() {
        let mut model = SeasonalNaiveModel::new(None);
        let result = model.fit_predict(&[], &[], 3);
        assert!(result.is_err());
    }
}
