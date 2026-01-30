use chrono::NaiveDateTime;
use common::{ChronosError, ForecastModel, ForecastOutput, ModelCategory, Result};
use tracing::debug;

/// NPTS (Non-Parametric Time Series) model.
///
/// K-nearest-neighbor forecasting: finds similar subsequences in history
/// and uses their subsequent values as forecasts. Weights by distance.
pub struct NptsModel {
    k: usize,
}

impl NptsModel {
    pub fn new(k: Option<usize>) -> Self {
        Self { k: k.unwrap_or(5) }
    }
}

impl Default for NptsModel {
    fn default() -> Self {
        Self::new(None)
    }
}

impl ForecastModel for NptsModel {
    fn name(&self) -> &str {
        "NPTS"
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Medium
    }

    fn fit_predict(
        &mut self,
        values: &[f64],
        _timestamps: &[NaiveDateTime],
        horizon: usize,
    ) -> Result<ForecastOutput> {
        let n = values.len();
        if n < 3 {
            return Err(ChronosError::InsufficientData(
                "NPTS requires at least 3 data points".into(),
            ));
        }

        // Use the last `context_len` values as the query pattern
        let context_len = horizon.min(n / 2).max(1);
        let query = &values[n - context_len..];

        debug!(
            k = self.k,
            context_len = context_len,
            horizon = horizon,
            "NPTS forecasting"
        );

        // Find K nearest neighbors: subsequences of length context_len
        // that are followed by at least `horizon` values.
        let max_start = n.saturating_sub(context_len + horizon);
        if max_start == 0 {
            // Not enough history → fall back to repeating last values
            let mean: Vec<f64> = (0..horizon).map(|i| values[n - 1 - (i % n)]).collect();
            return Ok(ForecastOutput {
                mean,
                lower_quantile: None,
                upper_quantile: None,
                model_name: "NPTS".into(),
            });
        }

        // Compute distances for all valid windows
        let mut candidates: Vec<(usize, f64)> = (0..=max_start)
            .map(|start| {
                let window = &values[start..start + context_len];
                let dist = euclidean_distance(query, window);
                (start, dist)
            })
            .collect();

        // Sort by distance and take top K
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.k.min(candidates.len());
        let top_k = &candidates[..k];

        // Inverse-distance weighting
        let weights: Vec<f64> = top_k.iter().map(|(_, d)| 1.0 / (d + 1e-10)).collect();
        let total_weight: f64 = weights.iter().sum();

        // Weighted average of the subsequent values
        let mut mean = vec![0.0; horizon];
        for (idx, &(start, _)) in top_k.iter().enumerate() {
            let forecast_start = start + context_len;
            let w = weights[idx] / total_weight;
            for (h, mean_val) in mean.iter_mut().enumerate() {
                let source_idx = forecast_start + h;
                let val = if source_idx < n {
                    values[source_idx]
                } else {
                    // If we run out of future values, use the last available
                    values[n - 1]
                };
                *mean_val += w * val;
            }
        }

        Ok(ForecastOutput {
            mean,
            lower_quantile: None,
            upper_quantile: None,
            model_name: "NPTS".into(),
        })
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
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
    fn test_npts_repeating_pattern() {
        let mut model = NptsModel::new(Some(3));
        // Repeating pattern: should find similar subsequences
        let values: Vec<f64> = (0..60).map(|i| (i % 10) as f64 * 5.0 + 100.0).collect();
        let ts = make_timestamps(60);
        let output = model.fit_predict(&values, &ts, 5).unwrap();
        assert_eq!(output.mean.len(), 5);
    }

    #[test]
    fn test_npts_linear() {
        let mut model = NptsModel::new(None);
        let values: Vec<f64> = (0..50).map(|i| i as f64 * 2.0).collect();
        let ts = make_timestamps(50);
        let output = model.fit_predict(&values, &ts, 3).unwrap();
        assert_eq!(output.mean.len(), 3);
    }

    #[test]
    fn test_npts_insufficient() {
        let mut model = NptsModel::default();
        let result = model.fit_predict(&[1.0, 2.0], &make_timestamps(2), 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_npts_small_dataset() {
        let mut model = NptsModel::new(Some(2));
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let ts = make_timestamps(5);
        let output = model.fit_predict(&values, &ts, 2).unwrap();
        assert_eq!(output.mean.len(), 2);
    }
}
