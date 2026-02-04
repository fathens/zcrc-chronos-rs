use chrono::NaiveDateTime;
use common::{ChronosError, ForecastModel, ForecastOutput, ModelCategory, Result};
use rayon::prelude::*;
use scaler::{Scaler, StandardScaler};
use tracing::debug;

/// Minimum number of candidate windows to trigger parallel computation.
/// Below this threshold, sequential iteration is more efficient.
/// Set to 1000 based on benchmarks showing parallel overhead dominates for smaller datasets.
const PARALLEL_THRESHOLD: usize = 1000;

/// NPTS (Non-Parametric Time Series) model.
///
/// K-nearest-neighbor forecasting: finds similar subsequences in history
/// and uses their subsequent values as forecasts. Weights by distance.
#[derive(Default)]
pub struct NptsModel {
    /// User-specified K value. If None, K is determined adaptively based on data size.
    k: Option<usize>,
}

impl NptsModel {
    pub fn new(k: Option<usize>) -> Self {
        Self { k }
    }

    /// Compute adaptive K based on candidate count.
    /// Uses sqrt(candidates) as baseline, clamped to [3, 15].
    fn adaptive_k(&self, candidate_count: usize) -> usize {
        if let Some(k) = self.k {
            return k;
        }
        let k = (candidate_count as f64).sqrt().round() as usize;
        k.clamp(3, 15)
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

        // Normalize values for distance calculation
        let mut scaler = StandardScaler::new();
        let normalized = scaler.fit_transform(values)?;

        // Use the last `context_len` values as the query pattern
        let context_len = horizon.min(n / 2).max(1);
        let query = &normalized[n - context_len..];

        debug!(
            k = ?self.k,
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

        // Compute distances for all valid windows (on normalized values)
        // Use parallel iteration for large datasets, sequential for small ones
        let candidate_count = max_start + 1;
        let mut candidates: Vec<(usize, f64)> = if candidate_count >= PARALLEL_THRESHOLD {
            (0..candidate_count)
                .into_par_iter()
                .map(|start| {
                    let window = &normalized[start..start + context_len];
                    let dist = squared_euclidean_distance(query, window);
                    (start, dist)
                })
                .collect()
        } else {
            let mut cands = Vec::with_capacity(candidate_count);
            for start in 0..candidate_count {
                let window = &normalized[start..start + context_len];
                let dist = squared_euclidean_distance(query, window);
                cands.push((start, dist));
            }
            cands
        };

        // Partial sort: only need top K (O(n) vs O(n log n) for full sort)
        let k = self.adaptive_k(candidate_count).min(candidates.len());
        if k > 0 && k < candidates.len() {
            candidates.select_nth_unstable_by(k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let top_k = &candidates[..k];

        // Inverse-distance weighting (apply sqrt here since we stored squared distances)
        // Compute total weight first, then normalize inline to avoid Vec allocation
        let total_weight: f64 = top_k
            .iter()
            .map(|(_, d_sq)| 1.0 / (d_sq.sqrt() + 1e-10))
            .sum();

        // Weighted average of the subsequent values (on normalized values)
        let mut mean_normalized = vec![0.0; horizon];
        for &(start, d_sq) in top_k {
            let forecast_start = start + context_len;
            let w = (1.0 / (d_sq.sqrt() + 1e-10)) / total_weight;
            for (h, mean_val) in mean_normalized.iter_mut().enumerate() {
                let source_idx = forecast_start + h;
                let val = if source_idx < n {
                    normalized[source_idx]
                } else {
                    // If we run out of future values, use the last available
                    normalized[n - 1]
                };
                *mean_val += w * val;
            }
        }

        // Inverse transform to original scale
        let mean = scaler.inverse_transform(&mean_normalized)?;

        Ok(ForecastOutput {
            mean,
            lower_quantile: None,
            upper_quantile: None,
            model_name: "NPTS".into(),
        })
    }
}

/// Squared Euclidean distance (omits sqrt for K-NN ranking efficiency).
/// Since sqrt is monotonic, relative ordering is preserved without it.
fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
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

    /// Test that NPTS produces scale-invariant relative predictions.
    /// Without scaling, at tiny scales the epsilon (1e-10) in weight calculation
    /// dominates the actual distances, causing uniform weighting.
    #[test]
    fn test_npts_scale_invariance() {
        // Create data where neighbors have different distances (no exact matches)
        // The query pattern slightly differs from all historical patterns
        let mut base_pattern = Vec::new();
        for i in 0..50 {
            // Sinusoidal with slight drift - no exact repeats
            base_pattern.push(100.0 + 10.0 * (i as f64 * 0.3).sin() + i as f64 * 0.1);
        }
        let ts = make_timestamps(base_pattern.len());

        // At scale 1.0: distances are meaningful
        // At scale 1e-11: distances become ~1e-10, same order as epsilon
        let scales = [1.0, 1e-11];
        let mut predictions: Vec<Vec<f64>> = Vec::new();

        for &scale in &scales {
            let scaled: Vec<f64> = base_pattern.iter().map(|v| v * scale).collect();
            let mut model = NptsModel::new(Some(5));
            let output = model.fit_predict(&scaled, &ts, 3).unwrap();
            // Convert to relative scale for comparison
            let rel_pred: Vec<f64> = output.mean.iter().map(|v| v / scale).collect();
            predictions.push(rel_pred);
        }

        // With proper scaling, both should produce identical relative predictions
        let max_diff: f64 = predictions[0]
            .iter()
            .zip(predictions[1].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            max_diff < 0.001,
            "Scale invariance violated: predictions differ by {:.6} (scale1: {:?}, scale1e-11: {:?})",
            max_diff,
            predictions[0],
            predictions[1]
        );
    }

    /// Test NPTS with extremely small values (typical crypto prices).
    #[test]
    fn test_npts_extreme_small_scale() {
        // Simulating NEAR protocol-like prices: ~4e-9
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let base = 4e-9;
                let variation = (i as f64 * 0.1).sin() * 1e-9;
                base + variation
            })
            .collect();
        let ts = make_timestamps(100);

        let mut model = NptsModel::new(Some(5));
        let output = model.fit_predict(&values, &ts, 10).unwrap();

        // Predictions should be in reasonable range
        for pred in &output.mean {
            assert!(
                *pred > 1e-10 && *pred < 1e-7,
                "Prediction {} out of expected range for small scale data",
                pred
            );
        }
    }

    /// Test NPTS with extremely large values.
    #[test]
    fn test_npts_extreme_large_scale() {
        // Large values like Bitcoin market cap
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let base = 1e12;
                let variation = (i as f64 * 0.1).sin() * 1e11;
                base + variation
            })
            .collect();
        let ts = make_timestamps(100);

        let mut model = NptsModel::new(Some(5));
        let output = model.fit_predict(&values, &ts, 10).unwrap();

        // Predictions should be in reasonable range
        for pred in &output.mean {
            assert!(
                *pred > 1e11 && *pred < 1e13,
                "Prediction {} out of expected range for large scale data",
                pred
            );
        }
    }
}
