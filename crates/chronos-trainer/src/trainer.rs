use std::collections::HashMap;
use std::time::Instant;

use chrono::NaiveDateTime;
use chronos_core::{
    ChronosError, ForecastModel, ForecastOutput, ModelSelectionStrategy,
    ModelTrainingResult, Result, TimeAllocation,
};
use chronos_models::{EtsModel, NptsModel, SeasonalNaiveModel, ThetaModel};
use chronos_selector::AdaptiveModelSelector;
use tracing::{debug, info, warn};

/// Port of Python's `HierarchicalTrainer`.
///
/// Orchestrates staged model training: fast → medium → advanced.
/// Uses early stopping when improvement plateaus.
pub struct HierarchicalTrainer {
    early_stopping_threshold: f64,
    min_score_for_stop: f64,
    training_results: HashMap<String, ModelTrainingResult>,
    best_score: f64,
}

impl HierarchicalTrainer {
    pub fn new(early_stopping_threshold: Option<f64>, min_score_for_stop: Option<f64>) -> Self {
        Self {
            early_stopping_threshold: early_stopping_threshold.unwrap_or(0.02),
            min_score_for_stop: min_score_for_stop.unwrap_or(0.01),
            training_results: HashMap::new(),
            best_score: f64::INFINITY,
        }
    }

    /// Run hierarchical training and return the best ensemble forecast.
    pub fn train_hierarchically(
        &mut self,
        values: &[f64],
        timestamps: &[NaiveDateTime],
        strategy: &ModelSelectionStrategy,
        time_budget_secs: f64,
        horizon: usize,
    ) -> Result<(ForecastOutput, TrainingMetadata)> {
        info!("Starting hierarchical training");

        let start = Instant::now();
        self.training_results.clear();
        self.best_score = f64::INFINITY;

        let model_groups = AdaptiveModelSelector::get_hierarchical_model_groups(strategy);
        let time_allocation =
            calculate_time_allocation(&strategy.time_allocation, time_budget_secs);

        let mut all_forecasts: Vec<(ForecastOutput, f64)> = Vec::new(); // (forecast, score)
        let stages = ["fast", "medium", "advanced"];

        for stage in &stages {
            let models = match model_groups.get(*stage) {
                Some(m) if !m.is_empty() => m,
                _ => continue,
            };

            let stage_budget = *time_allocation.get(*stage).unwrap_or(&0.0);
            if stage_budget <= 0.0 {
                continue;
            }

            let elapsed = start.elapsed().as_secs_f64();
            let remaining = time_budget_secs - elapsed;
            if remaining <= 1.0 {
                warn!("Insufficient time budget, stopping hierarchical training");
                break;
            }

            info!(
                stage = *stage,
                models = ?models,
                budget = format!("{:.0}s", stage_budget.min(remaining)),
                "Starting stage"
            );

            let stage_start = Instant::now();
            let stage_forecasts =
                self.train_stage(values, timestamps, models, horizon, &strategy.excluded_models);
            let stage_time = stage_start.elapsed().as_secs_f64();

            for (forecast, score) in &stage_forecasts {
                if *score < self.best_score {
                    self.best_score = *score;
                    info!(
                        stage = *stage,
                        score = format!("{:.4}", score),
                        model = %forecast.model_name,
                        "New best score"
                    );
                }
            }

            // Record stage result
            if let Some((best_f, best_s)) = stage_forecasts
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                self.training_results.insert(
                    stage.to_string(),
                    ModelTrainingResult {
                        model_name: best_f.model_name.clone(),
                        score: *best_s,
                        training_time_secs: stage_time,
                        forecast: Some(best_f.clone()),
                    },
                );
            }

            all_forecasts.extend(stage_forecasts);

            // Early stopping check
            if self.should_stop_early(stage) {
                info!(stage = *stage, "Early stopping triggered");
                break;
            }
        }

        if all_forecasts.is_empty() {
            return Err(ChronosError::ModelError(
                "No models produced a valid forecast".into(),
            ));
        }

        // Ensemble: inverse-MAE weighted average
        let ensemble = inverse_mae_ensemble(&all_forecasts, horizon);

        let total_time = start.elapsed().as_secs_f64();
        let metadata = TrainingMetadata {
            total_training_time: total_time,
            strategy_used: strategy.strategy_name.clone(),
            stages_completed: self
                .training_results
                .keys()
                .cloned()
                .collect(),
            best_overall_score: self.best_score,
            results_summary: self.summarize_results(),
            model_count: all_forecasts.len(),
        };

        info!(
            total_time = format!("{:.1}s", total_time),
            best_score = format!("{:.4}", self.best_score),
            models = all_forecasts.len(),
            "Hierarchical training complete"
        );

        Ok((ensemble, metadata))
    }

    /// Train all models in a stage, returning forecasts with scores.
    fn train_stage(
        &self,
        values: &[f64],
        timestamps: &[NaiveDateTime],
        model_names: &[String],
        horizon: usize,
        excluded_models: &[String],
    ) -> Vec<(ForecastOutput, f64)> {
        let mut results = Vec::new();

        for model_name in model_names {
            if excluded_models.contains(model_name) {
                continue;
            }

            let mut model: Box<dyn ForecastModel> = match create_model(model_name, values) {
                Some(m) => m,
                None => {
                    debug!(model = %model_name, "Unknown model, skipping");
                    continue;
                }
            };

            match model.fit_predict(values, timestamps, horizon) {
                Ok(forecast) => {
                    let score = evaluate_forecast(values, &forecast);
                    debug!(
                        model = %model_name,
                        score = format!("{:.4}", score),
                        "Model training complete"
                    );
                    results.push((forecast, score));
                }
                Err(e) => {
                    warn!(model = %model_name, error = %e, "Model training failed");
                }
            }
        }

        results
    }

    fn should_stop_early(&self, current_stage: &str) -> bool {
        if current_stage == "fast" {
            return false;
        }

        if self.training_results.len() >= 2 {
            let scores: Vec<f64> = self.training_results.values().map(|r| r.score).collect();
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);

            if max_score > 0.0 {
                let improvement = (max_score - min_score) / max_score;
                if improvement < self.early_stopping_threshold {
                    debug!(
                        improvement = format!("{:.3}", improvement),
                        threshold = format!("{:.3}", self.early_stopping_threshold),
                        "Improvement below threshold"
                    );
                    return true;
                }
            }
        }

        if self.best_score < self.min_score_for_stop {
            debug!(
                score = format!("{:.4}", self.best_score),
                threshold = format!("{:.4}", self.min_score_for_stop),
                "Score sufficiently good"
            );
            return true;
        }

        false
    }

    fn summarize_results(&self) -> HashMap<String, StageSummary> {
        self.training_results
            .iter()
            .map(|(stage, result)| {
                (
                    stage.clone(),
                    StageSummary {
                        model_name: result.model_name.clone(),
                        score: result.score,
                        training_time: result.training_time_secs,
                    },
                )
            })
            .collect()
    }

    pub fn reset(&mut self) {
        self.training_results.clear();
        self.best_score = f64::INFINITY;
    }
}

impl Default for HierarchicalTrainer {
    fn default() -> Self {
        Self::new(None, None)
    }
}

/// Metadata from a hierarchical training run.
#[derive(Debug, Clone)]
pub struct TrainingMetadata {
    pub total_training_time: f64,
    pub strategy_used: String,
    pub stages_completed: Vec<String>,
    pub best_overall_score: f64,
    pub results_summary: HashMap<String, StageSummary>,
    pub model_count: usize,
}

#[derive(Debug, Clone)]
pub struct StageSummary {
    pub model_name: String,
    pub score: f64,
    pub training_time: f64,
}

// ---- Helper functions ----

fn calculate_time_allocation(
    allocation: &TimeAllocation,
    total_budget: f64,
) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    map.insert("fast".into(), total_budget * allocation.fast);
    map.insert("medium".into(), total_budget * allocation.medium);
    map.insert("advanced".into(), total_budget * allocation.advanced);
    map
}

/// Create a model instance by name.
fn create_model(name: &str, _values: &[f64]) -> Option<Box<dyn ForecastModel>> {
    match name {
        "SeasonalNaive" => Some(Box::new(SeasonalNaiveModel::new(None))),
        "AutoETS" | "ETS" => Some(Box::new(EtsModel::new(None))),
        "Theta" | "DynamicOptimizedTheta" => Some(Box::new(ThetaModel::new())),
        "NPTS" => Some(Box::new(NptsModel::default())),
        // Models not yet implemented return None
        _ => {
            debug!(model = name, "Model not implemented in Rust, skipping");
            None
        }
    }
}

/// Evaluate forecast quality using a simple metric.
/// Uses coefficient of variation of the predictions relative to the input data.
fn evaluate_forecast(values: &[f64], forecast: &ForecastOutput) -> f64 {
    if forecast.mean.is_empty() || values.is_empty() {
        return 1.0;
    }

    // Use the last N values as a rough "expected" range
    let n = values.len();
    let recent = &values[n.saturating_sub(forecast.mean.len())..];
    if recent.is_empty() {
        return 1.0;
    }

    // MAE between forecast and last-known values (naive benchmark)
    let mae: f64 = forecast
        .mean
        .iter()
        .zip(recent.iter().cycle())
        .map(|(pred, actual)| (pred - actual).abs())
        .sum::<f64>()
        / forecast.mean.len() as f64;

    let mean_abs = recent.iter().map(|v| v.abs()).sum::<f64>() / recent.len() as f64;
    if mean_abs > 1e-8 {
        mae / mean_abs
    } else {
        1.0
    }
}

/// Inverse-MAE weighted ensemble of multiple forecasts.
fn inverse_mae_ensemble(forecasts: &[(ForecastOutput, f64)], horizon: usize) -> ForecastOutput {
    if forecasts.len() == 1 {
        return forecasts[0].0.clone();
    }

    // Weights = 1 / (score + epsilon)
    let weights: Vec<f64> = forecasts.iter().map(|(_, s)| 1.0 / (s + 1e-10)).collect();
    let total_weight: f64 = weights.iter().sum();

    let mut mean = vec![0.0; horizon];
    let mut lower = vec![0.0; horizon];
    let mut upper = vec![0.0; horizon];
    let mut has_intervals = false;

    for (idx, (forecast, _)) in forecasts.iter().enumerate() {
        let w = weights[idx] / total_weight;
        for h in 0..horizon.min(forecast.mean.len()) {
            mean[h] += w * forecast.mean[h];
        }

        if let Some(ref lo) = forecast.lower_quantile {
            has_intervals = true;
            for h in 0..horizon.min(lo.len()) {
                lower[h] += w * lo[h];
            }
        }
        if let Some(ref hi) = forecast.upper_quantile {
            for h in 0..horizon.min(hi.len()) {
                upper[h] += w * hi[h];
            }
        }
    }

    let model_names: Vec<&str> = forecasts.iter().map(|(f, _)| f.model_name.as_str()).collect();

    ForecastOutput {
        mean,
        lower_quantile: if has_intervals { Some(lower) } else { None },
        upper_quantile: if has_intervals { Some(upper) } else { None },
        model_name: format!("Ensemble({})", model_names.join("+")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use chronos_core::TimeAllocation;

    fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
        let base = NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        (0..n)
            .map(|i| base + chrono::Duration::hours(i as i64))
            .collect()
    }

    fn make_strategy() -> ModelSelectionStrategy {
        ModelSelectionStrategy {
            strategy_name: "balanced".into(),
            priority_models: vec![
                "SeasonalNaive".into(),
                "AutoETS".into(),
                "NPTS".into(),
            ],
            excluded_models: vec!["Naive".into()],
            time_allocation: TimeAllocation {
                fast: 0.2,
                medium: 0.5,
                advanced: 0.3,
            },
            preset: "medium_quality".into(),
        }
    }

    #[test]
    fn test_hierarchical_basic() {
        let mut trainer = HierarchicalTrainer::default();
        let values: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let ts = make_timestamps(50);
        let strategy = make_strategy();

        let (forecast, metadata) = trainer
            .train_hierarchically(&values, &ts, &strategy, 60.0, 5)
            .unwrap();

        assert_eq!(forecast.mean.len(), 5);
        assert!(metadata.total_training_time > 0.0);
        assert!(metadata.model_count > 0);
    }

    #[test]
    fn test_hierarchical_constant_data() {
        let mut trainer = HierarchicalTrainer::default();
        let values = vec![42.0; 50];
        let ts = make_timestamps(50);
        let strategy = make_strategy();

        let (forecast, _) = trainer
            .train_hierarchically(&values, &ts, &strategy, 30.0, 3)
            .unwrap();

        assert_eq!(forecast.mean.len(), 3);
        // Constant data → predictions should be near 42
        for v in &forecast.mean {
            assert!(
                (*v - 42.0).abs() < 20.0,
                "Expected ~42, got {}",
                v
            );
        }
    }

    #[test]
    fn test_inverse_mae_ensemble() {
        let f1 = ForecastOutput {
            mean: vec![10.0, 20.0],
            lower_quantile: None,
            upper_quantile: None,
            model_name: "A".into(),
        };
        let f2 = ForecastOutput {
            mean: vec![30.0, 40.0],
            lower_quantile: None,
            upper_quantile: None,
            model_name: "B".into(),
        };

        // Equal scores → equal weights → simple average
        let ensemble = inverse_mae_ensemble(&[(f1, 0.5), (f2, 0.5)], 2);
        assert_eq!(ensemble.mean.len(), 2);
        // With equal weights: (10+30)/2=20, (20+40)/2=30
        assert!((ensemble.mean[0] - 20.0).abs() < 0.01);
        assert!((ensemble.mean[1] - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_inverse_mae_ensemble_weighted() {
        let f1 = ForecastOutput {
            mean: vec![100.0],
            lower_quantile: None,
            upper_quantile: None,
            model_name: "Good".into(),
        };
        let f2 = ForecastOutput {
            mean: vec![200.0],
            lower_quantile: None,
            upper_quantile: None,
            model_name: "Bad".into(),
        };

        // f1 has much better score (0.01) vs f2 (1.0) → f1 dominates
        let ensemble = inverse_mae_ensemble(&[(f1, 0.01), (f2, 1.0)], 1);
        // f1 weight: 1/0.01 = 100, f2 weight: 1/1.0 = 1 → f1 dominates
        assert!(ensemble.mean[0] < 110.0, "Expected ~100, got {}", ensemble.mean[0]);
    }

    #[test]
    fn test_early_stopping() {
        let mut trainer = HierarchicalTrainer::new(Some(0.02), Some(0.005));
        // Simulate: set best_score very low → should trigger early stop
        trainer.best_score = 0.001;
        assert!(trainer.should_stop_early("medium"));
    }

    #[test]
    fn test_create_model_known() {
        let vals = vec![1.0; 10];
        assert!(create_model("SeasonalNaive", &vals).is_some());
        assert!(create_model("ETS", &vals).is_some());
        assert!(create_model("Theta", &vals).is_some());
        assert!(create_model("NPTS", &vals).is_some());
        assert!(create_model("UnknownModel", &vals).is_none());
    }
}
