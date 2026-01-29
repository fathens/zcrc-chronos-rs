use std::collections::HashMap;

use chrono::NaiveDateTime;
use chronos_analyzer::TimeSeriesAnalyzer;
use chronos_core::{
    ModelSelectionStrategy, TimeAllocation, TimeSeriesCharacteristics,
};
use tracing::{debug, info};

/// Port of Python's `AdaptiveModelSelector` class.
pub struct AdaptiveModelSelector {
    analyzer: TimeSeriesAnalyzer,
    small_dataset_threshold: usize,
    large_dataset_threshold: usize,
    strategies: HashMap<String, ModelSelectionStrategy>,
}

impl AdaptiveModelSelector {
    pub fn new(
        small_dataset_threshold: Option<usize>,
        large_dataset_threshold: Option<usize>,
    ) -> Self {
        let small = small_dataset_threshold.unwrap_or(100);
        let large = large_dataset_threshold.unwrap_or(1000);

        Self {
            analyzer: TimeSeriesAnalyzer::new(),
            small_dataset_threshold: small,
            large_dataset_threshold: large,
            strategies: initialize_strategies(),
        }
    }

    /// Analyze data characteristics and select optimal strategy.
    /// Port of `select_optimal_strategy()`.
    pub fn select_optimal_strategy(
        &self,
        values: &[f64],
        timestamps: &[NaiveDateTime],
        _horizon: usize,
        time_budget: u64,
    ) -> ModelSelectionStrategy {
        let characteristics = self.analyzer.analyze(values, timestamps);
        self.select_strategy_from_characteristics(&characteristics, values.len(), time_budget)
    }

    /// Select optimal strategy from pre-computed characteristics.
    ///
    /// Use this when the caller already has `TimeSeriesCharacteristics`
    /// (e.g. to avoid running the analyzer twice).
    pub fn select_strategy_from_characteristics(
        &self,
        characteristics: &TimeSeriesCharacteristics,
        data_size: usize,
        time_budget: u64,
    ) -> ModelSelectionStrategy {
        info!("Selecting optimal strategy based on data characteristics");

        let strategy = self.determine_strategy(characteristics, data_size, time_budget);

        info!(
            strategy = %strategy.strategy_name,
            priority_models = ?strategy.priority_models,
            excluded_models = ?strategy.excluded_models,
            "Strategy selected"
        );

        strategy
    }

    /// Core strategy determination logic.
    /// Port of `_determine_strategy()`.
    fn determine_strategy(
        &self,
        characteristics: &TimeSeriesCharacteristics,
        data_size: usize,
        time_budget: u64,
    ) -> ModelSelectionStrategy {
        // Base strategy from data size
        let base_strategy = if data_size < self.small_dataset_threshold {
            "small_dataset"
        } else if data_size > self.large_dataset_threshold {
            "large_dataset"
        } else {
            "balanced"
        };

        // Score strategies based on characteristics
        let mut strategy_scores: HashMap<&str, f64> = HashMap::new();

        // Seasonality
        if characteristics.seasonality.strength == "strong"
            && characteristics.seasonality.score > 0.3
        {
            strategy_scores.insert(
                "strong_seasonal",
                characteristics.seasonality.score * 2.0,
            );
            debug!("Strong seasonality detected");
        }

        // Trend
        if characteristics.trend.strength == "strong"
            && characteristics.trend.r_squared > 0.7
        {
            strategy_scores.insert(
                "strong_trend",
                characteristics.trend.r_squared * 1.5,
            );
            debug!("Strong trend detected");
        }

        // Volatility
        if characteristics.volatility > 0.15 {
            strategy_scores.insert(
                "high_volatility",
                (characteristics.volatility * 2.0).min(2.0),
            );
            debug!(
                volatility = format!("{:.3}", characteristics.volatility),
                "High volatility detected"
            );
        }

        // Irregular spacing
        if !characteristics.density.regular {
            strategy_scores.insert("irregular", 1.0);
            debug!("Irregular time series detected");
        }

        // Small data constraint: remove complex strategies
        if data_size < 50 {
            strategy_scores.remove("high_volatility");
            strategy_scores.remove("large_dataset");
        }

        // Pick highest-scoring strategy
        let best_strategy = if let Some((&name, _)) = strategy_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            debug!(scores = ?strategy_scores, best = name, "Strategy scores");
            name
        } else {
            debug!(base = base_strategy, "No characteristic-based strategy, using base");
            base_strategy
        };

        let mut strategy = self
            .strategies
            .get(best_strategy)
            .cloned()
            .unwrap_or_else(|| self.strategies["balanced"].clone());

        // Time budget adjustments
        if time_budget < 300 {
            strategy = adjust_for_short_time(&strategy);
        } else if time_budget > 1800 {
            strategy = adjust_for_long_time(&strategy);
        }

        strategy
    }

    /// Get hierarchical model groups from a strategy.
    /// Port of `get_hierarchical_model_groups()`.
    pub fn get_hierarchical_model_groups(
        strategy: &ModelSelectionStrategy,
    ) -> HashMap<String, Vec<String>> {
        let mut fast = Vec::new();
        let mut medium = Vec::new();
        let mut advanced = Vec::new();

        for model in &strategy.priority_models {
            match model.as_str() {
                "SeasonalNaive" | "AutoETS" | "ETS" | "Theta" | "MSTL" => fast.push(model.clone()),
                "RecursiveTabular" | "DirectTabular" | "NPTS" | "ARIMA" => {
                    medium.push(model.clone())
                }
                _ => advanced.push(model.clone()),
            }
        }

        let mut groups = HashMap::new();
        if !fast.is_empty() {
            groups.insert("fast".into(), fast);
        }
        if !medium.is_empty() {
            groups.insert("medium".into(), medium);
        }
        if !advanced.is_empty() {
            groups.insert("advanced".into(), advanced);
        }

        debug!(groups = ?groups, "Hierarchical model groups");
        groups
    }

    /// Estimate model training time in seconds.
    /// Port of `estimate_model_training_time()`.
    pub fn estimate_model_training_time(model_name: &str, data_size: usize) -> f64 {
        let base_time: f64 = match model_name {
            "SeasonalNaive" => 1.0,
            "AutoETS" => 10.0,
            "ETS" => 8.0,
            "Theta" => 5.0,
            "ARIMA" => 15.0,
            "RecursiveTabular" => 30.0,
            "DirectTabular" => 25.0,
            "NPTS" => 20.0,
            "DeepAR" => 60.0,
            "TemporalFusionTransformer" => 120.0,
            "PatchTST" => 90.0,
            "TiDE" => 80.0,
            _ => 30.0,
        };

        let scale_factor = if data_size < 100 {
            1.0
        } else if data_size < 500 {
            1.5
        } else {
            2.0
        };

        base_time * scale_factor
    }
}

impl Default for AdaptiveModelSelector {
    fn default() -> Self {
        Self::new(None, None)
    }
}

/// Initialize all pre-defined strategies.
fn initialize_strategies() -> HashMap<String, ModelSelectionStrategy> {
    let mut strategies = HashMap::new();

    strategies.insert(
        "strong_seasonal".into(),
        ModelSelectionStrategy {
            strategy_name: "strong_seasonal".into(),
            priority_models: vec![
                "SeasonalNaive".into(),
                "AutoETS".into(),
                "MSTL".into(),
                "Theta".into(),
                "NPTS".into(),
            ],
            excluded_models: vec![
                "Naive".into(),
                "Chronos".into(),
                "TemporalFusionTransformer".into(),
            ],
            time_allocation: TimeAllocation {
                fast: 0.15,
                medium: 0.6,
                advanced: 0.25,
            },
            preset: "medium_quality".into(),
        },
    );

    strategies.insert(
        "strong_trend".into(),
        ModelSelectionStrategy {
            strategy_name: "strong_trend".into(),
            priority_models: vec![
                "ARIMA".into(),
                "ETS".into(),
                "RecursiveTabular".into(),
                "NPTS".into(),
                "AutoETS".into(),
            ],
            excluded_models: vec![
                "Naive".into(),
                "SeasonalNaive".into(),
                "Chronos".into(),
            ],
            time_allocation: TimeAllocation {
                fast: 0.1,
                medium: 0.5,
                advanced: 0.4,
            },
            preset: "medium_quality".into(),
        },
    );

    strategies.insert(
        "high_volatility".into(),
        ModelSelectionStrategy {
            strategy_name: "high_volatility".into(),
            priority_models: vec![
                "NPTS".into(),
                "RecursiveTabular".into(),
                "DirectTabular".into(),
                "DeepAR".into(),
            ],
            excluded_models: vec![
                "Naive".into(),
                "SeasonalNaive".into(),
                "Chronos".into(),
            ],
            time_allocation: TimeAllocation {
                fast: 0.1,
                medium: 0.4,
                advanced: 0.5,
            },
            preset: "high_quality".into(),
        },
    );

    strategies.insert(
        "small_dataset".into(),
        ModelSelectionStrategy {
            strategy_name: "small_dataset".into(),
            priority_models: vec![
                "AutoETS".into(),
                "ETS".into(),
                "MSTL".into(),
                "Theta".into(),
                "SeasonalNaive".into(),
            ],
            excluded_models: vec![
                "Naive".into(),
                "Chronos".into(),
                "TemporalFusionTransformer".into(),
                "DeepAR".into(),
            ],
            time_allocation: TimeAllocation {
                fast: 0.3,
                medium: 0.7,
                advanced: 0.0,
            },
            preset: "medium_quality".into(),
        },
    );

    strategies.insert(
        "large_dataset".into(),
        ModelSelectionStrategy {
            strategy_name: "large_dataset".into(),
            priority_models: vec![
                "RecursiveTabular".into(),
                "DirectTabular".into(),
                "NPTS".into(),
                "DeepAR".into(),
                "AutoETS".into(),
            ],
            excluded_models: vec!["Naive".into()],
            time_allocation: TimeAllocation {
                fast: 0.1,
                medium: 0.3,
                advanced: 0.6,
            },
            preset: "high_quality".into(),
        },
    );

    strategies.insert(
        "irregular".into(),
        ModelSelectionStrategy {
            strategy_name: "irregular".into(),
            priority_models: vec![
                "NPTS".into(),
                "RecursiveTabular".into(),
                "AutoETS".into(),
                "DirectTabular".into(),
            ],
            excluded_models: vec![
                "Naive".into(),
                "SeasonalNaive".into(),
                "Chronos".into(),
            ],
            time_allocation: TimeAllocation {
                fast: 0.2,
                medium: 0.5,
                advanced: 0.3,
            },
            preset: "medium_quality".into(),
        },
    );

    strategies.insert(
        "balanced".into(),
        ModelSelectionStrategy {
            strategy_name: "balanced".into(),
            priority_models: vec![
                "AutoETS".into(),
                "MSTL".into(),
                "RecursiveTabular".into(),
                "NPTS".into(),
                "SeasonalNaive".into(),
            ],
            excluded_models: vec![
                "Naive".into(),
                "Chronos".into(),
                "TemporalFusionTransformer".into(),
            ],
            time_allocation: TimeAllocation {
                fast: 0.2,
                medium: 0.5,
                advanced: 0.3,
            },
            preset: "medium_quality".into(),
        },
    );

    strategies
}

/// Adjust strategy for short time budget (<5 min).
fn adjust_for_short_time(strategy: &ModelSelectionStrategy) -> ModelSelectionStrategy {
    debug!("Adjusting strategy for short time budget");

    // Must match the "fast" category in get_hierarchical_model_groups
    let fast_models = ["SeasonalNaive", "AutoETS", "ETS", "Theta", "MSTL"];
    let mut priority: Vec<String> = strategy
        .priority_models
        .iter()
        .filter(|m| fast_models.contains(&m.as_str()))
        .cloned()
        .collect();

    if priority.is_empty() {
        priority = vec!["SeasonalNaive".into(), "AutoETS".into(), "MSTL".into()];
    }

    let mut excluded = strategy.excluded_models.clone();
    for extra in &["RecursiveTabular", "DirectTabular", "DeepAR"] {
        let s = extra.to_string();
        if !excluded.contains(&s) {
            excluded.push(s);
        }
    }

    ModelSelectionStrategy {
        strategy_name: format!("{}_fast", strategy.strategy_name),
        priority_models: priority,
        excluded_models: excluded,
        time_allocation: TimeAllocation {
            fast: 0.6,
            medium: 0.4,
            advanced: 0.0,
        },
        preset: "fast_training".into(),
    }
}

/// Adjust strategy for long time budget (>30 min).
fn adjust_for_long_time(strategy: &ModelSelectionStrategy) -> ModelSelectionStrategy {
    debug!("Adjusting strategy for long time budget");

    let mut extended = strategy.priority_models.clone();
    let candidates = [
        "DeepAR",
        "TemporalFusionTransformer",
        "PatchTST",
        "TiDE",
    ];

    for model in &candidates {
        let s = model.to_string();
        if !strategy.excluded_models.contains(&s) && !extended.contains(&s) {
            extended.push(s);
        }
    }

    let excluded: Vec<String> = strategy
        .excluded_models
        .iter()
        .filter(|m| m.as_str() != "TemporalFusionTransformer")
        .cloned()
        .collect();

    ModelSelectionStrategy {
        strategy_name: format!("{}_extended", strategy.strategy_name),
        priority_models: extended,
        excluded_models: excluded,
        time_allocation: TimeAllocation {
            fast: 0.1,
            medium: 0.3,
            advanced: 0.6,
        },
        preset: "high_quality".into(),
    }
}

#[cfg(test)]
mod tests;
