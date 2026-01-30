use serde::{Deserialize, Serialize};

/// Application-level configuration, mirrors config/app_config.yaml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub prediction: PredictionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    #[serde(default = "default_max_concurrent_tasks")]
    pub max_concurrent_tasks: usize,

    #[serde(default)]
    pub adaptive_selection: AdaptiveSelectionConfig,

    #[serde(default)]
    pub hierarchical_training: HierarchicalTrainingConfig,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: default_max_concurrent_tasks(),
            adaptive_selection: AdaptiveSelectionConfig::default(),
            hierarchical_training: HierarchicalTrainingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveSelectionConfig {
    #[serde(default = "default_small_dataset_threshold")]
    pub small_dataset_threshold: usize,
    #[serde(default = "default_large_dataset_threshold")]
    pub large_dataset_threshold: usize,
}

impl Default for AdaptiveSelectionConfig {
    fn default() -> Self {
        Self {
            small_dataset_threshold: default_small_dataset_threshold(),
            large_dataset_threshold: default_large_dataset_threshold(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalTrainingConfig {
    #[serde(default = "default_early_stopping_threshold")]
    pub early_stopping_threshold: f64,
    #[serde(default = "default_min_score_for_stop")]
    pub min_score_for_stop: f64,
    #[serde(default = "default_max_workers")]
    pub max_workers: usize,
}

impl Default for HierarchicalTrainingConfig {
    fn default() -> Self {
        Self {
            early_stopping_threshold: default_early_stopping_threshold(),
            min_score_for_stop: default_min_score_for_stop(),
            max_workers: default_max_workers(),
        }
    }
}

fn default_max_concurrent_tasks() -> usize {
    2
}
fn default_small_dataset_threshold() -> usize {
    100
}
fn default_large_dataset_threshold() -> usize {
    1000
}
fn default_early_stopping_threshold() -> f64 {
    0.02
}
fn default_min_score_for_stop() -> f64 {
    0.01
}
fn default_max_workers() -> usize {
    2
}
