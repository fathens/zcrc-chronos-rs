use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

/// Category of a forecast model based on its computational complexity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCategory {
    /// Fast statistical models (SeasonalNaive, AutoETS, ETS, Theta)
    Fast,
    /// Medium complexity models (RecursiveTabular, DirectTabular, NPTS, ARIMA)
    Medium,
    /// Advanced / deep learning models (DeepAR, TFT, PatchTST, TiDE)
    Advanced,
}

/// Trait that all forecast models must implement.
pub trait ForecastModel: Send + Sync {
    /// Returns the model's name.
    fn name(&self) -> &str;

    /// Returns the model's category (speed tier).
    fn category(&self) -> ModelCategory;

    /// Fit on the provided time series and produce a forecast.
    fn fit_predict(
        &mut self,
        values: &[f64],
        timestamps: &[NaiveDateTime],
        horizon: usize,
    ) -> crate::Result<ForecastOutput>;
}

/// Output of a forecast model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastOutput {
    /// Point forecast (mean).
    pub mean: Vec<f64>,
    /// Lower quantile (e.g. 10th percentile).
    pub lower_quantile: Option<Vec<f64>>,
    /// Upper quantile (e.g. 90th percentile).
    pub upper_quantile: Option<Vec<f64>>,
    /// Name of the model that produced this forecast.
    pub model_name: String,
}

/// Characteristics of a time series, as analyzed by the analyzer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimeSeriesCharacteristics {
    pub trend: TrendInfo,
    pub seasonality: SeasonalityInfo,
    pub volatility: f64,
    pub stationarity: StationarityInfo,
    pub frequency: FrequencyInfo,
    pub missing_pattern: MissingPatternInfo,
    pub density: DensityInfo,
    pub outliers: OutlierInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendInfo {
    pub strength: String,
    pub direction: String,
    pub slope: f64,
    pub r_squared: f64,
    pub p_value: f64,
    pub mann_kendall: MannKendallResult,
    /// True if the trend appears exponential (log-linear fit is significantly better).
    #[serde(default)]
    pub is_exponential: bool,
}

impl Default for TrendInfo {
    fn default() -> Self {
        Self {
            strength: "unknown".into(),
            direction: "unknown".into(),
            slope: 0.0,
            r_squared: 0.0,
            p_value: 1.0,
            mann_kendall: MannKendallResult::default(),
            is_exponential: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MannKendallResult {
    pub trend: String,
    pub p_value: f64,
    #[serde(rename = "S")]
    pub s_statistic: i64,
}

impl Default for MannKendallResult {
    fn default() -> Self {
        Self {
            trend: "unknown".into(),
            p_value: 1.0,
            s_statistic: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityInfo {
    pub strength: String,
    pub period: Option<usize>,
    pub score: f64,
    pub dominant_frequency: Option<f64>,
}

impl Default for SeasonalityInfo {
    fn default() -> Self {
        Self {
            strength: "unknown".into(),
            period: None,
            score: 0.0,
            dominant_frequency: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationarityInfo {
    pub is_stationary: Option<bool>,
    pub mean_difference: f64,
    pub variance_ratio: f64,
}

impl Default for StationarityInfo {
    fn default() -> Self {
        Self {
            is_stationary: None,
            mean_difference: 0.0,
            variance_ratio: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyInfo {
    pub estimated: String,
    pub confidence: f64,
    pub median_interval_seconds: f64,
}

impl Default for FrequencyInfo {
    fn default() -> Self {
        Self {
            estimated: "unknown".into(),
            confidence: 0.0,
            median_interval_seconds: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingPatternInfo {
    pub has_gaps: bool,
    pub gap_count: usize,
    pub gap_percentage: f64,
    pub expected_interval: f64,
    pub max_gap: f64,
}

impl Default for MissingPatternInfo {
    fn default() -> Self {
        Self {
            has_gaps: false,
            gap_count: 0,
            gap_percentage: 0.0,
            expected_interval: 0.0,
            max_gap: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityInfo {
    pub regular: bool,
    pub mean_interval: f64,
    pub interval_variance: f64,
    pub coefficient_of_variation: f64,
}

impl Default for DensityInfo {
    fn default() -> Self {
        Self {
            regular: false,
            mean_interval: 0.0,
            interval_variance: 0.0,
            coefficient_of_variation: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierInfo {
    pub count: usize,
    pub percentage: f64,
    pub indices: Vec<usize>,
    pub method: String,
}

impl Default for OutlierInfo {
    fn default() -> Self {
        Self {
            count: 0,
            percentage: 0.0,
            indices: vec![],
            method: "IQR_and_Z-score".into(),
        }
    }
}

/// Model selection strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionStrategy {
    pub strategy_name: String,
    pub priority_models: Vec<String>,
    pub excluded_models: Vec<String>,
    pub time_allocation: TimeAllocation,
    pub preset: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeAllocation {
    pub fast: f64,
    pub medium: f64,
    pub advanced: f64,
}

/// Result of training a single model stage.
#[derive(Debug, Clone)]
pub struct ModelTrainingResult {
    pub model_name: String,
    pub score: f64,
    pub training_time_secs: f64,
    pub forecast: Option<ForecastOutput>,
}
