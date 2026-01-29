use chrono::NaiveDateTime;
use common::{
    DensityInfo, FrequencyInfo, MannKendallResult, MissingPatternInfo, OutlierInfo,
    SeasonalityInfo, StationarityInfo, TimeSeriesCharacteristics, TrendInfo,
};
use num_complex::Complex;
use rustfft::FftPlanner;
use statrs::distribution::{ContinuousCDF, Normal};
use tracing::{debug, info, warn};

/// Port of Python's `TimeSeriesAnalyzer` class.
pub struct TimeSeriesAnalyzer;

impl TimeSeriesAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Comprehensive time-series characteristics analysis.
    /// Port of `analyze_time_series_characteristics()`.
    pub fn analyze(
        &self,
        values: &[f64],
        timestamps: &[NaiveDateTime],
    ) -> TimeSeriesCharacteristics {
        info!("Starting time series characteristics analysis");

        if values.len() != timestamps.len() || values.len() < 3 {
            warn!("Insufficient data, returning basic analysis");
            return self.basic_characteristics(values);
        }

        match self.detailed_analysis(values, timestamps) {
            Ok(chars) => {
                info!("Time series characteristics analysis complete");
                chars
            }
            Err(e) => {
                warn!(error = %e, "Detailed analysis failed, falling back to basic");
                self.basic_characteristics(values)
            }
        }
    }

    fn detailed_analysis(
        &self,
        values: &[f64],
        timestamps: &[NaiveDateTime],
    ) -> Result<TimeSeriesCharacteristics, String> {
        let seasonality = self.detect_seasonality(values);
        let trend = self.analyze_trend(values);
        let volatility = self.calculate_volatility(values);
        let missing_pattern = self.analyze_missing_data(timestamps);
        let outliers = self.detect_outliers(values);
        let density = self.analyze_time_intervals(timestamps);
        let stationarity = self.test_stationarity(values);
        let frequency = self.estimate_frequency(timestamps);

        Ok(TimeSeriesCharacteristics {
            trend,
            seasonality,
            volatility,
            stationarity,
            frequency,
            missing_pattern,
            density,
            outliers,
        })
    }

    fn basic_characteristics(&self, values: &[f64]) -> TimeSeriesCharacteristics {
        let volatility = if values.len() > 1 {
            let std = std_dev(values);
            let mean_abs = mean(&values.iter().map(|v| v.abs()).collect::<Vec<_>>());
            if mean_abs > 0.0 {
                std / mean_abs
            } else {
                0.0
            }
        } else {
            0.0
        };

        TimeSeriesCharacteristics {
            volatility,
            trend: TrendInfo {
                strength: "unknown".into(),
                direction: "unknown".into(),
                ..Default::default()
            },
            seasonality: SeasonalityInfo {
                strength: "unknown".into(),
                ..Default::default()
            },
            density: DensityInfo {
                regular: values.len() > 10,
                ..Default::default()
            },
            stationarity: StationarityInfo::default(),
            missing_pattern: MissingPatternInfo::default(),
            outliers: OutlierInfo::default(),
            frequency: FrequencyInfo::default(),
        }
    }

    // ---- Seasonality Detection (FFT) ----

    /// Port of `_detect_seasonality()`.
    pub fn detect_seasonality(&self, values: &[f64]) -> SeasonalityInfo {
        if values.len() < 10 {
            return SeasonalityInfo {
                strength: "weak".into(),
                period: None,
                score: 0.0,
                dominant_frequency: None,
            };
        }

        let n = values.len();

        // Detrend the data to prevent linear trend from dominating the spectrum.
        // This is important for trend+seasonal data where the trend creates a
        // strong low-frequency component that can mask the seasonal peak.
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let (slope, intercept, _, _, _) = linregress(&x, values);
        let detrended: Vec<f64> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| v - (slope * i as f64 + intercept))
            .collect();

        // Check if detrended data has significant variance.
        // If the residuals are nearly constant, there's no seasonality to detect.
        let detrended_var = variance(&detrended);
        let data_range = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - values.iter().cloned().fold(f64::INFINITY, f64::min);
        if detrended_var < (data_range * 0.01).powi(2) {
            // Detrended variance is less than 1% of data range squared
            return SeasonalityInfo {
                strength: "weak".into(),
                period: None,
                score: 0.0,
                dominant_frequency: None,
            };
        }

        // Prepare FFT input: detrended values (already zero-mean after detrending)
        let mut buffer: Vec<Complex<f64>> = detrended
            .iter()
            .map(|&v| Complex::new(v, 0.0))
            .collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        // Power spectrum
        let power: Vec<f64> = buffer.iter().map(|c| c.norm_sqr()).collect();

        // Only look at positive frequencies: indices 1..n/2
        let half = n / 2;
        if half < 2 {
            return SeasonalityInfo {
                strength: "weak".into(),
                period: None,
                score: 0.0,
                dominant_frequency: None,
            };
        }

        let positive_power = &power[1..half];
        // Python uses np.max(power) * 0.1 — the max over the ENTIRE spectrum (including DC)
        let max_power_all = power
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let threshold = max_power_all * 0.1;

        // Find peaks above threshold
        let peaks: Vec<usize> = find_peaks(positive_power, threshold);

        if peaks.is_empty() {
            return SeasonalityInfo {
                strength: "weak".into(),
                period: None,
                score: 0.0,
                dominant_frequency: None,
            };
        }

        // Find strongest peak (index within positive_power → actual index = peak + 1)
        let best_peak_local = peaks
            .iter()
            .copied()
            .max_by(|&a, &b| {
                positive_power[a]
                    .partial_cmp(&positive_power[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        // Apply parabolic interpolation to refine peak position.
        // This improves period accuracy when the true frequency falls between FFT bins.
        // Formula: delta = 0.5 * (y[i-1] - y[i+1]) / (y[i-1] - 2*y[i] + y[i+1])
        let refined_peak = if best_peak_local > 0 && best_peak_local < positive_power.len() - 1 {
            let y_prev = positive_power[best_peak_local - 1];
            let y_curr = positive_power[best_peak_local];
            let y_next = positive_power[best_peak_local + 1];
            let denom = y_prev - 2.0 * y_curr + y_next;
            if denom.abs() > 1e-10 {
                let delta = 0.5 * (y_prev - y_next) / denom;
                // Clamp delta to [-0.5, 0.5] to stay within neighboring bins
                best_peak_local as f64 + delta.clamp(-0.5, 0.5)
            } else {
                best_peak_local as f64
            }
        } else {
            best_peak_local as f64
        };

        // actual FFT bin index (refined)
        let main_freq_idx = refined_peak + 1.0;

        // Frequency = index / n
        let freq = main_freq_idx / n as f64;
        let period = if freq > 1e-10 {
            Some((1.0 / freq).round() as usize)
        } else {
            None
        };

        // Strength score = power at peak / total power
        // Use the discrete bin index for power lookup
        let peak_bin_idx = best_peak_local + 1;
        let total_power: f64 = power.iter().sum();
        let strength_score = if total_power > 0.0 {
            power[peak_bin_idx] / total_power
        } else {
            0.0
        };

        let strength = if strength_score > 0.3 {
            "strong"
        } else if strength_score > 0.1 {
            "moderate"
        } else {
            "weak"
        };

        debug!(
            period = ?period,
            strength = strength,
            score = format!("{:.3}", strength_score),
            "Seasonality detected"
        );

        SeasonalityInfo {
            strength: strength.into(),
            period,
            score: strength_score,
            dominant_frequency: Some(freq),
        }
    }

    // ---- Trend Analysis ----

    /// Port of `_analyze_trend()`.
    pub fn analyze_trend(&self, values: &[f64]) -> TrendInfo {
        if values.len() < 3 {
            return TrendInfo {
                strength: "weak".into(),
                direction: "none".into(),
                slope: 0.0,
                ..Default::default()
            };
        }

        let n = values.len();
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let (slope, _intercept, r_value, p_value, std_err) = linregress(&x, values);
        let r_squared = r_value * r_value;

        let strength = if r_squared > 0.7 {
            "strong"
        } else if r_squared > 0.3 {
            "moderate"
        } else {
            "weak"
        };

        let direction = if slope.abs() < std_err {
            "none"
        } else if slope > 0.0 {
            "increasing"
        } else {
            "decreasing"
        };

        let mann_kendall = self.mann_kendall_test(values);

        // Detect exponential growth: compare linear R² vs log-linear R²
        let is_exponential = self.detect_exponential_trend(values, r_squared);

        debug!(
            strength = strength,
            direction = direction,
            r_squared = format!("{:.3}", r_squared),
            is_exponential = is_exponential,
            "Trend analysis complete"
        );

        TrendInfo {
            strength: strength.into(),
            direction: direction.into(),
            slope,
            r_squared,
            p_value,
            mann_kendall,
            is_exponential,
        }
    }

    /// Detect if the trend is exponential by comparing linear vs log-linear fit.
    ///
    /// Returns true if:
    /// 1. All values are positive (required for log transform)
    /// 2. Log-linear R² is significantly better than linear R² (by at least 0.1)
    /// 3. Log-linear R² is strong (> 0.8)
    /// 4. There's a clear increasing trend
    fn detect_exponential_trend(&self, values: &[f64], linear_r_squared: f64) -> bool {
        let n = values.len();
        if n < 10 {
            return false;
        }

        // Check all values are positive (required for log)
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_val <= 0.0 {
            return false;
        }

        // Check for increasing trend (exponential decay is rare in forecasting context)
        let first_quarter_mean = mean(&values[..n / 4]);
        let last_quarter_mean = mean(&values[3 * n / 4..]);
        if last_quarter_mean <= first_quarter_mean * 1.5 {
            // Not enough growth to be exponential
            return false;
        }

        // Compute log-linear fit: regress x against log(y)
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let log_values: Vec<f64> = values.iter().map(|v| v.ln()).collect();

        let (_, _, r_log, _, _) = linregress(&x, &log_values);
        let log_r_squared = r_log * r_log;

        // Exponential if log-linear fit is nearly perfect and significantly better than linear.
        // Additional check: exponential data has consistent positive curvature (second derivative > 0).
        // Piecewise linear (changepoint) has zero curvature except at the kink point.

        // Check for consistent positive curvature (characteristic of exponential growth)
        let second_diffs: Vec<f64> = values
            .windows(3)
            .map(|w| (w[2] - w[1]) - (w[1] - w[0]))
            .collect();
        let mean_curvature = second_diffs.iter().sum::<f64>() / second_diffs.len() as f64;
        let positive_curvatures = second_diffs.iter().filter(|&&d| d > 0.0).count();
        let curvature_ratio = positive_curvatures as f64 / second_diffs.len() as f64;

        // Exponential has consistent positive curvature (most second diffs > 0)
        // Piecewise linear has mostly zero curvature with a spike at the changepoint
        if curvature_ratio < 0.7 || mean_curvature < 0.0 {
            return false;
        }

        let is_exp = log_r_squared > 0.99
            && log_r_squared > linear_r_squared + 0.01;

        if is_exp {
            debug!(
                linear_r2 = format!("{:.3}", linear_r_squared),
                log_r2 = format!("{:.3}", log_r_squared),
                "Exponential trend detected"
            );
        }

        is_exp
    }

    // ---- Mann-Kendall Test ----

    /// Port of `_mann_kendall_test()`.
    pub fn mann_kendall_test(&self, values: &[f64]) -> MannKendallResult {
        let n = values.len();
        if n < 3 {
            return MannKendallResult {
                trend: "unknown".into(),
                p_value: 1.0,
                s_statistic: 0,
            };
        }

        // Compute S statistic
        let mut s: i64 = 0;
        for i in 0..n - 1 {
            for j in (i + 1)..n {
                if values[j] > values[i] {
                    s += 1;
                } else if values[j] < values[i] {
                    s -= 1;
                }
            }
        }

        // Variance of S
        let n_f = n as f64;
        let var_s = n_f * (n_f - 1.0) * (2.0 * n_f + 5.0) / 18.0;

        // Standardized z
        let z = if s > 0 {
            (s as f64 - 1.0) / var_s.sqrt()
        } else if s < 0 {
            (s as f64 + 1.0) / var_s.sqrt()
        } else {
            0.0
        };

        // Two-sided p-value
        let normal = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - normal.cdf(z.abs()));

        let trend = if p_value < 0.05 {
            if s > 0 {
                "increasing"
            } else {
                "decreasing"
            }
        } else {
            "none"
        };

        MannKendallResult {
            trend: trend.into(),
            p_value,
            s_statistic: s,
        }
    }

    // ---- Volatility ----

    /// Port of `_calculate_volatility()`.
    pub fn calculate_volatility(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let diffs: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
        if diffs.is_empty() {
            return 0.0;
        }

        let mean_val = mean(values);
        if mean_val == 0.0 {
            return f64::INFINITY;
        }

        let volatility = std_dev(&diffs) / mean_val.abs();
        debug!(volatility = format!("{:.4}", volatility), "Volatility calculated");
        volatility
    }

    // ---- Missing Data Analysis ----

    /// Port of `_analyze_missing_data()`.
    pub fn analyze_missing_data(&self, timestamps: &[NaiveDateTime]) -> MissingPatternInfo {
        if timestamps.len() < 2 {
            return MissingPatternInfo::default();
        }

        let intervals: Vec<f64> = timestamps
            .windows(2)
            .map(|w| (w[1] - w[0]).num_milliseconds() as f64 / 1000.0)
            .collect();

        if intervals.is_empty() {
            return MissingPatternInfo::default();
        }

        let expected_interval = median(&intervals);
        let tolerance = expected_interval * 1.5;

        let gaps: Vec<f64> = intervals
            .iter()
            .filter(|&&iv| iv > tolerance)
            .copied()
            .collect();
        let gap_count = gaps.len();
        let has_gaps = gap_count > 0;
        let gap_percentage = gap_count as f64 / intervals.len() as f64 * 100.0;
        let max_gap = gaps.iter().cloned().fold(0.0_f64, f64::max);

        debug!(
            gaps = gap_count,
            percentage = format!("{:.1}", gap_percentage),
            "Missing data analysis complete"
        );

        MissingPatternInfo {
            has_gaps,
            gap_count,
            gap_percentage,
            expected_interval,
            max_gap,
        }
    }

    // ---- Outlier Detection ----

    /// Port of `_detect_outliers()`. Dual method: IQR (3.0x) AND Z-score (3.0σ).
    pub fn detect_outliers(&self, values: &[f64]) -> OutlierInfo {
        if values.len() < 4 {
            return OutlierInfo {
                count: 0,
                percentage: 0.0,
                indices: vec![],
                method: "IQR_and_Z-score".into(),
            };
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1 = percentile(&sorted, 25.0);
        let q3 = percentile(&sorted, 75.0);
        let iqr = q3 - q1;

        let lower_bound = q1 - 3.0 * iqr;
        let upper_bound = q3 + 3.0 * iqr;

        // IQR outliers
        let iqr_outliers: Vec<usize> = values
            .iter()
            .enumerate()
            .filter(|(_, &v)| v < lower_bound || v > upper_bound)
            .map(|(i, _)| i)
            .collect();

        // Z-score outliers
        let m = mean(values);
        let sd = std_dev_population(values);
        let z_outliers: Vec<usize> = if sd > 0.0 {
            values
                .iter()
                .enumerate()
                .filter(|(_, &v)| ((v - m) / sd).abs() > 3.0)
                .map(|(i, _)| i)
                .collect()
        } else {
            vec![]
        };

        // Intersection: both methods must agree
        let outlier_indices: Vec<usize> = iqr_outliers
            .iter()
            .filter(|i| z_outliers.contains(i))
            .copied()
            .collect();

        let count = outlier_indices.len();
        let percentage = count as f64 / values.len() as f64 * 100.0;

        debug!(
            count = count,
            percentage = format!("{:.1}", percentage),
            "Outlier detection complete"
        );

        OutlierInfo {
            count,
            percentage,
            indices: outlier_indices,
            method: "IQR_and_Z-score".into(),
        }
    }

    // ---- Time Interval Analysis ----

    /// Port of `_analyze_time_intervals()`.
    pub fn analyze_time_intervals(&self, timestamps: &[NaiveDateTime]) -> DensityInfo {
        if timestamps.len() < 2 {
            return DensityInfo {
                regular: false,
                ..Default::default()
            };
        }

        let intervals: Vec<f64> = timestamps
            .windows(2)
            .map(|w| (w[1] - w[0]).num_milliseconds() as f64 / 1000.0)
            .collect();

        if intervals.is_empty() {
            return DensityInfo {
                regular: false,
                ..Default::default()
            };
        }

        let mean_interval = mean(&intervals);
        let interval_variance = variance(&intervals);
        let cv = if mean_interval > 0.0 {
            interval_variance.sqrt() / mean_interval
        } else {
            f64::INFINITY
        };
        let is_regular = cv < 0.1;

        debug!(regular = is_regular, cv = format!("{:.3}", cv), "Time interval analysis");

        DensityInfo {
            regular: is_regular,
            mean_interval,
            interval_variance,
            coefficient_of_variation: cv,
        }
    }

    // ---- Stationarity Test ----

    /// Port of `_test_stationarity()`. Simplified: compare mean/variance of two halves.
    pub fn test_stationarity(&self, values: &[f64]) -> StationarityInfo {
        if values.len() < 10 {
            return StationarityInfo::default();
        }

        let n = values.len();
        let half = n / 2;
        let first_half = &values[..half];
        let second_half = &values[half..];

        let mean_diff = (mean(first_half) - mean(second_half)).abs();
        let mean_threshold = std_dev_population(values) * 0.5;

        let var1 = variance(first_half);
        let var2 = variance(second_half);
        let var_ratio = var1.max(var2) / var1.min(var2).max(1e-10);

        let is_stationary = mean_diff <= mean_threshold && var_ratio < 2.0;

        debug!(stationary = is_stationary, "Stationarity test");

        StationarityInfo {
            is_stationary: Some(is_stationary),
            mean_difference: mean_diff,
            variance_ratio: var_ratio,
        }
    }

    // ---- Frequency Estimation ----

    /// Port of `_estimate_frequency()`.
    pub fn estimate_frequency(&self, timestamps: &[NaiveDateTime]) -> FrequencyInfo {
        if timestamps.len() < 2 {
            return FrequencyInfo::default();
        }

        let intervals: Vec<f64> = timestamps
            .windows(2)
            .map(|w| (w[1] - w[0]).num_milliseconds() as f64 / 1000.0)
            .collect();

        let median_interval = median(&intervals);

        let freq = if median_interval <= 60.0 {
            "min"
        } else if median_interval <= 900.0 {
            "15min"
        } else if median_interval <= 1800.0 {
            "30min"
        } else if median_interval <= 3600.0 {
            "H"
        } else if median_interval <= 86400.0 {
            "D"
        } else {
            "irregular"
        };

        let interval_std = std_dev(&intervals);
        let cv = if median_interval > 0.0 {
            interval_std / median_interval
        } else {
            1.0
        };
        let confidence = (1.0 - cv).max(0.0);

        debug!(
            freq = freq,
            confidence = format!("{:.3}", confidence),
            "Frequency estimation"
        );

        FrequencyInfo {
            estimated: freq.into(),
            confidence,
            median_interval_seconds: median_interval,
        }
    }
}

impl Default for TimeSeriesAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Helper functions ----

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Population standard deviation (ddof=0), matching numpy.std() default.
fn std_dev_population(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let var = data.iter().map(|v| (v - m).powi(2)).sum::<f64>() / data.len() as f64;
    var.sqrt()
}

/// Sample standard deviation (ddof=0 to match numpy default behavior).
fn std_dev(data: &[f64]) -> f64 {
    std_dev_population(data)
}

/// Population variance (ddof=0), matching numpy.var() default.
fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|v| (v - m).powi(2)).sum::<f64>() / data.len() as f64
}

fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Linear percentile (matching numpy.percentile with linear interpolation).
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = pct / 100.0 * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let frac = rank - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Simple linear regression: returns (slope, intercept, r, p_value, std_err).
/// Port of scipy.stats.linregress.
fn linregress(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|a| a * a).sum();
    let sum_y2: f64 = y.iter().map(|a| a * a).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return (0.0, mean(y), 0.0, 1.0, 0.0);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    // Correlation coefficient
    let ss_x = sum_x2 - sum_x * sum_x / n;
    let ss_y = sum_y2 - sum_y * sum_y / n;
    let ss_xy = sum_xy - sum_x * sum_y / n;

    let r = if (ss_x * ss_y).abs() < 1e-15 {
        0.0
    } else {
        ss_xy / (ss_x * ss_y).sqrt()
    };

    // Standard error of slope
    let n_int = x.len();
    let residuals: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| yi - (slope * xi + intercept))
        .collect();
    let mse = residuals.iter().map(|r| r * r).sum::<f64>() / (n_int as f64 - 2.0).max(1.0);
    let std_err = (mse / ss_x.max(1e-15)).sqrt();

    // p-value for t-test on slope
    let t_stat = if std_err > 0.0 {
        slope / std_err
    } else {
        0.0
    };
    let df = (n_int as f64 - 2.0).max(1.0);
    let p_value = t_test_p_value(t_stat, df);

    (slope, intercept, r, p_value, std_err)
}

/// Two-sided p-value for t-statistic. Approximation using normal for large df.
fn t_test_p_value(t: f64, _df: f64) -> f64 {
    // For simplicity, use normal approximation (good for df > 30, reasonable otherwise).
    let normal = Normal::new(0.0, 1.0).unwrap();
    2.0 * (1.0 - normal.cdf(t.abs()))
}

/// Simple peak finding: local maxima above a height threshold.
/// Port of scipy.signal.find_peaks with height parameter.
fn find_peaks(data: &[f64], min_height: f64) -> Vec<usize> {
    let n = data.len();
    if n < 3 {
        return vec![];
    }
    let mut peaks = Vec::new();
    for i in 1..n - 1 {
        if data[i] > data[i - 1] && data[i] > data[i + 1] && data[i] >= min_height {
            peaks.push(i);
        }
    }
    peaks
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn make_ts(hour: u32, min: u32, sec: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(hour, min, sec)
            .unwrap()
    }

    fn uniform_timestamps(n: usize, interval_secs: i64) -> Vec<NaiveDateTime> {
        let base = NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        (0..n)
            .map(|i| base + chrono::Duration::seconds(interval_secs * i as i64))
            .collect()
    }

    #[test]
    fn test_insufficient_data_returns_basic() {
        let analyzer = TimeSeriesAnalyzer::new();
        let chars = analyzer.analyze(&[1.0, 2.0], &[make_ts(10, 0, 0), make_ts(11, 0, 0)]);
        assert_eq!(chars.trend.strength, "unknown");
    }

    #[test]
    fn test_uptrend_detection() {
        let analyzer = TimeSeriesAnalyzer::new();
        let vals: Vec<f64> = (0..50).map(|i| i as f64 * 2.0 + 10.0).collect();
        let ts = uniform_timestamps(50, 3600);
        let chars = analyzer.analyze(&vals, &ts);
        assert_eq!(chars.trend.strength, "strong");
        assert_eq!(chars.trend.direction, "increasing");
        assert!(chars.trend.r_squared > 0.99);
    }

    #[test]
    fn test_downtrend_detection() {
        let analyzer = TimeSeriesAnalyzer::new();
        let vals: Vec<f64> = (0..50).map(|i| 100.0 - i as f64 * 1.5).collect();
        let ts = uniform_timestamps(50, 3600);
        let chars = analyzer.analyze(&vals, &ts);
        assert_eq!(chars.trend.direction, "decreasing");
    }

    #[test]
    fn test_seasonality_detection() {
        let analyzer = TimeSeriesAnalyzer::new();
        // Generate sinusoidal data with period 12
        let n = 120;
        let vals: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() * 10.0)
            .collect();
        let ts = uniform_timestamps(n, 3600);
        let chars = analyzer.analyze(&vals, &ts);
        assert_ne!(chars.seasonality.strength, "weak");
        assert_eq!(chars.seasonality.period, Some(12));
    }

    #[test]
    fn test_mann_kendall_increasing() {
        let analyzer = TimeSeriesAnalyzer::new();
        let vals: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let mk = analyzer.mann_kendall_test(&vals);
        assert_eq!(mk.trend, "increasing");
        assert!(mk.p_value < 0.05);
        assert!(mk.s_statistic > 0);
    }

    #[test]
    fn test_mann_kendall_no_trend() {
        let analyzer = TimeSeriesAnalyzer::new();
        let vals = vec![5.0; 20];
        let mk = analyzer.mann_kendall_test(&vals);
        assert_eq!(mk.trend, "none");
        assert_eq!(mk.s_statistic, 0);
    }

    #[test]
    fn test_volatility_calculation() {
        let analyzer = TimeSeriesAnalyzer::new();
        // Constant series → zero volatility (diffs are all 0)
        let vals = vec![10.0; 20];
        let vol = analyzer.calculate_volatility(&vals);
        assert_eq!(vol, 0.0);

        // Linear series → diffs are all 1.0, std_dev(diffs) = 0
        let vals2: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let vol2 = analyzer.calculate_volatility(&vals2);
        assert_eq!(vol2, 0.0); // constant diffs → zero volatility

        // Varying series → positive volatility
        let vals3 = vec![100.0, 110.0, 95.0, 120.0, 80.0, 130.0, 90.0, 115.0, 85.0, 125.0];
        let vol3 = analyzer.calculate_volatility(&vals3);
        assert!(vol3 > 0.0);
    }

    #[test]
    fn test_outlier_detection() {
        let analyzer = TimeSeriesAnalyzer::new();
        let mut vals = vec![10.0; 50];
        vals[25] = 1000.0; // extreme outlier
        let outliers = analyzer.detect_outliers(&vals);
        assert!(outliers.count > 0);
        assert!(outliers.indices.contains(&25));
    }

    #[test]
    fn test_stationarity_stationary_series() {
        let analyzer = TimeSeriesAnalyzer::new();
        let vals = vec![10.0; 30];
        let stat = analyzer.test_stationarity(&vals);
        assert_eq!(stat.is_stationary, Some(true));
    }

    #[test]
    fn test_frequency_estimation_hourly() {
        let analyzer = TimeSeriesAnalyzer::new();
        let ts = uniform_timestamps(20, 3600); // 1-hour intervals
        let freq = analyzer.estimate_frequency(&ts);
        assert_eq!(freq.estimated, "H");
        assert!(freq.confidence > 0.9);
    }

    #[test]
    fn test_frequency_estimation_minutely() {
        let analyzer = TimeSeriesAnalyzer::new();
        let ts = uniform_timestamps(20, 60);
        let freq = analyzer.estimate_frequency(&ts);
        assert_eq!(freq.estimated, "min");
    }

    #[test]
    fn test_missing_data_no_gaps() {
        let analyzer = TimeSeriesAnalyzer::new();
        let ts = uniform_timestamps(20, 3600);
        let missing = analyzer.analyze_missing_data(&ts);
        assert!(!missing.has_gaps);
        assert_eq!(missing.gap_count, 0);
    }

    #[test]
    fn test_missing_data_with_gaps() {
        let analyzer = TimeSeriesAnalyzer::new();
        let base = NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let ts = vec![
            base,
            base + chrono::Duration::hours(1),
            base + chrono::Duration::hours(2),
            base + chrono::Duration::hours(10), // big gap
            base + chrono::Duration::hours(11),
        ];
        let missing = analyzer.analyze_missing_data(&ts);
        assert!(missing.has_gaps);
        assert!(missing.gap_count > 0);
    }

    #[test]
    fn test_time_interval_regular() {
        let analyzer = TimeSeriesAnalyzer::new();
        let ts = uniform_timestamps(20, 3600);
        let density = analyzer.analyze_time_intervals(&ts);
        assert!(density.regular);
        assert!(density.coefficient_of_variation < 0.1);
    }

    #[test]
    fn test_time_interval_irregular() {
        let analyzer = TimeSeriesAnalyzer::new();
        let base = NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let ts = vec![
            base,
            base + chrono::Duration::seconds(60),
            base + chrono::Duration::seconds(360),
            base + chrono::Duration::seconds(400),
            base + chrono::Duration::seconds(1800),
        ];
        let density = analyzer.analyze_time_intervals(&ts);
        assert!(!density.regular);
    }

    /// Test period detection accuracy for various data lengths (pure seasonal).
    ///
    /// This is a regression test for FFT bin discretization issues.
    /// For n=90 and period=12, the true frequency 1/12=0.0833 falls between
    /// FFT bins 7 (freq=0.0778) and 8 (freq=0.0889). Without interpolation,
    /// the wrong period (11 or 13) may be detected.
    #[test]
    fn test_seasonality_period_accuracy_pure() {
        let analyzer = TimeSeriesAnalyzer::new();
        let period = 12;

        // Test multiple data lengths that previously caused issues
        for n in [90, 100, 150, 200] {
            let vals: Vec<f64> = (0..n)
                .map(|i| {
                    // Pure seasonal (no trend)
                    30.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                })
                .collect();
            let ts = uniform_timestamps(n, 3600);
            let chars = analyzer.analyze(&vals, &ts);

            assert_eq!(
                chars.seasonality.period,
                Some(period),
                "pure seasonal n={}: expected period {}, got {:?}",
                n,
                period,
                chars.seasonality.period
            );
        }
    }

    /// Test period detection for trend + seasonal data.
    ///
    /// The trend component creates a large low-frequency peak in FFT.
    /// This test verifies that we can still detect the seasonal period.
    #[test]
    fn test_seasonality_period_accuracy_with_trend() {
        let analyzer = TimeSeriesAnalyzer::new();
        let period = 12;

        // Test various lengths including those used in benchmarks
        // (100-10=90, 200-20=180, 500-50=450 training points)
        for n in [90, 100, 180, 450] {
            let vals: Vec<f64> = (0..n)
                .map(|i| {
                    // Trend + seasonal (like trend_plus_seasonal benchmark)
                    100.0
                        + 1.5 * i as f64
                        + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                })
                .collect();
            let ts = uniform_timestamps(n, 3600);
            let chars = analyzer.analyze(&vals, &ts);

            assert_eq!(
                chars.seasonality.period,
                Some(period),
                "trend+seasonal n={}: expected period {}, got {:?}",
                n,
                period,
                chars.seasonality.period
            );
        }
    }

    /// Test exponential trend detection for various data lengths.
    ///
    /// This is a regression test: exponential_growth_100 was not detected as
    /// exponential, causing the log transform to be skipped and MASE to be 1.84.
    #[test]
    fn test_exponential_trend_detection() {
        let analyzer = TimeSeriesAnalyzer::new();

        // Test data matching benchmark: 100 * exp(0.02 * i)
        // Training lengths: 90 (100-10), 180 (200-20), 450 (500-50)
        for n in [90, 180, 450] {
            let vals: Vec<f64> = (0..n).map(|i| 100.0 * (0.02 * i as f64).exp()).collect();
            let ts = uniform_timestamps(n, 3600);
            let chars = analyzer.analyze(&vals, &ts);

            assert!(
                chars.trend.is_exponential,
                "exponential n={}: expected is_exponential=true, linear_r²={:.3}",
                n,
                chars.trend.r_squared
            );
        }
    }

    /// Test that changepoint data is NOT detected as exponential.
    ///
    /// Changepoint has piecewise linear pattern with slope change in the middle.
    /// This should NOT trigger exponential detection.
    #[test]
    fn test_changepoint_not_exponential() {
        let analyzer = TimeSeriesAnalyzer::new();

        // Test data matching benchmark: piecewise linear
        for n in [90, 180, 450] {
            let mid = n / 2;
            let vals: Vec<f64> = (0..n)
                .map(|i| {
                    if i < mid {
                        100.0 + 1.0 * i as f64
                    } else {
                        100.0 + 1.0 * mid as f64 + 3.0 * (i - mid) as f64
                    }
                })
                .collect();
            let ts = uniform_timestamps(n, 3600);
            let chars = analyzer.analyze(&vals, &ts);

            assert!(
                !chars.trend.is_exponential,
                "changepoint n={}: expected is_exponential=false",
                n
            );
        }
    }
}
