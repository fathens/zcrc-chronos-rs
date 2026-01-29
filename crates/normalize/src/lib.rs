use chrono::NaiveDateTime;
use bigdecimal::BigDecimal;
use common::{decimals_to_f64s, f64s_to_decimals, ChronosError, Result};
use tracing::info;

/// Normalize irregular time-series data to uniform intervals while preserving
/// price volatility.
///
/// Port of `normalize_time_series_data()` from Python routes.py (L1109-1224).
///
/// Algorithm:
/// 1. Check coefficient of variation (CV) of time deltas.
/// 2. If CV < 0.1 → already regular, return as-is.
/// 3. Otherwise, create a uniform grid from start to end with the same
///    number of points, and use nearest-neighbor resampling.
pub fn normalize_time_series_data(
    timestamps: &[NaiveDateTime],
    values: &[f64],
) -> Result<(Vec<NaiveDateTime>, Vec<f64>)> {
    if timestamps.is_empty() || values.is_empty() {
        return Ok((timestamps.to_vec(), values.to_vec()));
    }

    if timestamps.len() != values.len() {
        return Err(ChronosError::InvalidInput(
            "timestamps and values must have the same length".into(),
        ));
    }

    let num_points = timestamps.len();

    let start_time = *timestamps.iter().min().unwrap();
    let end_time = *timestamps.iter().max().unwrap();

    info!(
        start = %start_time,
        end = %end_time,
        count = num_points,
        "Normalizing time series data"
    );

    if num_points <= 1 {
        return Ok((timestamps.to_vec(), values.to_vec()));
    }

    // Calculate time deltas in seconds
    let time_diffs: Vec<f64> = timestamps
        .windows(2)
        .map(|w| (w[1] - w[0]).num_milliseconds() as f64 / 1000.0)
        .collect();

    // Coefficient of variation
    let mean_interval = mean(&time_diffs);
    let std_interval = std_dev(&time_diffs);
    let cv = if mean_interval > 0.0 {
        std_interval / mean_interval
    } else {
        0.0
    };

    // Already regular enough — skip normalization
    if cv < 0.1 {
        info!(cv = cv, "Intervals are regular (CV < 0.1), skipping normalization");
        return Ok((timestamps.to_vec(), values.to_vec()));
    }

    info!(cv = cv, "Intervals are irregular, performing normalization");

    let total_duration = (end_time - start_time).num_milliseconds() as f64 / 1000.0;

    if total_duration <= 0.0 {
        return Ok((timestamps.to_vec(), values.to_vec()));
    }

    // Create uniform grid
    let interval_seconds = total_duration / (num_points - 1).max(1) as f64;

    let mut new_timestamps = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let offset_ms = (interval_seconds * i as f64 * 1000.0) as i64;
        let new_time = start_time + chrono::Duration::milliseconds(offset_ms);
        new_timestamps.push(new_time);
    }
    // Ensure last timestamp matches end_time exactly
    if let Some(last) = new_timestamps.last_mut() {
        *last = end_time;
    }

    // Nearest-neighbor resampling: for each new timestamp, find the closest
    // original timestamp and take its value.
    let new_values = nearest_resample(timestamps, values, &new_timestamps);

    // Log retention rate
    let original_range = max_f64(values) - min_f64(values);
    let normalized_range = max_f64(&new_values) - min_f64(&new_values);
    let retention_rate = if original_range > 0.0 {
        normalized_range / original_range * 100.0
    } else {
        100.0
    };

    info!(
        retention_rate = format!("{:.1}", retention_rate),
        original_range = format!("{:.2}", original_range),
        normalized_range = format!("{:.2}", normalized_range),
        "Normalization complete"
    );

    Ok((new_timestamps, new_values))
}

/// Normalize irregular time-series data with `Decimal` values.
///
/// Delegates to `normalize_time_series_data` internally, converting between
/// `Decimal` and `f64` at the boundaries.
pub fn normalize_time_series_data_decimal(
    timestamps: &[NaiveDateTime],
    values: &[BigDecimal],
) -> Result<(Vec<NaiveDateTime>, Vec<BigDecimal>)> {
    let f64_values = decimals_to_f64s(values)?;
    let (norm_ts, norm_vals) = normalize_time_series_data(timestamps, &f64_values)?;
    let decimal_vals = f64s_to_decimals(&norm_vals)?;
    Ok((norm_ts, decimal_vals))
}

/// For each target timestamp, find the nearest source timestamp and return its value.
fn nearest_resample(
    src_timestamps: &[NaiveDateTime],
    src_values: &[f64],
    target_timestamps: &[NaiveDateTime],
) -> Vec<f64> {
    target_timestamps
        .iter()
        .map(|target| {
            let target_ms = target.and_utc().timestamp_millis();
            // Binary search for the closest timestamp
            let idx = match src_timestamps
                .binary_search_by_key(&target_ms, |ts| ts.and_utc().timestamp_millis())
            {
                Ok(exact) => exact,
                Err(insert_pos) => {
                    if insert_pos == 0 {
                        0
                    } else if insert_pos >= src_timestamps.len() {
                        src_timestamps.len() - 1
                    } else {
                        let left_ms = src_timestamps[insert_pos - 1].and_utc().timestamp_millis();
                        let right_ms = src_timestamps[insert_pos].and_utc().timestamp_millis();
                        if (target_ms - left_ms).abs() <= (right_ms - target_ms).abs() {
                            insert_pos - 1
                        } else {
                            insert_pos
                        }
                    }
                }
            };
            src_values[idx]
        })
        .collect()
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let variance = data.iter().map(|v| (v - m).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

fn max_f64(data: &[f64]) -> f64 {
    data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

fn min_f64(data: &[f64]) -> f64 {
    data.iter().cloned().fold(f64::INFINITY, f64::min)
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

    #[test]
    fn test_empty_input() {
        let (ts, vals) = normalize_time_series_data(&[], &[]).unwrap();
        assert!(ts.is_empty());
        assert!(vals.is_empty());
    }

    #[test]
    fn test_single_point() {
        let ts = vec![make_ts(10, 0, 0)];
        let vals = vec![42.0];
        let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();
        assert_eq!(out_ts.len(), 1);
        assert_eq!(out_vals, vec![42.0]);
    }

    #[test]
    fn test_regular_intervals_skip_normalization() {
        // 5 points at exactly 1-hour intervals → CV ≈ 0 → should return as-is
        let ts: Vec<NaiveDateTime> = (0..5).map(|i| make_ts(10 + i, 0, 0)).collect();
        let vals: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();
        assert_eq!(out_ts, ts);
        assert_eq!(out_vals, vals);
    }

    #[test]
    fn test_irregular_intervals_normalized() {
        // Irregular intervals: 60s, 300s, 60s, 600s → high CV
        let ts = vec![
            make_ts(10, 0, 0),
            make_ts(10, 1, 0),   // +60s
            make_ts(10, 6, 0),   // +300s
            make_ts(10, 7, 0),   // +60s
            make_ts(10, 17, 0),  // +600s
        ];
        let vals = vec![100.0, 200.0, 150.0, 300.0, 50.0];

        let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();
        assert_eq!(out_ts.len(), 5);
        assert_eq!(out_vals.len(), 5);
        // First timestamp should be preserved
        assert_eq!(out_ts[0], ts[0]);
        // Last timestamp should be preserved
        assert_eq!(out_ts[4], ts[4]);
        // All original values should appear (nearest-neighbor)
        for v in &out_vals {
            assert!(vals.contains(v));
        }
    }

    #[test]
    fn test_mismatched_lengths_error() {
        let ts = vec![make_ts(10, 0, 0)];
        let vals = vec![1.0, 2.0];
        assert!(normalize_time_series_data(&ts, &vals).is_err());
    }

    #[test]
    fn test_preserves_value_range() {
        // The normalization should not introduce values outside original range
        let ts = vec![
            make_ts(10, 0, 0),
            make_ts(10, 0, 30),  // +30s
            make_ts(10, 5, 0),   // +270s  (big gap)
            make_ts(10, 5, 20),  // +20s
            make_ts(10, 20, 0),  // +880s  (big gap)
        ];
        let vals = vec![10.0, 50.0, 20.0, 80.0, 30.0];

        let (_, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();
        let orig_min = min_f64(&vals);
        let orig_max = max_f64(&vals);
        for v in &out_vals {
            assert!(*v >= orig_min && *v <= orig_max);
        }
    }
}
