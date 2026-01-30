use super::*;
use chrono::NaiveDate;
use num_traits::{FromPrimitive, ToPrimitive};

fn make_data(n: usize) -> BTreeMap<NaiveDateTime, BigDecimal> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    (0..n)
        .map(|i| {
            let ts = base + TimeDelta::hours(i as i64);
            let val = BigDecimal::from_f64(100.0 + i as f64 * 2.0).unwrap();
            (ts, val)
        })
        .collect()
}

fn make_data_with_values(values: &[f64]) -> BTreeMap<NaiveDateTime, BigDecimal> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let ts = base + TimeDelta::hours(i as i64);
            let val = BigDecimal::from_f64(v).unwrap();
            (ts, val)
        })
        .collect()
}

#[test]
fn test_predict_uptrend() {
    let input = PredictionInput {
        data: make_data(100),
        horizon: TimeDelta::hours(10),
    };

    let result = predict(&input).unwrap();
    assert_eq!(result.forecast_values.len(), 10);
    assert!(!result.model_name.is_empty());
    assert!(!result.strategy_name.is_empty());
    assert!(result.processing_time_secs > 0.0);
    assert!(result.model_count > 0);

    // Verify timestamps are in ascending order (BTreeMap guarantees this)
    let timestamps: Vec<_> = result.forecast_values.keys().collect();
    for i in 1..timestamps.len() {
        assert!(timestamps[i] > timestamps[i - 1]);
    }
}

#[test]
fn test_predict_flat() {
    let input = PredictionInput {
        data: make_data_with_values(&vec![42.0; 50]),
        horizon: TimeDelta::hours(5),
    };

    let result = predict(&input).unwrap();
    assert_eq!(result.forecast_values.len(), 5);
    // Flat data → predictions near 42
    for v in result.forecast_values.values() {
        let f = v.to_f64().unwrap();
        assert!((f - 42.0).abs() < 20.0, "Expected ~42, got {}", f);
    }
}

#[test]
fn test_predict_seasonal() {
    let n = 120;
    let values: Vec<f64> = (0..n)
        .map(|i| 500.0 + (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() * 50.0)
        .collect();
    let input = PredictionInput {
        data: make_data_with_values(&values),
        horizon: TimeDelta::hours(12),
    };

    let result = predict(&input).unwrap();
    assert_eq!(result.forecast_values.len(), 12);
}

#[test]
fn test_predict_validation_errors() {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    // Too few points
    let result = predict(&PredictionInput {
        data: [(base, BigDecimal::from(1))].into_iter().collect(),
        horizon: TimeDelta::hours(5),
    });
    assert!(result.is_err());

    // Zero horizon
    let result = predict(&PredictionInput {
        data: make_data_with_values(&[1.0; 10]),
        horizon: TimeDelta::zero(),
    });
    assert!(result.is_err());

    // Negative horizon
    let result = predict(&PredictionInput {
        data: make_data_with_values(&[1.0; 10]),
        horizon: TimeDelta::hours(-1),
    });
    assert!(result.is_err());
}

#[test]
fn test_horizon_to_steps() {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    // Hourly data: 24 hours → 24 steps
    let hourly_ts: Vec<_> = (0..100).map(|i| base + TimeDelta::hours(i)).collect();
    assert_eq!(horizon_to_steps(&TimeDelta::hours(24), &hourly_ts), 24);

    // Daily data: 7 days → 7 steps
    let daily_ts: Vec<_> = (0..30).map(|i| base + TimeDelta::days(i)).collect();
    assert_eq!(horizon_to_steps(&TimeDelta::days(7), &daily_ts), 7);

    // Edge case: fewer than 2 timestamps → return 1
    let single_ts = vec![base];
    assert_eq!(horizon_to_steps(&TimeDelta::hours(24), &single_ts), 1);

    // Non-exact multiple: 2.5 hours on hourly data → 2 steps (truncated)
    assert_eq!(horizon_to_steps(&TimeDelta::minutes(150), &hourly_ts), 2);

    // Horizon smaller than interval → minimum 1 step
    assert_eq!(horizon_to_steps(&TimeDelta::minutes(30), &hourly_ts), 1);

    // Minutely data: 1 hour → 60 steps
    let minutely_ts: Vec<_> = (0..200).map(|i| base + TimeDelta::minutes(i)).collect();
    assert_eq!(horizon_to_steps(&TimeDelta::hours(1), &minutely_ts), 60);
}

#[test]
fn test_predict_horizon_steps_match() {
    // Verify that forecast length matches expected steps for various intervals

    // Daily data with 1 week horizon
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    let daily_data: BTreeMap<NaiveDateTime, BigDecimal> = (0..60)
        .map(|i| {
            let ts = base + TimeDelta::days(i);
            let val = BigDecimal::from_f64(100.0 + i as f64).unwrap();
            (ts, val)
        })
        .collect();

    let input = PredictionInput {
        data: daily_data,
        horizon: TimeDelta::days(7), // 7 days → 7 steps
    };

    let result = predict(&input).unwrap();
    assert_eq!(
        result.forecast_values.len(),
        7,
        "Daily data with 7-day horizon should produce 7 forecasts"
    );
}

#[test]
fn test_predict_non_exact_horizon() {
    // When horizon is not exact multiple of interval, should truncate
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    let hourly_data: BTreeMap<NaiveDateTime, BigDecimal> = (0..100)
        .map(|i| {
            let ts = base + TimeDelta::hours(i);
            let val = BigDecimal::from_f64(100.0 + i as f64).unwrap();
            (ts, val)
        })
        .collect();

    // 2.5 hours on hourly data → 2 steps
    let input = PredictionInput {
        data: hourly_data,
        horizon: TimeDelta::minutes(150),
    };

    let result = predict(&input).unwrap();
    assert_eq!(
        result.forecast_values.len(),
        2,
        "2.5 hours on hourly data should produce 2 forecasts (truncated)"
    );
}

#[test]
fn test_calculate_time_budget() {
    // 100 points → 60 + 10 = 70s
    assert!((calculate_time_budget(100) - 70.0).abs() < 0.1);

    // 500 points → 60 + 50 = 110s
    assert!((calculate_time_budget(500) - 110.0).abs() < 0.1);

    // 10000 points → capped at 900s
    assert!((calculate_time_budget(10000) - 900.0).abs() < 0.1);
}

#[test]
fn test_calculate_median_interval() {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    // Hourly data → 3600 seconds
    let hourly_ts: Vec<_> = (0..10).map(|i| base + TimeDelta::hours(i)).collect();
    assert_eq!(calculate_median_interval(&hourly_ts), 3600);

    // Daily data → 86400 seconds
    let daily_ts: Vec<_> = (0..10).map(|i| base + TimeDelta::days(i)).collect();
    assert_eq!(calculate_median_interval(&daily_ts), 86400);

    // Minutely data → 60 seconds
    let minutely_ts: Vec<_> = (0..10).map(|i| base + TimeDelta::minutes(i)).collect();
    assert_eq!(calculate_median_interval(&minutely_ts), 60);

    // Edge case: fewer than 2 timestamps → return 1
    let single_ts = vec![base];
    assert_eq!(calculate_median_interval(&single_ts), 1);

    // Empty slice → return 1
    let empty_ts: Vec<NaiveDateTime> = vec![];
    assert_eq!(calculate_median_interval(&empty_ts), 1);
}

#[test]
fn test_horizon_to_steps_with_timestamps() {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    // Hourly data: 5 hour horizon → 5 steps
    let hourly_ts: Vec<_> = (0..100).map(|i| base + TimeDelta::hours(i)).collect();
    let (steps, forecast_ts) = horizon_to_steps_with_timestamps(&TimeDelta::hours(5), &hourly_ts);

    assert_eq!(steps, 5);
    assert_eq!(forecast_ts.len(), 5);

    // First forecast timestamp = last training timestamp + 1 hour
    let last_train_ts = hourly_ts.last().unwrap();
    assert_eq!(forecast_ts[0], *last_train_ts + TimeDelta::hours(1));
    assert_eq!(forecast_ts[1], *last_train_ts + TimeDelta::hours(2));
    assert_eq!(forecast_ts[4], *last_train_ts + TimeDelta::hours(5));

    // Timestamps are in ascending order
    for i in 1..forecast_ts.len() {
        assert!(forecast_ts[i] > forecast_ts[i - 1]);
    }

    // Daily data: 7 day horizon → 7 steps
    let daily_ts: Vec<_> = (0..30).map(|i| base + TimeDelta::days(i)).collect();
    let (steps, forecast_ts) = horizon_to_steps_with_timestamps(&TimeDelta::days(7), &daily_ts);

    assert_eq!(steps, 7);
    assert_eq!(forecast_ts.len(), 7);

    let last_train_ts = daily_ts.last().unwrap();
    assert_eq!(forecast_ts[0], *last_train_ts + TimeDelta::days(1));
    assert_eq!(forecast_ts[6], *last_train_ts + TimeDelta::days(7));

    // Edge case: empty timestamps
    let (steps, forecast_ts) = horizon_to_steps_with_timestamps(&TimeDelta::hours(5), &[]);
    assert_eq!(steps, 1); // horizon_to_steps returns 1 for empty slice
    assert!(forecast_ts.is_empty());
}

#[test]
fn test_forecast_result_has_correct_timestamps() {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    // Create hourly data
    let hourly_data: BTreeMap<NaiveDateTime, BigDecimal> = (0..100)
        .map(|i| {
            let ts = base + TimeDelta::hours(i);
            let val = BigDecimal::from_f64(100.0 + i as f64).unwrap();
            (ts, val)
        })
        .collect();

    let input = PredictionInput {
        data: hourly_data.clone(),
        horizon: TimeDelta::hours(5),
    };

    let result = predict(&input).unwrap();

    // Verify forecast has correct number of entries
    assert_eq!(result.forecast_values.len(), 5);

    // Get last training timestamp
    let last_train_ts = *hourly_data.keys().last().unwrap();

    // Verify forecast timestamps start after training data
    let first_forecast_ts = *result.forecast_values.keys().next().unwrap();
    assert!(first_forecast_ts > last_train_ts);

    // Verify timestamps are evenly spaced (1 hour apart)
    let forecast_timestamps: Vec<_> = result.forecast_values.keys().collect();
    for i in 1..forecast_timestamps.len() {
        let diff = *forecast_timestamps[i] - *forecast_timestamps[i - 1];
        assert_eq!(diff.num_hours(), 1);
    }

    // Verify bounds have same timestamps as forecast_values if present
    if let Some(ref lower) = result.lower_bound {
        assert_eq!(lower.len(), result.forecast_values.len());
        for ts in lower.keys() {
            assert!(result.forecast_values.contains_key(ts));
        }
    }
    if let Some(ref upper) = result.upper_bound {
        assert_eq!(upper.len(), result.forecast_values.len());
        for ts in upper.keys() {
            assert!(result.forecast_values.contains_key(ts));
        }
    }
}

#[test]
fn test_forecast_result_with_daily_data() {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    // Create daily data
    let daily_data: BTreeMap<NaiveDateTime, BigDecimal> = (0..60)
        .map(|i| {
            let ts = base + TimeDelta::days(i);
            let val = BigDecimal::from_f64(100.0 + i as f64).unwrap();
            (ts, val)
        })
        .collect();

    let input = PredictionInput {
        data: daily_data.clone(),
        horizon: TimeDelta::days(7),
    };

    let result = predict(&input).unwrap();

    // Verify forecast has correct number of entries
    assert_eq!(result.forecast_values.len(), 7);

    // Get last training timestamp
    let last_train_ts = *daily_data.keys().last().unwrap();

    // Verify forecast timestamps are 1 day apart
    let forecast_timestamps: Vec<_> = result.forecast_values.keys().collect();

    // First forecast should be 1 day after last training point
    assert_eq!(*forecast_timestamps[0] - last_train_ts, TimeDelta::days(1));

    for i in 1..forecast_timestamps.len() {
        let diff = *forecast_timestamps[i] - *forecast_timestamps[i - 1];
        assert_eq!(diff.num_days(), 1);
    }
}

#[test]
fn test_forecast_timestamps_with_irregular_data() {
    // Test that forecast timestamps are evenly spaced based on normalized median interval,
    // even when input data has irregular spacing.
    // Note: The normalize module may resample data, so forecast step count depends on
    // the normalized timestamps, not the original input.
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    // Create irregular data: mostly hourly but with some gaps
    let mut irregular_data: BTreeMap<NaiveDateTime, BigDecimal> = BTreeMap::new();
    let mut ts = base;
    for i in 0..50 {
        irregular_data.insert(ts, BigDecimal::from_f64(100.0 + i as f64).unwrap());
        // Add irregular intervals: mostly 1 hour, occasionally 2 or 3 hours
        let interval = if i % 10 == 5 {
            TimeDelta::hours(3)
        } else if i % 7 == 0 {
            TimeDelta::hours(2)
        } else {
            TimeDelta::hours(1)
        };
        ts += interval;
    }

    let input = PredictionInput {
        data: irregular_data,
        horizon: TimeDelta::hours(5),
    };

    let result = predict(&input).unwrap();

    // Forecast should have at least 1 step
    assert!(!result.forecast_values.is_empty());

    // Key assertion: Timestamps should be evenly spaced based on median interval
    let forecast_timestamps: Vec<_> = result.forecast_values.keys().collect();
    if forecast_timestamps.len() >= 2 {
        let intervals: Vec<i64> = forecast_timestamps
            .windows(2)
            .map(|w| (*w[1] - *w[0]).num_seconds())
            .collect();

        // All intervals should be the same (median-based)
        let first_interval = intervals[0];
        for interval in &intervals {
            assert_eq!(
                *interval, first_interval,
                "Forecast timestamps should be evenly spaced"
            );
        }
    }
}

#[test]
fn test_forecast_with_small_data() {
    // Test with small dataset (10 points) to verify timestamps work correctly
    // Note: Models require more than 2 points to produce valid forecasts
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    let small_data: BTreeMap<NaiveDateTime, BigDecimal> = (0..10)
        .map(|i| {
            let ts = base + TimeDelta::hours(i);
            let val = BigDecimal::from_f64(100.0 + i as f64 * 5.0).unwrap();
            (ts, val)
        })
        .collect();

    let input = PredictionInput {
        data: small_data.clone(),
        horizon: TimeDelta::hours(3),
    };

    let result = predict(&input).unwrap();

    // Should produce 3 forecast points
    assert_eq!(result.forecast_values.len(), 3);

    // Last training timestamp
    let last_train_ts = *small_data.keys().last().unwrap();

    // First forecast should be 1 hour after last training point
    let first_forecast_ts = *result.forecast_values.keys().next().unwrap();
    assert_eq!(first_forecast_ts - last_train_ts, TimeDelta::hours(1));

    // Timestamps should be 1 hour apart
    let forecast_timestamps: Vec<_> = result.forecast_values.keys().collect();
    for i in 1..forecast_timestamps.len() {
        let diff = *forecast_timestamps[i] - *forecast_timestamps[i - 1];
        assert_eq!(diff.num_hours(), 1);
    }
}

#[test]
fn test_horizon_to_steps_with_timestamps_minutely() {
    // Test with minutely data to ensure various intervals work
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    let minutely_ts: Vec<_> = (0..100).map(|i| base + TimeDelta::minutes(i)).collect();
    let (steps, forecast_ts) =
        horizon_to_steps_with_timestamps(&TimeDelta::minutes(10), &minutely_ts);

    assert_eq!(steps, 10);
    assert_eq!(forecast_ts.len(), 10);

    let last_train_ts = minutely_ts.last().unwrap();

    // First forecast should be 1 minute after last training point
    assert_eq!(forecast_ts[0], *last_train_ts + TimeDelta::minutes(1));

    // All intervals should be 1 minute (60 seconds)
    for i in 1..forecast_ts.len() {
        let diff = (forecast_ts[i] - forecast_ts[i - 1]).num_seconds();
        assert_eq!(diff, 60);
    }
}

#[test]
fn test_forecast_timestamps_exact_first_value() {
    // Verify that first forecast timestamp is exactly last_train + median_interval
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    // Create regular hourly data
    let hourly_data: BTreeMap<NaiveDateTime, BigDecimal> = (0..50)
        .map(|i| {
            let ts = base + TimeDelta::hours(i);
            let val = BigDecimal::from_f64(100.0 + i as f64).unwrap();
            (ts, val)
        })
        .collect();

    let input = PredictionInput {
        data: hourly_data.clone(),
        horizon: TimeDelta::hours(3),
    };

    let result = predict(&input).unwrap();

    // Get last training timestamp (data is regular, so no normalization change)
    let last_train_ts = *hourly_data.keys().last().unwrap();

    // First forecast timestamp should be exactly 1 hour after last training
    let first_forecast_ts = *result.forecast_values.keys().next().unwrap();
    assert_eq!(
        first_forecast_ts,
        last_train_ts + TimeDelta::hours(1),
        "First forecast timestamp should be last_train + median_interval"
    );

    // Last forecast timestamp should be exactly 3 hours after last training
    let last_forecast_ts = *result.forecast_values.keys().last().unwrap();
    assert_eq!(
        last_forecast_ts,
        last_train_ts + TimeDelta::hours(3),
        "Last forecast timestamp should match horizon"
    );
}
