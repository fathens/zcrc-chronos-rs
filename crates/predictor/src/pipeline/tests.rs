use super::*;
use approx::assert_relative_eq;
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

    // make_data(100) is y = 100 + 2*i for i in 0..99 (Trending regime).
    // The forecast should preserve the linear trend, so values at i=100..109
    // should be approximately 300, 302, ..., 318. Each value should be > the
    // last training value (298) and trending upward.
    let forecast_values: Vec<f64> = result
        .forecast_values
        .values()
        .map(|v| v.to_f64().unwrap())
        .collect();
    assert!(
        forecast_values[0] > 290.0,
        "First forecast should be near 300 (trend extrapolation), got {}",
        forecast_values[0]
    );
    assert!(
        forecast_values.last().copied().unwrap() > forecast_values[0],
        "Forecast should continue the uptrend"
    );
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
fn test_predict_random_walk_no_detrend() {
    // Pseudo-random walk: the values drift but with no significant linear
    // trend. The span_ratio gate (|slope| × (n − 1) / |current_price|)
    // should stay well below DETREND_SPAN_RATIO_THRESHOLD here, so detrend
    // is not applied and the forecast stays near the last observed value
    // rather than extrapolating a spurious trend.
    let mut rng_state: u64 = 12345;
    let mut price = 100.0;
    let values: Vec<f64> = (0..200)
        .map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            price += (u - 0.5) * 0.5;
            price
        })
        .collect();
    let last_observed = values[values.len() - 1];

    let input = PredictionInput {
        data: make_data_with_values(&values),
        horizon: TimeDelta::hours(10),
    };

    let result = predict(&input).unwrap();
    let forecast_values: Vec<f64> = result
        .forecast_values
        .values()
        .map(|v| v.to_f64().unwrap())
        .collect();

    // For a random walk, the forecast should stay close to the last observed
    // value rather than extrapolating a trend. Allow a generous tolerance
    // since the underlying models do produce some drift.
    let last_forecast = forecast_values.last().copied().unwrap();
    assert!(
        (last_forecast - last_observed).abs() < 5.0,
        "Random walk forecast drifted too far: last_observed={}, last_forecast={}",
        last_observed,
        last_forecast
    );
}

#[test]
fn test_compute_detrend_state_gates_on_span_ratio() {
    use common::{RegimeInfo, TimeSeriesCharacteristics, TimeSeriesRegime, TrendInfo};

    fn chars(slope: f64, intercept: f64, regime: TimeSeriesRegime) -> TimeSeriesCharacteristics {
        TimeSeriesCharacteristics {
            trend: TrendInfo {
                slope,
                intercept,
                ..Default::default()
            },
            regime: RegimeInfo {
                regime,
                variance_ratio: 1.0,
                z_statistic: 0.0,
                p_value: 1.0,
                lag: 2,
            },
            ..Default::default()
        }
    }

    // 100 samples, current_price = 298, slope = 2.0
    // → span_ratio = |2.0| * 99 / 298 ≈ 0.665, well above 0.15 → detrend.
    let values: Vec<f64> = (0..100).map(|i| 100.0 + 2.0 * i as f64).collect();
    let state = compute_detrend_state(
        &chars(2.0, 100.0, TimeSeriesRegime::Trending),
        &values,
        false,
    )
    .expect("strong-trend series should be detrended");
    assert_eq!(state.training_len, 100);
    assert!((state.slope - 2.0).abs() < 1e-12);

    // Same series but tagged as MeanReverting: VR test is a negative
    // filter, so detrend must be skipped regardless of slope.
    assert!(
        compute_detrend_state(
            &chars(2.0, 100.0, TimeSeriesRegime::MeanReverting),
            &values,
            false,
        )
        .is_none(),
        "MeanReverting series must not be detrended even with a large slope"
    );

    // Weak-drift stable-like series: slope = 0.001, current ≈ 1.099
    // → span_ratio = 0.001 * 99 / 1.099 ≈ 0.09 < 0.15 → no detrend.
    let stable: Vec<f64> = (0..100).map(|i| 1.0 + 0.001 * i as f64).collect();
    assert!(
        compute_detrend_state(
            &chars(0.001, 1.0, TimeSeriesRegime::Trending),
            &stable,
            false,
        )
        .is_none(),
        "weak-drift series must not be detrended even when VR says Trending"
    );

    // Log-transformed series always skips detrend.
    assert!(
        compute_detrend_state(
            &chars(2.0, 100.0, TimeSeriesRegime::Trending),
            &values,
            true,
        )
        .is_none(),
        "log-transformed series must skip detrend"
    );

    // Degenerate / non-finite inputs are rejected without panicking.
    assert!(
        compute_detrend_state(
            &chars(f64::NAN, 100.0, TimeSeriesRegime::Trending),
            &values,
            false,
        )
        .is_none(),
        "NaN slope must be rejected"
    );
    assert!(
        compute_detrend_state(&chars(2.0, 100.0, TimeSeriesRegime::Trending), &[], false,)
            .is_none(),
        "empty log_values must be rejected"
    );
    let near_zero = vec![1e-200; 50];
    assert!(
        compute_detrend_state(
            &chars(2.0, 100.0, TimeSeriesRegime::Trending),
            &near_zero,
            false,
        )
        .is_none(),
        "near-zero baseline must be rejected"
    );
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

    // Non-exact multiple: 2.5 hours on hourly data → 3 steps (ceiling)
    assert_eq!(horizon_to_steps(&TimeDelta::minutes(150), &hourly_ts), 3);

    // Horizon smaller than interval → minimum 1 step
    assert_eq!(horizon_to_steps(&TimeDelta::minutes(30), &hourly_ts), 1);

    // Ceiling behavior: just over 1 interval → 2 steps
    assert_eq!(horizon_to_steps(&TimeDelta::minutes(61), &hourly_ts), 2);

    // Ceiling behavior: just under 1 interval → still 1 step
    assert_eq!(horizon_to_steps(&TimeDelta::minutes(59), &hourly_ts), 1);

    // Ceiling behavior: 25 hours on daily data → 2 steps
    assert_eq!(horizon_to_steps(&TimeDelta::hours(25), &daily_ts), 2);

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

    // 2.5 hours on hourly data → 3 steps (ceiling)
    let input = PredictionInput {
        data: hourly_data,
        horizon: TimeDelta::minutes(150),
    };

    let result = predict(&input).unwrap();
    assert_eq!(
        result.forecast_values.len(),
        3,
        "2.5 hours on hourly data should produce 3 forecasts (ceiling)"
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
fn test_predicted_std_matches_quantile_band() {
    // predicted_std must satisfy the contract
    //   std[t] ≈ (upper[t] - lower[t]) / (2 * Z_SCORE_80_INTERVAL)
    // whenever both bounds are present.
    let data = make_data(120);
    let input = PredictionInput {
        data,
        horizon: TimeDelta::hours(8),
    };
    let result = predict(&input).unwrap();

    // If the strategy emits no quantile bands the contract is vacuously
    // satisfied (predicted_std must also be None). Skip the inverse check.
    let (Some(lower), Some(upper), Some(std)) = (
        result.lower_bound.as_ref(),
        result.upper_bound.as_ref(),
        result.predicted_std.as_ref(),
    ) else {
        assert!(result.predicted_std.is_none());
        return;
    };

    assert_eq!(std.len(), result.forecast_values.len());
    for (ts, std_value) in std {
        let lo = lower.get(ts).expect("lower bound missing for std ts");
        let hi = upper.get(ts).expect("upper bound missing for std ts");
        let lo_f = lo.to_f64().unwrap();
        let hi_f = hi.to_f64().unwrap();
        let expected = ((hi_f - lo_f) / (2.0 * Z_SCORE_80_INTERVAL)).max(0.0);
        let actual = std_value.to_f64().unwrap();
        assert!(actual >= 0.0, "predicted_std must be non-negative");
        assert_relative_eq!(actual, expected, max_relative = 1e-9, epsilon = 1e-9);
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
            .array_windows()
            .map(|[a, b]| (**b - **a).num_seconds())
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

// --- Predictor pub API tests ---

#[test]
fn test_predictor_new_zero_threads_returns_error() {
    let result = Predictor::new(0);
    assert!(result.is_err());
    let err = format!("{}", result.err().unwrap());
    assert!(
        err.contains("max_model_threads must be > 0"),
        "Expected InvalidInput error, got: {err}"
    );
}

#[test]
fn test_predictor_new_and_predict() {
    let predictor = Predictor::new(1).expect("Failed to create Predictor");
    let input = PredictionInput {
        data: make_data(100),
        horizon: TimeDelta::hours(10),
    };
    let result = predictor.predict(&input).unwrap();
    assert_eq!(result.forecast_values.len(), 10);
    assert!(result.model_count > 0);
}

#[test]
fn test_predictor_predict_insufficient_data() {
    let predictor = Predictor::new(1).unwrap();
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let input = PredictionInput {
        data: [(base, BigDecimal::from(1))].into_iter().collect(),
        horizon: TimeDelta::hours(5),
    };
    assert!(predictor.predict(&input).is_err());
}

#[test]
fn test_predictor_predict_zero_horizon() {
    let predictor = Predictor::new(1).unwrap();
    let input = PredictionInput {
        data: make_data_with_values(&[1.0; 10]),
        horizon: TimeDelta::zero(),
    };
    assert!(predictor.predict(&input).is_err());
}

// --- safe_exp tests ---

#[test]
fn test_safe_exp_normal_values() {
    assert_relative_eq!(safe_exp(0.0), 1.0);
    assert_relative_eq!(safe_exp(1.0), 1.0_f64.exp());
    assert_relative_eq!(safe_exp(-1.0), (-1.0_f64).exp());
}

#[test]
fn test_safe_exp_nan_passthrough() {
    assert!(safe_exp(f64::NAN).is_nan());
}

#[test]
fn test_safe_exp_large_value_clamped() {
    assert_relative_eq!(safe_exp(710.0), 709.0_f64.exp());
}

#[test]
fn test_safe_exp_negative_infinity() {
    assert_eq!(safe_exp(f64::NEG_INFINITY), 0.0);
}

#[test]
fn test_safe_exp_boundary() {
    assert_relative_eq!(safe_exp(709.0), 709.0_f64.exp());
}

#[test]
fn test_safe_exp_positive_infinity_clamped() {
    assert_relative_eq!(safe_exp(f64::INFINITY), 709.0_f64.exp());
}
