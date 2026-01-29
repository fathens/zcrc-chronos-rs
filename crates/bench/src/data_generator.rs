use chrono::NaiveDateTime;

/// A time-series fixture with train/test split for benchmarking.
#[derive(Debug, Clone)]
pub struct TimeSeriesFixture {
    pub name: String,
    pub train_values: Vec<f64>,
    pub test_values: Vec<f64>,
    pub train_timestamps: Vec<NaiveDateTime>,
    pub test_timestamps: Vec<NaiveDateTime>,
    pub horizon: usize,
    pub expected_characteristics: ExpectedCharacteristics,
}

/// Expected high-level characteristics for sanity-checking analysis output.
#[derive(Debug, Clone)]
pub struct ExpectedCharacteristics {
    pub has_trend: bool,
    pub has_seasonality: bool,
    pub seasonal_period: Option<usize>,
}

/// Generate all standard benchmark fixtures.
///
/// Returns fixtures across multiple data lengths and pattern types.
pub fn generate_all_fixtures() -> Vec<TimeSeriesFixture> {
    let mut fixtures = Vec::new();
    for &length in &[100, 200, 500] {
        let horizon = (length / 10).max(5);
        fixtures.push(pure_trend(length, horizon));
        fixtures.push(trend_with_noise(length, horizon));
        fixtures.push(pure_seasonal(length, horizon));
        fixtures.push(trend_plus_seasonal(length, horizon));
        fixtures.push(multi_seasonal(length, horizon));
        fixtures.push(changepoint(length, horizon));
        fixtures.push(stationary_noise(length, horizon));
        fixtures.push(intermittent(length, horizon));
        fixtures.push(exponential_growth(length, horizon));
        fixtures.push(damped_trend(length, horizon));
    }
    fixtures
}

fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
    let base = chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::hours(i as i64))
        .collect()
}

fn split(
    values: &[f64],
    horizon: usize,
) -> (Vec<f64>, Vec<f64>, Vec<NaiveDateTime>, Vec<NaiveDateTime>) {
    let n = values.len();
    let train_end = n - horizon;
    let ts = make_timestamps(n);
    (
        values[..train_end].to_vec(),
        values[train_end..].to_vec(),
        ts[..train_end].to_vec(),
        ts[train_end..].to_vec(),
    )
}

/// Deterministic pseudo-random: simple LCG-based noise in [-amplitude, amplitude].
fn noise(seed: u64, n: usize, amplitude: f64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            // LCG parameters (Numerical Recipes)
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let frac = ((state >> 33) as f64) / (u32::MAX as f64); // 0..1
            (frac * 2.0 - 1.0) * amplitude
        })
        .collect()
}

pub fn pure_trend(n: usize, horizon: usize) -> TimeSeriesFixture {
    let values: Vec<f64> = (0..n).map(|i| 100.0 + 2.5 * i as f64).collect();
    let (tv, te, tts, tte) = split(&values, horizon);
    TimeSeriesFixture {
        name: format!("pure_trend_{n}"),
        train_values: tv,
        test_values: te,
        train_timestamps: tts,
        test_timestamps: tte,
        horizon,
        expected_characteristics: ExpectedCharacteristics {
            has_trend: true,
            has_seasonality: false,
            seasonal_period: None,
        },
    }
}

pub fn trend_with_noise(n: usize, horizon: usize) -> TimeSeriesFixture {
    let ns = noise(42, n, 5.0);
    let values: Vec<f64> = (0..n).map(|i| 100.0 + 2.0 * i as f64 + ns[i]).collect();
    let (tv, te, tts, tte) = split(&values, horizon);
    TimeSeriesFixture {
        name: format!("trend_with_noise_{n}"),
        train_values: tv,
        test_values: te,
        train_timestamps: tts,
        test_timestamps: tte,
        horizon,
        expected_characteristics: ExpectedCharacteristics {
            has_trend: true,
            has_seasonality: false,
            seasonal_period: None,
        },
    }
}

pub fn pure_seasonal(n: usize, horizon: usize) -> TimeSeriesFixture {
    let period = 12;
    let values: Vec<f64> = (0..n)
        .map(|i| 500.0 + 50.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
        .collect();
    let (tv, te, tts, tte) = split(&values, horizon);
    TimeSeriesFixture {
        name: format!("pure_seasonal_{n}"),
        train_values: tv,
        test_values: te,
        train_timestamps: tts,
        test_timestamps: tte,
        horizon,
        expected_characteristics: ExpectedCharacteristics {
            has_trend: false,
            has_seasonality: true,
            seasonal_period: Some(period),
        },
    }
}

pub fn trend_plus_seasonal(n: usize, horizon: usize) -> TimeSeriesFixture {
    let period = 12;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            100.0
                + 1.5 * i as f64
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
        })
        .collect();
    let (tv, te, tts, tte) = split(&values, horizon);
    TimeSeriesFixture {
        name: format!("trend_plus_seasonal_{n}"),
        train_values: tv,
        test_values: te,
        train_timestamps: tts,
        test_timestamps: tte,
        horizon,
        expected_characteristics: ExpectedCharacteristics {
            has_trend: true,
            has_seasonality: true,
            seasonal_period: Some(period),
        },
    }
}

pub fn multi_seasonal(n: usize, horizon: usize) -> TimeSeriesFixture {
    let p1 = 12;
    let p2 = 24;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            500.0
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / p1 as f64).sin()
                + 15.0 * (2.0 * std::f64::consts::PI * i as f64 / p2 as f64).sin()
        })
        .collect();
    let (tv, te, tts, tte) = split(&values, horizon);
    TimeSeriesFixture {
        name: format!("multi_seasonal_{n}"),
        train_values: tv,
        test_values: te,
        train_timestamps: tts,
        test_timestamps: tte,
        horizon,
        expected_characteristics: ExpectedCharacteristics {
            has_trend: false,
            has_seasonality: true,
            seasonal_period: Some(p1),
        },
    }
}

pub fn changepoint(n: usize, horizon: usize) -> TimeSeriesFixture {
    let mid = n / 2;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            if i < mid {
                100.0 + 1.0 * i as f64
            } else {
                100.0 + 1.0 * mid as f64 + 3.0 * (i - mid) as f64
            }
        })
        .collect();
    let (tv, te, tts, tte) = split(&values, horizon);
    TimeSeriesFixture {
        name: format!("changepoint_{n}"),
        train_values: tv,
        test_values: te,
        train_timestamps: tts,
        test_timestamps: tte,
        horizon,
        expected_characteristics: ExpectedCharacteristics {
            has_trend: true,
            has_seasonality: false,
            seasonal_period: None,
        },
    }
}

pub fn stationary_noise(n: usize, horizon: usize) -> TimeSeriesFixture {
    let ns = noise(123, n, 10.0);
    let values: Vec<f64> = (0..n).map(|i| 100.0 + ns[i]).collect();
    let (tv, te, tts, tte) = split(&values, horizon);
    TimeSeriesFixture {
        name: format!("stationary_noise_{n}"),
        train_values: tv,
        test_values: te,
        train_timestamps: tts,
        test_timestamps: tte,
        horizon,
        expected_characteristics: ExpectedCharacteristics {
            has_trend: false,
            has_seasonality: false,
            seasonal_period: None,
        },
    }
}

pub fn intermittent(n: usize, horizon: usize) -> TimeSeriesFixture {
    let ns = noise(999, n, 1.0);
    let values: Vec<f64> = (0..n)
        .map(|i| {
            // Mostly zero with occasional spikes
            if ns[i] > 0.6 {
                50.0 + ns[i] * 20.0
            } else {
                0.0
            }
        })
        .collect();
    let (tv, te, tts, tte) = split(&values, horizon);
    TimeSeriesFixture {
        name: format!("intermittent_{n}"),
        train_values: tv,
        test_values: te,
        train_timestamps: tts,
        test_timestamps: tte,
        horizon,
        expected_characteristics: ExpectedCharacteristics {
            has_trend: false,
            has_seasonality: false,
            seasonal_period: None,
        },
    }
}

pub fn exponential_growth(n: usize, horizon: usize) -> TimeSeriesFixture {
    let values: Vec<f64> = (0..n).map(|i| 100.0 * (0.02 * i as f64).exp()).collect();
    let (tv, te, tts, tte) = split(&values, horizon);
    TimeSeriesFixture {
        name: format!("exponential_growth_{n}"),
        train_values: tv,
        test_values: te,
        train_timestamps: tts,
        test_timestamps: tte,
        horizon,
        expected_characteristics: ExpectedCharacteristics {
            has_trend: true,
            has_seasonality: false,
            seasonal_period: None,
        },
    }
}

pub fn damped_trend(n: usize, horizon: usize) -> TimeSeriesFixture {
    let phi: f64 = 0.95;
    let mut values = vec![100.0];
    let slope_init = 2.0;
    for i in 1..n {
        let damped_slope = slope_init * phi.powi(i as i32);
        values.push(values[i - 1] + damped_slope);
    }
    let (tv, te, tts, tte) = split(&values, horizon);
    TimeSeriesFixture {
        name: format!("damped_trend_{n}"),
        train_values: tv,
        test_values: te,
        train_timestamps: tts,
        test_timestamps: tte,
        horizon,
        expected_characteristics: ExpectedCharacteristics {
            has_trend: true,
            has_seasonality: false,
            seasonal_period: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_fixtures_valid() {
        let fixtures = generate_all_fixtures();
        assert!(!fixtures.is_empty());
        for f in &fixtures {
            assert_eq!(
                f.train_values.len(),
                f.train_timestamps.len(),
                "fixture '{}': train len mismatch",
                f.name
            );
            assert_eq!(
                f.test_values.len(),
                f.test_timestamps.len(),
                "fixture '{}': test len mismatch",
                f.name
            );
            assert_eq!(
                f.test_values.len(),
                f.horizon,
                "fixture '{}': test len != horizon",
                f.name
            );
            assert!(
                f.train_values.len() >= 10,
                "fixture '{}': too little training data",
                f.name
            );
            // No NaN
            for v in &f.train_values {
                assert!(!v.is_nan(), "fixture '{}': NaN in train", f.name);
            }
            for v in &f.test_values {
                assert!(!v.is_nan(), "fixture '{}': NaN in test", f.name);
            }
        }
    }

    #[test]
    fn test_fixture_count() {
        let fixtures = generate_all_fixtures();
        // 10 patterns × 3 lengths = 30
        assert_eq!(fixtures.len(), 30);
    }
}
