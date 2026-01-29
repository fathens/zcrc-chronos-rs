use super::*;

fn make_seasonal_additive(n: usize, m: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            500.0
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / m as f64).sin()
        })
        .collect()
}

fn make_seasonal_multiplicative(n: usize, m: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let base = 500.0;
            let seasonal_ratio = 1.0 + 0.3 * (2.0 * std::f64::consts::PI * i as f64 / m as f64).sin();
            base * seasonal_ratio
        })
        .collect()
}

fn make_trend_seasonal(n: usize, m: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            100.0
                + 2.0 * i as f64
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / m as f64).sin()
        })
        .collect()
}

#[test]
fn test_initialize_additive_seasonal() {
    let m = 4;
    let values = vec![10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0, 45.0];
    let spec = HwSpec {
        trend: TrendType::Additive,
        seasonal: SeasonalType::Additive,
    };
    let (level, trend, seasonal) = initialize_state(&values, m, spec);

    // Level = mean of first cycle = (10+20+30+40)/4 = 25
    assert!((level - 25.0).abs() < 1e-10);

    // Trend = (mean2 - mean1) / m = ((15+25+35+45)/4 - 25) / 4 = (30-25)/4 = 1.25
    assert!((trend - 1.25).abs() < 1e-10);

    // Seasonal[i] = values[i] - level
    assert!((seasonal[0] - (-15.0)).abs() < 1e-10);
    assert!((seasonal[1] - (-5.0)).abs() < 1e-10);
    assert!((seasonal[2] - 5.0).abs() < 1e-10);
    assert!((seasonal[3] - 15.0).abs() < 1e-10);
}

#[test]
fn test_nelder_mead_quadratic() {
    // Minimize f(x,y) = (x-3)^2 + (y-5)^2
    let result = nelder_mead(
        |p| (p[0] - 3.0).powi(2) + (p[1] - 5.0).powi(2),
        &[0.0, 0.0],
        &NelderMeadBounds {
            lower: vec![-10.0, -10.0],
            upper: vec![10.0, 10.0],
        },
        500,
        1e-8,
    );
    assert!((result[0] - 3.0).abs() < 0.01, "x = {}", result[0]);
    assert!((result[1] - 5.0).abs() < 0.01, "y = {}", result[1]);
}

#[test]
fn test_nelder_mead_bounds() {
    // Minimum is at (3,5), but upper bound constrains to [0,2] × [0,2]
    let result = nelder_mead(
        |p| (p[0] - 3.0).powi(2) + (p[1] - 5.0).powi(2),
        &[1.0, 1.0],
        &NelderMeadBounds {
            lower: vec![0.0, 0.0],
            upper: vec![2.0, 2.0],
        },
        500,
        1e-8,
    );
    assert!(
        result[0] >= 0.0 && result[0] <= 2.0,
        "x out of bounds: {}",
        result[0]
    );
    assert!(
        result[1] >= 0.0 && result[1] <= 2.0,
        "y out of bounds: {}",
        result[1]
    );
    // Should be close to (2, 2) — the nearest corner to (3, 5) within bounds
    assert!((result[0] - 2.0).abs() < 0.1, "x = {}", result[0]);
    assert!((result[1] - 2.0).abs() < 0.1, "y = {}", result[1]);
}

#[test]
fn test_fit_additive_seasonal() {
    let m = 12;
    let values = make_seasonal_additive(120, m);
    let result = hw_fit_predict(&values, m, 12).unwrap();
    assert_eq!(result.mean.len(), 12);

    // Forecast should roughly follow the seasonal pattern
    for (i, &v) in result.mean.iter().enumerate() {
        let expected = 500.0
            + 30.0
                * (2.0 * std::f64::consts::PI * (120 + i) as f64 / m as f64).sin();
        assert!(
            (v - expected).abs() < 50.0,
            "h={}: forecast={:.1}, expected={:.1}",
            i,
            v,
            expected
        );
    }
}

#[test]
fn test_fit_multiplicative_seasonal() {
    let m = 12;
    let values = make_seasonal_multiplicative(120, m);
    let result = hw_fit_predict(&values, m, 12).unwrap();
    assert_eq!(result.mean.len(), 12);

    for &v in &result.mean {
        // Should be in a reasonable range around 500
        assert!(
            v > 200.0 && v < 800.0,
            "forecast out of range: {:.1}",
            v
        );
    }
}

#[test]
fn test_aicc_penalizes_complexity() {
    let sse = 1000.0;
    let n = 100;
    let m = 12;

    let simple = HwSpec {
        trend: TrendType::None,
        seasonal: SeasonalType::Additive,
    };
    let complex = HwSpec {
        trend: TrendType::AdditiveDamped,
        seasonal: SeasonalType::Additive,
    };

    let aicc_simple = compute_aicc(sse, n, simple, m);
    let aicc_complex = compute_aicc(sse, n, complex, m);

    // Same SSE, but complex model has more parameters → higher AICc
    assert!(
        aicc_complex > aicc_simple,
        "simple={:.2}, complex={:.2}",
        aicc_simple,
        aicc_complex
    );
}

#[test]
fn test_forecast_seasonal_shape() {
    let m = 4;
    let n = 40;
    let values = make_seasonal_additive(n, m);
    let result = hw_fit_predict(&values, m, 8).unwrap();

    // 2 full cycles of forecast; each cycle should have similar pattern
    let cycle1: Vec<f64> = result.mean[0..4].to_vec();
    let cycle2: Vec<f64> = result.mean[4..8].to_vec();

    for i in 0..4 {
        assert!(
            (cycle1[i] - cycle2[i]).abs() < 20.0,
            "Cycles differ at {}: {:.1} vs {:.1}",
            i,
            cycle1[i],
            cycle2[i]
        );
    }
}

#[test]
fn test_forecast_intervals_widen() {
    let m = 12;
    let values = make_seasonal_additive(120, m);
    let result = hw_fit_predict(&values, m, 12).unwrap();

    // Interval widths should increase with horizon
    let widths: Vec<f64> = result
        .upper
        .iter()
        .zip(result.lower.iter())
        .map(|(u, l)| u - l)
        .collect();

    for i in 1..widths.len() {
        assert!(
            widths[i] >= widths[i - 1] - 1e-10,
            "Width shrank at h={}: {:.2} < {:.2}",
            i,
            widths[i],
            widths[i - 1]
        );
    }
}

#[test]
fn test_negative_values_skip_multiplicative() {
    let m = 4;
    let values: Vec<f64> = (0..40)
        .map(|i| 10.0 * (2.0 * std::f64::consts::PI * i as f64 / m as f64).sin())
        .collect();
    // Ensure negatives are present
    assert!(values.iter().any(|&v| v < 0.0));

    // Should still produce a valid forecast (multiplicative candidates skipped)
    let result = hw_fit_predict(&values, m, 4).unwrap();
    assert_eq!(result.mean.len(), 4);
    for &v in &result.mean {
        assert!(!v.is_nan(), "NaN in forecast");
    }
}

#[test]
fn test_constant_series() {
    let m = 4;
    let values = vec![42.0; 40];
    let result = hw_fit_predict(&values, m, 4).unwrap();
    assert_eq!(result.mean.len(), 4);
    for &v in &result.mean {
        assert!(
            (v - 42.0).abs() < 5.0,
            "Expected ~42, got {:.1}",
            v
        );
    }
}

#[test]
fn test_insufficient_data() {
    let m = 12;
    let values = vec![1.0; 20]; // Less than 2*12=24
    let result = hw_fit_predict(&values, m, 5);
    assert!(result.is_err());
}

#[test]
fn test_trend_plus_seasonal() {
    let m = 12;
    let values = make_trend_seasonal(120, m);
    let result = hw_fit_predict(&values, m, 12).unwrap();
    assert_eq!(result.mean.len(), 12);

    // Should continue the upward trend
    let last_train = values.last().copied().unwrap();
    let forecast_mean: f64 = result.mean.iter().sum::<f64>() / result.mean.len() as f64;
    assert!(
        forecast_mean > last_train - 50.0,
        "Forecast mean ({:.1}) should be near or above last training value ({:.1})",
        forecast_mean,
        last_train
    );
}
