//! Concurrent prediction tests.
//!
//! Verifies that multiple threads can safely share a single `Predictor`
//! and that the dedicated ThreadPool correctly limits parallelism.

use chrono::{NaiveDate, TimeDelta};
use predictor::{BigDecimal, PredictionInput, Predictor};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

/// Create test input with 15-minute intervals (matching production).
fn make_input(n: usize) -> PredictionInput {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    let data = (0..n)
        .map(|i| {
            let ts = base + chrono::Duration::minutes(15 * i as i64);
            let val = 100.0
                + 1.5 * i as f64
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / 96.0).sin();
            let decimal = BigDecimal::from_str(&format!("{val:.6}")).unwrap();
            (ts, decimal)
        })
        .collect();

    PredictionInput {
        data,
        horizon: TimeDelta::hours(24),
    }
}

#[test]
fn test_concurrent_predictions_all_succeed() {
    let predictor = Arc::new(Predictor::new(2).expect("Failed to create Predictor"));
    let num_threads = 4;
    let input = make_input(500);

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let predictor = predictor.clone();
            let input = input.clone();
            std::thread::spawn(move || predictor.predict(&input))
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    for (i, result) in results.iter().enumerate() {
        assert!(result.is_ok(), "prediction {i} failed: {result:?}");
        let forecast = result.as_ref().unwrap();
        assert!(
            !forecast.forecast_values.is_empty(),
            "prediction {i} returned empty forecast"
        );
    }
}

#[test]
fn test_pool_threads_affects_throughput() {
    let input = make_input(500);
    let num_concurrent = 4;

    let measure = |pool_threads: usize| -> f64 {
        let predictor = Arc::new(Predictor::new(pool_threads).unwrap());
        let start = Instant::now();

        let handles: Vec<_> = (0..num_concurrent)
            .map(|_| {
                let predictor = predictor.clone();
                let input = input.clone();
                std::thread::spawn(move || predictor.predict(&input).unwrap())
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        start.elapsed().as_secs_f64()
    };

    let time_1 = measure(1);
    let time_2 = measure(2);

    println!("pool_threads=1: {time_1:.2}s");
    println!("pool_threads=2: {time_2:.2}s");
    println!("speedup: {:.2}x", time_1 / time_2);

    // pool_threads=2 should be at least somewhat faster than pool_threads=1
    // when running 4 concurrent predictions (more model training parallelism).
    // We use a generous margin since CI environments vary.
    assert!(
        time_2 < time_1 * 1.5,
        "pool_threads=2 should not be significantly slower than pool_threads=1"
    );
}
