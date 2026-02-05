//! Benchmarks for time-series analyzer functions.
//!
//! Covers: FFT seasonality detection, Mann-Kendall test, trend analysis, outlier detection.

use analyzer::TimeSeriesAnalyzer;
use chrono::{NaiveDate, NaiveDateTime};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::hours(i as i64))
        .collect()
}

fn generate_seasonal_data(n: usize, period: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 100.0 + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
        .collect()
}

fn generate_trend_data(n: usize) -> Vec<f64> {
    (0..n).map(|i| 100.0 + 2.0 * i as f64).collect()
}

fn generate_trend_seasonal_data(n: usize, period: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            100.0
                + 1.5 * i as f64
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
        })
        .collect()
}

fn bench_detect_seasonality(c: &mut Criterion) {
    let analyzer = TimeSeriesAnalyzer::new();
    let mut group = c.benchmark_group("detect_seasonality");

    for n in [100, 500, 1000, 5000] {
        let values = generate_seasonal_data(n, 12);
        group.bench_with_input(BenchmarkId::from_parameter(n), &values, |b, vals| {
            b.iter(|| analyzer.detect_seasonality(black_box(vals)))
        });
    }

    group.finish();
}

fn bench_mann_kendall(c: &mut Criterion) {
    let analyzer = TimeSeriesAnalyzer::new();
    let mut group = c.benchmark_group("mann_kendall_test");

    for n in [100, 500, 1000, 5000] {
        let values = generate_trend_data(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &values, |b, vals| {
            b.iter(|| analyzer.mann_kendall_test(black_box(vals)))
        });
    }

    group.finish();
}

fn bench_analyze_trend(c: &mut Criterion) {
    let analyzer = TimeSeriesAnalyzer::new();
    let mut group = c.benchmark_group("analyze_trend");

    for n in [100, 500, 1000, 5000] {
        let values = generate_trend_data(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &values, |b, vals| {
            b.iter(|| analyzer.analyze_trend(black_box(vals)))
        });
    }

    group.finish();
}

fn bench_detect_outliers(c: &mut Criterion) {
    let analyzer = TimeSeriesAnalyzer::new();
    let mut group = c.benchmark_group("detect_outliers");

    for n in [100, 500, 1000, 5000] {
        let mut values: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
            .collect();
        // Add some outliers
        if n > 10 {
            values[n / 4] = 1000.0;
            values[n / 2] = -500.0;
        }
        group.bench_with_input(BenchmarkId::from_parameter(n), &values, |b, vals| {
            b.iter(|| analyzer.detect_outliers(black_box(vals)))
        });
    }

    group.finish();
}

fn bench_full_analyze(c: &mut Criterion) {
    let analyzer = TimeSeriesAnalyzer::new();
    let mut group = c.benchmark_group("analyze_full");

    for n in [100, 500, 1000, 5000] {
        let values = generate_trend_seasonal_data(n, 12);
        let timestamps = make_timestamps(n);
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(values, timestamps),
            |b, (vals, ts)| b.iter(|| analyzer.analyze(black_box(vals), black_box(ts))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_detect_seasonality,
    bench_mann_kendall,
    bench_analyze_trend,
    bench_detect_outliers,
    bench_full_analyze
);
criterion_main!(benches);
