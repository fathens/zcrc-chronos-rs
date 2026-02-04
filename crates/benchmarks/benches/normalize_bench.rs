//! Benchmarks for time-series normalization functions.
//!
//! Covers: normalize_time_series_data, nearest_resample (via normalization).

use chrono::{NaiveDate, NaiveDateTime};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use normalize::normalize_time_series_data;

fn make_regular_timestamps(n: usize, interval_secs: i64) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::seconds(interval_secs * i as i64))
        .collect()
}

fn make_irregular_timestamps(n: usize, seed: u64) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    let mut state = seed;
    let mut timestamps = Vec::with_capacity(n);
    let mut current = base;

    for _ in 0..n {
        timestamps.push(current);
        // LCG-based random interval: 30s to 2h
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let frac = ((state >> 33) as f64) / (u32::MAX as f64);
        let interval = 30 + (frac * 7170.0) as i64; // 30s to 7200s (2h)
        current += chrono::Duration::seconds(interval);
    }

    timestamps
}

fn generate_values(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 100.0 + 1.5 * i as f64 + 20.0 * (i as f64 * 0.1).sin())
        .collect()
}

fn bench_normalize_regular(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_regular");

    for n in [100, 500, 1000, 5000] {
        let timestamps = make_regular_timestamps(n, 3600); // hourly
        let values = generate_values(n);
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(timestamps, values),
            |b, (ts, vals)| b.iter(|| normalize_time_series_data(black_box(ts), black_box(vals))),
        );
    }

    group.finish();
}

fn bench_normalize_irregular(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_irregular");

    for n in [100, 500, 1000, 5000] {
        let timestamps = make_irregular_timestamps(n, 42);
        let values = generate_values(n);
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(timestamps, values),
            |b, (ts, vals)| b.iter(|| normalize_time_series_data(black_box(ts), black_box(vals))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_normalize_regular, bench_normalize_irregular);
criterion_main!(benches);
