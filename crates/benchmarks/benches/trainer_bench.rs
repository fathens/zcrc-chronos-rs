//! Benchmarks for hierarchical trainer.
//!
//! Covers: stage training, full hierarchical training.

use chrono::{NaiveDate, NaiveDateTime};
use common::ModelSelectionStrategy;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use trainer::{HierarchicalTrainer, TrainingHints};

fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::hours(i as i64))
        .collect()
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

fn get_fast_strategy() -> ModelSelectionStrategy {
    // Minimal strategy for benchmarking
    ModelSelectionStrategy {
        strategy_name: "bench_fast".into(),
        priority_models: vec!["SeasonalNaive".into()],
        time_allocation: common::TimeAllocation {
            fast: 1.0,
            medium: 0.0,
            advanced: 0.0,
        },
        excluded_models: vec![],
        preset: "speed".into(),
    }
}

fn get_full_strategy() -> ModelSelectionStrategy {
    ModelSelectionStrategy {
        strategy_name: "bench_full".into(),
        priority_models: vec!["SeasonalNaive".into(), "ETS".into(), "NPTS".into()],
        time_allocation: common::TimeAllocation {
            fast: 0.4,
            medium: 0.6,
            advanced: 0.0,
        },
        excluded_models: vec![],
        preset: "balanced".into(),
    }
}

fn bench_train_fast_stage(c: &mut Criterion) {
    let mut group = c.benchmark_group("train_fast_stage");

    for n in [100, 500, 1000] {
        let values = generate_trend_seasonal_data(n, 12);
        let timestamps = make_timestamps(n);
        let strategy = get_fast_strategy();
        let horizon = (n / 10).max(5);

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(values, timestamps, strategy, horizon),
            |b, (vals, ts, strat, h)| {
                b.iter(|| {
                    let mut trainer = HierarchicalTrainer::default();
                    trainer.train_hierarchically(
                        black_box(vals),
                        black_box(ts),
                        black_box(strat),
                        60.0,
                        *h,
                        TrainingHints {
                            season_period: Some(12),
                            volatility: None,
                        },
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_train_hierarchically(c: &mut Criterion) {
    let mut group = c.benchmark_group("train_hierarchically");
    // Longer measurement time for comprehensive benchmarks
    group.sample_size(10);

    for n in [100, 500] {
        let values = generate_trend_seasonal_data(n, 12);
        let timestamps = make_timestamps(n);
        let strategy = get_full_strategy();
        let horizon = (n / 10).max(5);

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(values, timestamps, strategy, horizon),
            |b, (vals, ts, strat, h)| {
                b.iter(|| {
                    let mut trainer = HierarchicalTrainer::default();
                    trainer.train_hierarchically(
                        black_box(vals),
                        black_box(ts),
                        black_box(strat),
                        60.0,
                        *h,
                        TrainingHints {
                            season_period: Some(12),
                            volatility: None,
                        },
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_train_fast_stage, bench_train_hierarchically);
criterion_main!(benches);
