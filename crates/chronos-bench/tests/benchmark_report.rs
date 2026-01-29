use chronos_bench::backtester;
use chronos_bench::data_generator;
use chronos_bench::reporter;

#[test]
fn benchmark_all_fixtures() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("warn")
        .try_init();

    let fixtures = data_generator::generate_all_fixtures();
    let results = backtester::run_all_backtests(&fixtures);

    println!("\n========== BENCHMARK REPORT ==========\n");
    reporter::print_report(&results);

    // Sanity: every fixture should have at least one result
    for f in &fixtures {
        let count = results.iter().filter(|r| r.fixture_name == f.name).count();
        assert!(
            count > 0,
            "No results for fixture '{}'",
            f.name
        );
    }

    // Quality gates: ETS seasonal accuracy
    // ETS with Holt-Winters should produce reasonable forecasts on seasonal fixtures.
    // trend_plus_seasonal: ETS handles trend + seasonality well (MASE << 1)
    // multi_seasonal: ETS captures dominant period (MASE varies by data length)
    let seasonal_gates: &[(&str, &str, f64)] = &[
        ("trend_plus_seasonal_100", "ETS", 0.5),
        ("trend_plus_seasonal_200", "ETS", 0.5),
        ("trend_plus_seasonal_500", "ETS", 0.5),
        ("multi_seasonal_100", "ETS", 2.0),
        ("multi_seasonal_200", "ETS", 2.0),
        ("multi_seasonal_500", "ETS", 2.0),
    ];

    for &(fixture, model, threshold) in seasonal_gates {
        if let Some(r) = results
            .iter()
            .find(|r| r.fixture_name == fixture && r.model_name == model)
        {
            assert!(
                r.metrics.mase.is_finite() && r.metrics.mase < threshold,
                "Quality gate failed: {fixture}/{model} MASE = {:.3}, expected < {threshold}",
                r.metrics.mase
            );
        }
    }

    // Ensemble quality gate: ensemble MASE should be within 1.5x of best individual model
    // for seasonal fixtures. This validates that score-based filtering prevents
    // poor models from diluting ensemble accuracy.
    let seasonal_fixtures = [
        "pure_seasonal_100",
        "pure_seasonal_200",
        "pure_seasonal_500",
        "trend_plus_seasonal_100",
        "trend_plus_seasonal_200",
        "trend_plus_seasonal_500",
        "multi_seasonal_100",
        "multi_seasonal_200",
        "multi_seasonal_500",
    ];

    for fixture_name in &seasonal_fixtures {
        let fixture_results: Vec<_> = results
            .iter()
            .filter(|r| r.fixture_name == *fixture_name)
            .collect();

        if fixture_results.is_empty() {
            continue;
        }

        // Find best individual model (excluding ensemble)
        let best_individual = fixture_results
            .iter()
            .filter(|r| !r.model_name.starts_with("Ensemble"))
            .filter(|r| r.metrics.mase.is_finite())
            .min_by(|a, b| a.metrics.mase.partial_cmp(&b.metrics.mase).unwrap());

        // Find ensemble result
        let ensemble = fixture_results
            .iter()
            .find(|r| r.model_name.starts_with("Ensemble"));

        if let (Some(best), Some(ens)) = (best_individual, ensemble) {
            // Skip comparison if best MASE is near-zero (perfect fit or degenerate case)
            if ens.metrics.mase.is_finite()
                && best.metrics.mase.is_finite()
                && best.metrics.mase > 0.01
            {
                let ratio = ens.metrics.mase / best.metrics.mase;
                assert!(
                    ratio <= 1.5,
                    "Ensemble quality gate failed: {fixture_name} ensemble MASE ({:.3}) is {ratio:.2}x best individual ({} MASE={:.3}), expected <= 1.5x",
                    ens.metrics.mase,
                    best.model_name,
                    best.metrics.mase
                );
            }
        }
    }
}
