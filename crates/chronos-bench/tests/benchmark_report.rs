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
}
