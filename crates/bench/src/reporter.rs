use crate::backtester::BacktestResult;

/// Print a formatted table of backtest results to stdout.
pub fn print_report(results: &[BacktestResult]) {
    if results.is_empty() {
        println!("No results to report.");
        return;
    }

    // Header
    println!(
        "{:<30} {:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Fixture", "Model", "MAE", "RMSE", "MAPE%", "sMAPE%", "MASE", "WAPE%"
    );
    println!("{}", "-".repeat(110));

    // Group by fixture
    let mut current_fixture = String::new();
    for r in results {
        if r.fixture_name != current_fixture {
            if !current_fixture.is_empty() {
                println!("{}", "-".repeat(110));
            }
            current_fixture = r.fixture_name.clone();
        }

        let mape_str = if r.metrics.mape.is_infinite() {
            "Inf".to_string()
        } else {
            format!("{:.2}", r.metrics.mape)
        };
        let mase_str = if r.metrics.mase.is_infinite() {
            "Inf".to_string()
        } else {
            format!("{:.3}", r.metrics.mase)
        };
        let wape_str = if r.metrics.wape.is_infinite() {
            "Inf".to_string()
        } else {
            format!("{:.2}", r.metrics.wape)
        };

        println!(
            "{:<30} {:<20} {:>8.2} {:>8.2} {:>8} {:>8.2} {:>8} {:>8}",
            r.fixture_name,
            r.model_name,
            r.metrics.mae,
            r.metrics.rmse,
            mape_str,
            r.metrics.smape,
            mase_str,
            wape_str,
        );
    }
    println!("{}", "-".repeat(110));

    // Summary: average MASE per model across all fixtures
    println!("\n=== Average MASE by Model ===");
    let model_names = collect_model_names(results);
    for model in &model_names {
        let mases: Vec<f64> = results
            .iter()
            .filter(|r| r.model_name == *model && r.metrics.mase.is_finite())
            .map(|r| r.metrics.mase)
            .collect();
        if !mases.is_empty() {
            let avg = mases.iter().sum::<f64>() / mases.len() as f64;
            let finite_count = mases.len();
            let total_count = results.iter().filter(|r| r.model_name == *model).count();
            println!(
                "  {:<20} avg MASE = {:.3}  ({}/{} fixtures)",
                model, avg, finite_count, total_count
            );
        }
    }
}

fn collect_model_names(results: &[BacktestResult]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for r in results {
        if !names.contains(&r.model_name) {
            names.push(r.model_name.clone());
        }
    }
    names
}
