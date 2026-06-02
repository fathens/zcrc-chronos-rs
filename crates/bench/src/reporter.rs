use crate::backtester::BacktestResult;

/// Print a formatted table of backtest results to stdout.
pub fn print_report(results: &[BacktestResult]) {
    if results.is_empty() {
        println!("No results to report.");
        return;
    }

    // Header
    println!(
        "{:<30} {:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7} {:>7}",
        "Fixture", "Model", "MAE", "RMSE", "MAPE%", "MASE", "WAPE%", "DirAcc", "DirAcF", "IC"
    );
    println!("{}", "-".repeat(125));

    // Group by fixture
    let mut current_fixture = String::new();
    for r in results {
        if r.fixture_name != current_fixture {
            if !current_fixture.is_empty() {
                println!("{}", "-".repeat(125));
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

        let (dir_acc_str, dir_acf_str, ic_str) = match &r.direction_metrics {
            Some(d) => (
                format!("{:.2}", d.dir_acc),
                d.dir_acc_filtered
                    .map(|v| format!("{:.2}", v))
                    .unwrap_or_else(|| "-".to_string()),
                if d.ic.is_finite() {
                    format!("{:+.2}", d.ic)
                } else {
                    "-".to_string()
                },
            ),
            None => ("-".to_string(), "-".to_string(), "-".to_string()),
        };

        println!(
            "{:<30} {:<20} {:>8.2} {:>8.2} {:>8} {:>8} {:>8} {:>7} {:>7} {:>7}",
            r.fixture_name,
            r.model_name,
            r.metrics.mae,
            r.metrics.rmse,
            mape_str,
            mase_str,
            wape_str,
            dir_acc_str,
            dir_acf_str,
            ic_str,
        );
    }
    println!("{}", "-".repeat(125));

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

    // Summary: average DirAcc and IC per model across all fixtures
    println!("\n=== Average Direction Metrics by Model ===");
    for model in &model_names {
        let direction: Vec<_> = results
            .iter()
            .filter(|r| r.model_name == *model)
            .filter_map(|r| r.direction_metrics.as_ref())
            .collect();
        if direction.is_empty() {
            continue;
        }
        let avg_dir_acc = direction.iter().map(|d| d.dir_acc).sum::<f64>() / direction.len() as f64;
        let finite_ic: Vec<f64> = direction
            .iter()
            .map(|d| d.ic)
            .filter(|v| v.is_finite())
            .collect();
        let avg_ic = if finite_ic.is_empty() {
            f64::NAN
        } else {
            finite_ic.iter().sum::<f64>() / finite_ic.len() as f64
        };
        let ic_str = if avg_ic.is_nan() {
            "-".to_string()
        } else {
            format!("{:+.3}", avg_ic)
        };
        println!(
            "  {:<20} avg DirAcc = {:.3}  avg IC = {}  ({} fixtures)",
            model,
            avg_dir_acc,
            ic_str,
            direction.len()
        );
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
