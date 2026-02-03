//! Holt-Winters exponential smoothing with additive/multiplicative seasonality.
//!
//! Provides a full seasonal ETS implementation to supplement augurs AutoETS,
//! which does not yet support the seasonal component ("ZZZ").

use common::{ChronosError, Result};
use tracing::debug;

// ---------------------------------------------------------------------------
// Type definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrendType {
    None,
    Additive,
    AdditiveDamped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SeasonalType {
    None,
    Additive,
    Multiplicative,
}

#[derive(Debug, Clone, Copy)]
struct HwSpec {
    trend: TrendType,
    seasonal: SeasonalType,
}

#[derive(Debug, Clone)]
struct HwParams {
    alpha: f64,
    beta: Option<f64>,
    gamma: Option<f64>,
    phi: Option<f64>,
}

#[derive(Debug, Clone)]
struct FittedHw {
    level: f64,
    trend: f64,
    seasonal: Vec<f64>,
    sse: f64,
    n: usize,
    params: HwParams,
    spec: HwSpec,
}

pub(crate) struct HwResult {
    pub mean: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

// ---------------------------------------------------------------------------
// State initialization
// ---------------------------------------------------------------------------

fn initialize_state(values: &[f64], m: usize, spec: HwSpec) -> (f64, f64, Vec<f64>) {
    // Level: mean of first season
    let level: f64 = values[..m].iter().sum::<f64>() / m as f64;

    // Trend
    let trend = match spec.trend {
        TrendType::None => 0.0,
        _ => {
            if values.len() >= 2 * m {
                let mean1: f64 = values[..m].iter().sum::<f64>() / m as f64;
                let mean2: f64 = values[m..2 * m].iter().sum::<f64>() / m as f64;
                (mean2 - mean1) / m as f64
            } else {
                // Fewer than 2 cycles: simple slope from first to last
                let last = values.len() - 1;
                if last > 0 {
                    (values[last] - values[0]) / last as f64
                } else {
                    0.0
                }
            }
        }
    };

    // Seasonal indices
    let seasonal = match spec.seasonal {
        SeasonalType::None => vec![0.0; m],
        SeasonalType::Additive => values[..m].iter().map(|&v| v - level).collect(),
        SeasonalType::Multiplicative => {
            if level.abs() < 1e-10 {
                // Level near zero: fall back to additive-style init
                values[..m].iter().map(|&v| v - level).collect()
            } else {
                values[..m].iter().map(|&v| v / level).collect()
            }
        }
    };

    (level, trend, seasonal)
}

// ---------------------------------------------------------------------------
// Holt-Winters update equations & fitting
// ---------------------------------------------------------------------------

fn fit_single(values: &[f64], m: usize, spec: HwSpec, params: &HwParams) -> FittedHw {
    let n = values.len();
    let alpha = params.alpha;
    let beta = params.beta.unwrap_or(0.0);
    let gamma = params.gamma.unwrap_or(0.0);
    let phi = params.phi.unwrap_or(1.0);

    let (init_level, init_trend, init_seasonal) = initialize_state(values, m, spec);

    let mut level = init_level;
    let mut trend = init_trend;
    let mut seasonal = init_seasonal.clone();
    let mut sse = 0.0;

    for t in m..n {
        let s_prev = seasonal[t % m];

        // One-step-ahead forecast
        let forecast = match (spec.trend, spec.seasonal) {
            (TrendType::None, SeasonalType::None) => level,
            (TrendType::None, SeasonalType::Additive) => level + s_prev,
            (TrendType::None, SeasonalType::Multiplicative) => level * s_prev,
            (_, SeasonalType::None) => level + phi * trend,
            (_, SeasonalType::Additive) => level + phi * trend + s_prev,
            (_, SeasonalType::Multiplicative) => (level + phi * trend) * s_prev,
        };

        let error = values[t] - forecast;
        sse += error * error;

        let prev_level = level;
        let prev_trend = trend;

        // Update level
        level = match spec.seasonal {
            SeasonalType::None => {
                alpha * values[t] + (1.0 - alpha) * (prev_level + phi * prev_trend)
            }
            SeasonalType::Additive => {
                alpha * (values[t] - s_prev) + (1.0 - alpha) * (prev_level + phi * prev_trend)
            }
            SeasonalType::Multiplicative => {
                if s_prev.abs() < 1e-10 {
                    prev_level + phi * prev_trend
                } else {
                    alpha * (values[t] / s_prev) + (1.0 - alpha) * (prev_level + phi * prev_trend)
                }
            }
        };

        // Update trend
        trend = match spec.trend {
            TrendType::None => 0.0,
            _ => beta * (level - prev_level) + (1.0 - beta) * phi * prev_trend,
        };

        // Update seasonal
        seasonal[t % m] = match spec.seasonal {
            SeasonalType::None => 0.0,
            SeasonalType::Additive => gamma * (values[t] - level) + (1.0 - gamma) * s_prev,
            SeasonalType::Multiplicative => {
                let base = prev_level + phi * prev_trend;
                if base.abs() < 1e-10 {
                    s_prev
                } else {
                    gamma * (values[t] / base) + (1.0 - gamma) * s_prev
                }
            }
        };
    }

    FittedHw {
        level,
        trend,
        seasonal,
        sse,
        n,
        params: params.clone(),
        spec,
    }
}

// ---------------------------------------------------------------------------
// Forecasting
// ---------------------------------------------------------------------------

fn forecast(fitted: &FittedHw, m: usize, horizon: usize) -> HwResult {
    let n = fitted.n;
    let phi = fitted.params.phi.unwrap_or(1.0);
    let level = fitted.level;
    let trend = fitted.trend;

    let mut mean = Vec::with_capacity(horizon);
    for h in 1..=horizon {
        // Sum of damping factors: phi + phi^2 + ... + phi^h
        let phi_sum = if (phi - 1.0).abs() < 1e-12 {
            h as f64
        } else {
            phi * (1.0 - phi.powi(h as i32)) / (1.0 - phi)
        };

        let trend_comp = match fitted.spec.trend {
            TrendType::None => 0.0,
            _ => phi_sum * trend,
        };

        // Seasonal index for horizon step h
        // We need s_{n+h-m*ceil(h/m)} which maps to the stored seasonal indices.
        let s_idx = (n + h - 1) % m; // 0-based index into seasonal array
        let s = fitted.seasonal[s_idx];

        let point = match fitted.spec.seasonal {
            SeasonalType::None => level + trend_comp,
            SeasonalType::Additive => level + trend_comp + s,
            SeasonalType::Multiplicative => (level + trend_comp) * s,
        };

        mean.push(point);
    }

    // Residual-based prediction intervals (80%)
    let residual_var = if n > m {
        fitted.sse / (n - m) as f64
    } else {
        fitted.sse.max(1e-10)
    };
    let residual_std = residual_var.sqrt();
    let z = 1.2816; // z_{0.90} for 80% interval

    let lower: Vec<f64> = mean
        .iter()
        .enumerate()
        .map(|(h, &m_val)| m_val - z * residual_std * ((h + 1) as f64).sqrt())
        .collect();
    let upper: Vec<f64> = mean
        .iter()
        .enumerate()
        .map(|(h, &m_val)| m_val + z * residual_std * ((h + 1) as f64).sqrt())
        .collect();

    HwResult { mean, lower, upper }
}

// ---------------------------------------------------------------------------
// Nelder-Mead simplex optimizer (bounded)
// ---------------------------------------------------------------------------

struct NelderMeadBounds {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

fn clamp_point(point: &mut [f64], bounds: &NelderMeadBounds) {
    for (i, v) in point.iter_mut().enumerate() {
        *v = v.clamp(bounds.lower[i], bounds.upper[i]);
    }
}

fn nelder_mead<F>(
    f: F,
    initial: &[f64],
    bounds: &NelderMeadBounds,
    max_iter: usize,
    tol: f64,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let dim = initial.len();
    let n = dim + 1; // number of simplex vertices

    // Build initial simplex
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut clamped = initial.to_vec();
    clamp_point(&mut clamped, bounds);
    simplex.push(clamped.clone());

    for i in 0..dim {
        let mut vertex = clamped.clone();
        let step = (bounds.upper[i] - bounds.lower[i]) * 0.1;
        vertex[i] = (vertex[i] + step).min(bounds.upper[i]);
        if (vertex[i] - clamped[i]).abs() < 1e-12 {
            vertex[i] = (vertex[i] - step).max(bounds.lower[i]);
        }
        simplex.push(vertex);
    }

    let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _iter in 0..max_iter {
        // Sort by function value
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Check convergence: diameter of simplex
        let best = &simplex[indices[0]];
        let worst = &simplex[indices[n - 1]];
        let diameter: f64 = best
            .iter()
            .zip(worst.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, |acc, x| acc.max(x));
        if diameter < tol {
            return simplex[indices[0]].clone();
        }

        // Centroid of all points except worst
        let mut centroid = vec![0.0; dim];
        for &idx in &indices[..n - 1] {
            for (j, c) in centroid.iter_mut().enumerate() {
                *c += simplex[idx][j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= (n - 1) as f64;
        }

        let worst_idx = indices[n - 1];
        let second_worst_idx = indices[n - 2];

        // Reflection
        let mut reflected: Vec<f64> = centroid
            .iter()
            .zip(simplex[worst_idx].iter())
            .map(|(&c, &w)| 2.0 * c - w)
            .collect();
        clamp_point(&mut reflected, bounds);
        let f_reflected = f(&reflected);

        if f_reflected < values[indices[0]] {
            // Expansion
            let mut expanded: Vec<f64> = centroid
                .iter()
                .zip(reflected.iter())
                .map(|(&c, &r)| 2.0 * r - c)
                .collect();
            clamp_point(&mut expanded, bounds);
            let f_expanded = f(&expanded);

            if f_expanded < f_reflected {
                simplex[worst_idx] = expanded;
                values[worst_idx] = f_expanded;
            } else {
                simplex[worst_idx] = reflected;
                values[worst_idx] = f_reflected;
            }
        } else if f_reflected < values[second_worst_idx] {
            simplex[worst_idx] = reflected;
            values[worst_idx] = f_reflected;
        } else {
            // Contraction
            let use_reflected = f_reflected < values[worst_idx];
            let contract_from = if use_reflected {
                &reflected
            } else {
                &simplex[worst_idx].clone()
            };
            let f_contract_from = if use_reflected {
                f_reflected
            } else {
                values[worst_idx]
            };

            let mut contracted: Vec<f64> = centroid
                .iter()
                .zip(contract_from.iter())
                .map(|(&c, &w)| 0.5 * (c + w))
                .collect();
            clamp_point(&mut contracted, bounds);
            let f_contracted = f(&contracted);

            if f_contracted < f_contract_from {
                simplex[worst_idx] = contracted;
                values[worst_idx] = f_contracted;
            } else {
                // Shrink all towards best
                let best_point = simplex[indices[0]].clone();
                for &idx in &indices[1..] {
                    for j in 0..dim {
                        simplex[idx][j] = 0.5 * (simplex[idx][j] + best_point[j]);
                    }
                    clamp_point(&mut simplex[idx], bounds);
                    values[idx] = f(&simplex[idx]);
                }
            }
        }
    }

    // Return best point
    let best_idx = values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    simplex[best_idx].clone()
}

// ---------------------------------------------------------------------------
// Parameter optimization
// ---------------------------------------------------------------------------

fn optimize_params(values: &[f64], m: usize, spec: HwSpec) -> HwParams {
    let has_trend = spec.trend != TrendType::None;
    let has_seasonal = spec.seasonal != SeasonalType::None;
    let has_damping = spec.trend == TrendType::AdditiveDamped;

    // Build parameter vector: [alpha, (beta), (gamma), (phi)]
    let mut initial = vec![0.3];
    let mut lower = vec![0.001];
    let mut upper = vec![0.999];

    if has_trend {
        initial.push(0.05);
        lower.push(0.001);
        upper.push(0.5);
    }
    if has_seasonal {
        initial.push(0.1);
        lower.push(0.001);
        upper.push(0.999);
    }
    if has_damping {
        initial.push(0.95);
        lower.push(0.8);
        upper.push(0.999);
    }

    let bounds = NelderMeadBounds { lower, upper };

    // Reduced iterations (200 vs 500) and relaxed tolerance (1e-6 vs 1e-8)
    // for faster convergence with minimal accuracy loss
    let best = nelder_mead(
        |params| {
            let hw_params = unpack_params(params, spec);
            let fitted = fit_single(values, m, spec, &hw_params);
            let sse = fitted.sse;
            if sse.is_nan() || sse.is_infinite() {
                f64::MAX
            } else {
                sse
            }
        },
        &initial,
        &bounds,
        200,
        1e-6,
    );

    unpack_params(&best, spec)
}

fn unpack_params(raw: &[f64], spec: HwSpec) -> HwParams {
    let has_trend = spec.trend != TrendType::None;
    let has_seasonal = spec.seasonal != SeasonalType::None;
    let has_damping = spec.trend == TrendType::AdditiveDamped;

    let mut idx = 0;
    let alpha = raw[idx];
    idx += 1;

    let beta = if has_trend {
        let v = raw[idx];
        idx += 1;
        Some(v)
    } else {
        None
    };

    let gamma = if has_seasonal {
        let v = raw[idx];
        idx += 1;
        Some(v)
    } else {
        None
    };

    let phi = if has_damping {
        Some(raw[idx])
    } else if has_trend {
        Some(1.0) // Non-damped additive trend
    } else {
        None
    };

    HwParams {
        alpha,
        beta,
        gamma,
        phi,
    }
}

// ---------------------------------------------------------------------------
// AICc model selection
// ---------------------------------------------------------------------------

fn compute_aicc(sse: f64, n: usize, spec: HwSpec, m: usize) -> f64 {
    let n_f = n as f64;

    // Number of parameters k = smoothing params + initial states
    let n_smooth = 1 // alpha
        + if spec.trend != TrendType::None { 1 } else { 0 } // beta
        + if spec.seasonal != SeasonalType::None { 1 } else { 0 } // gamma
        + if spec.trend == TrendType::AdditiveDamped { 1 } else { 0 }; // phi

    let n_init = 1 // level
        + if spec.trend != TrendType::None { 1 } else { 0 } // trend
        + if spec.seasonal != SeasonalType::None { m } else { 0 }; // seasonal

    let k = (n_smooth + n_init) as f64;

    // Guard against SSE=0 (constant series): use a small floor to avoid ln(0)
    let mse = (sse / n_f).max(1e-300);

    let aic = n_f * mse.ln() + 2.0 * k;
    if n_f - k - 1.0 > 0.0 {
        aic + 2.0 * k * (k + 1.0) / (n_f - k - 1.0)
    } else {
        f64::INFINITY
    }
}

fn candidate_specs(has_negative: bool) -> Vec<HwSpec> {
    let trends = [
        TrendType::None,
        TrendType::Additive,
        TrendType::AdditiveDamped,
    ];
    let seasonals_all = [
        SeasonalType::None,
        SeasonalType::Additive,
        SeasonalType::Multiplicative,
    ];
    let seasonals_no_mult = [SeasonalType::None, SeasonalType::Additive];

    let mut specs = Vec::new();
    for &trend in &trends {
        let seasonals: &[SeasonalType] = if has_negative {
            &seasonals_no_mult
        } else {
            &seasonals_all
        };
        for &seasonal in seasonals {
            specs.push(HwSpec { trend, seasonal });
        }
    }
    specs
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub(crate) fn hw_fit_predict(
    values: &[f64],
    season_length: usize,
    horizon: usize,
) -> Result<HwResult> {
    let n = values.len();
    let m = season_length;

    if n < 2 * m {
        return Err(ChronosError::InsufficientData(format!(
            "Holt-Winters requires at least 2 full seasonal cycles ({} points), got {}",
            2 * m,
            n
        )));
    }

    let has_negative = values.iter().any(|&v| v < 0.0);
    let specs = candidate_specs(has_negative);

    debug!(
        season_length = m,
        horizon = horizon,
        data_length = n,
        n_candidates = specs.len(),
        has_negative = has_negative,
        "Holt-Winters model selection"
    );

    let mut best_fitted: Option<FittedHw> = None;
    let mut best_aicc = f64::INFINITY;

    for spec in &specs {
        let params = optimize_params(values, m, *spec);
        let fitted = fit_single(values, m, *spec, &params);

        if fitted.sse.is_nan() || fitted.sse.is_infinite() {
            continue;
        }

        let aicc = compute_aicc(fitted.sse, n, *spec, m);
        if aicc.is_nan() || aicc.is_infinite() {
            continue;
        }

        debug!(
            trend = ?spec.trend,
            seasonal = ?spec.seasonal,
            sse = fitted.sse,
            aicc = aicc,
            alpha = fitted.params.alpha,
            "Holt-Winters candidate"
        );

        if aicc < best_aicc {
            best_aicc = aicc;
            best_fitted = Some(fitted);
        }
    }

    let fitted = best_fitted
        .ok_or_else(|| ChronosError::ModelError("All Holt-Winters candidates failed".into()))?;

    debug!(
        trend = ?fitted.spec.trend,
        seasonal = ?fitted.spec.seasonal,
        aicc = best_aicc,
        "Holt-Winters selected model"
    );

    Ok(forecast(&fitted, m, horizon))
}

#[cfg(test)]
mod tests;
