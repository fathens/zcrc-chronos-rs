mod standard;

pub use standard::StandardScaler;

use common::Result;

/// Trait for value scaling/normalization.
pub trait Scaler: Send + Sync {
    /// Compute scaling parameters from the input values.
    fn fit(&mut self, values: &[f64]) -> Result<()>;

    /// Transform values using the fitted parameters.
    fn transform(&self, values: &[f64]) -> Result<Vec<f64>>;

    /// Inverse transform scaled values back to original scale.
    fn inverse_transform(&self, values: &[f64]) -> Result<Vec<f64>>;

    /// Fit and transform in one step.
    fn fit_transform(&mut self, values: &[f64]) -> Result<Vec<f64>> {
        self.fit(values)?;
        self.transform(values)
    }
}
