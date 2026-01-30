use crate::Scaler;
use common::{ChronosError, Result};

/// Standard scaler: z-score normalization.
///
/// Transforms values to have zero mean and unit variance: `(x - mean) / std`.
/// For constant series (std ≈ 0), transforms to zeros and inverse transforms
/// to the mean value.
#[derive(Debug, Clone)]
pub struct StandardScaler {
    mean: Option<f64>,
    std: Option<f64>,
}

impl StandardScaler {
    const EPSILON: f64 = 1e-10;

    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
        }
    }

    /// Returns true if the fitted series is constant (std ≈ 0).
    pub fn is_constant(&self) -> bool {
        self.std.map(|s| s < Self::EPSILON).unwrap_or(false)
    }

    /// Returns the fitted mean, if available.
    pub fn mean(&self) -> Option<f64> {
        self.mean
    }

    /// Returns the fitted standard deviation, if available.
    pub fn std(&self) -> Option<f64> {
        self.std
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scaler for StandardScaler {
    fn fit(&mut self, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            return Err(ChronosError::InsufficientData(
                "Cannot fit scaler on empty values".into(),
            ));
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;

        self.mean = Some(mean);
        self.std = Some(variance.sqrt());

        Ok(())
    }

    fn transform(&self, values: &[f64]) -> Result<Vec<f64>> {
        let mean = self
            .mean
            .ok_or_else(|| ChronosError::InvalidInput("Scaler not fitted".into()))?;
        let std = self
            .std
            .ok_or_else(|| ChronosError::InvalidInput("Scaler not fitted".into()))?;

        if std < Self::EPSILON {
            return Ok(vec![0.0; values.len()]);
        }

        Ok(values.iter().map(|v| (v - mean) / std).collect())
    }

    fn inverse_transform(&self, values: &[f64]) -> Result<Vec<f64>> {
        let mean = self
            .mean
            .ok_or_else(|| ChronosError::InvalidInput("Scaler not fitted".into()))?;
        let std = self
            .std
            .ok_or_else(|| ChronosError::InvalidInput("Scaler not fitted".into()))?;

        if std < Self::EPSILON {
            return Ok(vec![mean; values.len()]);
        }

        Ok(values.iter().map(|v| v * std + mean).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fit_transform_roundtrip() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut scaler = StandardScaler::new();

        let transformed = scaler.fit_transform(&values).unwrap();
        let restored = scaler.inverse_transform(&transformed).unwrap();

        for (original, restored_val) in values.iter().zip(restored.iter()) {
            assert_relative_eq!(original, restored_val, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_standard_scaling_properties() {
        let values = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let mut scaler = StandardScaler::new();

        let transformed = scaler.fit_transform(&values).unwrap();

        // Mean should be approximately 0
        let mean: f64 = transformed.iter().sum::<f64>() / transformed.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);

        // Std should be approximately 1
        let variance: f64 =
            transformed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / transformed.len() as f64;
        let std = variance.sqrt();
        assert_relative_eq!(std, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_constant_series() {
        let values = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let mut scaler = StandardScaler::new();

        let transformed = scaler.fit_transform(&values).unwrap();
        assert!(scaler.is_constant());

        // All transformed values should be 0
        for v in &transformed {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-10);
        }

        // Inverse should restore to the constant value
        let restored = scaler.inverse_transform(&transformed).unwrap();
        for v in &restored {
            assert_relative_eq!(*v, 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_empty_input() {
        let mut scaler = StandardScaler::new();
        let result = scaler.fit(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_transform_without_fit() {
        let scaler = StandardScaler::new();
        let result = scaler.transform(&[1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extreme_small_values() {
        let values: Vec<f64> = (0..10).map(|i| (i as f64 + 1.0) * 1e-9).collect();
        let mut scaler = StandardScaler::new();

        let transformed = scaler.fit_transform(&values).unwrap();
        let restored = scaler.inverse_transform(&transformed).unwrap();

        for (original, restored_val) in values.iter().zip(restored.iter()) {
            assert_relative_eq!(original, restored_val, epsilon = 1e-18);
        }
    }

    #[test]
    fn test_extreme_large_values() {
        let values: Vec<f64> = (0..10).map(|i| (i as f64 + 1.0) * 1e12).collect();
        let mut scaler = StandardScaler::new();

        let transformed = scaler.fit_transform(&values).unwrap();
        let restored = scaler.inverse_transform(&transformed).unwrap();

        for (original, restored_val) in values.iter().zip(restored.iter()) {
            assert_relative_eq!(original, restored_val, epsilon = 1e3);
        }
    }

    #[test]
    fn test_negative_values() {
        let values = vec![-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0];
        let mut scaler = StandardScaler::new();

        let transformed = scaler.fit_transform(&values).unwrap();
        let restored = scaler.inverse_transform(&transformed).unwrap();

        for (original, restored_val) in values.iter().zip(restored.iter()) {
            assert_relative_eq!(original, restored_val, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_single_value() {
        let values = vec![42.0];
        let mut scaler = StandardScaler::new();

        let transformed = scaler.fit_transform(&values).unwrap();
        assert!(scaler.is_constant());
        assert_relative_eq!(transformed[0], 0.0, epsilon = 1e-10);

        let restored = scaler.inverse_transform(&transformed).unwrap();
        assert_relative_eq!(restored[0], 42.0, epsilon = 1e-10);
    }
}
