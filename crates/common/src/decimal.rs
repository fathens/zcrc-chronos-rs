use bigdecimal::BigDecimal;
use num_traits::{FromPrimitive, ToPrimitive};

use crate::{ChronosError, Result};

/// Convert a slice of `BigDecimal` values to `Vec<f64>`.
///
/// Returns `InvalidInput` if any value cannot be represented as f64.
pub fn decimals_to_f64s(values: &[BigDecimal]) -> Result<Vec<f64>> {
    let mut result = Vec::with_capacity(values.len());
    for (i, d) in values.iter().enumerate() {
        let v = d.to_f64().ok_or_else(|| {
            ChronosError::InvalidInput(format!(
                "Cannot convert BigDecimal to f64 at index {i}: {d}"
            ))
        })?;
        result.push(v);
    }
    Ok(result)
}

/// Convert a slice of `f64` values to `Vec<BigDecimal>`.
///
/// Returns `InvalidInput` if any value is NaN or Infinity.
pub fn f64s_to_decimals(values: &[f64]) -> Result<Vec<BigDecimal>> {
    let mut result = Vec::with_capacity(values.len());
    for (i, &v) in values.iter().enumerate() {
        let d = BigDecimal::from_f64(v).ok_or_else(|| {
            ChronosError::InvalidInput(format!(
                "Cannot convert f64 to BigDecimal at index {i}: {v}"
            ))
        })?;
        result.push(d);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimals_to_f64s() {
        let decimals = vec![
            BigDecimal::from_f64(1.0).unwrap(),
            BigDecimal::from_f64(2.5).unwrap(),
            BigDecimal::from_f64(-0.375).unwrap(),
        ];
        let result = decimals_to_f64s(&decimals).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.5).abs() < 1e-10);
        assert!((result[2] - (-0.375)).abs() < 1e-10);
    }

    #[test]
    fn test_f64s_to_decimals() {
        let floats = vec![1.0, 2.5, -0.375];
        let result = f64s_to_decimals(&floats).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], BigDecimal::from_f64(1.0).unwrap());
        assert_eq!(result[1], BigDecimal::from_f64(2.5).unwrap());
        assert_eq!(result[2], BigDecimal::from_f64(-0.375).unwrap());
    }

    #[test]
    fn test_f64s_to_decimals_nan_error() {
        let floats = vec![1.0, f64::NAN, 3.0];
        assert!(f64s_to_decimals(&floats).is_err());
    }

    #[test]
    fn test_f64s_to_decimals_inf_error() {
        let floats = vec![1.0, f64::INFINITY, 3.0];
        assert!(f64s_to_decimals(&floats).is_err());
    }

    #[test]
    fn test_empty_slices() {
        assert!(decimals_to_f64s(&[]).unwrap().is_empty());
        assert!(f64s_to_decimals(&[]).unwrap().is_empty());
    }

    #[test]
    fn test_roundtrip() {
        let original = vec![42.0, 100.5, -0.01, 0.0];
        let decimals = f64s_to_decimals(&original).unwrap();
        let back = decimals_to_f64s(&decimals).unwrap();
        for (a, b) in original.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-10, "{a} != {b}");
        }
    }
}
