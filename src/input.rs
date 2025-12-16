//! Input abstraction for LOWESS inputs.
//!
//! This module defines the `LowessInput` trait which allows the `fit` method
//! to accept both standard slices and ndarray inputs interchangeably.

use lowess::testing::primitives::errors::LowessError;
use ndarray::{ArrayBase, Data, Ix1};
use num_traits::Float;

/// Trait for types that can be used as input for LOWESS smoothing.
///
/// This trait abstracts over slice-like inputs, allowing functions to accept
/// both `&[T]` and `&ArrayBase` (ndarray) seamlessly.
pub trait LowessInput<T: Float> {
    /// Convert the input to a contiguous slice.
    ///
    /// # Returns
    ///
    /// * `Ok(&[T])` - A reference to the underlying contiguous slice.
    /// * `Err(LowessError)` - If the data is not contiguous in memory.
    fn as_lowess_slice(&self) -> Result<&[T], LowessError>;
}

impl<T: Float> LowessInput<T> for [T] {
    fn as_lowess_slice(&self) -> Result<&[T], LowessError> {
        Ok(self)
    }
}

impl<T: Float> LowessInput<T> for Vec<T> {
    fn as_lowess_slice(&self) -> Result<&[T], LowessError> {
        Ok(self.as_slice())
    }
}

impl<T: Float, S> LowessInput<T> for ArrayBase<S, Ix1>
where
    S: Data<Elem = T>,
{
    fn as_lowess_slice(&self) -> Result<&[T], LowessError> {
        self.as_slice().ok_or_else(|| {
            LowessError::InvalidInput("ndarray input must be contiguous in memory".to_string())
        })
    }
}
