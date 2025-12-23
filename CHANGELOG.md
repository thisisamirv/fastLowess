# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Added

- Integrated `parallel(bool)` directly into the core `LowessBuilder`, simplifying the interface for enabling parallel execution.
- Added tests for ensuring consistency between serial and parallel execution.

### Changed

- Renamed `Extended*LowessBuilder` structs to `Parallel*LowessBuilder` for clarity (e.g., `ExtendedBatchLowessBuilder` → `ParallelBatchLowessBuilder`), given their changed behavior.
- Simplified extended builders by leveraging core crate field for `parallel` configuration.
- Standardized method names and parameter propagation across all execution adapters.
- Updated `prelude` to export enum variants directly (e.g., `Batch`, `Tricube`, `Bisquare`) instead of the enum types themselves, simplifying API usage.
- Updated `lowess` dependency to v0.6.0.

### Removed

- Removed `Adapter`, `BoundaryPolicy`, `CrossValidationStrategy`, `MergeStrategy`, `RobustnessMethod`, `UpdateMode`, `WeightFunction`, and `ZeroWeightFallback` type exports from `prelude`.
- Removed the `type Result<T>` alias (previously in `api.rs` and exported via prelude) which shadowed `std::result::Result`. This alias caused ambiguity by implicitly binding `LowessError`. We now strictly follow Rust idioms: explicit `Result<LowessResult<T>, LowessError>` return types using the standard library `Result`.

## [0.2.2]

### Changed

- **Binary search for delta optimization**: Replaced linear O(n) scan in `compute_anchor_points` with `partition_point` binary search, reducing anchor discovery complexity from O(n) to O(log n) per anchor point.
- **Precomputed slope in interpolation**: Eliminated per-iteration division in `interpolate_gap` by precomputing the slope once, reducing computational overhead in the interpolation loop.
- **Vectorized fill for tied values**: Replaced iterator-based assignment with `slice::fill` for tied x-values, enabling SIMD vectorization.
- Aligned with `lowess` crate v0.5.3 optimizations for consistent performance characteristics.

## [0.2.1]

### Changed

- Drop LaTeX formatting due to docs.rs rendering issues.
- Improve documentation.

## [0.2.0]

- For changes to the core logic and the API, see the [lowess](https://github.com/av746/lowess) crate.

### Added

- Added more explanation on how to use the streaming mode in the documentation.
- Added convenience wrappers to adapters, allowing for a more flexible API.
- Added support for the new features in the `lowess` crate version 0.5.1.

## [0.1.0]

### Added

- Initial release
