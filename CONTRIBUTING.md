# Contributing to fastLowess

Thank you for your interest in contributing to `fastLowess`! We welcome bug reports, feature suggestions, documentation improvements, and code contributions.

## Quick Links

- 🐛 [Report a bug](https://github.com/thisisamirv/fastLowess/issues/new?labels=bug)
- 💡 [Request a feature](https://github.com/thisisamirv/fastLowess/issues/new?labels=enhancement)
- 📖 [Documentation](https://docs.rs/fastLowess)
- 💬 [Discussions](https://github.com/thisisamirv/fastLowess/discussions)

## Code of Conduct

Be respectful, inclusive, and constructive. We're here to build great software together.

## Reporting Bugs

**Before submitting**, search existing issues to avoid duplicates.

Please include:

- Clear description of the problem
- Minimal reproducible example
- Expected vs actual behavior
- Environment details (OS, Rust version, feature flags)
- Backtrace if applicable (`RUST_BACKTRACE=1`)

**Example:**

```rust
// This produces incorrect output
let x = vec![1.0, 2.0, 3.0];
let y = vec![1.0, 2.0, 3.0];
let result = Lowess::new().fraction(0.5).fit(&x, &y)?;
// Expected: [1.0, 2.0, 3.0]
// Actual: [0.9, 2.1, 2.9]
```

## Suggesting Features

Feature requests are welcome! Please:

- **Check existing issues** first
- **Explain the use case** - why is this needed?
- **Provide examples** of how it would work
- **Consider alternatives** - have you tried existing features?

Areas of particular interest:

- Performance optimizations
- Better error messages
- Real-world use case examples

## Pull Requests

### Process

1. **Fork** the repository and create a feature branch

   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** with clear, focused commits

3. **Add tests** for new functionality

4. **Update documentation** (code comments, README, CHANGELOG)

5. **Ensure quality**

   ```bash
   make check
   ```

6. **Submit PR** with clear description of changes

### PR Checklist

- [ ] Tests added/updated and passing
- [ ] Documentation updated (if applicable)
- [ ] `cargo fmt` applied
- [ ] `cargo clippy` passes with no warnings
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] Commit messages follow [conventional commits](https://www.conventionalcommits.org/)

### What Makes a Good PR?

✅ **Do:**

- Keep changes focused and atomic
- Write descriptive commit messages
- Add tests that fail without your changes
- Update documentation for API changes
- Consider backward compatibility

❌ **Avoid:**

- Mixing unrelated changes
- Breaking existing APIs without discussion
- Adding dependencies without justification
- Submitting untested code

## Development Setup

We use a `Makefile` to standardize development tasks and ensure both `std` and `no-std` configurations are checked.

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fastLowess.git
cd fastLowess

# Create a development branch
git checkout -b feature/my-feature

# Build (std & no-std)
make build

# Run tests (std & no-std)
make test

# Check formatting and lints (std & no-std)
make fmt
make clippy

# Run all checks at once
make check
```

### Development Feature Flag

The project uses a `dev` feature flag to expose internal modules for testing and development. This is automatically enabled via `.cargo/config.toml` when working in the repository:

```toml
# .cargo/config.toml
[build]
rustflags = ["--cfg", "feature=\"dev\""]
```

**What the `dev` feature does:**

- Exposes internal modules under `fastLowess::internals::*`
- Allows testing of internal implementation details
- Provides access to primitives, math, algorithms, engine, and evaluation modules

**Using internal modules in tests:**

```rust
#[cfg(feature = "dev")]
use fastLowess::internals::adapters::batch::ExtendedBatchLowessBuilder;

#[test]
#[cfg(feature = "dev")]
fn test_internal_component() {
    // Test internal implementation details
}
```

**⚠️ Warning:** Internal APIs exposed via the `dev` feature have no stability guarantees and may change without notice. Only use them for testing within this repository.

## Testing Guidelines

### Running Tests

```bash
# All tests (dev feature automatically enabled via .cargo/config.toml)
cargo test --all-features

# Specific test file
cargo test --test integration_tests

# Specific test function
cargo test test_batch_smoothing

# With output
cargo test -- --nocapture

# Release mode (for benchmarks)
cargo test --release

# Run examples
cargo run --example batch_smoothing
cargo run --example online_smoothing
cargo run --example streaming_smoothing
```

### Writing Tests

Place tests in the appropriate `tests/` file:

- `tests/adapters_*_tests.rs` - Execution mode adapters (batch, streaming, online)
- `tests/api_tests.rs` - High-level builder API

**Test structure:**

```rust
#[test]
fn test_descriptive_name() {
    // Arrange
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

    // Act
    let result = Lowess::new()
        .fraction(0.5)
        .fit(&x, &y)
        .unwrap();

    // Assert
    assert_eq!(result.x.len(), x.len());
    assert!(result.y.iter().all(|&v| v.is_finite()));
}
```

**For floating-point comparisons**, use the `approx` crate:

```rust
use approx::assert_relative_eq;

assert_relative_eq!(result.y[0], 2.0, epsilon = 1e-10);
```

## Validation and Benchmarking

This project maintains a dedicated `bench` branch for validating correctness and benchmarking performance against the reference Python implementation (`statsmodels`).

**Note:** To run the full validation and benchmarking suite, you must switch to the `bench` branch. The `bench` branch is a separate branch that uses the latest version of `lowess` from `develop` branch (not `main` branch).

```bash
# Switch to bench branch
git switch bench
```

For more information, see the README.md file in the `bench` branch.

## Code Style

### Formatting

```bash
# Format all code
make fmt

# Check without modifying
cargo fmt --check
```

### Linting

```bash
# Run clippy (std & no-std)
make clippy

# Auto-fix some issues
cargo clippy --fix --all-features
```

### Documentation

All public APIs must have documentation:

````rust
/// Performs LOWESS smoothing on the given data.
///
/// # Arguments
///
/// * `x` - Independent variable values (must be finite)
/// * `y` - Dependent variable values (must be finite)
///
/// # Returns
///
/// A `LowessResult` containing smoothed values and diagnostics.
///
/// # Errors
///
/// Returns `LowessError::EmptyInput` if either array is empty.
/// Returns `LowessError::MismatchedInputs` if arrays have different lengths.
///
/// # Examples
///
/// ```
/// use fastLowess::Lowess;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
///
/// let result = Lowess::new()
///     .fraction(0.5)
///     .fit(&x, &y)?;
/// # Ok::<(), fastLowess::LowessError>(())
/// ```
pub fn fit(&self, x: &[T], y: &[T]) -> Result<LowessResult<T>> {
    // ...
}
````

**Documentation tips:**

- Start with a brief one-line summary
- Document all parameters and return values
- List all possible errors
- Include at least one example
- Use proper Markdown formatting
- Link to related functions with backticks

## Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, semicolons, etc.)
- `refactor`: Code restructuring (no functional changes)
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Build, dependencies, or maintenance

**Scopes (optional):**

- `api`: High-level builder API
- `adapters`: Execution mode adapters (batch, streaming, online)
- `engine`: Orchestration and execution
- `evaluation`: Post-processing (CV, diagnostics, intervals)
- `algorithms`: Core LOWESS algorithms (regression, robustness, interpolation)
- `math`: Pure mathematical functions (kernel, MAD)
- `primitives`: Data structures and utilities
- `docs`: Documentation

**Examples:**

```text
feat(adapters): add new adapter

Add a new adapter for batch smoothing.

Closes #42

fix(engine): handle NaN values in input arrays

Previously, NaN values would cause silent failures. Now returns
LowessError::InvalidInput with a descriptive message.

Fixes #58

perf(adapters): optimize batch smoothing

Reuse buffers across CV folds, reducing allocations by 70%.
Improves performance on large datasets.
```

## Project Structure

### Acyclic Hierarchical Layered Architecture

The codebase follows a strict 4-layer architecture where each layer has a single, well-defined responsibility and can only depend on layers below it.

```text
┌─────────────────────────────────────────────────────────────────┐
│                       Layer 4: API                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ api.rs (High-level fluent builder API)                    │  │
│  │  - LowessBuilder                                          │  │
│  │  - CrossValidationStrategy                                │  │
│  │  - Adapter selection (Batch, Streaming, Online)           │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ imports
┌────────────────────────────▼────────────────────────────────────┐
│                    Layer 3: Adapters                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  - batch.rs      (Standard in-memory processing)          │  │
│  │  - streaming.rs  (Chunked processing for large data)      │  │
│  │  - online.rs     (Incremental sliding window)             │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ imports
┌────────────────────────────▼────────────────────────────────────┐
│                        Layer 2: Engine                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  - executor.rs  (fastLowess Executor, iteration loop)     │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ imports
┌────────────────────────────▼────────────────────────────────────┐
│                      Layer 1: lowess crate                      │
└─────────────────────────────────────────────────────────────────┘
```

### Module Responsibility Matrix

| Module                         | Responsibility                    | Dependencies           |
| ------------------------------ | --------------------------------- | ---------------------- |
| **api.rs**                     | Fluent builder API                | All lower layers       |
| **adapters/batch**             | Standard batch execution          | All lower layers       |
| **adapters/streaming**         | Chunked processing                | All lower layers       |
| **adapters/online**            | Incremental sliding window        | All lower layers       |
| **engine/executor**            | Main iteration loop               | evaluation, algorithms |

**Key principles:**

- Lower layers never import from higher layers
- Each layer has a single, well-defined responsibility
- No circular dependencies

## Adding Examples

Examples are located in the `examples/` directory and serve as both documentation and practical demonstrations of the library's capabilities.

### Current Examples

- **`batch_smoothing.rs`** - Comprehensive batch processing examples (8 scenarios)
  - Basic smoothing, robust outlier handling, uncertainty quantification
  - Cross-validation, diagnostics, kernel comparisons, robustness methods
  - Quick and robust presets

- **`online_smoothing.rs`** - Online/incremental processing examples (6 scenarios)
  - Basic streaming, sensor data simulation, outlier handling
  - Window size effects, memory-bounded processing, sliding window behavior

- **`streaming_smoothing.rs`** - Chunked processing for large datasets (6 scenarios)
  - Basic chunking, chunk size comparison, overlap strategies
  - Large dataset processing, outlier handling, file-based simulation

### Adding a New Example

1. **Create the file** in `examples/your_example.rs`

2. **Follow the established pattern:**

   ```rust
   //! Brief description of what this example demonstrates
   //!
   //! Detailed explanation of use cases and scenarios covered.

   use fastLowess::prelude::*;

   fn main() -> Result<()> {
       println!("{}", "=".repeat(80));
       println!("Example Title");
       println!("{}", "=".repeat(80));
       println!();

       // Run example scenarios
       example_1()?;
       example_2()?;

       Ok(())
   }

   fn example_1() -> Result<()> {
       println!("Example 1: Description");
       println!("{}", "-".repeat(80));

       // Your example code here

       /* Expected Output:
       Document the expected output here as a comment
       This helps users verify the example works correctly
       */

       println!();
       Ok(())
   }
   ```

3. **Include expected outputs** as block comments after each example

4. **Test the example:**

   ```bash
   cargo run --example your_example
   ```

5. **Document it** in the main README.md if it demonstrates a key feature

### Example Guidelines

- **Be comprehensive** - Cover multiple scenarios in a single example file
- **Use realistic data** - Demonstrate practical use cases
- **Include comments** - Explain what's happening and why
- **Show expected output** - Help users verify correctness
- **Keep it runnable** - Examples should compile and run without errors
- **Follow the pattern** - Use the same structure as existing examples

## Adding Dependencies

We keep dependencies minimal. Before adding a dependency:

1. **Check if it's necessary** - Can we implement it ourselves?
2. **Verify maintenance** - Is it actively maintained?
3. **Check dependency tree** - What does it pull in?
4. **Discuss first** - Open an issue for non-trivial additions
5. **Use minimal features** - Only enable what's needed

**Current dependencies:**

- `num-traits`: Numeric traits abstraction (with `libm` for no_std support)
- `ndarray`: N-dimensional array library
- `rayon`: Parallel processing library

**Dev dependencies:**

- `approx`: Floating-point assertions for tests

**Feature flags:**

- `std` (default): Standard library support
- `dev`: Exposes internal modules for testing (automatically enabled in development via `.cargo/config.toml`)

## Release Process

(For maintainers)

1. Update version in `Cargo.toml` following [SemVer](https://semver.org/)
2. Update `CHANGELOG.md` with user-facing changes
3. Run full test suite with all features
4. Create git tag: `git tag -a v0.x.y -m "Release v0.x.y"`
5. Push tag: `git push origin v0.x.y`
6. Publish: `cargo publish`
7. Create GitHub release with changelog

## Getting Help

- 📖 **Documentation**: [docs.rs/fastLowess](https://docs.rs/fastLowess)
- 💬 **Discussions**: Ask questions in GitHub Discussions
- 📧 **Email**: <thisisamirv@gmail.com> (for private inquiries)
- 🐛 **Issues**: Report bugs via GitHub Issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! Every improvement, no matter how small, helps make `fastLowess` better for everyone. 🎉
