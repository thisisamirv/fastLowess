# Contributing to fastLowess

We welcome contributions via bug reports, feature requests, documentation improvements, and code changes.

## Issues

Before opening a new issue, please search existing ones.

- **Bugs**: Include a minimal reproducible example, environment details, and expected vs actual behavior.
- **Features**: Describe the use case and provide examples of the proposed API or behavior.

## Development Setup

The project uses a `Makefile` to standardize development tasks across `std` and `no-std` targets.

```bash
# Clone and branch
git clone https://github.com/thisisamirv/fastLowess.git
cd fastLowess
git checkout -b feature/your-feature

# Common commands
make build      # Build both std and no-std
make test       # Run test suite
make check      # Run all checks (fmt, clippy, test)
```

### The `dev` Feature Flag

A `dev` feature flag is used to expose internal modules for testing. It is automatically enabled via `.cargo/config.toml` during local development.

Internal modules are accessible under `fastLowess::internals::*`. Note that these APIs are unstable and intended only for internal testing.

## Pull Requests

1. **Focus**: Keep PRs small and focused on a single change.
2. **Tests**: Add or update tests for any logic changes.
3. **Linting**: Ensure `make check` passes. We follow [Conventional Commits](https://www.conventionalcommits.org/).
4. **Documentation**: Update doc comments for any changed public APIs.

### Commitment Guidelines

- Use `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, or `chore` types.
- Scopes are optional but helpful (e.g., `api`, `math`, `engine`).

## Project Structure

`fastLowess` follows a layered architecture to prevent circular dependencies:

1. **API**: High-level fluent builder (`api.rs`).
2. **Adapters**: Execution modes (`batch`, `streaming`, `online`).
3. **Engine**: Only contains the executor engine.

Higher layers may import from lower layers, but never vice versa.

Most operations and logics in the `fastLowess` crate are actually delegated to the `lowess` crate (<https://github.com/thisisamirv/lowess>).

## Testing

Tests are organized by component in the `tests/` directory:

- `api_tests.rs`: Integration tests for the public API.
- `adapters_*`: Mode-specific tests.

Use the `approx` crate for floating-point comparisons:

```rust
use approx::assert_relative_eq;
assert_relative_eq!(result.y[0], 2.0, epsilon = 1e-10);
```

Please do not add tests directly to the scripts in `src/` directory. Add them to the `tests/` directory instead to keep the codebase minimal, clean, and organized.

### Benchmarks and Validation

Correction and performance are validated against Python's `statsmodels` on the `bench` branch. Switch to that branch to run the full validation suite.

## License

By contributing, you agree that your work will be licensed under the project's existing license.
