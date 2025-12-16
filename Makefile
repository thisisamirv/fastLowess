# Run all local checks (formatting, linting, building, tests, docs)
check: fmt clippy build test doc
	@echo "All checks completed successfully!"

# --- Formatting ---
fmt:
	@echo "Checking code formatting..."
	@cargo fmt --all -- --check
	@echo "Formatting check complete!"

# --- Linter ---
clippy: clippy-default clippy-serial

clippy-default:
	@echo "Running clippy (default / parallel)..."
	@cargo clippy --all-targets -- -D warnings

clippy-serial:
	@echo "Running clippy (serial / no parallel)..."
	@cargo clippy --all-targets --no-default-features -- -D warnings
	@echo "Clippy check complete!"

# --- Build ---
build: build-default build-serial

build-default:
	@echo "Building crate (default / parallel)..."
	@cargo build

build-serial:
	@echo "Building crate (serial / no parallel)..."
	@cargo build --no-default-features
	@echo "Build complete!"

# --- Test ---
test: test-default test-serial

test-default:
	@echo "Running tests (default / parallel)..."
	@cargo test

test-serial:
	@echo "Running tests (serial / no parallel)..."
	@cargo test --no-default-features
	@echo "Tests complete!"

# --- Documentation ---
doc: doc-default doc-serial
	@echo "Documentation build complete!"

doc-default:
	@echo "Building documentation (default / parallel)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

doc-serial:
	@echo "Building documentation (serial / no parallel)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --no-default-features

# --- Clean ---
clean:
	@echo "Performing cargo clean..."
	@cargo clean
	@echo "Clean complete!"
