# Run all local checks (formatting, linting, building, tests, docs)
check: fmt clippy build test doc
	@echo "All checks completed successfully!"

# Coverage (requires cargo-llvm-cov and llvm)
coverage:
	@echo "Running coverage report (text)..."
	@LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov --features dev
	@echo "Coverage report complete!"

# Formatting
fmt:
	@echo "Checking code formatting..."
	@cargo fmt --all -- --check
	@echo "Formatting check complete!"

# Linter
clippy:
	@echo "Running clippy..."
	@cargo clippy --all-targets -- -D warnings
	@echo "Clippy check complete!"

# Build
build:
	@echo "Building crate..."
	@cargo build
	@echo "Build complete!"

# Test
test:
	@echo "Running tests..."
	@cargo test --workspace
	@echo "Tests complete!"

# Documentation
doc:
	@echo "Building documentation..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
	@echo "Documentation build complete!"

# Clean
clean:
	@echo "Performing cargo clean..."
	@cargo clean
	@echo "Clean complete!"
