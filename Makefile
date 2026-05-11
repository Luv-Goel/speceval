.PHONY: install install-hooks test lint typecheck format ci clean docs docs-serve

# Install the package in editable mode with dev dependencies
install:
	pip install -e ".[dev]"

# Install pre-commit hooks (run once after cloning)
install-hooks:
	pre-commit install

# Run all tests with coverage
test:
	pytest --cov=speceval --cov-report=term-missing

# Run linting with Ruff
lint:
	ruff check .

# Auto-format code with Ruff
format:
	ruff format .

# Run static type checking with mypy
typecheck:
	mypy src/speceval

# Run the same checks as CI (lint -> typecheck -> test)
# Use this locally before opening a PR to catch failures early.
ci: lint typecheck test

# Build documentation with MkDocs
docs:
	mkdocs build

# Serve documentation locally
docs-serve:
	mkdocs serve

# Clean build artifacts and caches
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .ruff_cache/ .mypy_cache/ site/ coverage.xml
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
