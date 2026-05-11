.PHONY: install test lint typecheck clean docs

# Install the package in editable mode with dev dependencies
install:
	pip install -e ".[dev]"

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

# Build documentation with MkDocs
docs:
	mkdocs build

# Serve documentation locally
docs-serve:
	mkdocs serve

# Clean build artifacts and caches
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	rm -rf **/__pycache__/
	rm -rf site/
	rm -rf coverage.xml
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
