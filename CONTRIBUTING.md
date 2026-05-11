# Contributing to SpecEval

Thanks for your interest in contributing! This document covers the essentials.

## Development Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/Luv-Goel/speceval.git
   cd speceval
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

3. **Install in editable mode with dev dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks** (optional but recommended)

   ```bash
   pre-commit install
   ```

## Running Tests

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=speceval --cov-report=term-missing
```

## Code Style

- We use **Ruff** for linting and formatting.
- Run lint checks before committing:

  ```bash
  ruff check .
  ```

- Auto-format:

  ```bash
  ruff format .
  ```

- Type hints are required for all public APIs. Run mypy for static type checking:

  ```bash
  mypy src/speceval
  ```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, add tests for new functionality.
3. Ensure all tests pass and lint checks are clean.
4. Update documentation if your change introduces new features or modifies existing behavior.
5. Open a pull request with a clear title and description.

### Commit Messages

Use conventional commit format:

```
feat: add model comparison engine
fix: correct numeric match for negative numbers
docs: update README with CI badges
```

## Project Structure

```
speceval/
├── src/
│   └── speceval/       # Main package source
│       ├── cli/        # Command-line interface
│       ├── core/       # Core evaluation engine
│       ├── datasets/   # Dataset loaders
│       ├── models/     # Model provider interfaces
│       ├── metrics/    # Scoring metrics
│       └── report/     # Report generation
├── tests/              # Test suite
├── docs/               # MkDocs documentation
└── speceval.yaml       # Example spec file
```

## Questions?

Open an issue on GitHub or start a discussion.
