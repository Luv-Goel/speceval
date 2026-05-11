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
   make install
   ```

4. **Install pre-commit hooks** (recommended — catches lint/format issues before push)

   ```bash
   make install-hooks
   ```

## Running Tests

```bash
make test
```

With coverage report:

```bash
pytest --cov=speceval --cov-report=term-missing
```

## Reproducing CI Failures Locally

Before opening a PR, run the same steps as GitHub Actions with:

```bash
make ci
```

This runs lint → type-check → tests in order and stops on the first failure,
mirroring the CI matrix exactly.

## Code Style

- We use **Ruff** for linting and formatting (`ruff>=0.4.9`).
- Run lint checks: `make lint`
- Auto-format: `make format`
- Type hints are required for all public APIs. Run mypy: `make typecheck`

## Adding a New Metric

1. Add your function to `src/speceval/metrics/generation.py` (or
   `classification.py` for label-based metrics).
2. Register it in `register_all()` inside `src/speceval/metrics/__init__.py`.
3. Add tests in `tests/test_metrics.py` — at minimum: perfect score,
   zero score, empty-list edge case, and length-mismatch raises.
4. Update `CHANGELOG.md` under `[Unreleased] > Added`.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, add tests for new functionality.
3. Run `make ci` and confirm it passes.
4. Update documentation if your change introduces new features or modifies existing behavior.
5. Open a pull request with a clear title and description.

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat: add LLM-as-judge metric
fix: correct numeric match for negative numbers
test: add normalize=True coverage for exact_match
docs: update CONTRIBUTING with make targets
```

## Project Structure

```
speceval/
├── src/
│   └── speceval/       # Main package source
│       ├── cli/        # Command-line interface
│       ├── core/       # Core evaluation engine
│       ├── datasets/   # Dataset loaders
│       ├── models/     # Model provider adapters
│       ├── metrics/    # Scoring metrics
│       └── report/     # Report generation
├── tests/              # Test suite
├── docs/               # MkDocs documentation
├── examples/           # Example spec files (gsm8k, mmlu, compare-models)
└── Makefile            # Dev shortcuts
```

## Questions?

Open an issue or start a [GitHub Discussion](https://github.com/Luv-Goel/speceval/discussions).
