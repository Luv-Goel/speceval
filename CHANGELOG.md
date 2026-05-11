# Changelog

All notable changes to SpecEval are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `ENV_API_KEY_PREFIX` typo (`SPECTEVAL_` → `SPECEVAL_`) that caused
  silent key-resolution failures when using env-var auth (#12)
- `config.py` `mkdir` calls now catch `OSError` so importing speceval in
  read-only CI containers no longer raises `PermissionError`

### Added
- `normalize=True` kwarg on `exact_match` for case-insensitive /
  punctuation-stripped comparison
- `__str__` and `__repr__` on all `SpecEvalError` sub-classes for
  greppable log output (e.g. `[MetricError] bleu: length mismatch`)
- `check-merge-conflict` and `check-added-large-files` pre-commit hooks
- `make ci` target that mirrors GitHub Actions steps locally
- `make install-hooks` for first-time pre-commit setup

## [0.1.0] — 2025-05-11

### Added

- Declarative evaluation specification format (YAML)
- CLI with `init`, `run`, `report`, and `list` commands
- Dataset loaders for common benchmarks (GSM8K, MMLU, HumanEval, etc.)
- Model providers: OpenAI, Anthropic, vLLM, Ollama, HuggingFace
- Metrics: exact match, numeric match, pass@k, BLEU, ROUGE, LLM-as-judge
- Head-to-head model comparison engine with significance testing
- Self-contained HTML report generation with interactive visualizations
- CI integration (exit codes for pass/fail thresholds)
- Provenance logging (dataset versions, model configs, timestamps)
- Deterministic seeding for reproducibility

[Unreleased]: https://github.com/Luv-Goel/speceval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Luv-Goel/speceval/releases/tag/v0.1.0
