# Changelog

All notable changes to SpecEval are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
