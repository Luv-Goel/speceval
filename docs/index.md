# SpecEval

[![CI](https://img.shields.io/github/actions/workflow/status/Luv-Goel/speceval/ci.yml?branch=main&style=flat-square)](https://github.com/Luv-Goel/speceval/actions)
[![PyPI](https://img.shields.io/pypi/v/speceval?style=flat-square)](https://pypi.org/project/speceval/)
[![Python Version](https://img.shields.io/pypi/pyversions/speceval?style=flat-square)](https://pypi.org/project/speceval/)
[![License](https://img.shields.io/github/license/Luv-Goel/speceval?style=flat-square)](https://github.com/Luv-Goel/speceval/blob/main/LICENSE)

**Reproducible evaluation specifications for AI systems.**  
Define evaluations as code — version-controlled, auditable, composable.

---

## Overview

SpecEval lets you write evaluation specs as declarative YAML files. Each spec describes:

- **What** dataset to evaluate on
- **Which** models to test (OpenAI, Anthropic, open-source, or local)
- **How** to score outputs (exact match, BLEU, ROUGE, LLM-as-judge, and more)
- **Comparisons** — head-to-head model comparisons with significance testing
- **Reports** — self-contained HTML reports with interactive charts

## Quick Start

```bash
pip install speceval
speceval init my-eval
speceval run my-eval/speceval.yaml
speceval report my-eval --open
```

## Example Spec

```yaml
name: gsm8k-eval
description: Grade-school math reasoning benchmark

dataset:
  path: speceval/gsm8k
  split: test

models:
  - id: openai/gpt-4o
    provider: openai
    params:
      temperature: 0

prompt:
  template: |
    Solve the following math problem step by step.
    {question}
    Answer:

metrics:
  - exact_match
  - numeric_match

report:
  format: html
  output: report.html
```

## Key Features

- **Declarative specs** — One YAML file defines the entire evaluation pipeline.
- **Reproducible by default** — Deterministic seeding, pinned datasets, full provenance logging.
- **Model-agnostic** — OpenAI, Anthropic, vLLM, Ollama, HuggingFace.
- **Built-in metrics** — Exact match, numeric match, pass@k, BLEU, ROUGE, LLM-as-judge.
- **Comparison engine** — Head-to-head with bootstrap p-values and Cohen's d.
- **CI-ready** — Exit codes for pass/fail thresholds in any CI system.
- **Portable reports** — Self-contained HTML with per-example breakdowns.

## Installation

```bash
pip install speceval
```

Requires Python 3.10+.

For development:

```bash
git clone https://github.com/Luv-Goel/speceval.git
cd speceval
pip install -e ".[dev]"
```

## Project Layout

```
speceval/
├── src/speceval/        # Main package
│   ├── adapters/        # Model providers (OpenAI, HuggingFace, etc.)
│   ├── cli/             # Command-line interface
│   ├── engine/          # Evaluation runner
│   ├── metrics/         # Scoring metrics
│   ├── spec/            # Spec parsing, validation, hashing
│   ├── store/           # Result storage (SQLite)
│   ├── report/          # HTML report generation
│   ├── compare/         # Model comparison engine
│   └── provenance/      # Environment provenance tracking
├── tests/               # Test suite
└── docs/                # Documentation
```

## Contributing

See [CONTRIBUTING.md](https://github.com/Luv-Goel/speceval/blob/main/CONTRIBUTING.md).

## License

Apache 2.0.
