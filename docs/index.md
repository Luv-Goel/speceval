# SpecEval

[![CI](https://img.shields.io/github/actions/workflow/status/Luv-Goel/speceval/ci.yml?branch=main&style=flat-square)](https://github.com/Luv-Goel/speceval/actions)
[![PyPI](https://img.shields.io/pypi/v/speceval?style=flat-square)](https://pypi.org/project/speceval/)
[![Python Version](https://img.shields.io/pypi/pyversions/speceval?style=flat-square)](https://pypi.org/project/speceval/)
[![License](https://img.shields.io/github/license/Luv-Goel/speceval?style=flat-square)](https://github.com/Luv-Goel/speceval/blob/main/LICENSE)

**Reproducible evaluation specifications for AI systems.**

---

## Quick Start

```bash
pip install speceval
speceval init my-eval
speceval run my-eval/speceval.yaml
speceval report my-eval --open
```

For development, use the Makefile shortcuts:

```bash
git clone https://github.com/Luv-Goel/speceval.git
cd speceval
make install          # pip install -e '.[dev]'
make install-hooks    # set up pre-commit hooks
make ci               # lint → typecheck → test (mirrors GitHub Actions)
```

---

## Why SpecEval?

Evaluation today is ad-hoc scripts scattered across notebooks, internal dashboards, and undocumented workflows. Every team reinvents the same pipeline — loading models, formatting prompts, scoring outputs — with subtle differences that make results impossible to compare or reproduce. Benchmarks are run once, screenshotted, and forgotten.

SpecEval brings declarative evaluation specifications to AI. You write a single YAML file that describes exactly what to evaluate, which models to test, what metrics to compute, and how to present results. That spec lives in your repository, gets run in CI, and produces portable HTML reports anyone can inspect. Same spec, same results, every time. No hidden randomness, no script drift.

---

## Example Spec

```yaml
# speceval.yaml — evaluate on GSM8K
name: gsm8k-eval
description: Grade-school math reasoning benchmark

dataset:
  path: speceval/gsm8k
  split: test
  subset: main

models:
  - id: openai/gpt-4o
    provider: openai
    params:
      temperature: 0
      max_tokens: 1024
  - id: anthropic/claude-3-opus-20240229
    provider: anthropic
    params:
      temperature: 0
      max_tokens: 1024

prompt:
  template: |
    Solve the following math problem step by step.

    {question}

    Answer:
  variables:
    question: question

metrics:
  - exact_match
  - numeric_match
  - pass_at_1

comparisons:
  - model_a: openai/gpt-4o
    model_b: anthropic/claude-3-opus-20240229
    metric: exact_match
    method: pairwise

report:
  format: html
  output: report.html
  include: [scores, comparisons, examples]
```

---

## Key Features

- **Declarative specs** — One YAML file defines the entire evaluation pipeline.
- **Reproducible by default** — Deterministic seeding, pinned datasets, full provenance logging.
- **Model-agnostic** — Works with OpenAI, Anthropic, open-source models (vLLM, Ollama), and local model servers.
- **Built-in metrics** — Exact match, numeric match, pass@k, BLEU, ROUGE, LLM-as-judge, and custom Python metrics.
- **Comparison engine** — Head-to-head model comparisons with statistical significance tests.
- **CI-ready** — Run evaluation specs directly in any CI system.
- **Portable reports** — Self-contained HTML reports with interactive charts and per-example breakdowns.

---

## Installation

```bash
pip install speceval
```

Requires Python 3.10+.

---

## License

Apache 2.0.
