<div align="center">
  <img src="https://raw.githubusercontent.com/Luv-Goel/speceval/main/docs/assets/speceval_terminal_demo.jpg" alt="SpecEval Terminal Demo" width="100%">
</div>

# SpecEval

[![CI](https://img.shields.io/github/actions/workflow/status/Luv-Goel/speceval/ci.yml?branch=main&style=for-the-badge&logo=github)](https://github.com/Luv-Goel/speceval/actions)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue?style=for-the-badge&logo=materialformkdocs)](https://Luv-Goel.github.io/speceval/)
[![PyPI](https://img.shields.io/pypi/v/speceval?style=for-the-badge&logo=pypi)](https://pypi.org/project/speceval/)
[![Python Version](https://img.shields.io/pypi/pyversions/speceval?style=for-the-badge&logo=python)](https://pypi.org/project/speceval/)
[![License](https://img.shields.io/github/license/Luv-Goel/speceval?style=for-the-badge)](https://github.com/Luv-Goel/speceval/blob/main/LICENSE)

**Reproducible evaluation specifications for AI systems.**

SpecEval lets you define AI evaluations as version-controlled, auditable, composable YAML — promoting reproducibility and rigour in ML research workflows.

---

## ⚡ Quick Start

Get started with SpecEval in less than a minute:

```bash
pip install speceval
speceval init my-eval
speceval run my-eval/speceval.yaml
speceval report my-eval --open
```

---

## 🎯 Why SpecEval?

Evaluation today consists of ad-hoc scripts scattered across notebooks, internal dashboards, and undocumented workflows. Every team reinvents the same pipeline — loading models, formatting prompts, scoring outputs — with subtle differences that make results impossible to compare or reproduce. Benchmarks are run once, screenshotted, and forgotten.

**SpecEval brings declarative evaluation specifications to AI.** You write a single YAML file that describes exactly what to evaluate, which models to test, what metrics to compute, and how to present results. That spec lives in your repository, gets run in CI, and produces portable HTML reports anyone can inspect. Same spec, same results, every time. No hidden randomness, no script drift.

---

## 🏗️ Architecture

<div align="center">
  <img src="https://raw.githubusercontent.com/Luv-Goel/speceval/main/docs/assets/architecture.svg" alt="SpecEval Architecture Diagram" width="80%">
</div>

SpecEval acts as an orchestration layer between your datasets, models, and metric engines.

---

## 🚀 Features

- **Declarative YAML Specs**: Define models, datasets, and metrics in one place.
- **Multi-Provider Support**: Out-of-the-box adapters for OpenAI, Anthropic, HuggingFace, and vLLM.
- **Extensible Metrics Engine**: Built-in support for exact match, BLEU, ROUGE, and more. Easily plug in your own.
- **Reproducible execution**: Locks seeds, tracks prompts, and guarantees consistent evaluation states.
- **Rich Reporting**: Generate beautiful, interactive HTML reports to share with your team.

---

## 📖 Example: GSM8K Math Reasoning

The following toy spec evaluates two LLMs on grade-school math reasoning and produces a head-to-head comparison report in one command:

```yaml
# speceval.yaml
name: gsm8k-mini-demo
description: Comparing GPT-4o vs Claude on 50 GSM8K samples

dataset:
  path: speceval/gsm8k
  split: test
  subset: main
  limit: 50  # quick smoke-test

models:
  - id: openai/gpt-4o
    provider: openai
    params: { temperature: 0, max_tokens: 512 }
  - id: anthropic/claude-3-opus-20240229
    provider: anthropic
    params: { temperature: 0, max_tokens: 512 }

prompt:
  template: |
    Solve step by step.

    {question}

    Answer:
  variables:
    question: question

metrics:
  - exact_match
  - numeric_match

report:
  format: html
  output: gsm8k-demo-report.html
  include: [scores, examples]
```

---

## 🛠️ Installation

**From PyPI (Recommended):**
```bash
pip install speceval
```

**From Source (For Development):**
```bash
git clone https://github.com/Luv-Goel/speceval.git
cd speceval
pip install -e ".[dev]"
```

---

## ⚙️ Detailed Configuration

SpecEval YAML files are composed of several key sections:

### 1. Dataset
Defines the source of the data to evaluate against.
- `path`: HuggingFace dataset path, local CSV, or JSONL.
- `split`: Which split to use (`train`, `test`, `validation`).
- `limit`: (Optional) Limit the number of samples for quick smoke testing.

### 2. Models
A list of models to evaluate. You can mix and match providers.
- `id`: The model identifier (e.g., `openai/gpt-4o`, `anthropic/claude-3-opus-20240229`).
- `provider`: The adapter backend (`openai`, `anthropic`, `huggingface`).
- `params`: Inference parameters (e.g., `temperature`, `max_tokens`).

### 3. Prompt
The template used to format the dataset rows into a prompt for the models.
- `template`: A string with placeholder variables (e.g., `{question}`).
- `variables`: A mapping of template placeholders to dataset column names.

### 4. Metrics
The metrics to compute over the model outputs.
- Can be simple strings (e.g., `exact_match`, `bleu`).
- Or complex objects with parameters for custom behavior.

---

## 🛠️ Extending Metrics

SpecEval is designed to be easily extensible. You can add your own custom metrics by subclassing `BaseMetric`.

```python
from speceval.metrics.base import BaseMetric

class MyCustomMetric(BaseMetric):
    @classmethod
    def name(cls) -> str:
        return "my_custom_metric"

    def compute(self, expected: str, prediction: str) -> dict[str, float]:
        score = 1.0 if expected.strip() == prediction.strip() else 0.0
        return {"score": score}
```
Simply register it or ensure it's loaded in your environment, and you can reference it in your YAML spec!

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details. 

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🛡️ Security

If you discover any security related issues, please refer to our [Security Policy](SECURITY.md) and report it accordingly.

## 📄 License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.
