# MMLU Knowledge Evaluation

This example evaluates a model on the **MMLU** (Massive Multitask Language Understanding) benchmark — 57 subjects covering STEM, humanities, social sciences, and more. Each question is a 4-way multiple-choice task.

## Spec Overview

| Field          | Value                                  |
|----------------|----------------------------------------|
| **Model**      | `meta-llama/Llama-3.2-3B` (Hugging Face) |
| **Dataset**    | `lukaemon/mmlu` — `test` split, all subjects |
| **Sample size**| 200 examples                           |
| **Metrics**    | `accuracy`                             |
| **Trials**     | 1 (deterministic for local models)     |
| **Assertion**  | accuracy ≥ 0.25 (random baseline)      |

## How to Run

```bash
# Validate the spec
speceval validate --spec examples/mmlu/speceval.yaml

# Run the evaluation (downloads the model + dataset on first run)
speceval run --spec examples/mmlu/speceval.yaml

# Generate a report
speceval report examples/mmlu --open
```

## Expected Output

```
Run ID: meta-llama_Llama-3.2-3B_b3c4d5e6
Model:  meta-llama/Llama-3.2-3B
Dataset: lukaemon/mmlu
Metrics: accuracy
Trials:  1

✔ Evaluation complete!

Summary — meta-llama_Llama-3.2-3B_b3c4d5e6
┏━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Item ┃ Accuracy ┃ Duration ┃
┡━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│    0 │   1.0000 │  3400 ms │
│    1 │   0.0000 │  3100 ms │
│  ... │     ...  │      ... │
└──────┴──────────┴──────────┘
```

## Per-Subject Breakdown

MMLU covers 57 subjects. For a more detailed analysis, consider removing `dataset.limit` or running separate specs per subject (set `dataset.subset` to a specific subject like `astronomy` or `college_physics`).

## Customisation

- **Swap model**: Change `model.name` to any Hugging Face model ID (e.g. `mistralai/Mistral-7B-v0.1`, `tiiuae/falcon-7b`).
- **Full evaluation**: Remove `dataset.limit` to run on the complete test set (~14,000 questions).
- **Specific subjects**: Set `dataset.subset` to a single subject name (e.g. `college_medicine`).
- **API models**: Change `model.provider` to `openai` or `anthropic` and adjust `model.name` accordingly.

## References

- [MMLU Paper](https://arxiv.org/abs/2009.03300) — Hendrycks et al., 2020
- [Hugging Face Dataset](https://huggingface.co/datasets/lukaemon/mmlu)
