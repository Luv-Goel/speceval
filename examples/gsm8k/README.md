# GSM8K Math Reasoning Evaluation

This example evaluates a model on the **GSM8K** (Grade School Math 8K) benchmark — a dataset of 8,500 grade-school-level math word problems.

## Spec Overview

| Field          | Value                        |
|----------------|------------------------------|
| **Model**      | `gpt-4o` (OpenAI)           |
| **Dataset**    | `gsm8k` — `test` split, `main` subset |
| **Sample size**| 100 examples (for quick eval) |
| **Metrics**    | `exact_match`, `accuracy`   |
| **Trials**     | 3 (measures stability)      |
| **Assertion**  | accuracy ≥ 0.5              |

## How to Run

```bash
# Validate the spec
speceval validate --spec examples/gsm8k/speceval.yaml

# Run the evaluation
speceval run --spec examples/gsm8k/speceval.yaml

# Generate a report (after results exist)
speceval report examples/gsm8k --open
```

## Expected Output

The command will print a summary table:

```
Run ID: gpt-4o_a1b2c3d4
Model:  gpt-4o
Dataset: gsm8k
Metrics: exact_match, accuracy
Trials:  3

✔ Evaluation complete!

Summary — gpt-4o_a1b2c3d4
┏━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Item ┃ Exact Match ┃ Accuracy ┃ Duration ┃
┡━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│    0 │      1.0000 │   1.0000 │  1200 ms │
│    1 │      0.0000 │   0.0000 │   980 ms │
│  ... │        ...  │     ...  │      ... │
└──────┴─────────────┴──────────┴──────────┘
```

A JSON results file is saved to `./speceval_results/`.

## Customisation

- **Change model**: Swap `model.provider` and `model.name` to test a different model.
- **Full dataset**: Remove or increase `dataset.limit` to run on all 1,319 test examples.
- **More metrics**: Add `bleu`, `rouge_l`, or custom metric entries.
- **Adjust assertions**: Change the `value` in `assertions` to set a different pass/fail threshold.

## References

- [GSM8K Paper](https://arxiv.org/abs/2110.14168) — Cobbe et al., 2021
- [Hugging Face Dataset](https://huggingface.co/datasets/gsm8k)
