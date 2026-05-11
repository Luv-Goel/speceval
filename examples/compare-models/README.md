# Model Comparison Template

This example provides a **reusable spec template** for head-to-head model comparisons. Instead of defining multiple models in one file (which the current spec format doesn't support), you create one spec per model and use `speceval compare` to analyse the results statistically.

## How It Works

The spec defines everything *except* the model: dataset, metrics, trials, environment, and assertions. You:

1. **Clone the spec** for each model you want to test.
2. **Set the model section** to your target model.
3. **Run each spec** separately.
4. **Compare the runs** with `speceval compare`.

## Step-by-Step Example

Compare **GPT-4o** against **Claude 3.5 Sonnet** on GSM8K:

```bash
# ── Run 1: GPT-4o ─────────────────────────────────────────────────
cp examples/compare-models/speceval.yaml speceval-gpt4o.yaml
# Edit speceval-gpt4o.yaml:
#   model.provider: openai
#   model.name: gpt-4o
speceval run --spec speceval-gpt4o.yaml

# ── Run 2: Claude 3.5 Sonnet ──────────────────────────────────────
cp examples/compare-models/speceval.yaml speceval-claude.yaml
# Edit speceval-claude.yaml:
#   model.provider: anthropic
#   model.name: claude-3-5-sonnet-20241022
speceval run --spec speceval-claude.yaml

# ── Compare results ───────────────────────────────────────────────
# Find run IDs from output, then:
speceval compare gpt-4o_a1b2c3d4 claude-3-5-sonnet-20241022_e5f6g7h8
```

## Expected Comparison Output

```
Comparison: gpt-4o_a1b2c3d4 vs claude-3-5-sonnet-20241022_e5f6g7h8

Metric Deltas
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Metric      ┃ Δ (B − A)    ┃ p-value  ┃ Cohen's d  ┃ Significance ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ accuracy    │ +0.0320      │  0.0021  │     0.423  │   p < 0.05   │
│ exact_match │ +0.0280      │  0.0038  │     0.387  │   p < 0.05   │
│ f1          │ +0.0300      │  0.0028  │     0.401  │   p < 0.05   │
└─────────────┴──────────────┴──────────┴────────────┴──────────────┘

Items in run A: 200
Items in run B: 200
Common metrics: 3
```

A positive Δ means model B scored higher. p-values < 0.05 indicate statistically significant differences (95% confidence). Cohen's d tells you the effect size (small ≈ 0.2, medium ≈ 0.5, large ≈ 0.8).

## Tips for Fair Comparisons

| Practice | Why |
|----------|-----|
| **Same seed** | The `env.seeds` block ensures both models see identical data ordering. |
| **Same dataset.limit** | Use identical sample sizes so deltas are directly comparable. |
| **Same trials** | Equal trials ensures variance estimates are comparable. |
| **temperature: 0** | Deterministic generation removes noise from sampling. |
| **Bootstrap comparison** | `speceval compare` uses bootstrap resampling for robust p-values. |

## Customisation

- **Change dataset** — Replace `dataset.path` with any Hugging Face dataset (e.g. `lukaemon/mmlu`, `truthful_qa`, `hellaswag`).
- **Add metrics** — Add `bleu`, `rouge_l`, `precision`, `recall`, or custom metrics.
- **Multiple models** — Create 3+ copies of the spec to compare several models in a sweep.
- **CI integration** — Run comparisons automatically in GitHub Actions and post results as PR comments.

## Related Commands

```bash
# Validate a spec
speceval validate --spec examples/compare-models/speceval.yaml

# List all completed runs (find run IDs for comparison)
speceval report --list

# Generate a detailed report for a specific run
speceval report <run_id> --open
```

## References

- [Speceval Compare Command](https://speceval.dev/docs/compare)
- [Bootstrap Hypothesis Testing](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
- [Cohen's d](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d)
