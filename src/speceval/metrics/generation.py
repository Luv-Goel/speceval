"""Generation metrics — exact match, BLEU, ROUGE-L, perplexity."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

try:
    import sacrebleu

    _HAS_SACREBLEU = True
except ImportError:  # pragma: no cover
    _HAS_SACREBLEU = False


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------


def exact_match(
    predictions: list[str],
    references: list[str],
    normalize: bool = False,
    **kwargs: Any,
) -> float:
    """Fraction of predictions that exactly match the reference (after stripping).

    Args:
        predictions: Model outputs.
        references: Ground-truth answers.
        normalize: If ``True``, fold to lower-case before comparison.  This
            matches the convention used in SQuAD-style QA benchmarks.

    Returns:
        Exact match score in ``[0, 1]``.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references length mismatch: "
            f"{len(predictions)} vs {len(references)}"
        )
    if len(predictions) == 0:
        return 0.0

    def _norm(s: str) -> str:
        s = s.strip()
        return s.lower() if normalize else s

    correct = sum(1 for p, r in zip(predictions, references) if _norm(p) == _norm(r))
    return correct / len(predictions)


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenization."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    """Return n-gram counter from token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _bleu_manual(predictions: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU-4 with a pure-Python / numpy implementation.

    Guards against division-by-zero when short sequences produce fewer than
    4 tokens, and clamps ``effective_n`` to at least 1.
    """
    max_n = 4
    pred_tokens = [_tokenize(p) for p in predictions]
    ref_tokens = [_tokenize(r) for r in references]

    total_clipped = [0] * max_n
    total_pred_count = [0] * max_n
    total_ref_len = 0
    total_pred_len = 0

    for pt, rt in zip(pred_tokens, ref_tokens):
        ref_len = len(rt)
        pred_len = len(pt)
        total_ref_len += ref_len
        total_pred_len += pred_len

        # clamp to 1 so the loop always executes at least once
        effective_n = max(1, min(max_n, max(ref_len, pred_len)))
        for n in range(1, effective_n + 1):
            pred_ng = _ngrams(pt, n)
            ref_ng = _ngrams(rt, n)
            clipped = sum(min(c, ref_ng.get(ng, 0)) for ng, c in pred_ng.items())
            total_clipped[n - 1] += clipped
            total_pred_count[n - 1] += max(pred_len - n + 1, 0)

    # Brevity penalty — guard against zero total_pred_len
    if total_pred_len == 0:
        return 0.0
    bp = min(1.0, math.exp(1 - total_ref_len / total_pred_len))

    nonzero_prec = []
    for n in range(max_n):
        if total_pred_count[n] == 0:
            continue  # n-gram order never occurred — skip
        if total_clipped[n] == 0:
            nonzero_prec.append(0.0)
        else:
            nonzero_prec.append(total_clipped[n] / total_pred_count[n])

    if not nonzero_prec:
        return 0.0

    positive = [p for p in nonzero_prec if p > 0]
    if not positive:
        return 0.0

    avg_log_prec = sum(math.log(p) for p in positive) / len(positive)
    return float(bp * math.exp(avg_log_prec))


def bleu(predictions: list[str], references: list[str], **kwargs: Any) -> float:
    """Compute corpus-level BLEU score.

    Uses ``sacrebleu`` if available, otherwise falls back to a pure-Python
    implementation.

    Args:
        predictions: Model outputs.
        references: Ground-truth texts (one per prediction).

    Returns:
        BLEU score in ``[0, 100]`` following the convention of sacrebleu.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references length mismatch: "
            f"{len(predictions)} vs {len(references)}"
        )
    if len(predictions) == 0:
        return 0.0

    if _HAS_SACREBLEU:
        score = sacrebleu.corpus_bleu(predictions, [references])
        return float(score.score)

    return _bleu_manual(predictions, references) * 100


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length between two token lists.

    Uses a standard DP table (O(mn) time, O(mn) space).  For very long
    sequences this can be memory-intensive; inputs are typically short
    NLG outputs so this is acceptable.
    """
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[m, n])


def rouge_l(predictions: list[str], references: list[str], **kwargs: Any) -> float:
    """Compute macro-averaged ROUGE-L F1 score.

    Uses token-level LCS-based recall, precision, and F1.

    Args:
        predictions: Model outputs.
        references: Ground-truth texts.

    Returns:
        ROUGE-L F1 in ``[0, 1]``.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references length mismatch: "
            f"{len(predictions)} vs {len(references)}"
        )
    if len(predictions) == 0:
        return 0.0

    f_scores = []
    for p, r in zip(predictions, references):
        pt = _tokenize(p)
        rt = _tokenize(r)
        if len(pt) == 0 and len(rt) == 0:
            f_scores.append(1.0)
            continue
        if len(pt) == 0 or len(rt) == 0:
            f_scores.append(0.0)
            continue

        lcs = _lcs_length(pt, rt)
        prec = lcs / len(pt)
        rec = lcs / len(rt)
        if prec + rec > 0:
            f_scores.append(2 * prec * rec / (prec + rec))
        else:
            f_scores.append(0.0)

    return float(np.mean(f_scores))


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


def perplexity(predictions: list[str], references: list[str], **kwargs: Any) -> float:
    """Compute token-level perplexity from log-probability strings.

    .. note::

        This metric expects predictions to contain **log-probabilities**
        separated by whitespace (one per token).  The reference strings are
        ignored.  This is a thin wrapper for when a model outputs per-token
        log-probabilities for computing perplexity.

    Args:
        predictions: Strings containing space-separated log-probabilities.
        references: Ignored (present for API compatibility).

    Returns:
        Perplexity (positive float). Returns ``0.0`` on empty input.
    """
    if len(predictions) == 0:
        return 0.0

    log_probs: list[float] = []
    for sent in predictions:
        parts = sent.strip().split()
        for p_str in parts:
            try:
                lp = float(p_str)
            except ValueError:
                continue
            log_probs.append(lp)

    if not log_probs:
        return 0.0

    avg_neg_log_lik = -float(np.mean(log_probs))
    return float(np.exp(avg_neg_log_lik))


__all__ = ["exact_match", "bleu", "rouge_l", "perplexity"]
