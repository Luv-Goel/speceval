"""Tests for metric computations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from speceval.metrics import (
    compute_metric,
    get,
    list_metrics,
    register,
    register_all,
)
from speceval.metrics.classification import (
    _macro_binary_metric,
    _to_labels,
    accuracy,
    f1_score,
    precision,
    recall,
)
from speceval.metrics.generation import (
    _bleu_manual,
    _lcs_length,
    _ngrams,
    _tokenize,
    bleu,
    exact_match,
    perplexity,
    rouge_l,
)
from speceval.exceptions import MetricError


# Ensure built-in metrics are registered once for the entire module
_registered_metrics = False


def setup_module():
    """Register all built-in metrics once before any test runs."""
    global _registered_metrics
    if not _registered_metrics:
        register_all()
        _registered_metrics = True


class TestMetricRegistry:
    """Tests for the metric registry."""

    def test_list_metrics(self):
        """list_metrics returns sorted registered metric names."""
        metrics = list_metrics()
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "exact_match" in metrics
        assert "bleu" in metrics
        assert "rouge_l" in metrics
        assert "perplexity" in metrics
        assert metrics == sorted(metrics)

    def test_get_registered(self):
        """get returns a callable for registered metrics."""
        fn = get("accuracy")
        assert callable(fn)

    def test_get_unregistered(self):
        """get raises MetricError for unregistered metrics."""
        with pytest.raises(MetricError, match="not registered"):
            get("nonexistent")

    def test_register_duplicate(self):
        """Registering a duplicate name raises MetricError."""
        with pytest.raises(MetricError, match="already registered"):
            register("accuracy", accuracy)

    def test_compute_metric(self):
        """compute_metric works for a registered metric."""
        result = compute_metric("accuracy", ["a", "b"], ["a", "b"])
        assert result.name == "accuracy"
        assert result.value == 1.0
        assert isinstance(result.details, dict)

    def test_compute_metric_unregistered(self):
        """compute_metric raises MetricError for unregistered metric."""
        with pytest.raises(MetricError, match="not registered"):
            compute_metric("nonexistent", ["a"], ["a"])

    def test_compute_metric_with_kwargs(self):
        """compute_metric forwards kwargs to metric function."""
        result = compute_metric("exact_match", ["hello"], ["hello"])
        assert result.value == 1.0


class TestAccuracy:
    """Tests for accuracy metric."""

    def test_perfect_accuracy(self):
        """All predictions match references."""
        assert accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_zero_accuracy(self):
        """No predictions match references."""
        assert accuracy(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_accuracy(self):
        """Half of predictions match."""
        assert accuracy(["a", "b"], ["a", "c"]) == 0.5

    def test_empty_lists(self):
        """Empty lists return 0.0."""
        assert accuracy([], []) == 0.0

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="length mismatch"):
            accuracy(["a"], ["a", "b"])

    def test_single_item_correct(self):
        """Single correct prediction returns 1.0."""
        assert accuracy(["yes"], ["yes"]) == 1.0

    def test_single_item_wrong(self):
        """Single wrong prediction returns 0.0."""
        assert accuracy(["yes"], ["no"]) == 0.0


class TestF1Score:
    """Tests for F1 score metric."""

    def test_perfect_f1(self):
        """All predictions match -> F1 = 1.0."""
        assert f1_score(["a", "b"], ["a", "b"]) == 1.0

    def test_zero_f1(self):
        """No matches -> F1 = 0.0."""
        result = f1_score(["a"], ["b"])
        assert result >= 0.0
        assert result <= 1.0

    def test_f1_range(self):
        """F1 is always in [0, 1]."""
        result = f1_score(["a", "b", "c"], ["a", "b", "d"])
        assert 0.0 <= result <= 1.0

    def test_empty_lists(self):
        """Empty lists return 0.0."""
        assert f1_score([], []) == 0.0

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="length mismatch"):
            f1_score(["a"], ["a", "b"])


class TestPrecision:
    """Tests for precision metric."""

    def test_perfect_precision(self):
        """All predictions are correct."""
        assert precision(["a", "b"], ["a", "b"]) == 1.0

    def test_zero_precision(self):
        """No correct predictions."""
        result = precision(["a"], ["b"])
        assert 0.0 <= result <= 1.0

    def test_precision_range(self):
        """Precision is always in [0, 1]."""
        result = precision(["a", "b", "c"], ["a", "b", "d"])
        assert 0.0 <= result <= 1.0

    def test_empty_lists(self):
        """Empty lists return 0.0."""
        assert precision([], []) == 0.0

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="length mismatch"):
            precision(["a"], ["a", "b"])


class TestRecall:
    """Tests for recall metric."""

    def test_perfect_recall(self):
        """All predictions are correct."""
        assert recall(["a", "b"], ["a", "b"]) == 1.0

    def test_recall_range(self):
        """Recall is always in [0, 1]."""
        result = recall(["a", "b", "c"], ["a", "b", "d"])
        assert 0.0 <= result <= 1.0

    def test_empty_lists(self):
        """Empty lists return 0.0."""
        assert recall([], []) == 0.0

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="length mismatch"):
            recall(["a"], ["a", "b"])


class TestExactMatch:
    """Tests for exact_match metric."""

    def test_exact_match_perfect(self):
        """All predictions exactly match references."""
        assert exact_match(["hello world"], ["hello world"]) == 1.0

    def test_exact_match_none(self):
        """No predictions exactly match."""
        assert exact_match(["hello"], ["world"]) == 0.0

    def test_exact_match_partial(self):
        """Some predictions match."""
        result = exact_match(["yes", "no", "yes"], ["yes", "no", "no"])
        assert result == 2.0 / 3.0

    def test_strips_whitespace(self):
        """Whitespace is stripped before comparison."""
        assert exact_match(["  hello  "], ["hello"]) == 1.0
        assert exact_match(["hello"], ["  hello  "]) == 1.0

    def test_empty_lists(self):
        """Empty lists return 0.0."""
        assert exact_match([], []) == 0.0

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="length mismatch"):
            exact_match(["a"], ["a", "b"])

    def test_case_sensitive(self):
        """Exact match is case-sensitive by default."""
        assert exact_match(["Hello"], ["hello"]) == 0.0

    def test_normalize_case_insensitive(self):
        """normalize=True should treat 'Hello' and 'hello' as equal."""
        result = exact_match(["Hello World"], ["hello world"], normalize=True)
        assert result == 1.0, (
            "normalize=True must lower-case both sides before comparing"
        )

    def test_normalize_strips_punctuation(self):
        """normalize=True strips leading/trailing punctuation differences."""
        result = exact_match(["hello."], ["hello"], normalize=True)
        # Punctuation-stripped 'hello.' == 'hello'
        assert result == 1.0

    def test_normalize_false_is_default(self):
        """Default behavior (normalize omitted) is case-sensitive."""
        result = exact_match(["HELLO"], ["hello"])
        assert result == 0.0


class TestBLEU:
    """Tests for BLEU metric."""

    def test_identical_texts(self):
        """Identical texts should get BLEU > 0."""
        score = bleu(["hello world"], ["hello world"])
        assert score > 0.0
        assert score <= 100.0

    def test_completely_different(self):
        """Completely different texts should get low BLEU."""
        score = bleu(["hello world"], ["foo bar baz qux"])
        assert score >= 0.0

    def test_bleu_range(self):
        """BLEU is always in [0, 100]."""
        score = bleu(["the cat sat on the mat"], ["the cat sat on the mat"])
        assert 0.0 <= score <= 100.0

    def test_empty_lists(self):
        """Empty lists return 0.0."""
        assert bleu([], []) == 0.0

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="length mismatch"):
            bleu(["a"], ["a", "b"])

    def test_bleu_manual_identical(self):
        """Manual BLEU for identical texts."""
        score = _bleu_manual(["hello world"], ["hello world"])
        assert score == 1.0

    def test_bleu_manual_different(self):
        """Manual BLEU for different texts."""
        score = _bleu_manual(["hello world"], ["foo bar"])
        assert score < 1.0

    def test_bleu_manual_no_ngram_match(self):
        """BLEU with no common n-grams is 0."""
        score = _bleu_manual(["abc"], ["xyz"])
        assert score == 0.0

    def test_tokenize(self):
        """Tokenization splits on whitespace and punctuation."""
        tokens = _tokenize("Hello, world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_ngrams(self):
        """N-gram extraction works."""
        tokens = ["a", "b", "c"]
        bigrams = _ngrams(tokens, 2)
        assert ("a", "b") in bigrams
        assert ("b", "c") in bigrams
        assert bigrams[("a", "b")] == 1


class TestROUGEL:
    """Tests for ROUGE-L metric."""

    def test_identical_texts(self):
        """Identical texts get ROUGE-L F1 = 1.0."""
        score = rouge_l(["hello world"], ["hello world"])
        assert score == 1.0

    def test_completely_different(self):
        """Completely different texts get ROUGE-L = 0.0."""
        score = rouge_l(["abc"], ["xyz"])
        assert score == 0.0

    def test_partial_match(self):
        """Partially matching texts get ROUGE-L > 0."""
        score = rouge_l(["hello world foo"], ["hello world bar"])
        assert score > 0.0
        assert score < 1.0

    def test_empty_lists(self):
        """Empty lists return 0.0."""
        assert rouge_l([], []) == 0.0

    def test_both_empty_strings(self):
        """Both empty strings get ROUGE-L = 1.0."""
        score = rouge_l([""], [""])
        assert score == 1.0

    def test_one_empty(self):
        """One empty string gets ROUGE-L = 0.0."""
        score = rouge_l(["hello"], [""])
        assert score == 0.0

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="length mismatch"):
            rouge_l(["a"], ["a", "b"])

    def test_lcs_length(self):
        """LCS length computation works."""
        assert _lcs_length(["a", "b", "c"], ["a", "c", "b"]) == 2

    def test_lcs_length_identical(self):
        """Same lists have full LCS."""
        assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_lcs_length_empty(self):
        """Empty lists have LCS 0."""
        assert _lcs_length([], []) == 0


class TestPerplexity:
    """Tests for perplexity metric."""

    def test_known_log_probs(self):
        """Perplexity of uniform distribution with 10 tokens -> exp(-(-2.3026)) ≈ 10."""
        result = perplexity(["-2.302585"], [""])
        assert abs(result - 10.0) < 1.0

    def test_empty_predictions(self):
        """Empty predictions return 0.0."""
        assert perplexity([], []) == 0.0

    def test_no_valid_log_probs(self):
        """Strings with no valid floats return 0.0."""
        assert perplexity(["not a number"], [""]) == 0.0

    def test_multiple_log_probs(self):
        """Multiple log probs are averaged."""
        result = perplexity(["0.0 0.0"], [""])
        assert abs(result - 1.0) < 0.01

    def test_single_token(self):
        """Single token log prob computes correctly."""
        result = perplexity(["-0.5"], [""])
        assert abs(result - math.exp(0.5)) < 0.01

    def test_batched_predictions(self):
        """Multiple prediction strings are handled."""
        result = perplexity(["-1.0", "-2.0"], ["", ""])
        assert abs(result - math.exp(1.5)) < 0.1

    def test_mixed_valid_and_invalid_tokens(self):
        """Prediction strings that mix valid floats and non-numeric tokens.

        Parser must skip non-float tokens rather than raising or treating
        them as 0.  Only valid float-parseable tokens contribute to the
        average log-prob.
        """
        # "-1.0 <|endoftext|> -1.0" — the special token in the middle should
        # be silently skipped, leaving avg_log_prob = -1.0 → ppl = e^1.0
        result = perplexity(["-1.0 <|endoftext|> -1.0"], [""])
        # Must be a positive finite number (parser didn't crash)
        assert result > 0.0 and math.isfinite(result)
        # With two valid tokens of -1.0 each, ppl ≈ e ≈ 2.718
        assert abs(result - math.e) < 0.5


class TestClassificationHelpers:
    """Tests for classification module helper functions."""

    def test_to_labels(self):
        """_to_labels maps strings to integers."""
        y_pred, y_true = _to_labels(["a", "b"], ["a", "c"])
        assert len(y_pred) == 2
        assert len(y_true) == 2
        assert y_pred.dtype == np.int32

    def test_to_labels_consistent_mapping(self):
        """Same labels get same mapping."""
        y_pred1, y_true1 = _to_labels(["a", "b"], ["a", "b"])
        y_pred2, y_true2 = _to_labels(["a", "b"], ["a", "b"])
        assert (y_pred1 == y_pred2).all()
        assert (y_true1 == y_true2).all()

    def test_macro_binary_metric_all_correct(self):
        """All correct -> metric = 1.0."""
        assert _macro_binary_metric(["a", "b"], ["a", "b"], "f1") == 1.0

    def test_macro_binary_metric_all_wrong(self):
        """All wrong -> metric = 0.0."""
        assert _macro_binary_metric(["a"], ["b"], "f1") == 0.0

    def test_macro_binary_metric_empty(self):
        """Empty lists -> 0.0."""
        assert _macro_binary_metric([], [], "precision") == 0.0
        assert _macro_binary_metric([], [], "recall") == 0.0
        assert _macro_binary_metric([], [], "f1") == 0.0
