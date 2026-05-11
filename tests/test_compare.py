"""Tests for comparison / delta computation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from speceval.compare.delta import (
    ComparisonResult,
    _aggregate_metrics,
    bootstrap_significance,
    cohens_d,
    compare_runs,
    compute_deltas,
)
from speceval.exceptions import CompareError
from speceval.store.sqlite import SQLiteStore


class TestComputeDeltas:
    """Tests for compute_deltas."""

    def test_basic_delta(self):
        """Delta is mean(B) - mean(A)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        delta = compute_deltas(a, b)
        assert delta == 3.0  # (5 - 2)

    def test_negative_delta(self):
        """Delta can be negative."""
        a = np.array([5.0, 6.0])
        b = np.array([1.0, 2.0])
        delta = compute_deltas(a, b)
        assert delta == -4.0

    def test_zero_delta(self):
        """Identical arrays give zero delta."""
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        delta = compute_deltas(a, b)
        assert delta == 0.0

    def test_single_element(self):
        """Single element arrays work."""
        delta = compute_deltas(np.array([5.0]), np.array([10.0]))
        assert delta == 5.0


class TestBootstrapSignificance:
    """Tests for bootstrap_significance."""

    def test_identical_samples_high_p(self):
        """Identical samples should have high p-value."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = bootstrap_significance(a, b, n_resamples=500)
        assert p > 0.05  # should not be significant

    def test_very_different_samples_low_p(self):
        """Very different samples should have low p-value."""
        a = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        b = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        p = bootstrap_significance(a, b, n_resamples=500)
        assert p < 0.05  # should be significant

    def test_empty_arrays(self):
        """Empty arrays return p=1.0."""
        p = bootstrap_significance(np.array([]), np.array([]))
        assert p == 1.0

    def test_length_mismatch_raises(self):
        """Length mismatch raises CompareError."""
        a = np.array([1.0, 2.0])
        b = np.array([1.0])
        with pytest.raises(CompareError, match="length mismatch"):
            bootstrap_significance(a, b)

    def test_deterministic_with_seed(self):
        """Same seed produces same p-value."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.5, 2.5, 3.5, 4.5])
        p1 = bootstrap_significance(a, b, n_resamples=500, random_seed=42)
        p2 = bootstrap_significance(a, b, n_resamples=500, random_seed=42)
        assert p1 == p2

    def test_p_value_bounds(self):
        """P-value is always in (0, 1]."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        b = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1])
        p = bootstrap_significance(a, b, n_resamples=200)
        assert 0.0 < p <= 1.0


class TestCohensD:
    """Tests for Cohen's d effect size."""

    def test_identical_samples(self):
        """Identical samples give d=0."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        d = cohens_d(a, b)
        assert d == 0.0

    def test_positive_effect(self):
        """B > A gives positive d."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        d = cohens_d(a, b)
        assert d > 0.0

    def test_negative_effect(self):
        """A > B gives negative d."""
        a = np.array([4.0, 5.0, 6.0])
        b = np.array([1.0, 2.0, 3.0])
        d = cohens_d(a, b)
        assert d < 0.0

    def test_large_effect(self):
        """Large difference should give large d."""
        # Use non-zero variance to avoid division by zero
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([100.0, 101.0, 102.0])
        d = cohens_d(a, b)
        assert abs(d) > 1.0

    def test_small_sample_returns_zero(self):
        """Samples with fewer than 2 elements return 0.0."""
        assert cohens_d(np.array([1.0]), np.array([1.0])) == 0.0
        assert cohens_d(np.array([1.0]), np.array([1.0, 2.0])) == 0.0
        assert cohens_d(np.array([1.0, 2.0]), np.array([1.0])) == 0.0

    def test_zero_variance(self):
        """Zero variance samples return 0.0."""
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([2.0, 2.0, 2.0])
        d = cohens_d(a, b)
        assert d == 0.0  # pooled std = 0


class TestAggregateMetrics:
    """Tests for _aggregate_metrics helper."""

    def test_basic_aggregation(self):
        """Aggregates metrics from multiple results."""
        results = [
            {"metrics_json": json.dumps({"accuracy": 0.9, "f1": 0.85})},
            {"metrics_json": json.dumps({"accuracy": 0.8, "f1": 0.75})},
        ]
        agg = _aggregate_metrics(results)
        assert "accuracy" in agg
        assert "f1" in agg
        assert agg["accuracy"] == [0.9, 0.8]
        assert agg["f1"] == [0.85, 0.75]

    def test_empty_results(self):
        """Empty results return empty dict."""
        assert _aggregate_metrics([]) == {}

    def test_missing_metrics_json(self):
        """Results without metrics_json are skipped."""
        results = [{"some_key": "value"}]
        agg = _aggregate_metrics(results)
        assert agg == {}

    def test_invalid_json(self):
        """Invalid JSON in metrics_json is skipped."""
        results = [{"metrics_json": "not valid json"}]
        agg = _aggregate_metrics(results)
        assert agg == {}

    def test_non_float_scores_skipped(self):
        """Non-numeric scores are skipped."""
        results = [
            {"metrics_json": json.dumps({"accuracy": "string_value"})},
        ]
        agg = _aggregate_metrics(results)
        assert agg == {}


class TestCompareRuns:
    """Tests for compare_runs integration."""

    def test_compare_two_runs(self, populated_store: SQLiteStore):
        """Compare two runs with common metrics."""
        result = compare_runs("run_a", "run_b", populated_store, n_resamples=200)
        assert isinstance(result, ComparisonResult)
        assert result.run_a == "run_a"
        assert result.run_b == "run_b"
        assert "accuracy" in result.metric_deltas
        assert "f1" in result.metric_deltas
        assert "accuracy" in result.significance
        assert "f1" in result.significance
        assert "accuracy" in result.effect_sizes
        assert "f1" in result.effect_sizes

    def test_compare_self(self, populated_store: SQLiteStore):
        """Comparing a run to itself gives zero deltas."""
        result = compare_runs("run_a", "run_a", populated_store, n_resamples=200)
        for metric, delta in result.metric_deltas.items():
            assert delta == 0.0

    def test_empty_run_a_raises(self, sqlite_store: SQLiteStore):
        """Run with no results raises CompareError."""
        sqlite_store.save_run(run_id="empty_run", spec_hash="a", model_name="m",
                              dataset_name="d", provenance_json="{}")
        sqlite_store.save_run(run_id="other_run", spec_hash="b", model_name="m",
                              dataset_name="d", provenance_json="{}")
        sqlite_store.save_result(
            run_id="other_run", item_index=0, input_json="{}",
            expected="", prediction="", metrics_json="{}", duration_ms=0.0,
        )
        with pytest.raises(CompareError, match="has no results"):
            compare_runs("empty_run", "other_run", sqlite_store)

    def test_no_common_metrics_raises(self, sqlite_store: SQLiteStore):
        """Runs with no common metrics raise CompareError."""
        sqlite_store.save_run(run_id="run_x", spec_hash="a", model_name="m",
                              dataset_name="d", provenance_json="{}")
        sqlite_store.save_run(run_id="run_y", spec_hash="b", model_name="m",
                              dataset_name="d", provenance_json="{}")
        sqlite_store.save_result(
            run_id="run_x", item_index=0, input_json="{}",
            expected="", prediction="",
            metrics_json=json.dumps({"metric_a": 0.5}),
            duration_ms=0.0,
        )
        sqlite_store.save_result(
            run_id="run_y", item_index=0, input_json="{}",
            expected="", prediction="",
            metrics_json=json.dumps({"metric_b": 0.6}),
            duration_ms=0.0,
        )
        with pytest.raises(CompareError, match="No common metrics"):
            compare_runs("run_x", "run_y", sqlite_store)

    def test_details_populated(self, populated_store: SQLiteStore):
        """Comparison result includes details."""
        result = compare_runs("run_a", "run_b", populated_store, n_resamples=100)
        assert "n_items_a" in result.details
        assert "n_items_b" in result.details
        assert "n_metrics" in result.details
        assert result.details["n_items_a"] == 5
        assert result.details["n_items_b"] == 5
        assert result.details["n_metrics"] == 2

    def test_delta_values(self, populated_store: SQLiteStore):
        """Deltas are correctly computed as mean(B) - mean(A)."""
        result = compare_runs("run_a", "run_b", populated_store, n_resamples=100)
        # run_a accuracy: [0.8, 0.85, 0.9, 0.95, 1.0] -> mean=0.9
        # run_b accuracy: [0.85, 0.9, 0.95, 1.0, 1.05] -> mean=0.95
        # delta = 0.05
        assert abs(result.metric_deltas["accuracy"] - 0.05) < 1e-10


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_create_comparison_result(self):
        """ComparisonResult can be instantiated."""
        result = ComparisonResult(
            run_a="run_a",
            run_b="run_b",
            metric_deltas={"accuracy": 0.05},
            significance={"accuracy": 0.01},
            effect_sizes={"accuracy": 0.5},
            details={"n_items": 10},
        )
        assert result.run_a == "run_a"
        assert result.run_b == "run_b"
        assert result.metric_deltas["accuracy"] == 0.05
        assert result.significance["accuracy"] == 0.01
        assert result.effect_sizes["accuracy"] == 0.5
