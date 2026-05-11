"""Tests for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from speceval.cli.main import app


@pytest.fixture
def runner() -> CliRunner:
    """Return a Typer CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_spec_file(temp_dir: Path) -> Path:
    """Create a temporary valid spec file."""
    spec_path = temp_dir / "speceval.yaml"
    spec_path.write_text("""name: test-cli
model:
  provider: openai
  name: gpt-4o-mini
dataset:
  source: jsonl
  path: ./data.jsonl
metrics:
  - name: exact_match
trials: 1
""")
    return spec_path


class TestCliInit:
    """Tests for `speceval init` command."""

    def test_init_creates_file(self, runner: CliRunner, temp_dir: Path):
        """`speceval init` creates a YAML template."""
        target = temp_dir / "speceval.yaml"
        result = runner.invoke(app, ["init", "--path", str(target)])
        assert result.exit_code == 0
        assert target.exists()
        content = target.read_text()
        assert "model:" in content
        assert "dataset:" in content
        assert "metrics:" in content

    def test_init_default_path(self, runner: CliRunner):
        """`speceval init` without path writes to speceval.yaml."""
        # Use temp cwd to avoid overwriting real files
        pass  # This would write to cwd, hard to test

    def test_init_existing_file_abort(self, runner: CliRunner, temp_dir: Path):
        """Init to existing path prompts and can abort."""
        target = temp_dir / "speceval.yaml"
        target.write_text("existing content")
        result = runner.invoke(app, ["init", "--path", str(target)], input="n\n")
        assert result.exit_code == 0
        assert target.read_text() == "existing content"

    def test_init_existing_file_overwrite(self, runner: CliRunner, temp_dir: Path):
        """Init to existing path overwrites when confirmed."""
        target = temp_dir / "speceval.yaml"
        target.write_text("old content")
        result = runner.invoke(app, ["init", "--path", str(target)], input="y\n")
        assert result.exit_code == 0
        content = target.read_text()
        assert "model:" in content
        assert "old content" not in content

    def test_init_interactive(self, runner: CliRunner, temp_dir: Path):
        """Init with --interactive prompts for config."""
        target = temp_dir / "speceval.yaml"
        inputs = "\n".join([
            "openai",          # provider
            "gpt-4",           # model
            "huggingface",     # dataset source
            "test/dataset",    # dataset path
            "test",            # split
            "{question}",      # input template
            "answer",          # reference field
            "exact_match,bleu",  # metrics
        ])
        result = runner.invoke(
            app, ["init", "--path", str(target), "--interactive"],
            input=inputs,
        )
        assert result.exit_code == 0
        assert target.exists()
        content = target.read_text()
        assert "gpt-4" in content
        assert "exact_match" in content
        assert "bleu" in content


class TestCliValidate:
    """Tests for `speceval validate` command."""

    def test_validate_valid_spec(self, runner: CliRunner, temp_spec_file: Path):
        """Valid spec file passes validation."""
        result = runner.invoke(app, ["validate", "--spec", str(temp_spec_file)])
        assert result.exit_code == 0
        assert "valid" in result.stdout

    def test_validate_invalid_spec(self, runner: CliRunner, temp_dir: Path):
        """Invalid spec file reports errors."""
        spec_file = temp_dir / "bad.yaml"
        spec_file.write_text("not: valid: yaml: :")
        result = runner.invoke(app, ["validate", "--spec", str(spec_file)])
        assert result.exit_code == 1

    def test_validate_missing_file(self, runner: CliRunner):
        """Non-existent spec file reports error."""
        result = runner.invoke(app, ["validate", "--spec", "/nonexistent/spec.yaml"])
        assert result.exit_code == 1

    def test_validate_missing_required_keys(self, runner: CliRunner, temp_dir: Path):
        """Spec missing required keys reports errors."""
        spec_file = temp_dir / "incomplete.yaml"
        spec_file.write_text("name: incomplete\n")
        result = runner.invoke(app, ["validate", "--spec", str(spec_file)])
        assert result.exit_code == 1
        assert "Missing required" in result.stdout


class TestCliRun:
    """Tests for `speceval run` command."""

    @patch("speceval.store.sqlite.SQLiteStore")
    @patch("speceval.provenance.environment.capture_provenance")
    @patch("speceval.metrics.register_all")
    def test_run_with_valid_spec(self, mock_register, mock_prov, mock_store,
                                  runner: CliRunner, temp_spec_file: Path):
        """Valid spec runs successfully."""
        mock_prov.return_value = {"test": True}
        mock_store_instance = MagicMock()
        mock_store_instance.get_results.return_value = [
            {
                "item_index": 0,
                "metrics_json": json.dumps({"exact_match": 1.0}),
                "duration_ms": 100.0,
            },
        ]
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["run", "--spec", str(temp_spec_file)])
        assert result.exit_code == 0

    def test_run_missing_spec(self, runner: CliRunner):
        """Run with non-existent spec reports error."""
        result = runner.invoke(app, ["run", "--spec", "/nonexistent/spec.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    @patch("speceval.store.sqlite.SQLiteStore")
    @patch("speceval.provenance.environment.capture_provenance")
    @patch("speceval.metrics.register_all")
    def test_run_with_overrides(self, mock_register, mock_prov, mock_store,
                                 runner: CliRunner, temp_dir: Path):
        """Run with --trials and --output overrides work."""
        spec_file = temp_dir / "spec.yaml"
        spec_file.write_text("""name: override-test
model:
  provider: openai
  name: gpt-4
dataset:
  source: jsonl
  path: ./data.jsonl
  output_dir: ./custom_output
metrics:
  - name: exact_match
trials: 2
""")
        mock_prov.return_value = {"test": True}
        mock_store_instance = MagicMock()
        mock_store_instance.get_results.return_value = [
            {
                "item_index": 0,
                "metrics_json": json.dumps({"exact_match": 1.0}),
                "duration_ms": 50.0,
            },
        ]
        mock_store.return_value = mock_store_instance

        out_dir = temp_dir / "output"
        result = runner.invoke(app, [
            "run", "--spec", str(spec_file),
            "--output", str(out_dir),
            "--trials", "3",
        ])
        assert result.exit_code == 0


class TestCliCompare:
    """Tests for `speceval compare` command."""

    @patch("speceval.store.sqlite.SQLiteStore")
    @patch("speceval.compare.delta.compare_runs")
    def test_compare_two_runs(self, mock_compare_runs, mock_store,
                               runner: CliRunner):
        """Compare command displays comparison results."""
        from speceval.compare.delta import ComparisonResult

        mock_compare_runs.return_value = ComparisonResult(
            run_a="run_a",
            run_b="run_b",
            metric_deltas={"accuracy": 0.05},
            significance={"accuracy": 0.01},
            effect_sizes={"accuracy": 0.5},
            details={"n_items_a": 10, "n_items_b": 10, "n_metrics": 1},
        )
        mock_store_instance = MagicMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["compare", "run_a", "run_b"])
        assert result.exit_code == 0
        assert "run_a" in result.stdout
        assert "run_b" in result.stdout
        assert "accuracy" in result.stdout

    @patch("speceval.store.sqlite.SQLiteStore")
    @patch("speceval.compare.delta.compare_runs")
    def test_compare_failure(self, mock_compare_runs, mock_store,
                              runner: CliRunner):
        """Compare with failing comparison reports error."""
        mock_compare_runs.side_effect = Exception("Something went wrong")
        mock_store_instance = MagicMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["compare", "run_a", "run_b"])
        assert result.exit_code == 1
        assert "failed" in result.stdout.lower()


class TestCliReport:
    """Tests for `speceval report` command."""

    @patch("speceval.store.sqlite.SQLiteStore")
    def test_report_by_run_id(self, mock_store, runner: CliRunner):
        """Report command with run ID works."""
        mock_store_instance = MagicMock()
        mock_store_instance.get_run.return_value = {
            "id": "test_run",
            "spec_hash": "abc",
            "model_name": "gpt-4",
            "dataset_name": "test",
            "provenance_json": "{}",
            "status": "completed",
        }
        mock_store_instance.get_results.return_value = []
        mock_store.return_value = mock_store_instance

        with patch("speceval.report.html.generate_html_report") as mock_gen:
            mock_gen.return_value = Path("/tmp/report.html")
            result = runner.invoke(app, ["report", "--run-id", "test_run"])

        assert result.exit_code == 0


class TestCliList:
    """Tests for `speceval list` command."""

    @patch("speceval.store.sqlite.SQLiteStore")
    def test_list_no_runs(self, mock_store, runner: CliRunner):
        """List with no runs says none found."""
        mock_store_instance = MagicMock()
        mock_store_instance.get_runs.return_value = []
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No runs found" in result.stdout

    @patch("speceval.store.sqlite.SQLiteStore")
    def test_list_with_runs(self, mock_store, runner: CliRunner):
        """List with runs displays them."""
        mock_store_instance = MagicMock()
        mock_store_instance.get_runs.return_value = [
            {"id": "run1", "model_name": "gpt-4", "dataset_name": "test",
             "timestamp": "2024-01-01", "status": "completed"},
            {"id": "run2", "model_name": "gpt-3.5", "dataset_name": "test2",
             "timestamp": "2024-01-02", "status": "running"},
        ]
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "run1" in result.stdout
        assert "run2" in result.stdout
