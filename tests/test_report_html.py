"""Tests for HTML report template rendering."""

from __future__ import annotations

import jinja2
import pytest

from speceval.report.html import _env, _from_json_filter


class TestReportEnvironment:
    """Jinja2 environment configuration."""

    def test_env_creation(self):
        """_env() returns a configured Environment."""
        env = _env()
        assert isinstance(env, jinja2.Environment)

    def test_env_has_report_template(self):
        """The environment can load the main report template."""
        env = _env()
        template = env.get_template("report.html")
        assert template is not None

    def test_env_autoescape_enabled(self):
        """Auto-escaping is on by default."""
        env = _env()
        assert env.autoescape is True

    def test_env_has_from_json_filter(self):
        """The 'from_json' custom filter is registered."""
        env = _env()
        assert "from_json" in env.filters


class TestFromJsonFilter:
    """The custom ``from_json`` Jinja2 filter."""

    def test_parse_valid_json(self):
        """Valid JSON string is parsed to a Python dict."""
        result = _from_json_filter('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_invalid_json(self):
        """Invalid JSON returns an empty dict."""
        result = _from_json_filter("not json")
        assert result == {}

    def test_parse_json_array(self):
        """JSON array parses correctly."""
        result = _from_json_filter("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_parse_none_value(self):
        """None input returns empty dict."""
        result = _from_json_filter(None)
        assert result == {}

    def test_parse_type_error(self):
        """Non-string non-None types return empty dict gracefully."""
        result = _from_json_filter(42)
        assert result == {}


class TestReportRendering:
    """End-to-end template rendering with sample data."""

    def test_basic_report_renders(self):
        """Minimal context renders a report without errors."""
        template = _env().get_template("report.html")
        html = template.render(
            title="Test Report",
            run_id="test-run-001",
            run={
                "spec_yaml": "name: test",
                "created_at": "2026-01-01T00:00:00",
                "duration_s": 12.5,
            },
            provenance={"timestamp": "2026-01-01T00:00:00"},
            aggregated_metrics={
                "accuracy": type("Stats", (), {"mean": 0.95, "std": 0.05, "min": 0.0, "max": 1.0})()
            },
            results=[],
            total_items=0,
        )
        assert isinstance(html, str)
        assert len(html) > 100
        assert "Test Report" in html
        assert "0.9500" in html

    def test_report_with_results(self):
        """Results are rendered in the table."""
        template = _env().get_template("report.html")
        html = template.render(
            title="Results Report",
            run_id="run-abc",
            run={
                "spec_yaml": "name: eval",
                "created_at": "2026-01-01T00:00:00",
                "duration_s": 5.0,
            },
            provenance={},
            aggregated_metrics={
                "accuracy": type("Stats", (), {"mean": 0.88, "std": 0.10, "min": 0.0, "max": 1.0})()
            },
            results=[
                {
                    "item_index": 0,
                    "duration_ms": 1200,
                    "metrics_json": '{"accuracy": 1.0}',
                },
                {
                    "item_index": 1,
                    "duration_ms": 800,
                    "metrics_json": '{"accuracy": 0.0}',
                },
            ],
            total_items=2,
        )
        assert "0.8800" in html
        assert "1200.0 ms" in html

    def test_report_empty(self):
        """Empty context renders without errors."""
        template = _env().get_template("report.html")
        html = template.render(
            title="Empty",
            run_id="e",
            run={"spec_yaml": "", "created_at": "", "duration_s": 0},
            provenance={},
            aggregated_metrics={},
            results=[],
            total_items=0,
        )
        assert isinstance(html, str)

    @pytest.fixture
    def template_dir(self, tmp_path):
        """Create a temporary template directory with a custom template."""
        d = tmp_path / "templates"
        d.mkdir()
        (d / "custom.html").write_text("<html><body>{{ message }}</body></html>")
        return d

    def test_custom_template_loading(self, template_dir):
        """Can load templates from a custom directory."""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=True,
        )
        template = env.get_template("custom.html")
        html = template.render(message="Hello from custom template")
        assert "Hello from custom template" in html
