"""Tests for provenance capture."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from speceval.provenance import ProvenanceInfo
from speceval.provenance.environment import (
    _git_commit,
    _gpu_info,
    _pip_packages,
    capture_provenance,
)


class TestProvenanceInfo:
    """Tests for the ProvenanceInfo dataclass."""

    def test_create_empty(self):
        """ProvenanceInfo can be created with defaults."""
        info = ProvenanceInfo()
        assert info.git_commit_hash is None
        assert info.python_version is None
        assert info.platform is None
        assert info.hostname is None
        assert info.gpu_info is None
        assert info.pip_packages is None
        assert info.timestamp is None
        assert info.additional == {}

    def test_to_dict(self):
        """to_dict returns a JSON-serialisable dict."""
        info = ProvenanceInfo(
            git_commit_hash="abc123",
            python_version="3.10.12",
            platform="Linux-5.15.0-x86_64",
            hostname="test-machine",
            gpu_info="NVIDIA A100",
            pip_packages=[{"name": "pytest", "version": "7.4.0"}],
            timestamp="2024-01-01T00:00:00Z",
            additional={"custom": "value"},
        )
        d = info.to_dict()
        assert d["git_commit_hash"] == "abc123"
        assert d["python_version"] == "3.10.12"
        assert d["additional"]["custom"] == "value"

    def test_to_json(self):
        """to_json returns a valid JSON string."""
        info = ProvenanceInfo(git_commit_hash="abc123")
        json_str = info.to_json()
        parsed = json.loads(json_str)
        assert parsed["git_commit_hash"] == "abc123"


class TestCaptureProvenance:
    """Tests for capture_provenance function."""

    def test_capture_returns_dict(self):
        """capture_provenance returns a dictionary."""
        info = capture_provenance(include_pip=False)
        assert isinstance(info, dict)

    def test_contains_expected_keys(self):
        """Provenance dict contains expected top-level keys."""
        info = capture_provenance(include_pip=False)
        assert "timestamp" in info
        assert "python_version" in info
        assert "platform" in info
        assert "hostname" in info
        assert "git_commit_hash" in info
        assert "gpu_info" in info

    def test_timestamp_format(self):
        """Timestamp is ISO-8601 formatted."""
        info = capture_provenance(include_pip=False)
        assert info["timestamp"].endswith("Z")
        assert "T" in info["timestamp"]

    def test_python_version(self):
        """Python version is a string."""
        info = capture_provenance(include_pip=False)
        assert isinstance(info["python_version"], str)
        assert len(info["python_version"]) > 0

    def test_platform(self):
        """Platform is a non-empty string."""
        info = capture_provenance(include_pip=False)
        assert isinstance(info["platform"], str)
        assert len(info["platform"]) > 0

    def test_hostname(self):
        """Hostname is a string."""
        info = capture_provenance(include_pip=False)
        assert isinstance(info["hostname"], str)

    def test_include_pip_true(self):
        """With include_pip=True, pip_packages list is included."""
        info = capture_provenance(include_pip=True)
        assert "pip_packages" in info
        if info["pip_packages"] is not None:
            assert isinstance(info["pip_packages"], list)
            if len(info["pip_packages"]) > 0:
                pkg = info["pip_packages"][0]
                assert "name" in pkg
                assert "version" in pkg

    def test_include_pip_false(self):
        """With include_pip=False, pip_packages may be absent or None."""
        capture_provenance(include_pip=False)
        # It's implementation-defined; check it doesn't raise
        assert True

    def test_git_commit(self):
        """Git commit is a string or None."""
        info = capture_provenance(include_pip=False)
        assert info["git_commit_hash"] is None or isinstance(info["git_commit_hash"], str)

    def test_multiple_calls(self):
        """Multiple calls return consistent structure."""
        info1 = capture_provenance(include_pip=False)
        info2 = capture_provenance(include_pip=False)
        assert set(info1.keys()) == set(info2.keys())


class TestGitCommit:
    """Tests for _git_commit internal helper."""

    @patch("speceval.provenance.environment.subprocess.run")
    def test_git_commit_found(self, mock_run):
        """Git commit returns hash when git succeeds."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "abc123\n"
        mock_run.return_value = mock_result

        result = _git_commit(cwd=None)
        assert result == "abc123"

    @patch("speceval.provenance.environment.subprocess.run")
    def test_git_commit_not_found(self, mock_run):
        """Git commit returns None when git fails."""
        mock_run.side_effect = FileNotFoundError()

        result = _git_commit(cwd=None)
        assert result is None

    @patch("speceval.provenance.environment.subprocess.run")
    def test_git_commit_timeout(self, mock_run):
        """Git commit returns None on timeout."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=5)

        result = _git_commit(cwd=None)
        assert result is None

    @patch("speceval.provenance.environment.subprocess.run")
    def test_git_not_in_repo(self, mock_run):
        """Git commit returns None when not in a repo."""
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_run.return_value = mock_result

        result = _git_commit(cwd=None)
        assert result is None


class TestGpuInfo:
    """Tests for _gpu_info internal helper."""

    @patch("speceval.provenance.environment.shutil.which")
    def test_no_nvidia_smi(self, mock_which):
        """GPU info returns None when nvidia-smi not found."""
        mock_which.return_value = None
        result = _gpu_info()
        assert result is None

    @patch("speceval.provenance.environment.shutil.which")
    @patch("speceval.provenance.environment.subprocess.run")
    def test_nvidia_smi_success(self, mock_run, mock_which):
        """GPU info returns GPU names when nvidia-smi works."""
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA A100\nNVIDIA V100\n"
        mock_run.return_value = mock_result

        result = _gpu_info()
        assert result == "NVIDIA A100, NVIDIA V100"

    @patch("speceval.provenance.environment.shutil.which")
    @patch("speceval.provenance.environment.subprocess.run")
    def test_nvidia_smi_fails(self, mock_run, mock_which):
        """GPU info returns None when nvidia-smi fails."""
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        result = _gpu_info()
        assert result is None

    @patch("speceval.provenance.environment.shutil.which")
    @patch("speceval.provenance.environment.subprocess.run")
    def test_nvidia_smi_timeout(self, mock_run, mock_which):
        """GPU info returns None on timeout."""
        import subprocess
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

        result = _gpu_info()
        assert result is None


class TestPipPackages:
    """Tests for _pip_packages internal helper."""

    def test_returns_list(self):
        """_pip_packages returns a list."""
        packages = _pip_packages()
        assert isinstance(packages, list)

    def test_packages_have_name_and_version(self):
        """Each package has name and version keys."""
        packages = _pip_packages()
        for pkg in packages:
            assert "name" in pkg
            assert "version" in pkg

    def test_sorted_by_name(self):
        """Packages are sorted alphabetically by name."""
        packages = _pip_packages()
        names = [p["name"].lower() for p in packages]
        assert names == sorted(names)

    def test_no_duplicates(self):
        """No duplicate package names."""
        packages = _pip_packages()
        names = [p["name"].lower() for p in packages]
        assert len(names) == len(set(names))

    def test_pytest_exists(self):
        """pytest should be in the list (since we're running with it)."""
        packages = _pip_packages()
        names = [p["name"].lower() for p in packages]
        assert "pytest" in names
