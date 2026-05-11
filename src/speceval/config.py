"""Global configuration for speceval."""

import os
from pathlib import Path


def get_config_dir() -> Path:
    """Return the speceval config directory (platform-appropriate)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "speceval"
    home = Path.home()
    if os.name == "nt":  # Windows
        return Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / "speceval"
    return home / ".config" / "speceval"


def get_cache_dir() -> Path:
    """Return the speceval cache directory."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "speceval"
    home = Path.home()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local")) / "speceval" / "cache"
    return home / ".cache" / "speceval"


def get_data_dir() -> Path:
    """Return the speceval data directory (for result stores, registries)."""
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "speceval"
    home = Path.home()
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / "speceval" / "data"
    return home / ".local" / "share" / "speceval"


# Default paths used by the framework
CONFIG_DIR = get_config_dir()
CACHE_DIR = get_cache_dir()
DATA_DIR = get_data_dir()

# Ensure directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default SQLite store path
DEFAULT_STORE_PATH = DATA_DIR / "results.db"

# Default report output directory
DEFAULT_REPORT_DIR = Path.cwd() / "speceval_reports"

# Env vars for runtime configuration
ENV_API_KEY_PREFIX = "SPECTEVAL_API_KEY_"
