"""
Chitti settings.

Every adjustable setting is read from an environment variable, or from a file named `.env`
that sits next to chitti.py. Real environment variables win over the file.
Secrets (API keys) belong ONLY in `.env` or in environment variables, never in the code.
Copy `.env.example` to `.env` and edit it.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ENV_FILE = APP_DIR / ".env"


def load_env(path=None) -> bool:
    """Read KEY=VALUE lines from .env into os.environ (existing variables are not overwritten)."""
    path = Path(path) if path else ENV_FILE
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        elif val.startswith("#"):
            val = ""                                  # KEY=   # comment   means: empty
        else:
            val = val.split(" #")[0].strip()          # allow  KEY=value  # comment
        if key:
            os.environ.setdefault(key, val)
    return True


def get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def get_int(name: str, default: int) -> int:
    try:
        return int(float(get(name, str(default))))
    except ValueError:
        return default


def get_float(name: str, default: float) -> float:
    try:
        return float(get(name, str(default)))
    except ValueError:
        return default


def get_bool(name: str, default: bool = False) -> bool:
    v = get(name, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on", "y")


def get_list(name: str, default: str = "") -> list:
    return [x.strip() for x in get(name, default).split(",") if x.strip()]


def mask(secret: str) -> str:
    """Show a secret safely in a message: 'sk-a...wxyz'. Never print the full value."""
    if not secret:
        return "(not set)"
    if len(secret) <= 8:
        return "****"
    return f"{secret[:3]}...{secret[-4:]}"


def redact(text: str) -> str:
    """Remove every configured secret (any setting named ..._KEY, ..._TOKEN, ..._SECRET) from text before it is shown."""
    for name, value in os.environ.items():
        if name.upper().endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")) and len(value) >= 8:
            text = text.replace(value, mask(value))
    return text


load_env()
