"""Dashboard components package.

Importing this package injects Streamlit Cloud secrets into ``os.environ``
so that ``packages.core.config.get_settings()`` can read them via
``os.getenv`` on any platform.
"""
from __future__ import annotations

import os

_SECRET_KEYS = ["MONGODB_URI", "MONGODB_DB", "CRYOSWARM_API_KEY"]


def inject_secrets() -> None:
    """Copy known Streamlit Cloud secrets into os.environ by explicit key."""
    try:
        import streamlit as st

        for key in _SECRET_KEYS:
            try:
                val = st.secrets[key]
                if val and key not in os.environ:
                    os.environ[key] = str(val)
            except KeyError:
                pass
    except Exception:
        pass


inject_secrets()

# Force-reset the settings cache so the next call picks up the new env vars.
try:
    from packages.core.config import _settings_cache
    import packages.core.config as _cfg

    if _cfg._settings_cache is not None and not _cfg._settings_cache.mongodb_uri:
        _cfg._settings_cache = None
except Exception:
    pass