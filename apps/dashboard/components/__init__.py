"""Dashboard components package.

Importing this package injects Streamlit Cloud secrets into ``os.environ``
so that ``packages.core.config.get_settings()`` can read them via
``os.getenv`` on any platform.
"""
from __future__ import annotations

import os


def inject_secrets() -> None:
    """Copy Streamlit Cloud secrets into os.environ."""
    try:
        import streamlit as st

        for key in st.secrets:
            if isinstance(st.secrets[key], str) and key not in os.environ:
                os.environ[key] = st.secrets[key]
    except Exception:
        pass


inject_secrets()