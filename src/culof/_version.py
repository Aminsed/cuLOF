"""Single source of truth for the package version.

pyproject.toml reads it via `tool.setuptools.dynamic`, and `culof.__version__`
re-exports it, so there is exactly one place to bump.
"""

__version__ = "0.2.0"
