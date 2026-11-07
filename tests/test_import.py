"""Smoke test: the package imports and exposes a version."""

import pfopt


def test_import_and_version() -> None:
    assert isinstance(pfopt.__version__, str)
    assert pfopt.__version__
