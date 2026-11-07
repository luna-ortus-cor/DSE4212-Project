"""Tests for reading the processed parquet files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfopt.data.cleaning import PROCESSED_FILES, read_processed


def _frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=5, name="Date")
    columns = pd.Index(["AAA", "BBB"], name="Symbol")
    return pd.DataFrame(rng.normal(size=(5, 2)), index=index, columns=columns)


def test_read_processed_roundtrip(tmp_path: Path) -> None:
    written = {name: _frame(i) for i, name in enumerate(PROCESSED_FILES)}
    for name, frame in written.items():
        frame.to_parquet(tmp_path / f"{name}.parquet")
    read = read_processed(tmp_path)
    assert set(read) == set(PROCESSED_FILES)
    for name, frame in written.items():
        pd.testing.assert_frame_equal(read[name], frame, check_freq=False, check_names=False)


def test_read_processed_missing_file_raises(tmp_path: Path) -> None:
    _frame(0).to_parquet(tmp_path / "log_returns.parquet")
    with pytest.raises(FileNotFoundError, match="01b_clean_data"):
        read_processed(tmp_path)
