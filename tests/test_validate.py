"""Tests for data validation guards."""

import pandas as pd
import pytest

from config.settings import DATE_COL, STATE_COL, TARGET_COL
from src.validate import (
    ValidationError,
    validate_raw_input,
    validate_weekly_panel,
    validate_splits,
)


class TestRawInputValidation:
    def test_missing_columns(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_raw_input(df)

    def test_empty_dataframe(self):
        df = pd.DataFrame({DATE_COL: [], STATE_COL: [], TARGET_COL: []})
        with pytest.raises(ValidationError, match="empty"):
            validate_raw_input(df)

    def test_all_null_target(self):
        df = pd.DataFrame({
            DATE_COL: ["2021-01-01"],
            STATE_COL: ["A"],
            TARGET_COL: [None],
        })
        with pytest.raises(ValidationError, match="entirely null"):
            validate_raw_input(df)

    def test_valid_input_passes(self):
        df = pd.DataFrame({
            DATE_COL: ["2021-01-01", "2021-01-02"],
            STATE_COL: ["A", "B"],
            TARGET_COL: [100, 200],
        })
        validate_raw_input(df)  # should not raise


class TestSplitValidation:
    def test_empty_split(self):
        splits = {
            "train": pd.DataFrame({DATE_COL: [], STATE_COL: [], TARGET_COL: []}),
            "val": pd.DataFrame({DATE_COL: [pd.Timestamp("2022-01-01")], STATE_COL: ["A"], TARGET_COL: [1]}),
        }
        with pytest.raises(ValidationError, match="empty"):
            validate_splits(splits)

    def test_temporal_overlap(self):
        splits = {
            "train": pd.DataFrame({DATE_COL: [pd.Timestamp("2022-06-01")], STATE_COL: ["A"], TARGET_COL: [1]}),
            "val": pd.DataFrame({DATE_COL: [pd.Timestamp("2022-01-01")], STATE_COL: ["A"], TARGET_COL: [1]}),
        }
        with pytest.raises(ValidationError, match="overlap"):
            validate_splits(splits)

    def test_valid_splits_pass(self):
        splits = {
            "train": pd.DataFrame({DATE_COL: [pd.Timestamp("2021-01-01")], STATE_COL: ["A"], TARGET_COL: [1]}),
            "val": pd.DataFrame({DATE_COL: [pd.Timestamp("2022-01-01")], STATE_COL: ["A"], TARGET_COL: [1]}),
            "test": pd.DataFrame({DATE_COL: [pd.Timestamp("2023-01-01")], STATE_COL: ["A"], TARGET_COL: [1]}),
        }
        validate_splits(splits)  # should not raise
