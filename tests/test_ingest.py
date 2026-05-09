"""Tests for data ingestion logic."""

import numpy as np
import pandas as pd
import pytest

from config.settings import DATE_COL, STATE_COL, TARGET_COL, FREQ
from src.ingest import parse_dates, validate, resample_to_weekly


class TestParseDates:
    def test_mixed_formats(self):
        df = pd.DataFrame({
            DATE_COL: ["2021-01-15", "15-02-2021", "2021-03-20"],
            STATE_COL: ["A", "A", "A"],
            TARGET_COL: [1, 2, 3],
        })
        result = parse_dates(df)
        assert result[DATE_COL].isna().sum() == 0
        assert pd.api.types.is_datetime64_any_dtype(result[DATE_COL])

    def test_drops_unparseable(self):
        """parse_dates raises on truly unparseable strings (pandas strict mode)."""
        df = pd.DataFrame({
            DATE_COL: ["2021-01-15", "not_a_date"],
            STATE_COL: ["A", "A"],
            TARGET_COL: [1, 2],
        })
        with pytest.raises(Exception):
            parse_dates(df)


class TestValidate:
    def test_drops_duplicates(self):
        df = pd.DataFrame({
            DATE_COL: pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-08"]),
            STATE_COL: ["A", "A", "A"],
            TARGET_COL: [10, 20, 30],
        })
        result = validate(df)
        assert len(result) == 2

    def test_drops_category_column(self):
        df = pd.DataFrame({
            DATE_COL: pd.to_datetime(["2021-01-01"]),
            STATE_COL: ["A"],
            TARGET_COL: [10],
            "category": ["Beverages"],
        })
        result = validate(df)
        assert "category" not in result.columns


class TestResample:
    def test_fills_gaps(self):
        """Sparse data should be expanded to complete weekly grid."""
        dates = pd.to_datetime(["2021-01-02", "2021-02-06"])  # ~5 weeks apart
        df = pd.DataFrame({
            DATE_COL: dates,
            STATE_COL: ["A", "A"],
            TARGET_COL: [100.0, 200.0],
        })
        result, _ = resample_to_weekly(df)
        # Should have filled in the missing weeks
        assert len(result) >= 3

    def test_no_nulls_after_imputation(self):
        dates = pd.to_datetime(["2021-01-02", "2021-03-06"])
        df = pd.DataFrame({
            DATE_COL: dates,
            STATE_COL: ["A", "A"],
            TARGET_COL: [100.0, 300.0],
        })
        result, _ = resample_to_weekly(df)
        assert result[TARGET_COL].isna().sum() == 0
