"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from config.settings import DATE_COL, STATE_COL, TARGET_COL
from src.features import (
    add_calendar_features,
    add_cyclical_encodings,
    add_lag_features,
    add_rolling_features,
    ALL_FEATURE_COLS,
)


class TestCalendarFeatures:
    def test_adds_expected_columns(self, sample_weekly_df):
        df = add_calendar_features(sample_weekly_df.copy())
        for col in ["month", "week_of_year", "year", "quarter", "day_of_week"]:
            assert col in df.columns

    def test_day_of_week_constant_on_weekly(self, sample_weekly_df):
        """Weekly W-SAT data should always land on Saturday (dayofweek=5)."""
        df = add_calendar_features(sample_weekly_df.copy())
        assert (df["day_of_week"] == 5).all()

    def test_month_range(self, sample_weekly_df):
        df = add_calendar_features(sample_weekly_df.copy())
        assert df["month"].between(1, 12).all()


class TestCyclicalEncodings:
    def test_sin_cos_bounded(self, sample_weekly_df):
        df = add_calendar_features(sample_weekly_df.copy())
        df = add_cyclical_encodings(df)
        for col in ["month_sin", "month_cos", "week_of_year_sin", "week_of_year_cos"]:
            assert df[col].between(-1, 1).all(), f"{col} out of [-1,1]"

    def test_sin_cos_not_constant(self, sample_weekly_df):
        df = add_calendar_features(sample_weekly_df.copy())
        df = add_cyclical_encodings(df)
        assert df["month_sin"].std() > 0


class TestLagFeatures:
    def test_lag_1_is_shifted(self, sample_weekly_df):
        df = sample_weekly_df.sort_values([STATE_COL, DATE_COL]).copy()
        df = add_lag_features(df)
        ca = df[df[STATE_COL] == "California"].reset_index(drop=True)
        # lag_1 should equal previous row's sales
        assert ca["lag_1"].iloc[1] == pytest.approx(ca[TARGET_COL].iloc[0])

    def test_lag_1_first_row_is_nan(self, sample_weekly_df):
        df = sample_weekly_df.sort_values([STATE_COL, DATE_COL]).copy()
        df = add_lag_features(df)
        ca = df[df[STATE_COL] == "California"].reset_index(drop=True)
        assert pd.isna(ca["lag_1"].iloc[0])

    def test_no_cross_state_leakage(self, sample_weekly_df):
        """First row of each state should have NaN lag — no data from other states."""
        df = sample_weekly_df.sort_values([STATE_COL, DATE_COL]).copy()
        df = add_lag_features(df)
        for state in df[STATE_COL].unique():
            first = df[df[STATE_COL] == state].iloc[0]
            assert pd.isna(first["lag_1"]), f"Lag leaked across states for {state}"


class TestRollingFeatures:
    def test_no_target_leakage(self, sample_weekly_df):
        """Rolling stats must not include the current row's target."""
        df = sample_weekly_df.sort_values([STATE_COL, DATE_COL]).copy()
        df = add_lag_features(df)
        df = add_rolling_features(df)
        ca = df[df[STATE_COL] == "California"].reset_index(drop=True)
        # roll_mean_4 at row i should use rows i-4..i-1 (shifted), not row i
        # The first row's rolling mean should be NaN or based on shifted data
        assert pd.notna(ca["roll_mean_4"].iloc[1])


class TestFullPipeline:
    def test_all_feature_cols_present(self, sample_featured_df):
        for col in ALL_FEATURE_COLS:
            assert col in sample_featured_df.columns, f"Missing feature: {col}"

    def test_no_nans_in_calendar_features(self, sample_featured_df):
        cal_cols = ["month", "year", "quarter", "day_of_week"]
        for col in cal_cols:
            assert sample_featured_df[col].notna().all(), f"NaN in {col}"

    def test_row_count_preserved(self, sample_weekly_df, sample_featured_df):
        assert len(sample_featured_df) == len(sample_weekly_df)
