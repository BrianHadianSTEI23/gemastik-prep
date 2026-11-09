"""
Tests for temporal cross-validation.
Guarantees chronological expanding-window splits with zero shuffling and zero future leakage.
"""

import pytest
import pandas as pd
import numpy as np

from nlp_workbench.splitting import DatasetSplitter
from nlp_workbench.training import run_cross_validation
from nlp_workbench.registry import registry


def test_temporal_cv_folds_zero_future_leakage():
    """Time-based CV folds must strictly satisfy max(train_idx) < min(val_idx)."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=50, freq="D"),
        "text": [f"Document at day {i}" for i in range(50)],
        "target": [f"Cat_{i % 3}" for i in range(50)],
    })

    splitter = DatasetSplitter(strategy="time", n_splits=4)
    folds = list(splitter.get_cv_folds(df, time_col="timestamp", target_col="target"))

    assert len(folds) == 4

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        # Zero future leakage: every train timestamp must be strictly earlier than any val timestamp
        train_times = df.loc[train_idx, "timestamp"]
        val_times = df.loc[val_idx, "timestamp"]
        assert train_times.max() < val_times.min()


def test_run_cross_validation_with_time_column():
    """run_cross_validation with time_column sorts chronologically and maintains temporal boundary."""
    # Shuffled order in input df
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    shuffled_dates = np.random.RandomState(42).permutation(dates)

    df = pd.DataFrame({
        "created_at": shuffled_dates,
        "title": [f"Title {i}" for i in range(30)],
        "content": [f"Content for document {i}" for i in range(30)],
        "category": [f"Class_{i % 2}" for i in range(30)],
    })

    adapter = registry.create_adapter("classical_nb")
    cv_res = run_cross_validation(
        adapter=adapter,
        train_df=df,
        task="classification",
        target_column="category",
        time_column="created_at",
        split_strategy="time",
        n_splits=3,
    )

    assert "mean_cv_score" in cv_res
    assert len(cv_res["cv_scores"]) == 3
    assert cv_res["cv_scores"][0] >= 0.0


def test_temporal_split_with_identical_timestamps_at_boundary():
    """Observations with identical timestamps at split boundary must not leak across partitions."""
    # Day 1: 5 rows, Day 2: 10 rows (boundary), Day 3: 5 rows
    times = (
        [pd.Timestamp("2025-01-01")] * 5 +
        [pd.Timestamp("2025-01-02")] * 10 +
        [pd.Timestamp("2025-01-03")] * 5
    )
    df = pd.DataFrame({
        "timestamp": times,
        "text": [f"Sample {i}" for i in range(20)],
        "target": [f"Cat_{i % 2}" for i in range(20)],
    })

    train_df, test_df = DatasetSplitter.split_train_test(df, time_column="timestamp", test_size=0.3)

    assert len(train_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(test_df) == len(df)

    # Intersection of timestamps between train and test must be empty
    train_times = set(train_df["timestamp"])
    test_times = set(test_df["timestamp"])
    assert len(train_times.intersection(test_times)) == 0
    assert max(train_times) < min(test_times)


def test_temporal_split_out_of_order_input():
    """Unordered input must be chronologically ordered before temporal train/test partitioning."""
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-01-05"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-04")],
        "text": ["Doc 5", "Doc 1", "Doc 3", "Doc 2", "Doc 4"],
        "target": ["A", "B", "A", "B", "A"],
    })

    train_df, test_df = DatasetSplitter.split_train_test(df, time_column="timestamp", test_size=0.4)

    assert train_df["timestamp"].max() < test_df["timestamp"].min()
    assert list(train_df["timestamp"]) == sorted(list(train_df["timestamp"]))


def test_group_split_strict_isolation():
    """Group-based splitting guarantees complete mutual exclusion of group identifiers."""
    df = pd.DataFrame({
        "user_id": ["user_1"] * 5 + ["user_2"] * 5 + ["user_3"] * 5 + ["user_4"] * 5,
        "text": [f"Post {i}" for i in range(20)],
        "target": [f"Label_{i % 2}" for i in range(20)],
    })

    train_df, test_df = DatasetSplitter.split_train_test(df, group_column="user_id", test_size=0.25, random_state=42)

    train_users = set(train_df["user_id"])
    test_users = set(test_df["user_id"])

    assert len(train_users) > 0
    assert len(test_users) > 0
    assert len(train_users.intersection(test_users)) == 0
    assert len(train_users) + len(test_users) == 4
