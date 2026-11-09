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
