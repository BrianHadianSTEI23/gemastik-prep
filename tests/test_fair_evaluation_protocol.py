"""
Tests for fair training and evaluation protocol.
Guarantees that all candidate models are evaluated on an identical validation slice (val_df),
that CV scores are reported strictly as diagnostics, and that the selected trainable model
is refitted on train_df + val_df before single held-out test evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from nlp_workbench.config import WorkbenchConfig
from nlp_workbench.orchestration import ExperimentRunner
from nlp_workbench.tasks.classification import ClassicalClassificationAdapter


def test_fair_evaluation_scores_on_val_df():
    """All candidate models must report Validation Score measured strictly on val_df."""
    cfg = WorkbenchConfig(
        task="classification",
        data_path="synthetic",
        text_columns=["title", "content"],
        target_column="category",
        model_ids=["classical_nb", "classical_logistic"],
        experimental_preferences={"enable_cv": True, "cv_folds": 2},
    )
    runner = ExperimentRunner(cfg)
    prepared = runner.prepare()

    leaderboard_df, records = runner.run_benchmark(prepared.train_df, prepared.val_df)

    assert "Validation Score" in leaderboard_df.columns
    assert "CV Mean Score" in leaderboard_df.columns
    assert "CV Std Score" in leaderboard_df.columns

    for mid in ["classical_nb", "classical_logistic"]:
        record = records[mid]
        # The primary_score must be the Validation Score on val_df
        val_score = record["primary_score"]
        adapter = record["adapter"]
        direct_eval = adapter.evaluate(prepared.val_df, target_column="category")
        assert round(float(val_score), 4) == round(float(direct_eval["Macro-F1"]), 4)


def test_trainable_model_refitted_on_train_plus_val():
    """Winning trainable model must be refitted on train_df + val_df before test evaluation."""
    cfg = WorkbenchConfig(
        task="classification",
        data_path="synthetic",
        text_columns=["title", "content"],
        target_column="category",
        model_ids=["classical_nb"],
        experimental_preferences={"enable_cv": False},
    )
    runner = ExperimentRunner(cfg)
    prepared = runner.prepare()

    train_len = len(prepared.train_df)
    val_len = len(prepared.val_df)
    combined_len = train_len + val_len

    fitted_lengths = []
    original_fit = ClassicalClassificationAdapter.fit

    def tracking_fit(self, train_df, val_df=None, **kwargs):
        fitted_lengths.append(len(train_df))
        return original_fit(self, train_df, val_df=val_df, **kwargs)

    with patch.object(ClassicalClassificationAdapter, "fit", side_effect=tracking_fit, autospec=True):
        result = runner.run(prepared)

    # First fit on train_df during benchmark, second fit on train_df + val_df after selection
    assert train_len in fitted_lengths
    assert combined_len in fitted_lengths
    assert result.best_model_id == "classical_nb"
    assert "Macro-F1" in result.test_metrics
