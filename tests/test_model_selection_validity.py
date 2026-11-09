"""
Tests for direction-aware model selection, candidate metric compatibility,
and fair evaluation benchmarking.
"""

import pytest
import pandas as pd
import numpy as np
from nlp_workbench.training import select_best_model
from nlp_workbench.evaluation import get_task_evaluation_spec, is_higher_better


def test_regression_model_selection_minimizes_error():
    """Regression primary metrics like MAE / RMSE must select the lowest error candidate."""
    # List of candidate dictionaries
    candidates = [
        {"model_id": "model_worse", "primary_metric": "Mean Absolute Error", "primary_score": 4.25},
        {"model_id": "model_best", "primary_metric": "Mean Absolute Error", "primary_score": 2.10},
        {"model_id": "model_mid", "primary_metric": "Mean Absolute Error", "primary_score": 3.40},
    ]
    best = select_best_model(candidates, task="regression")
    assert best.model_id == "model_best"
    assert best.score == pytest.approx(2.10)

    # DataFrame leaderboard
    df = pd.DataFrame({
        "Model ID": ["ridge_linear", "gbr_tree", "mlp_neural"],
        "Validation Score": [3.8, 1.95, 2.7],
        "Primary Metric": ["Mean Absolute Error", "Mean Absolute Error", "Mean Absolute Error"],
    })
    best_df = select_best_model(df, task="regression")
    assert best_df.model_id == "gbr_tree"
    assert best_df.score == pytest.approx(1.95)


def test_classification_model_selection_maximizes_score():
    """Classification primary metrics (Macro-F1, Accuracy) must select the highest score candidate."""
    df = pd.DataFrame({
        "Model ID": ["logistic", "naive_bayes", "transformer"],
        "Validation Score": [0.82, 0.74, 0.93],
        "Primary Metric": ["Macro-F1", "Macro-F1", "Macro-F1"],
    })
    best = select_best_model(df, task="classification")
    assert best.model_id == "transformer"
    assert best.score == pytest.approx(0.93)


def test_retrieval_model_selection_maximizes_ndcg():
    """Retrieval primary metric (NDCG@5) must select highest candidate."""
    candidates = [
        {"model_id": "bm25", "primary_metric": "NDCG@5", "primary_score": 0.65},
        {"model_id": "dense", "primary_metric": "NDCG@5", "primary_score": 0.78},
        {"model_id": "hybrid", "primary_metric": "NDCG@5", "primary_score": 0.88},
    ]
    best = select_best_model(candidates, task="retrieval")
    assert best.model_id == "hybrid"
    assert best.score == pytest.approx(0.88)


def test_incompatible_metrics_raises_error():
    """Candidates reporting differing primary metrics cannot be compared on the same leaderboard."""
    incompatible_candidates = [
        {"model_id": "model_a", "primary_metric": "Accuracy", "primary_score": 0.85},
        {"model_id": "model_b", "primary_metric": "Macro-F1", "primary_score": 0.82},
    ]
    with pytest.raises(ValueError, match="Cannot compare candidates with incompatible primary metrics"):
        select_best_model(incompatible_candidates)

    df_incompatible = pd.DataFrame({
        "Model ID": ["model_a", "model_b"],
        "Validation Score": [0.85, 0.82],
        "Primary Metric": ["Accuracy", "Macro-F1"],
    })
    with pytest.raises(ValueError, match="Cannot compare candidates with incompatible primary metrics"):
        select_best_model(df_incompatible)


def test_missing_or_nan_scores_raises_error():
    """Candidates with missing or NaN scores must raise error instead of converting to fake scores."""
    candidates_with_nan = [
        {"model_id": "model_ok", "primary_metric": "Macro-F1", "primary_score": 0.80},
        {"model_id": "model_bad", "primary_metric": "Macro-F1", "primary_score": float("nan")},
    ]
    with pytest.raises(ValueError, match="null/NaN"):
        select_best_model(candidates_with_nan, task="classification")

    df_nan = pd.DataFrame({
        "Model ID": ["model_ok", "model_bad"],
        "Validation Score": [0.80, np.nan],
        "Primary Metric": ["Macro-F1", "Macro-F1"],
    })
    with pytest.raises(ValueError, match="null/NaN"):
        select_best_model(df_nan, task="classification")


def test_unsupported_auto_selection_task_raises_error():
    """Unsupervised or non-comparable tasks cannot be auto-selected without explicit primary metric."""
    df = pd.DataFrame({
        "Model ID": ["lda_model", "nmf_model"],
        "Validation Score": [0.5, 0.6],
    })
    with pytest.raises(ValueError, match="does not support automatic"):
        select_best_model(df, task="topic_modeling")
