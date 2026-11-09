"""
Tests for automatic selection enforcement, metric integrity, and CV support guards.
Verifies that the workbench refuses to fabricate winners for non-scalar or qualitative tasks.
"""

from unittest.mock import patch
import pytest
import pandas as pd
from nlp_workbench.config import WorkbenchConfig
from nlp_workbench.orchestration import ExperimentRunner
from nlp_workbench.training import select_best_model, run_cross_validation
from nlp_workbench.evaluation import get_task_evaluation_spec
from nlp_workbench.registry import registry


def test_generation_cannot_use_word_count_as_selection_metric():
    """Word count or average generated words cannot be used as an automatic quality metric."""
    leaderboard = [
        {"model_id": "model_verbose", "Average Generated Words": 150.0},
        {"model_id": "model_concise", "Average Generated Words": 30.0},
    ]
    with pytest.raises(ValueError, match="cannot be used as an automatic quality metric"):
        select_best_model(leaderboard, primary_metric="Average Generated Words")


def test_topic_modeling_refuses_auto_selection():
    """Topic modeling cannot automatically declare a winner on topic diversity alone."""
    leaderboard = [
        {"model_id": "topic_lda", "Topic Diversity": 0.85},
        {"model_id": "topic_nmf", "Topic Diversity": 0.70},
    ]
    with pytest.raises(ValueError, match="does not support automatic winner selection"):
        select_best_model(leaderboard, task="topic_modeling")


def test_orchestrator_refuses_winner_for_topic_modeling():
    """ExperimentRunner for topic modeling must complete without declaring a fake winner."""
    cfg = WorkbenchConfig(
        task="topic_modeling",
        data_path="synthetic",
        text_columns=["content"],
        target_column=None,
        model_ids=["topic_lda"],
    )
    runner = ExperimentRunner(cfg)
    result = runner.run()

    assert result.best_model_id is None
    assert result.best_adapter is None
    assert "Notice" in result.test_metrics
    assert not result.leaderboard.empty


def test_orchestrator_refuses_winner_for_generation_without_custom_metric():
    """ExperimentRunner for open generation without custom metric must not declare a fake winner."""
    cfg = WorkbenchConfig(
        task="generation",
        language="english",
        data_path="synthetic",
        text_columns=["prompt"],
        target_column=None,
        model_ids=["generation_gpt"],
    )
    runner = ExperimentRunner(cfg)
    from nlp_workbench.tasks.generation import GenerationAdapter
    with patch.object(GenerationAdapter, "fit", return_value={"status": "mocked"}), \
         patch.object(GenerationAdapter, "evaluate", return_value={"ROUGE-L F1": None, "Avg Output Words (descriptive)": 15.0}):
        result = runner.run()

    assert result.best_model_id is None
    assert result.best_adapter is None
    assert "Notice" in result.test_metrics


def test_cross_validation_rejected_for_unsupported_tasks():
    """Tasks with supports_cv=False must raise ValueError when CV is attempted."""
    for task_name in ["topic_modeling", "generation", "summarization", "translation", "zeroshot"]:
        spec = get_task_evaluation_spec(task_name)
        assert not spec.supports_cv
        adapter = registry.create_adapter(registry.list_models(task=task_name)[0].model_id)
        df = pd.DataFrame({"text": ["a", "b", "c", "d"], "target": [1, 2, 3, 4]})
        with pytest.raises(ValueError, match="does not support cross-validation"):
            run_cross_validation(adapter=adapter, df=df, task=task_name)
