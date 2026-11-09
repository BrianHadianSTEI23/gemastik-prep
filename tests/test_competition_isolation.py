"""
Tests for strict Gemastik competition isolation, unlabelled test handling,
OOF probability alignment, and submission verification.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from nlp_workbench.config import WorkbenchConfig
from nlp_workbench.orchestration import ExperimentRunner
from nlp_workbench.competition import GemastikRunner
from nlp_workbench.data import generate_synthetic_classification_dataset


def test_missing_competition_test_path_raises_error():
    """gemastik_mode=True strictly requires competition_test_path to prevent falling back to evaluation test set."""
    cfg = WorkbenchConfig(
        task="classification",
        gemastik_mode=True,
        competition_test_path=None,
    )
    with pytest.raises(ValueError, match="gemastik_mode=True requires 'competition_test_path'"):
        cfg.validate()


def test_competition_test_isolation_from_held_out_test(tmp_path):
    """When competition_test_path is synthetic, it must generate a distinct unlabelled set without touching test_df."""
    cfg = WorkbenchConfig(
        task="classification",
        data_path="synthetic",
        model_ids=["classical_logistic"],
        output_dir=str(tmp_path / "model_out"),
        gemastik_mode=True,
        competition_test_path="synthetic",
        submission_path=str(tmp_path / "submission.csv"),
        random_state=42,
    )
    runner = ExperimentRunner(cfg)
    result = runner.run()

    # Held out evaluation test set has ground-truth target
    assert "target" in result.test_df.columns or "category" in result.test_df.columns
    assert len(result.test_df) > 0

    # Submission file exists and is validated
    assert Path(result.competition_summary["submission_path"]).exists()
    sub_df = pd.read_csv(result.competition_summary["submission_path"])
    assert len(sub_df) == result.competition_summary["row_count"]
    assert "id" in sub_df.columns
    tgt_col = result.competition_summary["target_column"]
    assert sub_df[tgt_col].isna().sum() == 0

    # Ensure competition test set row count was independent of held-out test split size
    assert result.competition_summary["row_count"] == 40


def test_oof_ensemble_class_universe_alignment():
    """
    OOF fold models with differing class universes or permutations must be aligned
    to a global class universe before probability averaging.
    """
    class MockFoldModel:
        def __init__(self, classes, probs):
            self.classes_ = np.array(classes)
            self._probs = np.array(probs)

        def predict_proba(self, X):
            return self._probs

    # Suppose fold 1 only saw ['A', 'B']
    model_1 = MockFoldModel(
        classes=["A", "B"],
        probs=[
            [0.8, 0.2],
            [0.3, 0.7],
        ],
    )
    # Suppose fold 2 saw ['B', 'C'] in that order
    model_2 = MockFoldModel(
        classes=["B", "C"],
        probs=[
            [0.1, 0.9],
            [0.6, 0.4],
        ],
    )

    test_df = pd.DataFrame({
        "id": [1, 2],
        "text": ["sample one", "sample two"],
    })

    runner = GemastikRunner(id_column="id", target_column="target", task="classification")
    final_preds = runner.generate_predictions(
        adapter=model_1,
        test_df=test_df,
        text_column="text",
        ensemble_adapters=[model_1, model_2],
    )

    # Global classes: ['A', 'B', 'C']
    # Sample 1:
    # Model 1 probs: [A: 0.8, B: 0.2, C: 0.0]
    # Model 2 probs: [A: 0.0, B: 0.1, C: 0.9]
    # Mean: [A: 0.4, B: 0.15, C: 0.45] -> Argmax is 'C'
    # Sample 2:
    # Model 1 probs: [A: 0.3, B: 0.7, C: 0.0]
    # Model 2 probs: [A: 0.0, B: 0.6, C: 0.4]
    # Mean: [A: 0.15, B: 0.65, C: 0.2] -> Argmax is 'B'
    assert list(final_preds) == ["C", "B"]


def test_submission_strict_validation(tmp_path):
    """Submissions with NaNs, row count discrepancies, or misaligned IDs must be rejected."""
    runner = GemastikRunner(id_column="id", target_column="target")
    test_df = pd.DataFrame({"id": [10, 20, 30], "text": ["a", "b", "c"]})

    # NaN in predictions
    with pytest.raises(ValueError, match="NaN/null"):
        runner.create_submission(test_df, [1.0, np.nan, 2.0], output_path=tmp_path / "nan.csv")

    # Mismatched row count
    with pytest.raises(ValueError, match="Prediction count mismatch"):
        runner.create_submission(test_df, [1, 2], output_path=tmp_path / "short.csv")
