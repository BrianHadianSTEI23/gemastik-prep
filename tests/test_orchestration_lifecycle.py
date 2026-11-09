"""
Tests for canonical experiment lifecycle:
Staged execution (prepare -> run), direction-aware regression runner,
retrieval bundle runner, and export/reload verification.
"""

import pytest
import pandas as pd
from pathlib import Path
from nlp_workbench.config import WorkbenchConfig, get_default_config
from nlp_workbench.orchestration import ExperimentRunner, PreparedDataBundle, ExperimentResult
from nlp_workbench.export import ModelExporter
from nlp_workbench.data import RetrievalDatasetBundle


def test_staged_preparation_lifecycle(tmp_path):
    """Test runner.prepare() returns PreparedDataBundle with complete audits and splits."""
    cfg = WorkbenchConfig(
        task="classification",
        data_path="synthetic",
        model_ids=["classical_nb", "classical_logistic"],
        output_dir=str(tmp_path / "model_out"),
    )
    runner = ExperimentRunner(cfg)
    prepared = runner.prepare()

    assert isinstance(prepared, PreparedDataBundle)
    assert len(prepared.train_df) > 0
    assert len(prepared.val_df) > 0
    assert len(prepared.test_df) > 0
    assert prepared.audit_report is not None
    assert prepared.leakage_report is not None
    assert prepared.leakage_report["leakage_detected"] is False

    # Tuple unpacking check
    tr, va, te = prepared
    assert len(tr) == len(prepared.train_df)

    # Continue execution with runner.run(prepared)
    res = runner.run(prepared)
    assert isinstance(res, ExperimentResult)
    assert res.best_model_id in ["classical_nb", "classical_logistic"]
    assert "Macro-F1" in res.leaderboard["Primary Metric"].values
    assert Path(res.exported_path).exists()


def test_regression_runner_end_to_end(tmp_path):
    """Regression experiment must run end-to-end, minimizing MAE."""
    cfg = get_default_config("regression")
    cfg.data_path = "synthetic"
    cfg.model_ids = ["regression_tfidf_ridge"]
    cfg.output_dir = str(tmp_path / "reg_out")

    runner = ExperimentRunner(cfg)
    result = runner.run()

    assert result.best_model_id == "regression_tfidf_ridge"
    assert "MAE" in result.leaderboard["Primary Metric"].values
    assert "MAE" in result.test_metrics
    assert Path(result.exported_path).exists()

    # Verify reload integrity
    loaded = ModelExporter.load(result.exported_path)
    sample_preds = loaded.predict(result.test_df.head(3))
    assert len(sample_preds) == 3


def test_retrieval_bundle_runner_end_to_end(tmp_path):
    """Retrieval experiment must execute end-to-end preserving RetrievalDatasetBundle."""
    cfg = get_default_config("retrieval")
    cfg.data_path = "synthetic"
    cfg.model_ids = ["ir_bm25"]
    cfg.output_dir = str(tmp_path / "ir_out")

    runner = ExperimentRunner(cfg)
    prepared = runner.prepare()

    assert isinstance(prepared.train_df, RetrievalDatasetBundle)
    assert isinstance(prepared.val_df, RetrievalDatasetBundle)
    assert isinstance(prepared.test_df, RetrievalDatasetBundle)

    result = runner.run(prepared)
    assert result.best_model_id == "ir_bm25"
    assert "NDCG@5" in result.leaderboard["Primary Metric"].values
    assert "NDCG@5" in result.test_metrics
    assert Path(result.exported_path).exists()
