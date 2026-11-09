"""
Unit tests for GemastikRunner competition submission pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from nlp_workbench.competition import GemastikRunner
from nlp_workbench.registry import registry
from nlp_workbench.data import generate_synthetic_classification_dataset


def test_gemastik_submission_generation(temp_dir):
    # Setup training data and train adapter
    train_df = generate_synthetic_classification_dataset(n_samples=40, random_state=42)
    adapter = registry.create_adapter("classical_logistic", ngram_range=(1, 1), max_features=100)
    adapter.train(train_df, text_col="text", target_col="target")

    # Setup unlabelled competition test data
    test_df = pd.DataFrame({
        "id": [101, 102, 103, 104, 105],
        "text": [
            "Presiden menghadiri peresmian gedung baru",
            "Pelatih sepak bola memuji performa tim",
            "Penelitian terbaru menemukan obat herbal",
            "Masyarakat antusias menyambut liburan akhir tahun",
            "IHSG menguat di penutupan perdagangan hari ini",
        ],
    })

    runner = GemastikRunner(id_column="id", target_column="target", task="classification")
    prepared_test = runner.prepare_test_dataset(test_df)
    assert len(prepared_test) == 5

    preds = runner.generate_predictions(adapter, prepared_test, text_column="text")
    assert len(preds) == 5

    out_csv = temp_dir / "submission.csv"
    summary = runner.create_submission(prepared_test, preds, output_path=out_csv)

    assert out_csv.exists()
    assert summary["row_count"] == 5
    assert summary["id_column"] == "id"
    assert summary["target_column"] == "target"

    # Verify loaded CSV
    sub_df = pd.read_csv(out_csv)
    assert len(sub_df) == 5
    assert list(sub_df["id"]) == [101, 102, 103, 104, 105]
    assert sub_df["target"].isna().sum() == 0


def test_gemastik_mismatched_rows_raises_error(temp_dir):
    runner = GemastikRunner()
    test_df = pd.DataFrame({"id": [1, 2, 3], "text": ["a", "b", "c"]})
    mismatched_preds = ["pos", "neg"]  # only 2 predictions for 3 rows

    with pytest.raises(ValueError, match="Prediction count mismatch"):
        runner.create_submission(test_df, mismatched_preds, output_path=temp_dir / "bad.csv")


def test_gemastik_nan_prediction_raises_error(temp_dir):
    runner = GemastikRunner()
    test_df = pd.DataFrame({"id": [1, 2], "text": ["a", "b"]})
    nan_preds = ["pos", None]

    with pytest.raises(ValueError, match="NaN/null values"):
        runner.create_submission(test_df, nan_preds, output_path=temp_dir / "nan.csv")
