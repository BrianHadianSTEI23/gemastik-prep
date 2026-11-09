"""
Tests for export and reload equivalence verification.
Guarantees that a loaded model bundle reproduces the exact predictions and probabilities
of the in-memory fitted adapter.
"""

import pytest
import numpy as np
import pandas as pd
from nlp_workbench.registry import registry
from nlp_workbench.export import ModelExporter
from nlp_workbench.data import DataLoader


def test_export_reload_classification_equivalence(tmp_path):
    """Loaded classification bundle must reproduce identical predictions and probabilities."""
    train_df = DataLoader.generate_synthetic_dataset("classification", n_samples=30, random_state=42)
    test_df = DataLoader.generate_synthetic_dataset("classification", n_samples=10, random_state=123)

    adapter = registry.create_adapter("classical_logistic")
    adapter.fit(train_df, target_column="category")

    orig_preds = adapter.predict(test_df)
    orig_probs = adapter.predict_proba(test_df)

    export_dir = tmp_path / "exported_logistic"
    saved = ModelExporter.export(adapter, output_dir=export_dir, task="classification")
    loaded = ModelExporter.load(saved)

    loaded_preds = loaded.predict(test_df)
    loaded_probs = loaded.predict_proba(test_df)

    assert (orig_preds == loaded_preds).all()
    np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5, atol=1e-5)


def test_export_reload_regression_equivalence(tmp_path):
    """Loaded regression bundle must reproduce identical continuous predictions."""
    train_df = DataLoader.generate_synthetic_dataset("regression", n_samples=30, random_state=42)
    test_df = DataLoader.generate_synthetic_dataset("regression", n_samples=10, random_state=123)

    adapter = registry.create_adapter("regression_tfidf_ridge")
    adapter.fit(train_df, target_column="target")

    orig_preds = adapter.predict(test_df)

    export_dir = tmp_path / "exported_ridge"
    saved = ModelExporter.export(adapter, output_dir=export_dir, task="regression")
    loaded = ModelExporter.load(saved)

    loaded_preds = loaded.predict(test_df)

    np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5, atol=1e-5)
