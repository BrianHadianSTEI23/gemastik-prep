"""
Unit tests for ModelExporter and predict_new inference API.
"""

import pytest
import pandas as pd
from pathlib import Path
from nlp_workbench.registry import registry
from nlp_workbench.export import ModelExporter
from nlp_workbench.inference import predict_new
from nlp_workbench.data import generate_synthetic_classification_dataset


def test_export_and_load_classical_model(temp_dir):
    df = generate_synthetic_classification_dataset(n_samples=30, random_state=42)
    adapter = registry.create_adapter("classical_nb", ngram_range=(1, 1), max_features=100)
    adapter.train(df, text_col="text", target_col="target")

    export_path = temp_dir / "exported_model"
    metadata = {
        "task": "classification",
        "text_column": "text",
        "target_column": "target",
        "classes": list(adapter.classes_),
    }

    # Export
    bundle_path = ModelExporter.export(
        adapter=adapter,
        output_dir=export_path,
        task="classification",
        metadata=metadata,
        cleaner_config={"lowercase": True},
    )
    assert Path(bundle_path).exists()

    # Load back
    loaded_bundle = ModelExporter.load(bundle_path)
    assert loaded_bundle["task"] == "classification"
    assert loaded_bundle["adapter"] is not None

    # Inference with loaded adapter
    new_texts = ["Pemerintah membuka pendaftaran beasiswa", "Film ini sangat membosankan"]
    res = predict_new(
        texts=new_texts,
        adapter=loaded_bundle["adapter"],
        cleaner_config=loaded_bundle.get("cleaner_config"),
    )
    assert "predictions" in res
    assert len(res["predictions"]) == 2
    assert "probabilities" in res
