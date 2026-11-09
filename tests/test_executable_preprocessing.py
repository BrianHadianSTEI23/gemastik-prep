"""
Tests for executable accepted preprocessing decisions.
Ensures user-configured preprocessing decisions directly govern the cleaning
pipeline during training, inference, and serialization.
"""

import pytest
import pandas as pd
from nlp_workbench.preprocessing import TextCleaner
from nlp_workbench.tasks.classification import ClassicalClassificationAdapter
from nlp_workbench.export import ModelExporter


def test_text_cleaner_honors_decisions():
    """TextCleaner behavior must change according to configured decisions."""
    cleaner_lower = TextCleaner(lowercase=True, remove_html=True)
    cleaner_raw = TextCleaner(lowercase=False, remove_html=False)

    sample = "<b>Kementerian KESEHATAN</b> https://kemkes.go.id"
    out_lower = cleaner_lower.clean_text(sample)
    out_raw = cleaner_raw.clean_text(sample)

    assert "kementerian kesehatan" in out_lower
    assert "KESEHATAN" in out_raw
    assert "<b>" in out_raw


def test_adapter_uses_custom_cleaner():
    """ClassicalClassificationAdapter must use the injected cleaner for text preparation."""
    cleaner = TextCleaner(lowercase=True, remove_stopwords=True, language="indonesian")
    adapter = ClassicalClassificationAdapter(
        model_id="classical_nb",
        text_columns=["text"],
        cleaner=cleaner,
    )

    df = pd.DataFrame({
        "text": ["Ini adalah sebuah uji coba sistem"],
        "target": ["test"],
    })
    prepared = adapter._prepare_text(df)
    # Stopwords 'ini', 'adalah' must be removed
    assert "ini" not in prepared.iloc[0].split()
    assert "uji" in prepared.iloc[0].split()


def test_exported_bundle_persists_cleaner(tmp_path):
    """Model bundle must serialize and restore the cleaner configuration."""
    cleaner = TextCleaner(lowercase=True, remove_stopwords=True, language="indonesian")
    adapter = ClassicalClassificationAdapter(
        model_id="classical_nb",
        text_columns=["text"],
        cleaner=cleaner,
    )
    df = pd.DataFrame({
        "text": ["Teks berita teknologi terbaru", "Laporan kesehatan masyarakat"],
        "target": ["tek", "kes"],
    })
    adapter.fit(df, target_column="target")

    export_path = tmp_path / "model_bundle"
    saved = ModelExporter.export(adapter, output_dir=export_path, task="classification")
    loaded = ModelExporter.load(saved)

    assert loaded.adapter.cleaner.remove_stopwords is True
    assert loaded.adapter.cleaner.lowercase is True
