"""
Unit tests for DataLoader, schema validation, and synthetic generators.
"""

import pytest
import pandas as pd
from pathlib import Path
from nlp_workbench.data import (
    DataLoader,
    validate_dataset_schema,
    generate_synthetic_classification_dataset,
    generate_synthetic_qa_dataset,
    generate_synthetic_ner_dataset,
    generate_synthetic_summarization_dataset,
    generate_synthetic_translation_dataset,
    generate_synthetic_retrieval_dataset,
)


def test_missing_file_raises_error():
    loader = DataLoader()
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        loader.load("non_existent_dataset_file_12345.csv")


def test_load_csv_and_parquet(temp_dir):
    df_orig = pd.DataFrame({
        "id": [1, 2, 3],
        "text": ["teks satu", "teks dua", "teks tiga"],
        "target": ["pos", "neg", "pos"],
    })
    csv_p = temp_dir / "test.csv"
    df_orig.to_csv(csv_p, index=False)

    loader = DataLoader()
    df_loaded = loader.load(csv_p)
    assert len(df_loaded) == 3
    assert list(df_loaded.columns) == ["id", "text", "target"]

    parquet_p = temp_dir / "test.parquet"
    df_orig.to_parquet(parquet_p, index=False)
    df_parquet = loader.load(parquet_p)
    assert len(df_parquet) == 3


def test_schema_validator():
    df = pd.DataFrame({"text": ["halo", "dunia"], "label": [0, 1]})
    
    # Valid schema check
    validate_dataset_schema(df, task="classification", text_col="text", target_col="label")

    # Missing text column
    with pytest.raises(KeyError, match="Text column 'missing_text' not found"):
        validate_dataset_schema(df, task="classification", text_col="missing_text", target_col="label")

    # Missing target column for classification
    with pytest.raises(KeyError, match="Target column 'missing_target' not found"):
        validate_dataset_schema(df, task="classification", text_col="text", target_col="missing_target")


def test_synthetic_generators():
    clf_df = generate_synthetic_classification_dataset(n_samples=20)
    assert len(clf_df) == 20
    assert "text" in clf_df.columns
    assert "target" in clf_df.columns

    qa_df = generate_synthetic_qa_dataset(n_samples=10)
    assert len(qa_df) == 10
    assert "context" in qa_df.columns
    assert "question" in qa_df.columns
    assert "answer" in qa_df.columns

    ner_df = generate_synthetic_ner_dataset(n_samples=10)
    assert len(ner_df) == 10
    assert "tokens" in ner_df.columns
    assert "ner_tags" in ner_df.columns

    sum_df = generate_synthetic_summarization_dataset(n_samples=10)
    assert len(sum_df) == 10
    assert "text" in sum_df.columns
    assert "summary" in sum_df.columns

    trans_df = generate_synthetic_translation_dataset(n_samples=10)
    assert len(trans_df) == 10
    assert "source_text" in trans_df.columns
    assert "target_text" in trans_df.columns

    ret_df = generate_synthetic_retrieval_dataset(n_samples=10)
    assert len(ret_df) == 10
    assert "query" in ret_df.columns
    assert "doc_text" in ret_df.columns
    assert "relevance" in ret_df.columns
