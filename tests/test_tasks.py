"""
Unit tests and smoke tests for task execution adapters.
"""

import pytest
import numpy as np
import pandas as pd
from nlp_workbench.registry import registry
from nlp_workbench.data import (
    generate_synthetic_classification_dataset,
    generate_synthetic_regression_dataset,
    generate_synthetic_multilabel_dataset,
    generate_synthetic_pair_classification_dataset,
)


def test_classical_classification_nb():
    df = generate_synthetic_classification_dataset(n_samples=40, random_state=42)
    adapter = registry.create_adapter("classical_nb", ngram_range=(1, 1), max_features=100)
    
    # Train
    result = adapter.train(df, text_col="text", target_col="target")
    assert "train_time_sec" in result
    
    # Predict
    preds = adapter.predict(df["text"].tolist())
    assert len(preds) == len(df)
    assert set(preds).issubset(set(df["target"].unique()))

    # Predict proba
    probs = adapter.predict_proba(df["text"].tolist())
    assert probs.shape == (len(df), len(adapter.classes_))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_classical_classification_svm():
    df = generate_synthetic_classification_dataset(n_samples=40, random_state=42)
    adapter = registry.create_adapter("classical_linear_svm", ngram_range=(1, 1), max_features=100)
    adapter.train(df, text_col="text", target_col="target")
    preds = adapter.predict(df["text"].tolist())
    assert len(preds) == len(df)


def test_regression_tfidf_ridge():
    df = generate_synthetic_regression_dataset(n_samples=40, random_state=42)
    adapter = registry.create_adapter("regression_tfidf_ridge", max_features=100)
    adapter.train(df, text_col="text", target_col="target")
    preds = adapter.predict(df["text"].tolist())
    assert len(preds) == len(df)
    assert isinstance(preds[0], (float, np.floating, int, np.integer))


def test_multilabel_binary_relevance():
    df = generate_synthetic_multilabel_dataset(n_samples=30, random_state=42)
    target_cols = [c for c in df.columns if c.startswith("label_")]
    adapter = registry.create_adapter("multilabel_binary_relevance", max_features=50)
    adapter.train(df, text_col="text", target_cols=target_cols)
    preds = adapter.predict(df["text"].tolist())
    assert preds.shape == (len(df), len(target_cols))


def test_topic_modeling_lda():
    df = generate_synthetic_classification_dataset(n_samples=30, random_state=42)
    adapter = registry.create_adapter("topic_lda", n_topics=3, max_features=100)
    res = adapter.train(df, text_col="text")
    assert "top_words_per_topic" in res
    assert len(res["top_words_per_topic"]) == 3


def test_ir_bm25():
    adapter = registry.create_adapter("ir_bm25")
    corpus = [
        "Jokowi meresmikan jalan tol trans sumatera di palembang",
        "Kucing persia bermain bola di taman kota",
        "Pemerintah membuka seleksi cpns tahun 2026",
    ]
    adapter.index_corpus(corpus)
    results = adapter.retrieve(query="presiden meresmikan jalan tol", top_k=2)
    assert len(results) == 2
    # First retrieved doc should be the first corpus text
    assert results[0]["corpus_id"] == 0


def test_pair_classification():
    df = generate_synthetic_pair_classification_dataset(n_samples=20, random_state=42)
    adapter = registry.create_adapter("pair_cross_encoder", device="cpu")
    # Smoke test adapter methods
    assert adapter.approach == "cross_encoder"
