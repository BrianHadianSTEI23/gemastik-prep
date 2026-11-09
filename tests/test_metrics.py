"""
Unit tests for evaluation metrics across task categories.
"""

import pytest
import numpy as np
from nlp_workbench.evaluation import (
    evaluate_classification,
    evaluate_regression,
    evaluate_multilabel,
    evaluate_ner,
    evaluate_qa,
    evaluate_summarization,
    evaluate_translation,
    evaluate_similarity,
    evaluate_retrieval,
    evaluate_topic_modeling,
    evaluate_generation,
    evaluate_predictions,
)


def test_classification_metrics():
    y_true = np.array(["pos", "neg", "pos", "neu"])
    y_pred = np.array(["pos", "neg", "neg", "neu"])
    metrics = evaluate_classification(y_true, y_pred)
    assert "Accuracy" in metrics
    assert "Macro-F1" in metrics
    assert 0.0 <= metrics["Accuracy"] <= 1.0
    assert metrics["Accuracy"] == 0.75


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    metrics = evaluate_regression(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["MAE"] < 0.3


def test_multilabel_metrics():
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    metrics = evaluate_multilabel(y_true, y_pred)
    assert "Hamming Loss" in metrics
    assert "Micro-F1" in metrics
    assert "Macro-F1" in metrics


def test_qa_metrics():
    refs = ["Jakarta", "Presiden Soekarno"]
    preds = ["jakarta", "Soekarno"]
    metrics = evaluate_qa(refs, preds)
    assert "Exact Match (EM)" in metrics
    assert "Token F1" in metrics
    assert metrics["Exact Match (EM)"] == 50.0  # Case-insensitive match on "Jakarta" vs "jakarta"


def test_similarity_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.2, 1.8, 3.1, 4.2, 4.9])
    metrics = evaluate_similarity(y_true, y_pred)
    assert "Pearson Correlation" in metrics
    assert "Spearman Correlation" in metrics
    assert metrics["Pearson Correlation"] > 0.95


def test_retrieval_metrics():
    # qrels: query_id -> {doc_id: relevance}
    qrels = {
        "q1": {"doc1": 1, "doc2": 1},
        "q2": {"doc3": 1},
    }
    # ranked_results: query_id -> [doc_id_1, doc_id_2, ...]
    ranked_results = {
        "q1": ["doc1", "doc4", "doc2"],
        "q2": ["doc5", "doc3"],
    }
    metrics = evaluate_retrieval(qrels, ranked_results, k=3)
    assert "Recall@3" in metrics
    assert "MRR" in metrics
    assert metrics["MRR"] > 0.0


def test_topic_modeling_metrics():
    keywords = [
        ["data", "mining", "model", "teks"],
        ["pasar", "saham", "ekonomi", "uang"],
        ["kesehatan", "dokter", "obat", "imun"],
    ]
    metrics = evaluate_topic_modeling(keywords, top_n=4)
    assert "Topic Diversity" in metrics
    assert metrics["Topic Diversity"] == 1.0  # all words are distinct
    # When reference corpus is absent, Topic Coherence must be None, never fabricated (e.g. diversity * 0.9)
    assert metrics["Topic Coherence"] is None
    assert "Topic Coherence Notice" in metrics


def test_topic_coherence_distinguishes_cooccurrence_with_identical_diversity():
    """
    Demonstrates that genuine empirical coherence differs based on reference corpus co-occurrence,
    even when topic diversity is 100% identical.
    """
    keywords = [
        ["kucing", "anjing", "kelinci"],
        ["mobil", "motor", "sepeda"],
    ]

    # Corpus A: High co-occurrence — words within the same topic appear together in documents
    corpus_high_cooccurrence = [
        "kucing anjing kelinci bermain bersama di taman",
        "kelinci dan kucing serta anjing tidur siang",
        "mobil motor sepeda melaju di jalan tol",
        "sepeda dan motor parkir dekat mobil merah",
        "anjing kucing kelinci makan bersama",
        "motor mobil sepeda adalah alat transportasi",
    ]

    # Corpus B: Disjoint / Zero co-occurrence — words appear strictly isolated in separate documents
    corpus_zero_cooccurrence = [
        "hanya ada kucing di sini",
        "seekor anjing menggonggong",
        "kelinci putih melompat",
        "mobil melaju di jalan",
        "sebuah motor baru",
        "sepeda tua di garasi",
    ]

    res_high = evaluate_topic_modeling(keywords, reference_corpus=corpus_high_cooccurrence, top_n=3)
    res_zero = evaluate_topic_modeling(keywords, reference_corpus=corpus_zero_cooccurrence, top_n=3)

    # Topic Diversity is identical across both evaluations
    assert res_high["Topic Diversity"] == res_zero["Topic Diversity"] == 1.0

    # Coherence scores MUST differ based on empirical co-occurrence
    assert res_high["Topic Coherence"] is not None
    assert res_zero["Topic Coherence"] is not None
    assert res_high["Topic Coherence"] > res_zero["Topic Coherence"]
    assert res_zero["Topic Coherence"] == -1.0  # complete absence of pair co-occurrence produces minimum NPMI


def test_generation_without_references_never_reports_fake_quality_score():
    """
    Verifies that generation without references reports descriptive metrics and explicit notice,
    without computing a fake quality score.
    """
    predictions = ["Model output text for test sample", "Another generated response"]
    metrics = evaluate_generation(references=None, predictions=predictions, latencies=[25.0, 30.0])

    assert metrics["ROUGE-L F1"] is None
    assert "Evaluation Notice" in metrics
    assert "Avg Output Words (descriptive)" in metrics
    assert metrics["Avg Output Words (descriptive)"] > 0
    assert "Sample Outputs" in metrics
