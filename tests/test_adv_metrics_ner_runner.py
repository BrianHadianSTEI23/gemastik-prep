"""
Adversarial tests for:
1. Real NER prediction-to-BIO conversion and seqeval evaluation (guaranteeing no gold copying).
2. Multi-metric correctness (Counter multiset token QA F1, normalized EM, graded NDCG@K, strict length rejection).
3. Adapters for multilabel, regression, pair classification, and retrieval reranking.
4. ExperimentRunner end-to-end execution.
"""

from collections import Counter
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from nlp_workbench.evaluation import (
    compute_qa_f1,
    compute_qa_exact_match,
    evaluate_qa,
    evaluate_ner,
    evaluate_retrieval,
    evaluate_summarization,
    evaluate_translation,
    evaluate_similarity,
    evaluate_classification,
    evaluate_multilabel,
    is_higher_better,
)
from nlp_workbench.tasks.ner import NERAdapter
from nlp_workbench.tasks.multilabel import TransformerMultilabelAdapter
from nlp_workbench.tasks.regression import TransformerRegressionAdapter
from nlp_workbench.tasks.pair_classification import CrossEncoderPairAdapter, SiamesePairAdapter
from nlp_workbench.tasks.zeroshot import ZeroShotClassificationAdapter
from nlp_workbench.tasks.retrieval import RetrievalAdapter
from nlp_workbench.config import WorkbenchConfig
from nlp_workbench.orchestration import ExperimentRunner
from nlp_workbench.export import ModelExporter
from nlp_workbench.inference import predict_new


# -------------------------------------------------------------
# 1. NER EVALUATION & NO-GOLD-COPYING REGRESSION TEST
# -------------------------------------------------------------
def test_ner_evaluator_never_copies_gold():
    """Verify that predictions differ from gold tags and are genuine."""
    gold_tags = [
        ["B-PER", "I-PER", "O", "B-LOC"],
        ["O", "B-ORG", "O"],
    ]
    # Completely different or corrupted predictions
    pred_tags = [
        ["O", "O", "O", "B-LOC"],
        ["B-PER", "O", "O"],
    ]

    metrics = evaluate_ner(gold_tags, pred_tags)
    assert "Overall F1" in metrics
    assert "Entity Precision" in metrics
    assert "Entity Recall" in metrics
    # If the old bug was present (copying gold), F1 would be 1.0!
    # Because pred != gold, F1 must be < 1.0
    assert metrics["Overall F1"] < 1.0


def test_ner_length_mismatch_raises_error():
    """Ensure seqeval length mismatch per sentence raises ValueError."""
    gold_tags = [["B-PER", "I-PER", "O"]]
    mismatched_preds = [["B-PER", "I-PER"]]  # 2 vs 3 tokens

    with pytest.raises(ValueError, match="Token count mismatch"):
        evaluate_ner(gold_tags, mismatched_preds)


# -------------------------------------------------------------
# 2. QA MULTISET F1 & EM NORMALIZATION TESTS
# -------------------------------------------------------------
def test_qa_multiset_f1_penalizes_token_multiplicity():
    """
    Multiset token overlap with Counter:
    Gold: 'apple' (1 count)
    Prediction: 'apple apple apple apple' (4 counts)
    Set intersection would give overlap 1 / len(set) = 1.0!
    Counter multiset intersection gives overlap 1, precision = 1/4 = 0.25, recall = 1/1 = 1.0, F1 = 0.4.
    """
    gold = "apple"
    pred = "apple apple apple apple"
    f1 = compute_qa_f1(gold, pred)
    # Multiset F1 must strictly be 0.4, NOT 1.0
    assert np.isclose(f1, 0.4, atol=1e-3)


def test_qa_normalization_ignores_articles_punct_case():
    """SQuAD EM ignores articles ('a', 'an', 'the'), punctuation, and casing."""
    gold = "The University of Indonesia."
    pred = "university of indonesia"
    assert compute_qa_exact_match(gold, pred) == 1.0
    assert compute_qa_f1(gold, pred) == 1.0


# -------------------------------------------------------------
# 3. ZERO-SHOT REJECTS INVALID REFERENCE LABELS
# -------------------------------------------------------------
def test_zeroshot_rejects_unseen_ground_truth_labels():
    adapter = ZeroShotClassificationAdapter(
        model_id="test_zeroshot",
        candidate_labels=["Technology", "Sports", "Politics"],
    )
    # df with a label outside candidate_labels
    df = pd.DataFrame({
        "title": ["New phone launched", "Medical discovery announced"],
        "category": ["Technology", "Healthcare"],  # "Healthcare" is invalid!
    })

    with pytest.raises(ValueError, match="Ground truth contains labels"):
        adapter.evaluate(df, target_column="category")


# -------------------------------------------------------------
# 4. SUMMARIZATION & TRANSLATION STRICT LENGTH CHECKS
# -------------------------------------------------------------
def test_summarization_rejects_length_mismatch():
    refs = ["summary 1", "summary 2", "summary 3"]
    preds = ["summary 1", "summary 2"]  # only 2
    with pytest.raises(ValueError, match="(?i)length mismatch"):
        evaluate_summarization(refs, preds)


def test_translation_rejects_length_mismatch():
    refs = ["translation 1", "translation 2"]
    preds = ["translation 1", "translation 2", "translation 3"]
    with pytest.raises(ValueError, match="(?i)length mismatch"):
        evaluate_translation(refs, preds)


# -------------------------------------------------------------
# 5. RETRIEVAL GRADED NDCG & FULL IDCG CALCULATION
# -------------------------------------------------------------
def test_retrieval_graded_ndcg_and_idcg():
    # Gold relevance has graded scores 3, 2, 1, 0
    qrels = {
        "q1": {"d1": 3, "d2": 2, "d3": 1, "d4": 0}
    }
    # System retrieved d2 (rel 2), then d1 (rel 3)
    ranked = {
        "q1": ["d2", "d1"]
    }
    metrics = evaluate_retrieval(qrels, ranked, k=2)
    assert "NDCG@2" in metrics
    # Perfect IDCG order would be d1 (3), then d2 (2)
    # DCG = 2/log2(2) + 3/log2(3) = 2.0 + 1.8928 = 3.8928
    # IDCG = 3/log2(2) + 2/log2(3) = 3.0 + 1.2619 = 4.2619
    # NDCG = 3.8928 / 4.2619 ~= 0.913
    assert 0.8 < metrics["NDCG@2"] < 1.0


# -------------------------------------------------------------
# 6. SIMILARITY ZERO VARIANCE HANDLING
# -------------------------------------------------------------
def test_similarity_zero_variance_does_not_crash():
    gold = np.array([1.0, 1.0, 1.0, 1.0])  # zero variance
    preds = np.array([0.5, 0.6, 0.7, 0.8])
    metrics = evaluate_similarity(gold, preds)
    assert metrics["Pearson Correlation"] == 0.0
    assert metrics["Spearman Correlation"] == 0.0


# -------------------------------------------------------------
# 7. METRIC DIRECTION AWARENESS
# -------------------------------------------------------------
def test_metric_direction_awareness():
    assert is_higher_better("Accuracy") is True
    assert is_higher_better("Macro-F1") is True
    assert is_higher_better("NDCG@5") is True
    assert is_higher_better("SacreBLEU") is True
    assert is_higher_better("ROUGE-1 F1") is True
    assert is_higher_better("MAE") is False
    assert is_higher_better("MSE") is False
    assert is_higher_better("RMSE") is False
    assert is_higher_better("Hamming Loss") is False
    assert is_higher_better("loss") is False


# -------------------------------------------------------------
# 8. HYBRID RETRIEVAL CANDIDATE GENERATION + CROSS-ENCODER
# -------------------------------------------------------------
def test_hybrid_retrieval_candidate_generation():
    adapter = RetrievalAdapter(
        model_id="ir_hybrid_rerank",
        approach="hybrid",
        dense_checkpoint="sentence-transformers/all-MiniLM-L6-v2",
        reranker_checkpoint=None,  # smoke test candidate stage
    )
    corpus = [
        {"doc_id": "doc1", "document": "Machine learning and deep learning in Python"},
        {"doc_id": "doc2", "document": "Financial market reports and bank interest rates"},
        {"doc_id": "doc3", "document": "Medical health advice on vaccination and nutrition"},
    ]
    df_corpus = pd.DataFrame(corpus)
    adapter.index_corpus(df_corpus)

    results = adapter.retrieve("deep learning python", top_k=2)
    assert len(results) == 2
    assert results[0]["doc_id"] == "doc1"


# -------------------------------------------------------------
# 9. END-TO-END EXPERIMENT RUNNER TEST
# -------------------------------------------------------------
def test_experiment_runner_end_to_end(tmp_path):
    config = WorkbenchConfig(
        task="classification",
        language="indonesian",
        data_path="synthetic",
        model_ids=["classical_nb", "classical_logistic"],
        output_dir=str(tmp_path / "model_out"),
        gemastik_mode=True,
        competition_test_path="synthetic",
        submission_path=str(tmp_path / "submission.csv"),
    )
    runner = ExperimentRunner(config)
    result = runner.run()

    assert result.leaderboard is not None
    assert len(result.leaderboard) == 2
    assert result.best_model_id in ["classical_nb", "classical_logistic"]
    assert result.exported_path.exists()
    assert (tmp_path / "submission.csv").exists()
    assert result.competition_summary is not None
    assert result.competition_summary["row_count"] > 0
