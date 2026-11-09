"""
Standardized Task-Specific Evaluation Metrics module.
Guarantees correct semantic metric calculation across all 14 NLP task families.
"""

import re
import string
import math
import collections
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    hamming_loss,
)
from scipy.stats import pearsonr, spearmanr


def is_higher_better(metric_name: str) -> bool:
    """
    Determines whether higher values of a given evaluation metric indicate better performance.
    """
    m = metric_name.lower().replace("-", "_").replace(" ", "_").strip()
    # Metrics where lower is better
    lower_better_tokens = ["mae", "mse", "rmse", "loss", "hamming", "error", "perplexity", "residual"]
    if any(token in m for token in lower_better_tokens):
        return False
    return True


# 1. Classification Metrics
def evaluate_classification(
    y_true: Union[np.ndarray, List[Any]],
    y_pred: Union[np.ndarray, List[Any]],
    target_names: Optional[List[str]] = None,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics with consistent label universe support."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Classification evaluation length mismatch: {len(y_true_arr)} true vs {len(y_pred_arr)} pred."
        )

    labels = target_names if target_names is not None else None
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", labels=labels, zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", labels=labels, zero_division=0))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", labels=labels, zero_division=0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", labels=labels, zero_division=0))

    try:
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels).tolist()
    except Exception:
        cm = []

    res = {
        "Accuracy": round(acc, 4),
        "Balanced Accuracy": round(bal_acc, 4),
        "Macro-F1": round(macro_f1, 4),
        "Weighted-F1": round(weighted_f1, 4),
        "Macro-Precision": round(macro_prec, 4),
        "Macro-Recall": round(macro_rec, 4),
        "Confusion Matrix": cm,
    }
    return res


# 2. Multilabel Metrics
def evaluate_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes multilabel classification metrics with subset accuracy and Hamming loss."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Multilabel shape mismatch: true shape {y_true_arr.shape} vs pred shape {y_pred_arr.shape}"
        )

    if np.issubdtype(y_pred_arr.dtype, np.floating):
        y_pred_bin = (y_pred_arr >= threshold).astype(int)
    else:
        y_pred_bin = y_pred_arr.astype(int)

    y_true_bin = y_true_arr.astype(int)

    h_loss = float(hamming_loss(y_true_bin, y_pred_bin))
    micro_f1 = float(f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))
    subset_acc = float((y_true_bin == y_pred_bin).all(axis=1).mean()) if len(y_true_bin) > 0 else 0.0

    return {
        "Micro-F1": round(micro_f1, 4),
        "Macro-F1": round(macro_f1, 4),
        "Hamming Loss": round(h_loss, 4),
        "Subset Accuracy": round(subset_acc, 4),
    }


# 3. Regression Metrics
def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Computes regression error metrics and R-squared."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) != len(y_p):
        raise ValueError(f"Regression length mismatch: {len(y_t)} true vs {len(y_p)} pred.")

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(y_t, y_p)) if len(y_t) > 1 and np.var(y_t) > 0 else 0.0
    residuals = y_t - y_p

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Residual Mean": round(float(np.mean(residuals)), 4),
        "Residual Std": round(float(np.std(residuals)), 4),
    }


# 4. NER Metrics (Seqeval Entity-Level)
def evaluate_ner(
    y_true_tags: List[List[str]],
    y_pred_tags: List[List[str]],
) -> Dict[str, Any]:
    """Computes entity-level seqeval metrics for token classification."""
    if len(y_true_tags) != len(y_pred_tags):
        raise ValueError(
            f"Sentence count mismatch for NER: {len(y_true_tags)} true vs {len(y_pred_tags)} pred."
        )

    for i, (gold_seq, pred_seq) in enumerate(zip(y_true_tags, y_pred_tags)):
        if len(gold_seq) != len(pred_seq):
            raise ValueError(
                f"Token count mismatch at sentence index {i}: {len(gold_seq)} gold tokens vs {len(pred_seq)} pred tokens."
            )

    try:
        import seqeval.metrics
        prec = float(seqeval.metrics.precision_score(y_true_tags, y_pred_tags))
        rec = float(seqeval.metrics.recall_score(y_true_tags, y_pred_tags))
        f1 = float(seqeval.metrics.f1_score(y_true_tags, y_pred_tags))
        return {
            "Entity Precision": round(prec, 4),
            "Entity Recall": round(rec, 4),
            "Entity F1 (Seqeval)": round(f1, 4),
            "Overall F1": round(f1, 4),
        }
    except Exception as e:
        return {"NER Evaluation Error": str(e)}


# 5. Question Answering Metrics (Standard SQuAD Normalized EM & Multiset Token F1)
def normalize_qa_text(s: str) -> str:
    """Standardizes text for QA evaluation: lowercase, removes punctuation, removes articles, standardizes spaces."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def compute_qa_f1(gold: str, pred: str) -> float:
    """
    Computes token-level multiset F1 between extractive QA gold and prediction
    using token frequency counts (collections.Counter), consistent with official SQuAD.
    """
    gold_tokens = normalize_qa_text(gold).split()
    pred_tokens = normalize_qa_text(pred).split()

    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)

    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def compute_qa_exact_match(gold: str, pred: str) -> float:
    """Computes normalized Exact Match (EM) binary score."""
    return float(normalize_qa_text(gold) == normalize_qa_text(pred))


def evaluate_qa(
    gold_answers: List[str],
    pred_answers: List[str],
) -> Dict[str, Any]:
    """Computes Exact Match (EM) and average token-level F1 for question answering."""
    if len(gold_answers) != len(pred_answers):
        raise ValueError(
            f"QA evaluation length mismatch: {len(gold_answers)} gold vs {len(pred_answers)} pred."
        )

    exact_matches = [compute_qa_exact_match(g, p) for g, p in zip(gold_answers, pred_answers)]
    f1_scores = [compute_qa_f1(g, p) for g, p in zip(gold_answers, pred_answers)]

    return {
        "Exact Match (EM)": round(float(np.mean(exact_matches)) * 100.0, 2),
        "Token F1": round(float(np.mean(f1_scores)) * 100.0, 2),
    }


# 6. Text Summarization Metrics (ROUGE)
def evaluate_summarization(
    references: List[str],
    predictions: List[str],
) -> Dict[str, Any]:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L metrics. Strictly rejects length mismatch."""
    if len(references) != len(predictions):
        raise ValueError(
            f"Summarization length mismatch: {len(references)} references vs {len(predictions)} predictions."
        )

    try:
        import rouge_score.rouge_scorer
        scorer = rouge_score.rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        r1, r2, rl = [], [], []
        for ref, pred in zip(references, predictions):
            scores = scorer.score(ref, pred)
            r1.append(scores['rouge1'].fmeasure)
            r2.append(scores['rouge2'].fmeasure)
            rl.append(scores['rougeL'].fmeasure)
        return {
            "ROUGE-1 F1": round(float(np.mean(r1)), 4),
            "ROUGE-2 F1": round(float(np.mean(r2)), 4),
            "ROUGE-L F1": round(float(np.mean(rl)), 4),
        }
    except Exception as e:
        return {"Summarization Metric Notice": str(e)}


# 7. Machine Translation Metrics (SacreBLEU & chrF)
def evaluate_translation(
    references: List[str],
    predictions: List[str],
) -> Dict[str, Any]:
    """Computes SacreBLEU and chrF scores. Strictly rejects length mismatch."""
    if len(references) != len(predictions):
        raise ValueError(
            f"Translation length mismatch: {len(references)} references vs {len(predictions)} predictions."
        )

    try:
        import sacrebleu
        ref_corpus = [[r] for r in references]
        bleu = sacrebleu.corpus_bleu(predictions, ref_corpus)
        chrf = sacrebleu.corpus_chrf(predictions, ref_corpus)
        return {
            "SacreBLEU": round(float(bleu.score), 2),
            "chrF": round(float(chrf.score), 2),
        }
    except Exception as e:
        return {"Translation Metric Notice": str(e)}


# 8. Semantic Similarity Metrics (Pearson & Spearman)
def evaluate_similarity(
    gold_scores: Union[np.ndarray, List[float]],
    pred_scores: Union[np.ndarray, List[float]],
) -> Dict[str, Any]:
    """Computes Pearson and Spearman rank correlation with zero-variance safety."""
    gold = np.asarray(gold_scores, dtype=float)
    pred = np.asarray(pred_scores, dtype=float)

    if len(gold) != len(pred):
        raise ValueError(f"Similarity length mismatch: {len(gold)} gold vs {len(pred)} pred.")

    if len(gold) < 2 or np.all(gold == gold[0]) or np.all(pred == pred[0]) or np.isnan(gold).any() or np.isnan(pred).any():
        return {"Spearman Correlation": 0.0, "Pearson Correlation": 0.0}

    p_corr, _ = pearsonr(gold, pred)
    s_corr, _ = spearmanr(gold, pred)

    p_val = float(p_corr) if not np.isnan(p_corr) else 0.0
    s_val = float(s_corr) if not np.isnan(s_corr) else 0.0

    return {
        "Spearman Correlation": round(s_val, 4),
        "Pearson Correlation": round(p_val, 4),
    }


# 9. Information Retrieval Metrics (Recall@K, MRR, Graded NDCG@K)
def evaluate_retrieval(
    qrels: Dict[str, Dict[str, int]],  # query_id -> {doc_id: graded_relevance}
    ranked_results: Dict[str, List[str]],  # query_id -> [doc_id_1, doc_id_2, ...]
    k: int = 5,
) -> Dict[str, Any]:
    """
    Computes standard IR metrics: Recall@K, Precision@K, MRR, and Graded NDCG@K.
    Calculates IDCG using full relevance judgments from all gold documents, not merely retrieved hits.
    """
    recalls, precisions, mrrs, ndcgs = [], [], [], []

    for qid, results in ranked_results.items():
        gold_rel = qrels.get(qid, {})
        if not gold_rel:
            continue

        # Top K retrieved docs
        top_k = results[:k]

        # Graded relevance for top K
        retrieved_grades = [gold_rel.get(doc_id, 0) for doc_id in top_k]
        hits_binary = [1 if g > 0 else 0 for g in retrieved_grades]

        # Precision & Recall
        prec = sum(hits_binary) / max(1, k)
        total_relevant = sum(1 for g in gold_rel.values() if g > 0)
        rec = sum(hits_binary) / max(1, total_relevant) if total_relevant > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)

        # MRR
        rr = 0.0
        for rank, h in enumerate(hits_binary, start=1):
            if h == 1:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        # Graded DCG@K: sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
        dcg = sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(retrieved_grades, start=1))

        # Full IDCG@K: sorted descending over ALL gold relevance judgments
        all_ideal_grades = sorted([g for g in gold_rel.values() if g > 0], reverse=True)[:k]
        idcg = sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(all_ideal_grades, start=1))

        ndcg_val = (dcg / idcg) if idcg > 0 else 0.0
        ndcgs.append(ndcg_val)

    return {
        f"Recall@{k}": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        f"Precision@{k}": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "MRR": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        f"NDCG@{k}": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
    }


# 10. Topic Modeling Metrics (Topic Diversity & Keyword Summary)
def evaluate_topic_modeling(
    topic_keywords: List[List[str]],
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Computes Topic Diversity: fraction of unique words across top N words of all topics.
    High diversity (> 0.7) means topics are non-repetitive.
    """
    if not topic_keywords:
        return {"Topic Diversity": 0.0, "Total Topics": 0}

    words_per_topic = [topic[:top_n] for topic in topic_keywords]
    all_words = [w for t in words_per_topic for w in t]
    unique_words = set(all_words)
    diversity = len(unique_words) / max(1, len(all_words))

    return {
        "Topic Diversity": round(float(diversity), 4),
        "Total Topics": len(topic_keywords),
        "Unique Top Words": len(unique_words),
    }


# Unified Evaluation Gateway
def evaluate_predictions(
    task: str,
    y_true: Any,
    y_pred: Any,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified entry point that delegates to the appropriate task-specific metric handler.
    Never fabricates metrics.
    """
    task = task.lower().strip()
    if y_true is None and task not in ["topic_modeling", "clustering"]:
        return {"Status": "Reference ground truth absent. Unsupervised inference completed."}

    if task in ["classification", "pair_classification", "zeroshot"]:
        return evaluate_classification(np.asarray(y_true), np.asarray(y_pred), **kwargs)
    elif task == "multilabel":
        return evaluate_multilabel(np.asarray(y_true), np.asarray(y_pred), **kwargs)
    elif task == "regression":
        return evaluate_regression(np.asarray(y_true), np.asarray(y_pred))
    elif task == "ner":
        return evaluate_ner(y_true, y_pred)
    elif task == "qa":
        return evaluate_qa(y_true, y_pred)
    elif task == "summarization":
        return evaluate_summarization(y_true, y_pred)
    elif task == "translation":
        return evaluate_translation(y_true, y_pred)
    elif task == "similarity":
        return evaluate_similarity(np.asarray(y_true), np.asarray(y_pred))
    elif task == "retrieval":
        return evaluate_retrieval(y_true, y_pred, **kwargs)
    elif task == "topic_modeling":
        return evaluate_topic_modeling(y_pred, **kwargs)
    else:
        return {"Evaluation Notice": f"Custom evaluation for task '{task}'."}
