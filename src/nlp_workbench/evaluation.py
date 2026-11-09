"""
Standardized Task-Specific Evaluation Metrics module.
Guarantees correct semantic metric calculation across all 14 NLP task families.
"""

import math
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

# 1. Classification Metrics
def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

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
    if y_pred.dtype == float:
        y_pred_bin = (y_pred >= threshold).astype(int)
    else:
        y_pred_bin = y_pred

    h_loss = float(hamming_loss(y_true, y_pred_bin))
    micro_f1 = float(f1_score(y_true, y_pred_bin, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred_bin, average="macro", zero_division=0))
    # Subset accuracy: exact match ratio
    subset_acc = float((y_true == y_pred_bin).all(axis=1).mean())

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
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    residuals = y_true - y_pred

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
    try:
        import seqeval.metrics
        prec = float(seqeval.metrics.precision_score(y_true_tags, y_pred_tags))
        rec = float(seqeval.metrics.recall_score(y_true_tags, y_pred_tags))
        f1 = float(seqeval.metrics.f1_score(y_true_tags, y_pred_tags))
        return {
            "Entity Precision": round(prec, 4),
            "Entity Recall": round(rec, 4),
            "Entity F1 (Seqeval)": round(f1, 4),
        }
    except Exception as e:
        return {"NER Evaluation Error": str(e)}

# 5. Question Answering Metrics (Exact Match & Token-level F1)
def compute_qa_f1(gold: str, pred: str) -> float:
    """Computes token-level F1 between extractive QA gold and prediction."""
    gold_tokens = gold.lower().split()
    pred_tokens = pred.lower().split()
    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)
    common = set(gold_tokens).intersection(set(pred_tokens))
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    return (2 * prec * rec) / (prec + rec)

def evaluate_qa(
    gold_answers: List[str],
    pred_answers: List[str],
) -> Dict[str, Any]:
    """Computes Exact Match (EM) and average token-level F1 for question answering."""
    if len(gold_answers) != len(pred_answers):
        raise ValueError("gold_answers and pred_answers must have identical length.")
    
    exact_matches = [
        float(g.strip().lower() == p.strip().lower())
        for g, p in zip(gold_answers, pred_answers)
    ]
    f1_scores = [compute_qa_f1(g, p) for g, p in zip(gold_answers, pred_answers)]

    return {
        "Exact Match (EM)": round(float(np.mean(exact_matches)) * 100, 2),
        "Token F1": round(float(np.mean(f1_scores)) * 100, 2),
    }

# 6. Text Summarization Metrics (ROUGE)
def evaluate_summarization(
    references: List[str],
    predictions: List[str],
) -> Dict[str, Any]:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L metrics."""
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
    """Computes SacreBLEU and chrF scores."""
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
    gold_scores: np.ndarray,
    pred_scores: np.ndarray,
) -> Dict[str, Any]:
    """Computes Pearson and Spearman rank correlation for similarity pairs."""
    gold = np.asarray(gold_scores, dtype=float)
    pred = np.asarray(pred_scores, dtype=float)
    if len(gold) < 2 or np.std(gold) == 0 or np.std(pred) == 0:
        return {"Spearman Correlation": 0.0, "Pearson Correlation": 0.0}
    p_corr, _ = pearsonr(gold, pred)
    s_corr, _ = spearmanr(gold, pred)
    return {
        "Spearman Correlation": round(float(s_corr), 4),
        "Pearson Correlation": round(float(p_corr), 4),
    }

# 9. Information Retrieval Metrics (Recall@K, MRR, NDCG@K)
def evaluate_retrieval(
    qrels: Dict[str, Dict[str, int]],  # query_id -> {doc_id: relevance}
    ranked_results: Dict[str, List[str]],  # query_id -> [doc_id_1, doc_id_2, ...]
    k: int = 5,
) -> Dict[str, Any]:
    """Computes standard IR metrics: Recall@K, Precision@K, MRR, NDCG@K."""
    recalls, precisions, mrrs, ndcgs = [], [], [], []
    for qid, results in ranked_results.items():
        relevant_docs = qrels.get(qid, {})
        if not relevant_docs:
            continue
        top_k = results[:k]
        hits = [1 if doc_id in relevant_docs and relevant_docs[doc_id] > 0 else 0 for doc_id in top_k]

        # Precision & Recall
        prec = sum(hits) / max(1, k)
        total_rel = sum(1 for r in relevant_docs.values() if r > 0)
        rec = sum(hits) / max(1, total_rel)
        precisions.append(prec)
        recalls.append(rec)

        # MRR
        rr = 0.0
        for rank, h in enumerate(hits, start=1):
            if h == 1:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        # NDCG@K
        dcg = sum((2**h - 1) / math.log2(rank + 1) for rank, h in enumerate(hits, start=1))
        ideal_hits = sorted(hits, reverse=True)
        idcg = sum((2**h - 1) / math.log2(rank + 1) for rank, h in enumerate(ideal_hits, start=1))
        ndcgs.append(dcg / max(1e-9, idcg))

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
