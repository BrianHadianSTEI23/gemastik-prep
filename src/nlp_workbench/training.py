"""
Training, Cross-Validation, and Model Selection module.
Guarantees fold-local feature fitting and test-set isolation during model selection.
"""

import time
import copy
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from .tasks.base import BaseTaskAdapter
from .splitting import DatasetSplitter
from .registry import registry
from .evaluation import is_higher_better, get_task_evaluation_spec


class BestModelSelection(str):
    """Result object that is both a string model_id and iterable tuple (model_id, score)."""
    def __new__(cls, model_id: str, score: float = 0.0):
        obj = str.__new__(cls, model_id)
        obj.score = score
        obj.model_id = model_id
        return obj

    def __iter__(self):
        return iter((self.model_id, self.score))


def run_cross_validation(
    adapter: Optional[Union[BaseTaskAdapter, Callable[[], BaseTaskAdapter]]] = None,
    train_df: Optional[pd.DataFrame] = None,
    df: Optional[pd.DataFrame] = None,
    task: str = "classification",
    target_column: Optional[str] = None,
    target_col: Optional[str] = None,
    text_columns: Optional[List[str]] = None,
    cv_folds: Optional[int] = None,
    n_splits: Optional[int] = None,
    random_state: int = 42,
    group_column: Optional[str] = None,
    adapter_factory: Optional[Callable[[], BaseTaskAdapter]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Executes leak-free cross-validation.
    Each fold instantiates an independent model adapter and fits all feature encoders
    exclusively on the fold's training slice.
    """
    data = train_df if train_df is not None else df
    if data is None:
        raise ValueError("DataFrame (train_df or df) must be provided.")

    target = target_column or target_col or "category"
    if target not in data.columns and "target" in data.columns:
        target = "target"

    folds_count = n_splits or cv_folds or 5

    # Check task cross-validation support
    try:
        spec = get_task_evaluation_spec(task)
        if not spec.supports_cv:
            raise ValueError(
                f"Task '{task}' does not support cross-validation (supports_cv=False). "
                f"Cross-validation is only valid for tasks with repeatable supervised splits (e.g. classification, regression, ner)."
            )
        primary_metric_key = spec.primary_metric
    except KeyError:
        primary_metric_key = "Macro-F1" if task == "classification" else ("MAE" if task == "regression" else "Score")

    # Time-based ordering if configured
    time_column = kwargs.get("time_column") or kwargs.get("time_col")
    split_strategy = kwargs.get("split_strategy") or kwargs.get("strategy")
    if time_column:
        if time_column not in data.columns:
            raise ValueError(f"Configured time_column '{time_column}' not found in cross-validation dataset.")
        data = data.sort_values(by=time_column).reset_index(drop=True)

    # Determine factory
    effective_adapter = adapter or adapter_factory
    if effective_adapter is None:
        raise ValueError("Either 'adapter' or 'adapter_factory' must be provided.")

    if callable(effective_adapter) and not isinstance(effective_adapter, BaseTaskAdapter):
        factory = effective_adapter
        base_model_id = "custom_adapter"
    else:
        base_model_id = effective_adapter.model_id
        # Extract attributes from effective_adapter to preserve constructor kwargs across folds
        adapter_kwargs = {}
        for attr in [
            "approach", "checkpoint", "dense_checkpoint", "reranker_checkpoint",
            "source_lang", "target_lang", "model_type", "n_topics", "random_state",
            "text_columns", "target_column", "context_column", "question_column",
            "answer_column", "text_a_column", "text_b_column", "candidate_labels",
            "epochs", "num_epochs", "batch_size", "lr", "learning_rate", "max_seq_length",
            "device", "threshold", "cleaner"
        ]:
            if hasattr(effective_adapter, attr):
                adapter_kwargs[attr] = getattr(effective_adapter, attr)
        adapter_kwargs.update(kwargs)
        factory = lambda: registry.create_adapter(base_model_id, **adapter_kwargs)

    cv = DatasetSplitter.get_cv_splitter(
        task=task,
        n_splits=folds_count,
        random_state=random_state,
        group_column=group_column,
        time_column=time_column,
        strategy=split_strategy,
    )

    y_series = data[target] if target in data.columns else None
    fold_scores = []
    fold_results = []
    fold_adapters = []
    oof_predictions = [None] * len(data)

    split_generator = cv.split(data, y=y_series, groups=data[group_column] if group_column else None)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(split_generator):
        fold_train = data.iloc[train_idx].copy().reset_index(drop=True)
        fold_val = data.iloc[val_idx].copy().reset_index(drop=True)

        # Fresh adapter instance for this fold
        fold_adapter = factory()
        if text_columns:
            fold_adapter.text_columns = text_columns

        fold_adapter.fit(fold_train, val_df=None, target_column=target, target_col=target)

        # Predict on validation fold
        preds = fold_adapter.predict(fold_val)
        for i, val_pos in enumerate(val_idx):
            oof_predictions[val_pos] = preds[i]

        val_metrics = fold_adapter.evaluate(fold_val, target_column=target, target_col=target)
        fold_adapters.append(fold_adapter)

        # Extract primary metric
        if primary_metric_key in val_metrics:
            score = val_metrics[primary_metric_key]
        elif task == "classification":
            score = val_metrics.get("Macro-F1", val_metrics.get("Accuracy", 0.0))
        elif task == "regression":
            score = val_metrics.get("MAE", 0.0)
        elif task == "multilabel":
            score = val_metrics.get("Micro-F1", val_metrics.get("Macro-F1", 0.0))
        else:
            score = list(val_metrics.values())[0] if val_metrics else 0.0

        fold_scores.append(score)
        fold_results.append({
            "fold": fold + 1,
            "score": score,
            "metrics": val_metrics,
        })

    total_cv_duration = time.time() - t0
    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    return {
        "model_id": base_model_id,
        "cv_scores": fold_scores,
        "mean_cv_score": mean_score,
        "cv_mean": mean_score,
        "std_cv_score": std_score,
        "cv_std": std_score,
        "fold_results": fold_results,
        "fold_adapters": fold_adapters,
        "primary_metric": primary_metric_key,
        "oof_predictions": np.array(oof_predictions),
        "cv_duration_s": round(total_cv_duration, 2),
    }


def select_best_model(
    leaderboard: Union[pd.DataFrame, List[Dict[str, Any]]],
    task: Optional[str] = None,
    primary_metric: Optional[str] = None,
    higher_is_better: Optional[bool] = None,
    user_defined_objective: bool = False,
) -> BestModelSelection:
    """
    Selects winning model strictly based on validation scores.
    The held-out test set is NEVER used for model selection.
    Automatically respects metric direction (e.g. minimizing MAE/RMSE/loss, maximizing F1/accuracy),
    enforces fair comparison across candidates on the same metric, and rejects incompatible comparisons.
    Refuses auto-selection for tasks without a defensible scalar objective unless user_defined_objective=True.
    """
    if task:
        spec = get_task_evaluation_spec(task)
        if not spec.supports_auto_selection and not user_defined_objective:
            raise ValueError(
                f"Task '{task}' does not support automatic winner selection without an explicit user-defined objective preference. "
                f"Evaluation results must be reviewed qualitatively."
            )

    # Reject non-defensible quality metrics
    if primary_metric in ["Average Generated Words", "Avg Output Words (descriptive)", "Word Count"]:
        raise ValueError(
            f"Metric '{primary_metric}' cannot be used as an automatic quality metric for model selection. "
            f"Word count reflects verbosity, not generation quality."
        )

    if isinstance(leaderboard, list):
        if not leaderboard:
            raise ValueError("Leaderboard is empty; cannot select best model.")

        # Check candidate reported primary metrics for comparability
        reported_metrics = [e.get("primary_metric") for e in leaderboard if e.get("primary_metric")]
        unique_reported = sorted(list(set(reported_metrics)))
        if len(unique_reported) > 1 and primary_metric is None:
            raise ValueError(
                f"Cannot compare candidates with incompatible primary metrics: {unique_reported}. "
                f"All candidates within one leaderboard must be evaluated on the same primary metric."
            )

        # Resolve primary metric name
        if primary_metric is not None:
            metric_key = primary_metric
        elif unique_reported:
            metric_key = unique_reported[0]
        elif task:
            metric_key = get_task_evaluation_spec(task).primary_metric
        else:
            metric_key = leaderboard[0].get("primary_metric")
            if not metric_key:
                raise ValueError("Cannot determine primary evaluation metric for model selection.")

        # Resolve higher_is_better direction from metric metadata
        if higher_is_better is not None:
            hib = higher_is_better
        elif task:
            hib = get_task_evaluation_spec(task).higher_is_better
        else:
            hib = is_higher_better(metric_key)

        # Extract score for each candidate strictly without fake zeroes
        scored_candidates = []
        for e in leaderboard:
            if metric_key in e:
                score = e[metric_key]
            elif "val_metrics" in e and metric_key in e["val_metrics"]:
                score = e["val_metrics"][metric_key]
            elif "mean_cv_score" in e and (e.get("primary_metric") == metric_key or not e.get("primary_metric")):
                score = e["mean_cv_score"]
            elif "primary_score" in e and (e.get("primary_metric") == metric_key or not e.get("primary_metric")):
                score = e["primary_score"]
            elif "Validation Score" in e and (e.get("Primary Metric") == metric_key or not e.get("Primary Metric")):
                score = e["Validation Score"]
            else:
                raise ValueError(
                    f"Candidate '{e.get('model_id')}' is missing evaluation metric '{metric_key}'. "
                    f"Missing metrics cannot be silently converted to fake scores."
                )

            if score is None or (isinstance(score, float) and np.isnan(score)):
                raise ValueError(
                    f"Candidate '{e.get('model_id')}' has null/NaN for evaluation metric '{metric_key}'."
                )
            scored_candidates.append((e.get("model_id", "best_model"), float(score), e))

        sorted_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=hib)
        best_id, best_score, _ = sorted_candidates[0]
        return BestModelSelection(best_id, best_score)

    elif isinstance(leaderboard, pd.DataFrame):
        leaderboard_df = leaderboard
        if leaderboard_df.empty:
            raise ValueError("Leaderboard is empty; cannot select best model.")

        # Check candidate reported primary metrics in DataFrame
        metric_col_name = (
            "Primary Metric" if "Primary Metric" in leaderboard_df.columns
            else ("primary_metric" if "primary_metric" in leaderboard_df.columns else None)
        )
        if metric_col_name:
            unique_metrics = leaderboard_df[metric_col_name].dropna().unique()
            if len(unique_metrics) > 1 and primary_metric is None:
                raise ValueError(
                    f"Cannot compare candidates with incompatible primary metrics: {list(unique_metrics)}. "
                    f"All candidates within one leaderboard must be evaluated on the same primary metric."
                )

        if primary_metric is not None:
            metric_name = primary_metric
        elif metric_col_name and len(leaderboard_df[metric_col_name].dropna().unique()) == 1:
            metric_name = leaderboard_df[metric_col_name].dropna().unique()[0]
        elif task:
            metric_name = get_task_evaluation_spec(task).primary_metric
        else:
            metric_name = "Validation Score"

        # Locate score column in DataFrame
        if metric_name in leaderboard_df.columns:
            score_col = metric_name
        elif "Validation Score" in leaderboard_df.columns:
            score_col = "Validation Score"
        elif "CV Mean Score" in leaderboard_df.columns:
            score_col = "CV Mean Score"
        elif "mean_cv_score" in leaderboard_df.columns:
            score_col = "mean_cv_score"
        else:
            numeric_cols = [
                c for c in leaderboard_df.columns
                if "score" in c.lower() or "val" in c.lower() or "cv" in c.lower()
            ]
            if numeric_cols:
                score_col = numeric_cols[0]
            else:
                raise ValueError(f"Leaderboard DataFrame missing score column for '{metric_name}'.")

        if leaderboard_df[score_col].isna().any():
            raise ValueError(
                f"Leaderboard contains null/NaN values in score column '{score_col}'. Cannot silently evaluate missing metrics."
            )

        if higher_is_better is not None:
            hib = higher_is_better
        elif task:
            hib = get_task_evaluation_spec(task).higher_is_better
        else:
            hib = is_higher_better(metric_name)

        sorted_df = leaderboard_df.sort_values(by=score_col, ascending=(not hib))
        id_col = (
            "Model ID" if "Model ID" in sorted_df.columns
            else ("model_id" if "model_id" in sorted_df.columns else sorted_df.columns[0])
        )
        best_id = str(sorted_df[id_col].iloc[0])
        best_score = float(sorted_df[score_col].iloc[0])
        return BestModelSelection(best_id, best_score)
    else:
        raise TypeError(f"Unsupported leaderboard type: {type(leaderboard)}")
