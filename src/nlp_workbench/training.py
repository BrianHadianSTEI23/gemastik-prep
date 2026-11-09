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

    # Determine factory
    effective_adapter = adapter or adapter_factory
    if effective_adapter is None:
        raise ValueError("Either 'adapter' or 'adapter_factory' must be provided.")

    if callable(effective_adapter) and not isinstance(effective_adapter, BaseTaskAdapter):
        factory = effective_adapter
        base_model_id = "custom_adapter"
    else:
        base_model_id = effective_adapter.model_id
        # Define factory that recreates adapter with same kwargs
        factory = lambda: registry.create_adapter(effective_adapter.model_id)

    cv = DatasetSplitter.get_cv_splitter(
        task=task,
        n_splits=folds_count,
        random_state=random_state,
        group_column=group_column,
    )

    y_series = data[target] if target in data.columns else None
    fold_scores = []
    fold_results = []
    oof_predictions = [None] * len(data)
    primary_metric_key = "Macro-F1" if task == "classification" else ("MAE" if task == "regression" else "Score")

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

        # Extract primary metric
        if task == "classification":
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
        "primary_metric": primary_metric_key,
        "oof_predictions": np.array(oof_predictions),
        "cv_duration_s": round(total_cv_duration, 2),
    }


def select_best_model(
    leaderboard: Union[pd.DataFrame, List[Dict[str, Any]]],
    primary_metric: Optional[str] = None,
    higher_is_better: bool = True,
) -> BestModelSelection:
    """
    Selects winning model strictly based on validation/cross-validation scores.
    The held-out test set is NEVER used for model selection.
    """
    if isinstance(leaderboard, list):
        if not leaderboard:
            raise ValueError("Leaderboard is empty; cannot select best model.")
        metric_key = primary_metric or "mean_cv_score"
        sorted_board = sorted(
            leaderboard,
            key=lambda x: x.get(metric_key, x.get("cv_mean", 0.0)),
            reverse=higher_is_better,
        )
        best = sorted_board[0]
        best_id = str(best.get("model_id", "best_model"))
        best_score = float(best.get(metric_key, best.get("cv_mean", 0.0)))
        return BestModelSelection(best_id, best_score)

    elif isinstance(leaderboard, pd.DataFrame):
        leaderboard_df = leaderboard
        if leaderboard_df.empty:
            raise ValueError("Leaderboard is empty; cannot select best model.")

        metric_col = primary_metric or "CV Score (Mean)"
        if metric_col not in leaderboard_df.columns:
            numeric_cols = [c for c in leaderboard_df.columns if "CV" in c or "Val" in c or "Score" in c]
            metric_col = numeric_cols[0] if numeric_cols else leaderboard_df.columns[1]

        sorted_df = leaderboard_df.sort_values(by=metric_col, ascending=(not higher_is_better))
        id_col = "Model ID" if "Model ID" in sorted_df.columns else sorted_df.columns[0]
        best_id = str(sorted_df[id_col].iloc[0])
        best_score = float(sorted_df[metric_col].iloc[0])
        return BestModelSelection(best_id, best_score)
    else:
        raise TypeError(f"Unsupported leaderboard type: {type(leaderboard)}")
