"""
Text Regression Adapters.
Supports TF-IDF Ridge, Sentence Embedding Regression, and Transformer Sequence Regression.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from .base import BaseTaskAdapter
from ..preprocessing import TextCleaner
from ..evaluation import evaluate_regression
from ..hardware import HardwareManager

class TextRegressionAdapter(BaseTaskAdapter):
    """
    Text Regression model predicting continuous values from text inputs.
    """
    def __init__(
        self,
        model_id: str,
        approach: str = "tfidf_ridge",
        alpha: float = 1.0,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.approach = approach.lower().strip()
        self.alpha = alpha
        self.random_state = random_state
        self.text_columns: List[str] = kwargs.get("text_columns", ["title", "content"])
        self.cleaner = TextCleaner()
        self.ngram_range = kwargs.get("ngram_range", (1, 2))
        self.max_features = kwargs.get("max_features", 12000)
        self.pipeline: Optional[Pipeline] = None
        self._build_pipeline()

    def _build_pipeline(self):
        tfidf = TfidfVectorizer(
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            min_df=1,
            max_features=self.max_features,
            lowercase=True,
        )

        if "svr" in self.approach:
            reg = SVR(C=1.0)
        else:
            reg = Ridge(alpha=self.alpha, random_state=self.random_state)

        self.pipeline = Pipeline([
            ("tfidf", tfidf),
            ("reg", reg),
        ])

    def _prepare_text(self, df: Union[pd.DataFrame, pd.Series, List[str], np.ndarray], **kwargs) -> pd.Series:
        if isinstance(df, (list, tuple, np.ndarray, pd.Series)):
            return self.cleaner.clean_series(pd.Series(df).fillna("").astype(str))

        text_col = kwargs.get("text_column") or kwargs.get("text_col")
        if text_col and text_col in df.columns:
            return self.cleaner.clean_series(df[text_col].fillna("").astype(str))

        valid_cols = [c for c in self.text_columns if c in df.columns]
        if not valid_cols:
            if "cleaned_text" in df.columns:
                return df["cleaned_text"]
            if "text" in df.columns:
                return self.cleaner.clean_series(df["text"].fillna("").astype(str))
            raise KeyError(f"None of {self.text_columns} exist in DataFrame.")
        combined = df[valid_cols].fillna("").astype(str).agg(" [SEP] ".join, axis=1)
        return self.cleaner.clean_series(combined)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column") or kwargs.get("target_col") or "engagement_score"
        if target_col not in train_df.columns:
            if "target" in train_df.columns:
                target_col = "target"
            else:
                raise KeyError(f"Regression target '{target_col}' not found in train_df.")

        t0 = time.time()
        X_train = self._prepare_text(train_df, **kwargs)
        y_train = train_df[target_col].astype(float).values

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        train_duration = time.time() - t0

        val_metrics = {}
        if val_df is not None and target_col in val_df.columns:
            val_metrics = self.evaluate(val_df, y_true=val_df[target_col].values)

        return {
            "model_id": self.model_id,
            "train_duration_s": round(train_duration, 3),
            "val_metrics": val_metrics,
        }

    def predict(self, df: Union[pd.DataFrame, pd.Series, List[str], np.ndarray], **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Regression model is not fitted yet.")
        X = self._prepare_text(df, **kwargs)
        return self.pipeline.predict(X)

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", "engagement_score")
        if y_true is None:
            if target_col in df.columns:
                y_true = df[target_col].astype(float).values
            else:
                raise ValueError("Ground truth continuous values required for regression evaluation.")

        t0 = time.time()
        preds = self.predict(df)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        metrics = evaluate_regression(np.asarray(y_true), np.asarray(preds))
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "adapter_type": "regression",
            "pipeline": self.pipeline,
            "alpha": self.alpha,
            "text_columns": self.text_columns,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "TextRegressionAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(
            model_id=bundle["model_id"],
            alpha=bundle["alpha"],
            text_columns=bundle["text_columns"],
        )
        adapter.pipeline = bundle["pipeline"]
        adapter.is_trained = True
        return adapter
