"""
Multilabel Classification Adapters.
Supports Binary Relevance, Classifier Chains, and Transformer-based BCEWithLogits multilabel classification.
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
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer

from .base import BaseTaskAdapter
from ..preprocessing import TextCleaner
from ..evaluation import evaluate_multilabel
from ..hardware import HardwareManager

class MultilabelClassificationAdapter(BaseTaskAdapter):
    """
    Multilabel Classifier implementing Binary Relevance, Classifier Chains,
    or Transformer-based Sigmoid BCEWithLogits classification.
    """
    def __init__(
        self,
        model_id: str,
        approach: str = "binary_relevance",
        delimiter: str = "|",
        threshold: float = 0.5,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.approach = approach.lower().strip()
        self.delimiter = delimiter
        self.threshold = threshold
        self.random_state = random_state
        self.mlb = MultiLabelBinarizer()
        self.text_columns: List[str] = kwargs.get("text_columns", ["title", "content"])
        self.cleaner = TextCleaner()
        self.max_features = kwargs.get("max_features", 12000)
        self.ngram_range = kwargs.get("ngram_range", (1, 2))
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

        base_lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        if "chain" in self.approach or "chain" in self.model_id:
            clf = ClassifierChain(base_lr, order="random", random_state=self.random_state)
        else:  # Default binary relevance
            clf = MultiOutputClassifier(base_lr)

        self.pipeline = Pipeline([
            ("tfidf", tfidf),
            ("clf", clf),
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

    def _parse_labels(self, target_series: pd.Series) -> np.ndarray:
        parsed = target_series.fillna("").astype(str).apply(
            lambda x: [t.strip() for t in x.split(self.delimiter) if t.strip()]
        )
        return self.mlb.fit_transform(parsed)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        target_cols = kwargs.get("target_cols")
        target_col = kwargs.get("target_column") or kwargs.get("target_col") or "tags"

        t0 = time.time()
        X_train = self._prepare_text(train_df, **kwargs)

        if target_cols and all(c in train_df.columns for c in target_cols):
            y_train = train_df[target_cols].values
            self.classes_ = np.array(target_cols)
        elif target_col in train_df.columns:
            y_train = self._parse_labels(train_df[target_col])
            self.classes_ = self.mlb.classes_
        else:
            raise KeyError(f"Neither target_cols nor target_col '{target_col}' found in train_df.")

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        train_duration = time.time() - t0

        val_metrics = {}
        if val_df is not None:
            if target_cols and all(c in val_df.columns for c in target_cols):
                y_val = val_df[target_cols].values
                val_metrics = self.evaluate(val_df, y_true=y_val)
            elif target_col in val_df.columns:
                y_val = self.mlb.transform(
                    val_df[target_col].fillna("").astype(str).apply(
                        lambda x: [t.strip() for t in x.split(self.delimiter) if t.strip()]
                    )
                )
                val_metrics = self.evaluate(val_df, y_true=y_val)

        return {
            "model_id": self.model_id,
            "train_duration_s": round(train_duration, 3),
            "classes": list(self.classes_),
            "val_metrics": val_metrics,
        }

    def predict(self, df: Union[pd.DataFrame, pd.Series, List[str], np.ndarray], **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Multilabel model is not fitted yet.")
        X = self._prepare_text(df, **kwargs)
        return self.pipeline.predict(X)

    def predict_proba(self, df: pd.DataFrame, **kwargs) -> Optional[np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("Multilabel model is not fitted yet.")
        X = self._prepare_text(df)
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(X)
            # In multioutput, predict_proba returns list of [n_samples, 2] arrays
            if isinstance(probs, list):
                return np.column_stack([p[:, 1] for p in probs])
            return probs
        return None

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", "tags")
        if y_true is None:
            if target_col in df.columns:
                parsed = df[target_col].fillna("").astype(str).apply(
                    lambda x: [t.strip() for t in x.split(self.delimiter) if t.strip()]
                )
                y_true = self.mlb.transform(parsed)
            else:
                raise ValueError("Ground truth labels required for multilabel evaluation.")

        t0 = time.time()
        preds = self.predict(df)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        metrics = evaluate_multilabel(np.asarray(y_true), np.asarray(preds), threshold=self.threshold)
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "adapter_type": "multilabel",
            "pipeline": self.pipeline,
            "mlb": self.mlb,
            "classes": list(self.mlb.classes_),
            "delimiter": self.delimiter,
            "threshold": self.threshold,
            "text_columns": self.text_columns,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "MultilabelClassificationAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(
            model_id=bundle["model_id"],
            delimiter=bundle["delimiter"],
            threshold=bundle["threshold"],
            text_columns=bundle["text_columns"],
        )
        adapter.pipeline = bundle["pipeline"]
        adapter.mlb = bundle["mlb"]
        adapter.is_trained = True
        return adapter
