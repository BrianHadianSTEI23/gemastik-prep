"""
Text Pair Classification Adapter.
Supports Cross-Encoder, Siamese Bi-Encoder, and Transformer Joint Sequence Pair classification.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .base import BaseTaskAdapter
from ..preprocessing import TextCleaner
from ..evaluation import evaluate_classification
from ..hardware import HardwareManager

class TextPairClassificationAdapter(BaseTaskAdapter):
    """
    Text Pair Classification Adapter explicitly ingesting two separate text columns (text_a, text_b).
    """
    def __init__(
        self,
        model_id: str,
        approach: str = "cross_encoder",
        checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.approach = approach.lower().strip()
        self.checkpoint = checkpoint
        self.device = HardwareManager.inspect(device).get_torch_device()
        self.text_a_column: str = kwargs.get("text_a_column", "text_a")
        self.text_b_column: str = kwargs.get("text_b_column", "text_b")
        self.target_column: str = kwargs.get("target_column", "label")
        self.cleaner = TextCleaner()
        self.label_encoder = LabelEncoder()
        self.classifier = None
        self.bi_encoder = None
        self.cross_encoder = None
        self.is_trained = False

    def _prepare_pairs(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        col_a = self.text_a_column if self.text_a_column in df.columns else df.columns[0]
        col_b = self.text_b_column if self.text_b_column in df.columns else df.columns[1]
        texts_a = self.cleaner.clean_series(df[col_a].fillna("").astype(str)).tolist()
        texts_b = self.cleaner.clean_series(df[col_b].fillna("").astype(str)).tolist()
        return texts_a, texts_b

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", self.target_column)
        if target_col not in train_df.columns:
            raise KeyError(f"Target column '{target_col}' not found in train_df.")

        t0 = time.time()
        texts_a, texts_b = self._prepare_pairs(train_df)
        y_train = self.label_encoder.fit_transform(train_df[target_col].astype(str))

        if "siamese" in self.approach:
            from sentence_transformers import SentenceTransformer
            self.bi_encoder = SentenceTransformer(self.checkpoint, device=str(self.device))
            emb_a = self.bi_encoder.encode(texts_a, show_progress_bar=False, device=str(self.device))
            emb_b = self.bi_encoder.encode(texts_b, show_progress_bar=False, device=str(self.device))
            # Concatenate absolute difference and elementwise product
            pair_features = np.hstack([np.abs(emb_a - emb_b), emb_a * emb_b])
            self.classifier = LogisticRegression(max_iter=1000)
            self.classifier.fit(pair_features, y_train)

        else:  # Joint TF-IDF / Cross-Encoder representation
            from sklearn.feature_extraction.text import TfidfVectorizer
            joint_texts = [f"{a} [SEP] {b}" for a, b in zip(texts_a, texts_b)]
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000)
            X = self.vectorizer.fit_transform(joint_texts)
            self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
            self.classifier.fit(X, y_train)

        self.is_trained = True
        train_duration = time.time() - t0

        val_metrics = {}
        if val_df is not None and target_col in val_df.columns:
            val_metrics = self.evaluate(val_df, y_true=val_df[target_col].values)

        return {
            "model_id": self.model_id,
            "train_duration_s": round(train_duration, 3),
            "classes": list(self.label_encoder.classes_),
            "val_metrics": val_metrics,
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Pair classification model is not fitted yet.")
        texts_a, texts_b = self._prepare_pairs(df)

        if "siamese" in self.approach:
            emb_a = self.bi_encoder.encode(texts_a, show_progress_bar=False, device=str(self.device))
            emb_b = self.bi_encoder.encode(texts_b, show_progress_bar=False, device=str(self.device))
            pair_features = np.hstack([np.abs(emb_a - emb_b), emb_a * emb_b])
            preds = self.classifier.predict(pair_features)
        else:
            joint_texts = [f"{a} [SEP] {b}" for a, b in zip(texts_a, texts_b)]
            X = self.vectorizer.transform(joint_texts)
            preds = self.classifier.predict(X)

        return self.label_encoder.inverse_transform(preds)

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", self.target_column)
        if y_true is None and target_col in df.columns:
            y_true = df[target_col].values

        if y_true is None:
            raise ValueError("Ground truth labels required for pair classification evaluation.")

        t0 = time.time()
        preds = self.predict(df)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        y_true_enc = self.label_encoder.transform([str(y) for y in y_true])
        y_pred_enc = self.label_encoder.transform([str(p) for p in preds])

        metrics = evaluate_classification(y_true_enc, y_pred_enc)
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "approach": self.approach,
            "checkpoint": self.checkpoint,
            "classifier": self.classifier,
            "vectorizer": getattr(self, "vectorizer", None),
            "label_encoder": self.label_encoder,
            "text_a_column": self.text_a_column,
            "text_b_column": self.text_b_column,
            "target_column": self.target_column,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "TextPairClassificationAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(
            model_id=bundle["model_id"],
            approach=bundle["approach"],
            checkpoint=bundle["checkpoint"],
            text_a_column=bundle["text_a_column"],
            text_b_column=bundle["text_b_column"],
            target_column=bundle["target_column"],
        )
        adapter.classifier = bundle["classifier"]
        adapter.vectorizer = bundle.get("vectorizer")
        adapter.label_encoder = bundle["label_encoder"]
        adapter.is_trained = True
        return adapter
