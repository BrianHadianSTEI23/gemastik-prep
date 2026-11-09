"""
Text Regression Adapters.
Supports TF-IDF Ridge, Sentence Embedding Regression, and genuine Transformer Sequence Regression.
"""

import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
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
    Classical and Sentence-Embedding Text Regression model predicting continuous values from text.
    """
    def __init__(
        self,
        model_id: str,
        approach: str = "tfidf_ridge",
        alpha: float = 1.0,
        random_state: int = 42,
        embedding_checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.approach = approach.lower().strip()
        self.alpha = alpha
        self.random_state = random_state
        self.embedding_checkpoint = embedding_checkpoint
        self.text_columns: List[str] = kwargs.get("text_columns", ["title", "content"])
        self.cleaner = kwargs.get("cleaner") or TextCleaner()
        self.ngram_range = kwargs.get("ngram_range", (1, 2))
        self.max_features = kwargs.get("max_features", 12000)
        self.pipeline: Optional[Pipeline] = None
        self.embedder = None
        self.regressor = None
        self._build_pipeline()

    def _build_pipeline(self):
        if "sentence" in self.approach or "svr" in self.approach:
            self.regressor = SVR(C=1.0)
        else:
            tfidf = TfidfVectorizer(
                ngram_range=self.ngram_range,
                sublinear_tf=True,
                min_df=1,
                max_features=self.max_features,
                lowercase=True,
            )
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
        X_train_text = self._prepare_text(train_df, **kwargs)
        y_train = train_df[target_col].astype(float).values

        if "sentence" in self.approach or "svr" in self.approach:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_checkpoint)
            emb = self.embedder.encode(X_train_text.tolist(), show_progress_bar=False)
            self.regressor.fit(emb, y_train)
        else:
            self.pipeline.fit(X_train_text, y_train)

        self.is_trained = True
        train_duration = time.time() - t0

        val_metrics = {}
        if val_df is not None and target_col in val_df.columns:
            val_metrics = self.evaluate(val_df, y_true=val_df[target_col].values)

        return {
            "model_id": self.model_id,
            "mode": f"classical_regression_{self.approach}",
            "train_duration_s": round(train_duration, 3),
            "val_metrics": val_metrics,
        }

    def predict(self, df: Union[pd.DataFrame, pd.Series, List[str], np.ndarray], **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Regression model is not fitted yet.")
        X_text = self._prepare_text(df, **kwargs)

        if "sentence" in self.approach or "svr" in self.approach:
            emb = self.embedder.encode(X_text.tolist(), show_progress_bar=False)
            return self.regressor.predict(emb)
        else:
            return self.pipeline.predict(X_text)

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column") or kwargs.get("target_col") or "engagement_score"
        if y_true is None:
            if target_col in df.columns:
                y_true = df[target_col].values
            elif "target" in df.columns:
                y_true = df["target"].values
            else:
                raise ValueError("Ground truth continuous values required for regression evaluation.")

        t0 = time.time()
        preds = self.predict(df, **kwargs)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        metrics = evaluate_regression(np.asarray(y_true, dtype=float), np.asarray(preds, dtype=float))
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "adapter_type": "text_regression",
            "approach": self.approach,
            "pipeline": self.pipeline,
            "regressor": self.regressor,
            "embedding_checkpoint": self.embedding_checkpoint,
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
            approach=bundle["approach"],
            alpha=bundle.get("alpha", 1.0),
            embedding_checkpoint=bundle.get("embedding_checkpoint", "sentence-transformers/all-MiniLM-L6-v2"),
            text_columns=bundle["text_columns"],
        )
        adapter.pipeline = bundle["pipeline"]
        adapter.regressor = bundle.get("regressor")
        if adapter.regressor is not None:
            from sentence_transformers import SentenceTransformer
            adapter.embedder = SentenceTransformer(adapter.embedding_checkpoint)
        adapter.is_trained = True
        return adapter


class TransformerRegressionDataset(Dataset):
    def __init__(self, texts: List[str], targets: Optional[np.ndarray], tokenizer: Any, max_length: int = 128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item


class TransformerRegressionAdapter(BaseTaskAdapter):
    """
    Genuine Transformer Sequence Regression Adapter.
    Uses AutoModelForSequenceClassification with num_labels=1 and problem_type="regression".
    """
    def __init__(
        self,
        model_id: str,
        checkpoint: str = "distilbert-base-uncased",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 3e-5,
        max_seq_length: int = 128,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.checkpoint = checkpoint
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.device = HardwareManager.inspect(device).get_torch_device()
        self.text_columns: List[str] = kwargs.get("text_columns", ["title", "content"])
        self.cleaner = kwargs.get("cleaner") or TextCleaner()
        self.tokenizer = None
        self.model = None

    def _prepare_text(self, df: Union[pd.DataFrame, pd.Series, List[str], np.ndarray], **kwargs) -> List[str]:
        if isinstance(df, (list, tuple, np.ndarray, pd.Series)):
            return self.cleaner.clean_series(pd.Series(df).fillna("").astype(str)).tolist()

        text_col = kwargs.get("text_column") or kwargs.get("text_col")
        if text_col and text_col in df.columns:
            return self.cleaner.clean_series(df[text_col].fillna("").astype(str)).tolist()

        valid_cols = [c for c in self.text_columns if c in df.columns]
        if not valid_cols:
            if "cleaned_text" in df.columns:
                return df["cleaned_text"].tolist()
            if "text" in df.columns:
                return self.cleaner.clean_series(df["text"].fillna("").astype(str)).tolist()
            raise KeyError(f"None of {self.text_columns} exist in DataFrame.")
        combined = df[valid_cols].fillna("").astype(str).agg(" [SEP] ".join, axis=1)
        return self.cleaner.clean_series(combined).tolist()

    def _init_components(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint,
                num_labels=1,
                problem_type="regression",
                ignore_mismatched_sizes=True,
            )
            self.model.to(self.device)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column") or kwargs.get("target_col") or "engagement_score"
        if target_col not in train_df.columns:
            if "target" in train_df.columns:
                target_col = "target"
            else:
                raise KeyError(f"Regression target '{target_col}' not found in train_df.")

        t0 = time.time()
        self._init_components()

        train_texts = self._prepare_text(train_df, **kwargs)
        y_train = train_df[target_col].astype(float).values

        train_dataset = TransformerRegressionDataset(train_texts, y_train, self.tokenizer, self.max_seq_length)
        train_loader = TorchDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.num_epochs):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device).unsqueeze(-1)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

        self.is_trained = True
        duration = time.time() - t0

        val_metrics = {}
        if val_df is not None and target_col in val_df.columns:
            val_metrics = self.evaluate(val_df, y_true=val_df[target_col].values)

        return {
            "model_id": self.model_id,
            "mode": "transformer_regression",
            "train_duration_s": round(duration, 3),
            "val_metrics": val_metrics,
        }

    def predict(self, df: Union[pd.DataFrame, pd.Series, List[str], np.ndarray], **kwargs) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Transformer regression model is not fitted yet.")
        texts = self._prepare_text(df, **kwargs)
        self.model.eval()

        dataset = TransformerRegressionDataset(texts, None, self.tokenizer, self.max_seq_length)
        loader = TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        preds_list = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_preds = outputs.logits.squeeze(-1).cpu().numpy()
                preds_list.append(batch_preds)

        return np.concatenate(preds_list) if preds_list else np.empty(0)

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column") or kwargs.get("target_col") or "engagement_score"
        if y_true is None:
            if target_col in df.columns:
                y_true = df[target_col].values
            elif "target" in df.columns:
                y_true = df["target"].values
            else:
                raise ValueError("Ground truth values required for regression evaluation.")

        t0 = time.time()
        preds = self.predict(df, **kwargs)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        metrics = evaluate_regression(np.asarray(y_true, dtype=float), np.asarray(preds, dtype=float))
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.mkdir(parents=True, exist_ok=True)
        if self.model is not None and self.tokenizer is not None:
            self.model.save_pretrained(p)
            self.tokenizer.save_pretrained(p)
        meta = {
            "model_id": self.model_id,
            "adapter_type": "transformer_regression",
            "checkpoint": self.checkpoint,
            "max_seq_length": self.max_seq_length,
            "text_columns": self.text_columns,
        }
        joblib.dump(meta, p / "metadata.joblib")
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "TransformerRegressionAdapter":
        p = Path(artifact_path)
        meta_file = p / "metadata.joblib" if p.is_dir() else p
        meta = joblib.load(meta_file)

        adapter = cls(
            model_id=meta["model_id"],
            checkpoint=str(p.resolve()) if p.is_dir() else meta["checkpoint"],
            max_seq_length=meta.get("max_seq_length", 128),
            text_columns=meta.get("text_columns", ["title", "content"]),
        )
        adapter._init_components()
        adapter.is_trained = True
        return adapter
