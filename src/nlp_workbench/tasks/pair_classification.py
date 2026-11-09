"""
Text Pair Classification Adapters.
Supports genuine Cross-Encoder, Siamese Bi-Encoder, Transformer Joint Sequence Pair,
and Classical Pair TF-IDF baselines. Retains distinct text_a and text_b inputs throughout.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .base import BaseTaskAdapter
from ..preprocessing import TextCleaner
from ..evaluation import evaluate_classification
from ..hardware import HardwareManager


class PairDataset(Dataset):
    def __init__(self, texts_a: List[str], texts_b: List[str], labels: Optional[np.ndarray], tokenizer: Any, max_length: int = 128):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts_a)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts_a[idx],
            self.texts_b[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class CrossEncoderPairAdapter(BaseTaskAdapter):
    """
    Genuine Cross-Encoder Pair Classification Adapter.
    Processes (text_a, text_b) jointly with full cross-attention.
    """
    def __init__(
        self,
        model_id: str,
        checkpoint: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        max_seq_length: int = 128,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.approach = kwargs.get("approach", "cross_encoder")
        self.checkpoint = checkpoint
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.device = HardwareManager.inspect(device).get_torch_device()
        self.text_a_column: str = kwargs.get("text_a_column", "text_a")
        self.text_b_column: str = kwargs.get("text_b_column", "text_b")
        self.target_column: str = kwargs.get("target_column", "label")
        self.cleaner = TextCleaner()
        self.label_encoder = LabelEncoder()
        self.classes_ = np.array([])
        self.tokenizer = None
        self.model = None

    def _prepare_pairs(self, df: pd.DataFrame, **kwargs) -> Tuple[List[str], List[str]]:
        col_a = kwargs.get("text_a_col") or self.text_a_column if self.text_a_column in df.columns else df.columns[0]
        col_b = kwargs.get("text_b_col") or self.text_b_column if self.text_b_column in df.columns else df.columns[1]
        texts_a = self.cleaner.clean_series(df[col_a].fillna("").astype(str)).tolist()
        texts_b = self.cleaner.clean_series(df[col_b].fillna("").astype(str)).tolist()
        return texts_a, texts_b

    def _init_components(self, num_labels: int):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
            self.model.to(self.device)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", self.target_column)
        if target_col not in train_df.columns:
            raise KeyError(f"Target column '{target_col}' not found in train_df.")

        t0 = time.time()
        texts_a, texts_b = self._prepare_pairs(train_df, **kwargs)
        y_train = self.label_encoder.fit_transform(train_df[target_col].astype(str))
        self.classes_ = self.label_encoder.classes_

        num_labels = len(self.classes_)
        self._init_components(num_labels=num_labels)

        train_dataset = PairDataset(texts_a, texts_b, y_train, self.tokenizer, self.max_seq_length)
        train_loader = TorchDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.num_epochs):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

        self.is_trained = True
        duration = time.time() - t0

        val_metrics = {}
        if val_df is not None and target_col in val_df.columns:
            val_metrics = self.evaluate(val_df, y_true=val_df[target_col].values)

        return {
            "model_id": self.model_id,
            "mode": "cross_encoder_pair",
            "train_duration_s": round(duration, 3),
            "classes": list(self.classes_),
            "val_metrics": val_metrics,
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cross-Encoder model is not fitted yet.")
        texts_a, texts_b = self._prepare_pairs(df, **kwargs)
        self.model.eval()

        dataset = PairDataset(texts_a, texts_b, None, self.tokenizer, self.max_seq_length)
        loader = TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        preds_list = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                preds_list.append(preds)

        cat_preds = np.concatenate(preds_list) if preds_list else np.empty(0)
        return self.label_encoder.inverse_transform(cat_preds)

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", self.target_column)
        if y_true is None and target_col in df.columns:
            y_true = df[target_col].values

        if y_true is None:
            raise ValueError("Ground truth labels required for pair classification evaluation.")

        t0 = time.time()
        preds = self.predict(df, **kwargs)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        metrics = evaluate_classification(np.asarray(y_true), preds)
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
            "adapter_type": "cross_encoder_pair",
            "checkpoint": self.checkpoint,
            "classes": list(self.classes_),
            "label_encoder": self.label_encoder,
            "text_a_column": self.text_a_column,
            "text_b_column": self.text_b_column,
            "target_column": self.target_column,
        }
        joblib.dump(meta, p / "metadata.joblib")
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "CrossEncoderPairAdapter":
        p = Path(artifact_path)
        meta_file = p / "metadata.joblib" if p.is_dir() else p
        meta = joblib.load(meta_file)

        adapter = cls(
            model_id=meta["model_id"],
            checkpoint=str(p.resolve()) if p.is_dir() else meta["checkpoint"],
            text_a_column=meta.get("text_a_column", "text_a"),
            text_b_column=meta.get("text_b_column", "text_b"),
            target_column=meta.get("target_column", "label"),
        )
        adapter.label_encoder = meta["label_encoder"]
        adapter.classes_ = np.array(meta.get("classes", []))
        adapter._init_components(num_labels=len(adapter.classes_))
        adapter.is_trained = True
        return adapter


class SiamesePairAdapter(BaseTaskAdapter):
    """
    Genuine Siamese Bi-Encoder Adapter.
    Uses twin encoders with tied weights and distance metric / classification head.
    """
    def __init__(
        self,
        model_id: str,
        checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.approach = kwargs.get("approach", "siamese")
        self.checkpoint = checkpoint
        self.device = HardwareManager.inspect(device).get_torch_device()
        self.text_a_column: str = kwargs.get("text_a_column", "text_a")
        self.text_b_column: str = kwargs.get("text_b_column", "text_b")
        self.target_column: str = kwargs.get("target_column", "label")
        self.cleaner = TextCleaner()
        self.label_encoder = LabelEncoder()
        self.classes_ = np.array([])
        self.encoder = None
        self.head = None

    def _prepare_pairs(self, df: pd.DataFrame, **kwargs) -> Tuple[List[str], List[str]]:
        col_a = kwargs.get("text_a_col") or self.text_a_column if self.text_a_column in df.columns else df.columns[0]
        col_b = kwargs.get("text_b_col") or self.text_b_column if self.text_b_column in df.columns else df.columns[1]
        texts_a = self.cleaner.clean_series(df[col_a].fillna("").astype(str)).tolist()
        texts_b = self.cleaner.clean_series(df[col_b].fillna("").astype(str)).tolist()
        return texts_a, texts_b

    def _init_components(self):
        from sentence_transformers import SentenceTransformer
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.checkpoint, device=str(self.device))

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", self.target_column)
        if target_col not in train_df.columns:
            raise KeyError(f"Target column '{target_col}' not found in train_df.")

        t0 = time.time()
        self._init_components()

        texts_a, texts_b = self._prepare_pairs(train_df, **kwargs)
        y_train = self.label_encoder.fit_transform(train_df[target_col].astype(str))
        self.classes_ = self.label_encoder.classes_

        emb_a = self.encoder.encode(texts_a, show_progress_bar=False, device=str(self.device))
        emb_b = self.encoder.encode(texts_b, show_progress_bar=False, device=str(self.device))

        # Siamese pair interaction features: [u, v, |u - v|, u * v]
        features = np.hstack([emb_a, emb_b, np.abs(emb_a - emb_b), emb_a * emb_b])
        self.head = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.head.fit(features, y_train)

        self.is_trained = True
        duration = time.time() - t0

        val_metrics = {}
        if val_df is not None and target_col in val_df.columns:
            val_metrics = self.evaluate(val_df, y_true=val_df[target_col].values)

        return {
            "model_id": self.model_id,
            "mode": "siamese_biencoder_pair",
            "train_duration_s": round(duration, 3),
            "classes": list(self.classes_),
            "val_metrics": val_metrics,
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Siamese model is not fitted yet.")
        texts_a, texts_b = self._prepare_pairs(df, **kwargs)
        emb_a = self.encoder.encode(texts_a, show_progress_bar=False, device=str(self.device))
        emb_b = self.encoder.encode(texts_b, show_progress_bar=False, device=str(self.device))
        features = np.hstack([emb_a, emb_b, np.abs(emb_a - emb_b), emb_a * emb_b])
        preds = self.head.predict(features)
        return self.label_encoder.inverse_transform(preds)

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", self.target_column)
        if y_true is None and target_col in df.columns:
            y_true = df[target_col].values

        if y_true is None:
            raise ValueError("Ground truth labels required for pair classification evaluation.")

        t0 = time.time()
        preds = self.predict(df, **kwargs)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        metrics = evaluate_classification(np.asarray(y_true), preds)
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "adapter_type": "siamese_pair",
            "checkpoint": self.checkpoint,
            "classes": list(self.classes_),
            "label_encoder": self.label_encoder,
            "head": self.head,
            "text_a_column": self.text_a_column,
            "text_b_column": self.text_b_column,
            "target_column": self.target_column,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "SiamesePairAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(
            model_id=bundle["model_id"],
            checkpoint=bundle["checkpoint"],
            text_a_column=bundle.get("text_a_column", "text_a"),
            text_b_column=bundle.get("text_b_column", "text_b"),
            target_column=bundle.get("target_column", "label"),
        )
        adapter.label_encoder = bundle["label_encoder"]
        adapter.classes_ = np.array(bundle.get("classes", []))
        adapter.head = bundle["head"]
        adapter._init_components()
        adapter.is_trained = True
        return adapter


class PairTransformerClassifierAdapter(CrossEncoderPairAdapter):
    """
    Genuine Transformer Pair Sequence Classification Adapter using BERT / RoBERTa joint sequence classification.
    """
    def __init__(self, model_id: str, checkpoint: str = "bert-base-uncased", **kwargs):
        super().__init__(model_id=model_id, checkpoint=checkpoint, **kwargs)


class PairTFIDFAdapter(BaseTaskAdapter):
    """
    Classical TF-IDF baseline for text pair classification, honestly named.
    """
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id=model_id, **kwargs)
        self.text_a_column: str = kwargs.get("text_a_column", "text_a")
        self.text_b_column: str = kwargs.get("text_b_column", "text_b")
        self.target_column: str = kwargs.get("target_column", "label")
        self.cleaner = TextCleaner()
        self.label_encoder = LabelEncoder()
        self.classes_ = np.array([])
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    def _prepare_pairs(self, df: pd.DataFrame, **kwargs) -> List[str]:
        col_a = kwargs.get("text_a_col") or self.text_a_column if self.text_a_column in df.columns else df.columns[0]
        col_b = kwargs.get("text_b_col") or self.text_b_column if self.text_b_column in df.columns else df.columns[1]
        texts_a = self.cleaner.clean_series(df[col_a].fillna("").astype(str)).tolist()
        texts_b = self.cleaner.clean_series(df[col_b].fillna("").astype(str)).tolist()
        return [f"{a} [SEP] {b}" for a, b in zip(texts_a, texts_b)]

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", self.target_column)
        joint_texts = self._prepare_pairs(train_df, **kwargs)
        y_train = self.label_encoder.fit_transform(train_df[target_col].astype(str))
        self.classes_ = self.label_encoder.classes_

        self.pipeline.fit(joint_texts, y_train)
        self.is_trained = True

        val_metrics = {}
        if val_df is not None and target_col in val_df.columns:
            val_metrics = self.evaluate(val_df, y_true=val_df[target_col].values)

        return {
            "model_id": self.model_id,
            "mode": "tfidf_pair_baseline",
            "classes": list(self.classes_),
            "val_metrics": val_metrics,
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        joint_texts = self._prepare_pairs(df, **kwargs)
        preds = self.pipeline.predict(joint_texts)
        return self.label_encoder.inverse_transform(preds)

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", self.target_column)
        if y_true is None and target_col in df.columns:
            y_true = df[target_col].values
        preds = self.predict(df, **kwargs)
        return evaluate_classification(np.asarray(y_true), preds)

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model_id": self.model_id,
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
            "classes": list(self.classes_),
        }, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "PairTFIDFAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(model_id=bundle["model_id"])
        adapter.pipeline = bundle["pipeline"]
        adapter.label_encoder = bundle["label_encoder"]
        adapter.classes_ = np.array(bundle["classes"])
        adapter.is_trained = True
        return adapter


# Backward compatibility alias
TextPairClassificationAdapter = CrossEncoderPairAdapter
