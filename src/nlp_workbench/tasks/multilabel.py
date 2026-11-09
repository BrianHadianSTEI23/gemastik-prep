"""
Multilabel Classification Adapters.
Supports Binary Relevance, Classifier Chains, and genuine Transformer-based BCEWithLogits
multilabel classification with validation-based threshold optimization.
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
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer

from .base import BaseTaskAdapter
from ..preprocessing import TextCleaner
from ..evaluation import evaluate_multilabel
from ..hardware import HardwareManager


class MultilabelClassificationAdapter(BaseTaskAdapter):
    """
    Classical Multilabel Classifier implementing Binary Relevance and Classifier Chains.
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
        self.classes_ = np.array([])
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

    def predict_proba(self, df: Union[pd.DataFrame, pd.Series, List[str], np.ndarray], **kwargs) -> Optional[np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("Multilabel model is not fitted yet.")
        X = self._prepare_text(df, **kwargs)
        if hasattr(self.pipeline, "predict_proba"):
            probs_list = self.pipeline.predict_proba(X)
            if isinstance(probs_list, list):
                # MultiOutputClassifier returns a list of (n_samples, 2) arrays
                probs = np.column_stack([p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p for p in probs_list])
            else:
                probs = np.asarray(probs_list)
            return probs
        return None

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_cols = kwargs.get("target_cols")
        target_col = kwargs.get("target_column") or kwargs.get("target_col") or "tags"

        if y_true is None:
            if target_cols and all(c in df.columns for c in target_cols):
                y_true = df[target_cols].values
            elif target_col in df.columns:
                parsed = df[target_col].fillna("").astype(str).apply(
                    lambda x: [t.strip() for t in x.split(self.delimiter) if t.strip()]
                )
                y_true = self.mlb.transform(parsed)
            else:
                raise ValueError("Ground truth labels required for multilabel evaluation.")

        t0 = time.time()
        preds = self.predict(df, **kwargs)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        metrics = evaluate_multilabel(np.asarray(y_true), np.asarray(preds), threshold=self.threshold)
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "adapter_type": "multilabel_classical",
            "pipeline": self.pipeline,
            "mlb": self.mlb,
            "classes": list(self.classes_),
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
        adapter.classes_ = np.array(bundle["classes"])
        adapter.is_trained = True
        return adapter


class TransformerMultilabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[np.ndarray], tokenizer: Any, max_length: int = 128):
        self.texts = texts
        self.labels = labels
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
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class TransformerMultilabelAdapter(BaseTaskAdapter):
    """
    Genuine Transformer Multilabel Sequence Classification Adapter.
    Uses AutoModelForSequenceClassification with problem_type="multi_label_classification"
    and BCEWithLogitsLoss. Implements validation-based threshold optimization.
    """
    def __init__(
        self,
        model_id: str,
        checkpoint: str = "distilbert-base-uncased",
        delimiter: str = "|",
        threshold: float = 0.5,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 3e-5,
        max_seq_length: int = 128,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.checkpoint = checkpoint
        self.delimiter = delimiter
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.device = HardwareManager.inspect(device).get_torch_device()
        self.text_columns: List[str] = kwargs.get("text_columns", ["title", "content"])
        self.cleaner = TextCleaner()
        self.mlb = MultiLabelBinarizer()
        self.classes_ = np.array([])
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

    def _parse_labels(self, target_series: pd.Series) -> np.ndarray:
        parsed = target_series.fillna("").astype(str).apply(
            lambda x: [t.strip() for t in x.split(self.delimiter) if t.strip()]
        )
        return self.mlb.fit_transform(parsed)

    def _init_components(self, num_labels: int):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint,
                num_labels=num_labels,
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True,
            )
            self.model.to(self.device)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        target_cols = kwargs.get("target_cols")
        target_col = kwargs.get("target_column") or kwargs.get("target_col") or "tags"

        t0 = time.time()
        train_texts = self._prepare_text(train_df, **kwargs)

        if target_cols and all(c in train_df.columns for c in target_cols):
            y_train = train_df[target_cols].values.astype(float)
            self.classes_ = np.array(target_cols)
        elif target_col in train_df.columns:
            y_train = self._parse_labels(train_df[target_col]).astype(float)
            self.classes_ = self.mlb.classes_
        else:
            raise KeyError(f"Target column '{target_col}' not found in train_df.")

        num_labels = y_train.shape[1]
        self._init_components(num_labels=num_labels)

        train_dataset = TransformerMultilabelDataset(train_texts, y_train, self.tokenizer, self.max_seq_length)
        train_loader = TorchDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

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
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

        self.is_trained = True

        # Validation-based threshold optimization (using validation data only)
        val_metrics = {}
        if val_df is not None:
            if target_cols and all(c in val_df.columns for c in target_cols):
                y_val = val_df[target_cols].values.astype(float)
            elif target_col in val_df.columns:
                parsed_val = val_df[target_col].fillna("").astype(str).apply(
                    lambda x: [t.strip() for t in x.split(self.delimiter) if t.strip()]
                )
                y_val = self.mlb.transform(parsed_val).astype(float)
            else:
                y_val = None

            if y_val is not None:
                val_probas = self.predict_proba(val_df, **kwargs)
                best_thresh = self.threshold
                best_f1 = -1.0
                for cand_thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                    cand_preds = (val_probas >= cand_thresh).astype(int)
                    m = evaluate_multilabel(y_val, cand_preds, threshold=cand_thresh)
                    if m["Macro-F1"] > best_f1:
                        best_f1 = m["Macro-F1"]
                        best_thresh = cand_thresh
                self.threshold = best_thresh
                val_metrics = evaluate_multilabel(y_val, (val_probas >= self.threshold).astype(int), threshold=self.threshold)

        duration = time.time() - t0

        return {
            "model_id": self.model_id,
            "mode": "transformer_multilabel",
            "train_duration_s": round(duration, 3),
            "optimal_threshold": self.threshold,
            "classes": list(self.classes_),
            "val_metrics": val_metrics,
        }

    def predict_proba(self, df: Union[pd.DataFrame, pd.Series, List[str], np.ndarray], **kwargs) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Transformer multilabel model is not fitted yet.")
        texts = self._prepare_text(df, **kwargs)
        self.model.eval()

        dataset = TransformerMultilabelDataset(texts, None, self.tokenizer, self.max_seq_length)
        loader = TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probas_list = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_probs = torch.sigmoid(outputs.logits).cpu().numpy()
                probas_list.append(batch_probs)

        return np.vstack(probas_list) if probas_list else np.empty((0, len(self.classes_)))

    def predict(self, df: Union[pd.DataFrame, pd.Series, List[str], np.ndarray], **kwargs) -> np.ndarray:
        probas = self.predict_proba(df, **kwargs)
        return (probas >= self.threshold).astype(int)

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_cols = kwargs.get("target_cols")
        target_col = kwargs.get("target_column") or kwargs.get("target_col") or "tags"

        if y_true is None:
            if target_cols and all(c in df.columns for c in target_cols):
                y_true = df[target_cols].values
            elif target_col in df.columns:
                parsed = df[target_col].fillna("").astype(str).apply(
                    lambda x: [t.strip() for t in x.split(self.delimiter) if t.strip()]
                )
                y_true = self.mlb.transform(parsed)
            else:
                raise ValueError("Ground truth labels required for multilabel evaluation.")

        t0 = time.time()
        preds = self.predict(df, **kwargs)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        metrics = evaluate_multilabel(np.asarray(y_true), np.asarray(preds), threshold=self.threshold)
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
            "adapter_type": "transformer_multilabel",
            "checkpoint": self.checkpoint,
            "classes": list(self.classes_),
            "delimiter": self.delimiter,
            "threshold": self.threshold,
            "max_seq_length": self.max_seq_length,
            "text_columns": self.text_columns,
            "mlb": self.mlb,
        }
        joblib.dump(meta, p / "metadata.joblib")
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "TransformerMultilabelAdapter":
        p = Path(artifact_path)
        meta_file = p / "metadata.joblib" if p.is_dir() else p
        meta = joblib.load(meta_file)

        adapter = cls(
            model_id=meta["model_id"],
            checkpoint=str(p.resolve()) if p.is_dir() else meta["checkpoint"],
            delimiter=meta.get("delimiter", "|"),
            threshold=meta.get("threshold", 0.5),
            max_seq_length=meta.get("max_seq_length", 128),
            text_columns=meta.get("text_columns", ["title", "content"]),
        )
        adapter.classes_ = np.array(meta.get("classes", []))
        adapter.mlb = meta.get("mlb", MultiLabelBinarizer())
        adapter._init_components(num_labels=len(adapter.classes_))
        adapter.is_trained = True
        return adapter
