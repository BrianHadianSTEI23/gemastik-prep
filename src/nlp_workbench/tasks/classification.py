"""
Classification Adapters for Classical ML and Transformer architectures.
Ensures learned preprocessing is contained strictly inside the training pipeline.
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
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

from .base import BaseTaskAdapter
from ..preprocessing import TextCleaner
from ..evaluation import evaluate_classification
from ..hardware import HardwareManager

class ClassicalClassificationAdapter(BaseTaskAdapter):
    """
    Classical Machine Learning Text Classifier.
    Integrates TF-IDF vectorization and estimator into a leak-free scikit-learn Pipeline.
    """
    def __init__(self, model_id: str, model_type: str = "logistic", class_weight: Optional[str] = "balanced", random_state: int = 42, **kwargs):
        super().__init__(model_id=model_id, **kwargs)
        self.model_type = model_type.lower()
        self.class_weight = class_weight
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.pipeline: Optional[Pipeline] = None
        self.text_columns: List[str] = kwargs.get("text_columns", ["title", "content"])
        self.ngram_range = kwargs.get("ngram_range", (1, 2))
        self.max_features = kwargs.get("max_features", 15000)
        self.cleaner = TextCleaner()
        self.classes_ = np.array([])
        self._build_pipeline()

    def _build_pipeline(self):
        tfidf = TfidfVectorizer(
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            min_df=1,
            max_features=self.max_features,
            strip_accents="unicode",
            lowercase=True,
        )

        if "nb" in self.model_id or "naive_bayes" in self.model_id:
            clf = ComplementNB(alpha=1.0)
        elif "svm" in self.model_id or "svc" in self.model_id:
            base_svc = LinearSVC(class_weight=self.class_weight, random_state=self.random_state, max_iter=3000)
            clf = CalibratedClassifierCV(estimator=base_svc)
        else:  # Logistic Regression
            clf = LogisticRegression(
                max_iter=1500,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver="lbfgs"
            )

        self.pipeline = Pipeline([
            ("tfidf", tfidf),
            ("clf", clf)
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
        target_col = kwargs.get("target_column") or kwargs.get("target_col") or "category"
        if target_col not in train_df.columns:
            if "target" in train_df.columns:
                target_col = "target"
            else:
                raise KeyError(f"Target column '{target_col}' not found in train_df.")

        t0 = time.time()
        X_train = self._prepare_text(train_df, **kwargs)
        y_train = self.label_encoder.fit_transform(train_df[target_col].astype(str))
        self.classes_ = self.label_encoder.classes_

        # Pipeline fits TF-IDF strictly on train fold
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        train_duration = time.time() - t0

        val_metrics = {}
        if val_df is not None and target_col in val_df.columns:
            val_metrics = self.evaluate(val_df, y_true=val_df[target_col].values)

        return {
            "model_id": self.model_id,
            "train_duration_s": round(train_duration, 3),
            "train_time_sec": round(train_duration, 3),
            "classes": list(self.label_encoder.classes_),
            "val_metrics": val_metrics,
        }

    def predict(self, df: Union[pd.DataFrame, pd.Series, List[str], np.ndarray], **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model is not fitted yet.")
        X = self._prepare_text(df, **kwargs)
        encoded_preds = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(encoded_preds)

    def predict_proba(self, df: Union[pd.DataFrame, pd.Series, List[str], np.ndarray], **kwargs) -> Optional[np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("Model is not fitted yet.")
        X = self._prepare_text(df, **kwargs)
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        return None

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", "category")
        if y_true is None:
            if target_col in df.columns:
                y_true = df[target_col].values
            else:
                raise ValueError("Ground truth labels required for classification evaluation.")

        t0 = time.time()
        preds = self.predict(df)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        y_true_encoded = self.label_encoder.transform([str(y) for y in y_true])
        y_pred_encoded = self.label_encoder.transform([str(p) for p in preds])

        metrics = evaluate_classification(y_true_encoded, y_pred_encoded)
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "adapter_type": "classical",
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
            "text_columns": self.text_columns,
            "classes": list(self.label_encoder.classes_),
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "ClassicalClassificationAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(model_id=bundle["model_id"], text_columns=bundle["text_columns"])
        adapter.pipeline = bundle["pipeline"]
        adapter.label_encoder = bundle["label_encoder"]
        adapter.is_trained = True
        return adapter


class TransformerClassificationAdapter(BaseTaskAdapter):
    """
    Transformer Text Classifier for BERT, RoBERTa, DistilBERT, and IndoBERT.
    Hardware-optimized for RTX 3050 (mixed precision FP16, dynamic padding).
    """
    def __init__(
        self,
        model_id: str,
        checkpoint: str = "distilbert-base-uncased",
        device: str = "auto",
        batch_size: int = 8,
        grad_accum_steps: int = 4,
        num_epochs: int = 2,
        learning_rate: float = 3e-5,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.checkpoint = checkpoint
        self.device_pref = device
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.label_encoder = LabelEncoder()
        self.text_columns: List[str] = kwargs.get("text_columns", ["title", "content"])
        self.cleaner = TextCleaner(lowercase=False)  # Preserve case for cased transformers
        self.model = None
        self.tokenizer = None
        self.device = HardwareManager.inspect(device).get_torch_device()

    def _prepare_text(self, df: pd.DataFrame) -> List[str]:
        valid_cols = [c for c in self.text_columns if c in df.columns]
        if not valid_cols:
            if "cleaned_text" in df.columns:
                return df["cleaned_text"].tolist()
            if "text" in df.columns:
                return df["text"].tolist()
            raise KeyError(f"None of {self.text_columns} exist in DataFrame.")
        combined = df[valid_cols].fillna("").astype(str).agg(" [SEP] ".join, axis=1)
        return self.cleaner.clean_series(combined).tolist()

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", "category")
        if target_col not in train_df.columns:
            raise KeyError(f"Target column '{target_col}' not found in train_df.")

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        t0 = time.time()
        X_train = self._prepare_text(train_df)
        y_train = self.label_encoder.fit_transform(train_df[target_col].astype(str))
        num_labels = len(self.label_encoder.classes_)

        # Lazy loading of model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint, num_labels=num_labels
        ).to(self.device)

        # Tokenize with dynamic max length
        train_encodings = self.tokenizer(
            X_train, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )

        dataset = torch.utils.data.TensorDataset(
            train_encodings["input_ids"],
            train_encodings.get("attention_mask", torch.ones_like(train_encodings["input_ids"])),
            torch.tensor(y_train, dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        use_cuda = (self.device.type == "cuda")

        self.model.train()
        for epoch in range(self.num_epochs):
            for step, (b_input_ids, b_mask, b_labels) in enumerate(loader):
                b_input_ids = b_input_ids.to(self.device)
                b_mask = b_mask.to(self.device)
                b_labels = b_labels.to(self.device)

                with HardwareManager.get_autocast_context(device=self.device.type, enabled=use_cuda):
                    outputs = self.model(input_ids=b_input_ids, attention_mask=b_mask, labels=b_labels)
                    loss = outputs.loss / self.grad_accum_steps

                loss.backward()
                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()

        self.is_trained = True
        train_duration = time.time() - t0
        HardwareManager.clear_cuda_cache()

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
        if not self.is_trained or self.model is None:
            raise RuntimeError("Transformer model is not fitted yet.")
        X = self._prepare_text(df)
        self.model.eval()

        encodings = self.tokenizer(X, truncation=True, padding=True, max_length=128, return_tensors="pt")
        input_ids = encodings["input_ids"].to(self.device)
        mask = encodings.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        with torch.no_grad():
            with HardwareManager.get_autocast_context(device=self.device.type, enabled=(self.device.type == "cuda")):
                outputs = self.model(input_ids=input_ids, attention_mask=mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        return self.label_encoder.inverse_transform(preds)

    def predict_proba(self, df: pd.DataFrame, **kwargs) -> Optional[np.ndarray]:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Transformer model is not fitted yet.")
        X = self._prepare_text(df)
        self.model.eval()

        encodings = self.tokenizer(X, truncation=True, padding=True, max_length=128, return_tensors="pt")
        input_ids = encodings["input_ids"].to(self.device)
        mask = encodings.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=mask)
            probas = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return probas

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", "category")
        if y_true is None:
            if target_col in df.columns:
                y_true = df[target_col].values
            else:
                raise ValueError("Ground truth labels required for classification evaluation.")

        t0 = time.time()
        preds = self.predict(df)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        y_true_encoded = self.label_encoder.transform([str(y) for y in y_true])
        y_pred_encoded = self.label_encoder.transform([str(p) for p in preds])

        metrics = evaluate_classification(y_true_encoded, y_pred_encoded)
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.mkdir(parents=True, exist_ok=True)
        if self.model is not None and self.tokenizer is not None:
            self.model.save_pretrained(p)
            self.tokenizer.save_pretrained(p)
        joblib.dump(
            {
                "model_id": self.model_id,
                "checkpoint": self.checkpoint,
                "label_encoder": self.label_encoder,
                "text_columns": self.text_columns,
                "classes": list(self.label_encoder.classes_),
            },
            p / "metadata.joblib",
        )
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "TransformerClassificationAdapter":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        p = Path(artifact_path)
        meta = joblib.load(p / "metadata.joblib")
        adapter = cls(model_id=meta["model_id"], checkpoint=str(p), text_columns=meta["text_columns"])
        adapter.label_encoder = meta["label_encoder"]
        adapter.tokenizer = AutoTokenizer.from_pretrained(str(p))
        adapter.model = AutoModelForSequenceClassification.from_pretrained(str(p)).to(adapter.device)
        adapter.is_trained = True
        return adapter
