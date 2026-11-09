"""
Zero-Shot Classification Adapter.
Supports BART-MNLI, DeBERTa-v3-MNLI, and XLM-RoBERTa XNLI with custom hypothesis templates.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import joblib
import numpy as np
import pandas as pd
import torch

from .base import BaseTaskAdapter
from ..preprocessing import TextCleaner
from ..evaluation import evaluate_classification
from ..hardware import HardwareManager

class ZeroShotClassificationAdapter(BaseTaskAdapter):
    """
    Zero-Shot Inference Adapter using entailment formulation.
    Does not require training samples; classifies text directly into candidate labels.
    """
    def __init__(
        self,
        model_id: str,
        checkpoint: str = "MoritzLaurer/DeBERTa-v3-xsmall-mnli",
        candidate_labels: Optional[List[str]] = None,
        hypothesis_template: str = "Teks ini mengenai {}.",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.checkpoint = checkpoint
        self.candidate_labels = candidate_labels or [
            "Teknologi", "Ekonomi", "Kesehatan", "Pendidikan", "Olahraga"
        ]
        self.hypothesis_template = hypothesis_template
        self.text_columns: List[str] = kwargs.get("text_columns", ["title", "content"])
        self.cleaner = TextCleaner(lowercase=False)
        self.pipeline = None
        self.device = HardwareManager.inspect(device).get_torch_device()
        self.is_trained = True  # Pretrained zero-shot models are ready for inference

    def _init_pipeline(self):
        if self.pipeline is None:
            from transformers import pipeline
            device_idx = 0 if self.device.type == "cuda" else -1
            self.pipeline = pipeline(
                "zero-shot-classification",
                model=self.checkpoint,
                device=device_idx,
            )

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
        """Zero-shot models do not require supervised fitting; verifies pipeline and returns status."""
        self._init_pipeline()
        return {
            "model_id": self.model_id,
            "mode": "zero-shot-inference",
            "candidate_labels": self.candidate_labels,
            "status": "Ready for zero-shot inference without training.",
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        self._init_pipeline()
        texts = self._prepare_text(df)
        results = self.pipeline(
            texts,
            candidate_labels=self.candidate_labels,
            hypothesis_template=self.hypothesis_template,
        )
        if isinstance(results, dict):
            results = [results]
        return np.array([res["labels"][0] for res in results])

    def predict_proba(self, df: pd.DataFrame, **kwargs) -> Optional[np.ndarray]:
        self._init_pipeline()
        texts = self._prepare_text(df)
        results = self.pipeline(
            texts,
            candidate_labels=self.candidate_labels,
            hypothesis_template=self.hypothesis_template,
        )
        if isinstance(results, dict):
            results = [results]

        probas = []
        for res in results:
            score_dict = dict(zip(res["labels"], res["scores"]))
            probas.append([score_dict[c] for c in self.candidate_labels])
        return np.array(probas)

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", "category")
        if y_true is None and target_col in df.columns:
            y_true = df[target_col].values

        if y_true is None:
            return {"Status": "Zero-shot inference completed. Reference ground truth absent."}

        # Validate that y_true labels are in candidate_labels before running prediction
        candidate_set = set(str(c) for c in self.candidate_labels)
        invalid_labels = set(str(y) for y in y_true) - candidate_set
        if invalid_labels:
            raise ValueError(
                f"Evaluation failed: Ground truth contains labels {invalid_labels} "
                f"that are not in candidate_labels: {self.candidate_labels}"
            )

        t0 = time.time()
        preds = self.predict(df)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        class_to_idx = {str(c): i for i, c in enumerate(self.candidate_labels)}
        y_true_enc = [class_to_idx[str(y)] for y in y_true]
        y_pred_enc = [class_to_idx[str(p)] for p in preds]

        labels = list(range(len(self.candidate_labels)))
        metrics = evaluate_classification(np.array(y_true_enc), np.array(y_pred_enc), labels=labels)
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "adapter_type": "zeroshot",
            "checkpoint": self.checkpoint,
            "candidate_labels": self.candidate_labels,
            "hypothesis_template": self.hypothesis_template,
            "text_columns": self.text_columns,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "ZeroShotClassificationAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(
            model_id=bundle["model_id"],
            checkpoint=bundle["checkpoint"],
            candidate_labels=bundle["candidate_labels"],
            hypothesis_template=bundle["hypothesis_template"],
            text_columns=bundle["text_columns"],
        )
        return adapter
