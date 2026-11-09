"""
Semantic Similarity and Sentence Embedding Adapter.
Supports MiniLM, MPNet, and Multilingual E5 with Cosine similarity, Pearson, and Spearman metrics.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import joblib
import numpy as np
import pandas as pd
import torch

from .base import BaseTaskAdapter
from ..evaluation import evaluate_similarity
from ..hardware import HardwareManager

class SemanticSimilarityAdapter(BaseTaskAdapter):
    """
    Sentence Transformers Embedding & Similarity Adapter.
    """
    def __init__(
        self,
        model_id: str,
        checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.checkpoint = checkpoint
        self.device = HardwareManager.inspect(device).get_torch_device()
        self.text_a_column: str = kwargs.get("text_a_column", "text_a")
        self.text_b_column: str = kwargs.get("text_b_column", "text_b")
        self.target_column: str = kwargs.get("target_column", "similarity_score")
        self.model = None
        self.is_trained = True

    def _init_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.checkpoint, device=str(self.device))

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        self._init_model()
        return {
            "model_id": self.model_id,
            "mode": "sentence-embedding-extraction",
            "checkpoint": self.checkpoint,
            "status": "SentenceTransformer model initialized.",
        }

    def encode(self, texts: List[str], batch_size: int = 16, is_query: bool = True) -> np.ndarray:
        self._init_model()
        if "e5" in self.checkpoint.lower():
            prefix = "query: " if is_query else "passage: "
            texts = [t if t.startswith(("query: ", "passage: ")) else prefix + t for t in texts]
        with HardwareManager.get_autocast_context(device=self.device.type, enabled=(self.device.type == "cuda")):
            return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, device=str(self.device))

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        self._init_model()
        col_a = self.text_a_column if self.text_a_column in df.columns else df.columns[0]
        col_b = self.text_b_column if self.text_b_column in df.columns else df.columns[1]

        texts_a = df[col_a].astype(str).tolist()
        texts_b = df[col_b].astype(str).tolist()

        emb_a = self.encode(texts_a, is_query=True)
        emb_b = self.encode(texts_b, is_query=kwargs.get("is_passage", False) is False)

        # Dot product / cosine similarity
        norm_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-9)
        norm_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-9)
        similarities = (norm_a * norm_b).sum(axis=1)
        return similarities

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", self.target_column)
        if y_true is None and target_col in df.columns:
            y_true = df[target_col].astype(float).values

        if y_true is None:
            return {"Status": "Similarity scores generated. Ground truth absent."}

        t0 = time.time()
        preds = self.predict(df)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        metrics = evaluate_similarity(np.asarray(y_true), np.asarray(preds))
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "adapter_type": "similarity",
            "checkpoint": self.checkpoint,
            "text_a_column": self.text_a_column,
            "text_b_column": self.text_b_column,
            "target_column": self.target_column,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "SemanticSimilarityAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(
            model_id=bundle["model_id"],
            checkpoint=bundle["checkpoint"],
            text_a_column=bundle["text_a_column"],
            text_b_column=bundle["text_b_column"],
            target_column=bundle["target_column"],
        )
        return adapter
