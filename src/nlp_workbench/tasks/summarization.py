"""
Text Summarization Adapter.
Supports BART, Pegasus, and T5 encoder-decoder architectures with ROUGE metrics.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import joblib
import pandas as pd

from .base import BaseTaskAdapter
from ..evaluation import evaluate_summarization
from ..hardware import HardwareManager

class SummarizationAdapter(BaseTaskAdapter):
    """
    Abstractive Summarization Adapter for BART, Pegasus, and T5.
    """
    def __init__(
        self,
        model_id: str,
        checkpoint: str = "sshleifer/distilbart-cnn-12-6",
        device: str = "auto",
        max_length: int = 64,
        min_length: int = 10,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.checkpoint = checkpoint
        self.device = HardwareManager.inspect(device).get_torch_device()
        self.max_length = max_length
        self.min_length = min_length
        self.text_columns: List[str] = kwargs.get("text_columns", ["document"])
        self.target_column: str = kwargs.get("target_column", "summary")
        self.pipeline = None
        self.is_trained = True

    def _init_pipeline(self):
        if self.pipeline is None:
            from transformers import pipeline
            device_idx = 0 if self.device.type == "cuda" else -1
            self.pipeline = pipeline(
                "summarization",
                model=self.checkpoint,
                device=device_idx,
            )

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        self._init_pipeline()
        return {
            "model_id": self.model_id,
            "mode": "summarization-inference",
            "checkpoint": self.checkpoint,
            "status": "Summarization pipeline initialized.",
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> List[str]:
        self._init_pipeline()
        doc_col = self.text_columns[0] if self.text_columns[0] in df.columns else "content"
        docs = df[doc_col].astype(str).tolist()
        if "t5" in self.checkpoint.lower():
            docs = [f"summarize: {d}" if not d.lower().startswith("summarize:") else d for d in docs]
        summaries = self.pipeline(
            docs,
            max_length=self.max_length,
            min_length=self.min_length,
            truncation=True,
        )
        return [s["summary_text"] for s in summaries]

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", self.target_column)
        if y_true is None and target_col in df.columns:
            y_true = df[target_col].astype(str).tolist()

        if y_true is None:
            return {"Status": "Summarization completed. Reference summaries absent."}

        t0 = time.time()
        preds = self.predict(df)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        metrics = evaluate_summarization(y_true, preds)
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "adapter_type": "summarization",
            "checkpoint": self.checkpoint,
            "text_columns": self.text_columns,
            "target_column": self.target_column,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "SummarizationAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(
            model_id=bundle["model_id"],
            checkpoint=bundle["checkpoint"],
            text_columns=bundle["text_columns"],
            target_column=bundle["target_column"],
        )
        return adapter
