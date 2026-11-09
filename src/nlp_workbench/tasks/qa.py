"""
Extractive Question Answering (QA) Adapter.
Supports BERT, RoBERTa, and XLM-RoBERTa extractive span extraction with EM and Token F1 metrics.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import joblib
import numpy as np
import pandas as pd

from .base import BaseTaskAdapter
from ..evaluation import evaluate_qa
from ..hardware import HardwareManager

class QuestionAnsweringAdapter(BaseTaskAdapter):
    """
    Extractive QA Adapter predicting answer text spans within context documents.
    """
    def __init__(
        self,
        model_id: str,
        checkpoint: str = "deepset/roberta-base-squad2",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.checkpoint = checkpoint
        self.device = HardwareManager.inspect(device).get_torch_device()
        self.context_column: str = kwargs.get("context_column", "context")
        self.question_column: str = kwargs.get("question_column", "question")
        self.answer_column: str = kwargs.get("answer_column", "answer_text")
        self.pipeline = None
        self.is_trained = True

    def _init_pipeline(self):
        if self.pipeline is None:
            from transformers import pipeline
            device_idx = 0 if self.device.type == "cuda" else -1
            self.pipeline = pipeline(
                "question-answering",
                model=self.checkpoint,
                device=device_idx,
            )

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        self._init_pipeline()
        return {
            "model_id": self.model_id,
            "mode": "extractive-qa-inference",
            "checkpoint": self.checkpoint,
            "status": "QA pipeline initialized.",
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> List[str]:
        self._init_pipeline()
        ctx_col = self.context_column if self.context_column in df.columns else "content"
        q_col = self.question_column if self.question_column in df.columns else "title"

        preds = []
        for _, row in df.iterrows():
            q = str(row.get(q_col, ""))
            c = str(row.get(ctx_col, ""))
            if not q or not c:
                preds.append("")
                continue
            res = self.pipeline(question=q, context=c)
            preds.append(res["answer"])
        return preds

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        ans_col = self.answer_column if self.answer_column in df.columns else "answer_text"
        if y_true is None and ans_col in df.columns:
            y_true = df[ans_col].astype(str).tolist()

        if y_true is None:
            return {"Status": "QA inference completed. Reference answers absent."}

        t0 = time.time()
        preds = self.predict(df)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        metrics = evaluate_qa(y_true, preds)
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "adapter_type": "qa",
            "checkpoint": self.checkpoint,
            "context_column": self.context_column,
            "question_column": self.question_column,
            "answer_column": self.answer_column,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "QuestionAnsweringAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(
            model_id=bundle["model_id"],
            checkpoint=bundle["checkpoint"],
            context_column=bundle["context_column"],
            question_column=bundle["question_column"],
            answer_column=bundle["answer_column"],
        )
        return adapter
