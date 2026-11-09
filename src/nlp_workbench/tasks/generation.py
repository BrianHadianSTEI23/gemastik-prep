"""
Text Generation / Instruction Following Adapter.
Supports Qwen2.5, TinyLlama, and DistilGPT-2 causal language models with generate() controls.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import joblib
import pandas as pd
import torch

from .base import BaseTaskAdapter
from ..hardware import HardwareManager

class GenerationAdapter(BaseTaskAdapter):
    """
    Causal Language Modeling and Instruction-Following Adapter.
    """
    def __init__(
        self,
        model_id: str,
        checkpoint: str = "distilgpt2",
        device: str = "auto",
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.checkpoint = checkpoint
        self.device = HardwareManager.inspect(device).get_torch_device()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.text_columns: List[str] = kwargs.get("text_columns", ["prompt"])
        self.target_column: str = kwargs.get("target_column", "reference_output")
        self.model = None
        self.tokenizer = None
        self.is_trained = True

    def _init_model(self):
        if self.model is None or self.tokenizer is None:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint).to(self.device)
            self.model.eval()

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        self._init_model()
        return {
            "model_id": self.model_id,
            "mode": "causal-generation-inference",
            "checkpoint": self.checkpoint,
            "status": "Causal LM initialized for text generation.",
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> List[str]:
        self._init_model()
        prompt_col = self.text_columns[0] if self.text_columns[0] in df.columns else "content"
        prompts = df[prompt_col].astype(str).tolist()

        outputs = []
        for p in prompts:
            inputs = self.tokenizer(p, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                with HardwareManager.get_autocast_context(device=self.device.type, enabled=(self.device.type == "cuda")):
                    gen_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        repetition_penalty=self.repetition_penalty,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True,
                    )
            # Decode generated tokens only
            input_len = inputs["input_ids"].shape[1]
            generated = self.tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True)
            outputs.append(generated.strip())
        return outputs

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", self.target_column)
        t0 = time.time()
        preds = self.predict(df)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        avg_len = sum(len(p.split()) for p in preds) / max(1, len(preds))
        res = {
            "Average Generated Words": round(float(avg_len), 2),
            "Latency/Sample (ms)": round(latency_ms, 3),
        }
        if y_true is not None or target_col in df.columns:
            from ..evaluation import evaluate_summarization
            refs = y_true if y_true is not None else df[target_col].astype(str).tolist()
            rouge_metrics = evaluate_summarization(refs, preds)
            res.update(rouge_metrics)
        return res

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "adapter_type": "generation",
            "checkpoint": self.checkpoint,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "text_columns": self.text_columns,
            "target_column": self.target_column,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "GenerationAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(
            model_id=bundle["model_id"],
            checkpoint=bundle["checkpoint"],
            max_new_tokens=bundle["max_new_tokens"],
            temperature=bundle["temperature"],
            top_p=bundle["top_p"],
            repetition_penalty=bundle["repetition_penalty"],
            text_columns=bundle["text_columns"],
            target_column=bundle["target_column"],
        )
        return adapter
