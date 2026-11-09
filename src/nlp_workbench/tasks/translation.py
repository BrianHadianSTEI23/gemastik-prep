"""
Machine Translation Adapter.
Supports MarianMT, mT5, and NLLB sequence-to-sequence translation with SacreBLEU and chrF metrics.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import joblib
import pandas as pd

from .base import BaseTaskAdapter
from ..evaluation import evaluate_translation
from ..hardware import HardwareManager

class TranslationAdapter(BaseTaskAdapter):
    """
    Neural Machine Translation Adapter supporting bilingual and multilingual models.
    """
    def __init__(
        self,
        model_id: str,
        checkpoint: str = "Helsinki-NLP/opus-mt-id-en",
        source_lang: str = "id",
        target_lang: str = "en",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.checkpoint = checkpoint
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = HardwareManager.inspect(device).get_torch_device()
        self.text_columns: List[str] = kwargs.get("text_columns", ["source_text"])
        self.target_column: str = kwargs.get("target_column", "target_text")
        self.pipeline = None
        self.is_trained = True

    def _map_nllb_lang(self, lang: str) -> str:
        lang_map = {
            "id": "ind_Latn",
            "indonesian": "ind_Latn",
            "en": "eng_Latn",
            "english": "eng_Latn",
            "jv": "jav_Latn",
            "javanese": "jav_Latn",
            "su": "sun_Latn",
            "sundanese": "sun_Latn",
        }
        return lang_map.get(lang.lower().strip(), lang)

    def _init_pipeline(self):
        if self.pipeline is None:
            from transformers import pipeline
            device_idx = 0 if self.device.type == "cuda" else -1
            is_nllb = "nllb" in self.checkpoint.lower()
            is_mt5 = "mt5" in self.checkpoint.lower()

            if is_nllb:
                src_code = self._map_nllb_lang(self.source_lang)
                tgt_code = self._map_nllb_lang(self.target_lang)
                self.pipeline = pipeline(
                    "translation",
                    model=self.checkpoint,
                    src_lang=src_code,
                    tgt_lang=tgt_code,
                    device=device_idx,
                )
            elif is_mt5:
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.checkpoint,
                    device=device_idx,
                )
            else:
                self.pipeline = pipeline(
                    "translation",
                    model=self.checkpoint,
                    device=device_idx,
                )

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        self._init_pipeline()
        return {
            "model_id": self.model_id,
            "mode": "translation-inference",
            "checkpoint": self.checkpoint,
            "language_pair": f"{self.source_lang}->{self.target_lang}",
            "status": "Translation pipeline initialized.",
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> List[str]:
        self._init_pipeline()
        src_col = self.text_columns[0] if self.text_columns[0] in df.columns else "content"
        if src_col not in df.columns:
            # fallback to first string column
            src_col = df.columns[0]
        sources = df[src_col].astype(str).tolist()

        is_mt5 = "mt5" in self.checkpoint.lower()
        if is_mt5:
            prefix = f"translate {self.source_lang} to {self.target_lang}: "
            sources = [prefix + s for s in sources]

        translations = self.pipeline(sources, truncation=True)
        results = []
        for t in translations:
            if "translation_text" in t:
                results.append(t["translation_text"].strip())
            elif "generated_text" in t:
                results.append(t["generated_text"].strip())
            else:
                results.append(str(t))
        return results

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.get("target_column", self.target_column)
        if y_true is None and target_col in df.columns:
            y_true = df[target_col].astype(str).tolist()

        if y_true is None:
            return {"Status": "Translation completed. Reference targets absent."}

        t0 = time.time()
        preds = self.predict(df)
        latency_ms = ((time.time() - t0) / max(1, len(df))) * 1000

        metrics = evaluate_translation(y_true, preds)
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "adapter_type": "translation",
            "checkpoint": self.checkpoint,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "text_columns": self.text_columns,
            "target_column": self.target_column,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "TranslationAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(
            model_id=bundle["model_id"],
            checkpoint=bundle["checkpoint"],
            source_lang=bundle["source_lang"],
            target_lang=bundle["target_lang"],
            text_columns=bundle["text_columns"],
            target_column=bundle["target_column"],
        )
        return adapter
