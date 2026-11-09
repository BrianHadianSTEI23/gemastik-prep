"""
Production Inference module.
Provides an easy-to-use API for generating predictions on raw text inputs using exported artifacts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd

from .export import ModelExporter, LoadedModelBundle
from .tasks.base import BaseTaskAdapter


class InferenceResult(dict):
    """
    Result container providing dictionary, attribute, and DataFrame access.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        predictions: Any,
        probabilities: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            dataframe=dataframe,
            predictions=list(predictions) if hasattr(predictions, "__iter__") and not isinstance(predictions, str) else [predictions],
            probabilities=probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities,
            **kwargs,
        )
        self.df = dataframe
        self.predictions = self["predictions"]
        self.probabilities = self["probabilities"]

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'InferenceResult' has no attribute '{name}'")


def predict_new(
    raw_inputs: Optional[Union[str, List[str], List[Dict[str, Any]], pd.DataFrame]] = None,
    texts: Optional[Union[str, List[str]]] = None,
    artifact_path: Union[str, Path] = "output_model/nlp_model_bundle.joblib",
    adapter: Optional[Union[BaseTaskAdapter, LoadedModelBundle]] = None,
    cleaner_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> InferenceResult:
    """
    Production inference entry point.
    Accepts raw text, list of strings, dictionaries, or DataFrame,
    and returns predictions and probabilities in an InferenceResult container.
    """
    inputs = texts if texts is not None else raw_inputs
    if inputs is None:
        raise ValueError("Either 'raw_inputs' or 'texts' must be provided.")

    if adapter is None:
        loaded = ModelExporter.load(artifact_path)
        adapter = loaded.adapter
    elif isinstance(adapter, LoadedModelBundle):
        adapter = adapter.adapter

    # Standardize input to pandas DataFrame
    if isinstance(inputs, str):
        df_in = pd.DataFrame([{"text": inputs}])
    elif isinstance(inputs, list):
        if inputs and isinstance(inputs[0], str):
            df_in = pd.DataFrame([{"text": t} for t in inputs])
        elif inputs and isinstance(inputs[0], dict):
            df_in = pd.DataFrame(inputs)
        else:
            df_in = pd.DataFrame()
    elif isinstance(inputs, pd.DataFrame):
        df_in = inputs.copy()
    else:
        raise TypeError(f"Unsupported input type '{type(inputs)}' for inference.")

    if df_in.empty:
        return InferenceResult(pd.DataFrame(), predictions=[], probabilities=[])

    preds = adapter.predict(df_in)

    # Build output table
    results_df = pd.DataFrame()
    display_cols = [c for c in ["title", "content", "document", "prompt", "text", "text_a"] if c in df_in.columns]
    if display_cols:
        results_df["input_text"] = df_in[display_cols[0]].astype(str)
    else:
        results_df["input_text"] = df_in.iloc[:, 0].astype(str)

    results_df["prediction"] = preds

    # Check for probability support
    probas = None
    try:
        if hasattr(adapter, "predict_proba"):
            probas = adapter.predict_proba(df_in)
            if probas is not None and len(probas) == len(df_in):
                results_df["confidence"] = np.max(probas, axis=1).round(4)
                if hasattr(adapter, "label_encoder") and hasattr(adapter.label_encoder, "classes_"):
                    for c_idx, class_name in enumerate(adapter.label_encoder.classes_):
                        if c_idx < probas.shape[1]:
                            results_df[f"proba_{class_name}"] = probas[:, c_idx].round(4)
    except Exception:
        pass  # Model does not support probabilities

    return InferenceResult(
        dataframe=results_df,
        predictions=preds,
        probabilities=probas,
    )
