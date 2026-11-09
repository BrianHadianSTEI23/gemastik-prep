"""
Base Task Adapter interface for all NLP architecture categories.
Defines standard training, inference, evaluation, and serialization contracts.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np

class BaseTaskAdapter(ABC):
    def __init__(self, model_id: str, config: Optional[Any] = None, **kwargs):
        self.model_id = model_id
        self.config = config
        self.is_trained = False
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """Fits or fine-tunes the model on training data. Preprocessing must be fold-local."""
        pass

    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """Convenience alias for fit()."""
        return self.fit(*args, **kwargs)

    @abstractmethod
    def predict(self, df: pd.DataFrame, **kwargs) -> Any:
        """Generates predictions for input DataFrame."""
        pass

    def predict_proba(self, df: pd.DataFrame, **kwargs) -> Optional[np.ndarray]:
        """Returns prediction probabilities if supported by architecture."""
        return None

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """Evaluates model performance using task-specific metrics."""
        pass

    @abstractmethod
    def export(self, export_path: Union[str, Path]) -> Path:
        """Serializes model, weights, vectorizers/tokenizers, and preprocessing to artifact path."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, artifact_path: Union[str, Path]) -> "BaseTaskAdapter":
        """Loads serialized model artifact from disk for standalone inference."""
        pass
