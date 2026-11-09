"""
NLP Experimentation Workbench
A modular, hardware-aware NLP experimentation framework for research and Gemastik Data Mining preparation.
"""

from .config import WorkbenchConfig, get_default_config
from .hardware import HardwareManager, HardwareProfile
from .data import DataLoader, validate_dataset_schema
from .eda import DataQualityAuditor
from .preprocessing import TextCleaner, recommend_preprocessing
from .splitting import DatasetSplitter, LeakageAuditor
from .evaluation import evaluate_predictions
from .registry import ArchitectureRegistry, registry
from .training import run_cross_validation, select_best_model
from .inference import predict_new
from .export import ModelExporter
from .competition import GemastikRunner

__version__ = "1.0.0"

__all__ = [
    "WorkbenchConfig",
    "get_default_config",
    "HardwareManager",
    "HardwareProfile",
    "DataLoader",
    "validate_dataset_schema",
    "DataQualityAuditor",
    "TextCleaner",
    "recommend_preprocessing",
    "DatasetSplitter",
    "LeakageAuditor",
    "evaluate_predictions",
    "ArchitectureRegistry",
    "registry",
    "run_cross_validation",
    "select_best_model",
    "predict_new",
    "ModelExporter",
    "GemastikRunner",
]
