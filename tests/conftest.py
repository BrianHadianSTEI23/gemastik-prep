"""
Pytest configuration and shared fixtures for NLP Experimentation Workbench.
"""

import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from nlp_workbench.config import WorkbenchConfig
from nlp_workbench.hardware import HardwareManager
from nlp_workbench.data import (
    generate_synthetic_classification_dataset,
    generate_synthetic_regression_dataset,
    generate_synthetic_multilabel_dataset,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def hardware_mgr():
    return HardwareManager(target_device="cpu")


@pytest.fixture
def sample_classification_df():
    return generate_synthetic_classification_dataset(n_samples=60, random_state=42)


@pytest.fixture
def sample_regression_df():
    return generate_synthetic_regression_dataset(n_samples=60, random_state=42)


@pytest.fixture
def sample_multilabel_df():
    return generate_synthetic_multilabel_dataset(n_samples=60, random_state=42)


@pytest.fixture
def default_config():
    return WorkbenchConfig(
        task="classification",
        dataset_path="synthetic",
        text_column="text",
        target_column="target",
        device="cpu",
        n_splits=3,
        random_seed=42,
    )
