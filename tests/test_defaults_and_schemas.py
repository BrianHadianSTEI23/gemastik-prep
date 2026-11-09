"""
Tests for comprehensive task default configurations and schema validation contracts
across all 13 supported NLP tasks.
"""

import pytest
import pandas as pd
import numpy as np
from nlp_workbench.config import get_default_config, WorkbenchConfig
from nlp_workbench.data import (
    DataLoader,
    validate_dataset_schema,
    RetrievalDatasetBundle,
)

ALL_TASKS = [
    "classification",
    "zeroshot",
    "multilabel",
    "regression",
    "ner",
    "qa",
    "summarization",
    "translation",
    "generation",
    "pair_classification",
    "similarity",
    "retrieval",
    "topic_modeling",
]


@pytest.mark.parametrize("task", ALL_TASKS)
def test_default_config_validity_for_all_tasks(task):
    """Every supported task must have a ready-to-run default config that passes validation."""
    cfg = get_default_config(task=task, language="indonesian")
    assert cfg.task == task
    assert cfg.language == "indonesian"
    assert len(cfg.model_ids) > 0
    # Must pass validation without error
    cfg.validate()


@pytest.mark.parametrize("task", ALL_TASKS)
def test_synthetic_data_matches_schema_for_all_tasks(task):
    """Synthetic generator for each task must produce data compliant with its default schema."""
    cfg = get_default_config(task=task)
    data = DataLoader.generate_synthetic_dataset(task=task, n_samples=25, random_state=42)

    is_valid, errors = validate_dataset_schema(data, config=cfg, raise_on_error=False)
    assert is_valid, f"Schema validation failed for task '{task}': {errors}"


def test_retrieval_bundle_schema_validation():
    """RetrievalDatasetBundle must satisfy retrieval schema checks."""
    bundle = DataLoader.generate_synthetic_dataset("retrieval", n_samples=20)
    assert isinstance(bundle, RetrievalDatasetBundle)
    assert len(bundle) == 20
    assert len(bundle.corpus_df) > 0

    is_valid, errors = validate_dataset_schema(bundle, task="retrieval")
    assert is_valid

    # Corrupt corpus column
    bad_bundle = RetrievalDatasetBundle(
        corpus_df=pd.DataFrame({"doc_id": ["D1"], "bad_col": ["text"]}),
        queries_df=bundle.queries_df,
        doc_text_col="bad_col",
    )
    # Valid with custom doc_text_col
    assert validate_dataset_schema(bad_bundle, task="retrieval")[0]


def test_schema_null_targets_rejection():
    """Supervised tasks must reject null/NaN targets rather than converting to strings."""
    df = pd.DataFrame({
        "text": ["dokumen 1", "dokumen 2", "dokumen 3"],
        "target": ["pos", None, "neg"],
    })
    cfg = get_default_config("classification")
    with pytest.raises(ValueError, match="missing values"):
        validate_dataset_schema(df, config=cfg, raise_on_error=True)


def test_regression_non_numeric_target_rejection():
    """Regression tasks must reject non-numeric target columns."""
    df = pd.DataFrame({
        "text": ["dokumen 1", "dokumen 2"],
        "target": ["not_a_number", "also_text"],
    })
    cfg = get_default_config("regression")
    with pytest.raises(ValueError, match="must be numeric"):
        validate_dataset_schema(df, config=cfg, raise_on_error=True)
