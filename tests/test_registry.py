"""
Unit tests for ArchitectureRegistry and adapter instantiation.
"""

import pytest
from nlp_workbench.registry import ArchitectureRegistry, registry
from nlp_workbench.tasks.base import BaseTaskAdapter


def test_registry_size_and_completeness():
    assert len(registry.models) == 42
    df = registry.to_dataframe()
    assert len(df) == 42
    assert "model_id" in df.columns
    assert "category" in df.columns
    assert "framework" in df.columns


def test_registry_filter_by_task():
    clf_models = registry.list_models(task="classification")
    assert len(clf_models) >= 6  # 3 classical + 3 transformer (+ zeroshot)

    retrieval_models = registry.list_models(task="retrieval")
    assert len(retrieval_models) == 3

    qa_models = registry.list_models(task="qa")
    assert len(qa_models) == 3


def test_registry_model_validation():
    # Valid model & task
    registry.validate_model_configuration("classical_nb", task="classification", language="indonesian")

    # Incompatible task
    with pytest.raises(ValueError, match="does not support task"):
        registry.validate_model_configuration("classical_nb", task="retrieval", language="indonesian")


def test_create_adapter():
    adapter = registry.create_adapter("classical_nb")
    assert isinstance(adapter, BaseTaskAdapter)
    assert adapter.model_id == "classical_nb"
