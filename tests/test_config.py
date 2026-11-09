"""
Unit tests for WorkbenchConfig and task configuration validation.
"""

import pytest
from nlp_workbench.config import WorkbenchConfig, get_default_config


def test_default_config_creation():
    cfg = get_default_config("classification")
    assert cfg.task == "classification"
    assert cfg.text_column == "text"
    assert cfg.target_column == "target"
    assert cfg.split_strategy == "stratified"
    assert cfg.n_splits == 5


def test_invalid_task_rejection():
    with pytest.raises(ValueError, match="Unsupported NLP task"):
        WorkbenchConfig(task="unsupported_quantum_nlp_task")


def test_schema_validation_classification():
    # Target column is required for classification
    with pytest.raises(ValueError, match="target_column is required"):
        WorkbenchConfig(task="classification", target_column=None).validate_for_execution()


def test_schema_validation_pair_classification():
    # text_b_column is required for pair_classification
    with pytest.raises(ValueError, match="text_b_column"):
        WorkbenchConfig(task="pair_classification", text_b_column=None).validate_for_execution()

    # Valid config with text_b_column
    cfg = WorkbenchConfig(task="pair_classification", text_b_column="premise", target_column="label")
    cfg.validate_for_execution()


def test_config_to_dict_and_yaml():
    cfg = WorkbenchConfig(task="ner", text_column="tokens", target_column="ner_tags")
    d = cfg.to_dict()
    assert d["task"] == "ner"
    assert d["text_column"] == "tokens"
    assert d["target_column"] == "ner_tags"

    yaml_str = cfg.to_yaml()
    assert "task: ner" in yaml_str
