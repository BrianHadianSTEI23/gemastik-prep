"""
Unit tests for DatasetSplitter and LeakageAuditor.
"""

import pytest
import pandas as pd
import numpy as np
from nlp_workbench.splitting import DatasetSplitter, LeakageAuditor


def test_stratified_split(sample_classification_df):
    splitter = DatasetSplitter(strategy="stratified", n_splits=3, test_size=0.2, random_state=42)
    train_df, val_df, test_df = splitter.split(sample_classification_df, target_col="target")

    assert len(train_df) + len(val_df) + len(test_df) == len(sample_classification_df)
    # Check class representation
    orig_classes = set(sample_classification_df["target"].unique())
    train_classes = set(train_df["target"].unique())
    assert orig_classes == train_classes


def test_kfold_split(sample_classification_df):
    splitter = DatasetSplitter(strategy="kfold", n_splits=4, random_state=42)
    folds = list(splitter.get_cv_folds(sample_classification_df, target_col="target"))
    assert len(folds) == 4

    for train_idx, val_idx in folds:
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(train_idx) + len(val_idx) == len(sample_classification_df)


def test_group_split():
    df = pd.DataFrame({
        "text": [f"sentence {i}" for i in range(20)],
        "target": [i % 2 for i in range(20)],
        "author_id": [i // 4 for i in range(20)],  # 5 groups
    })
    splitter = DatasetSplitter(strategy="group", n_splits=3, test_size=0.2, random_state=42)
    train_df, val_df, test_df = splitter.split(df, group_col="author_id")

    train_groups = set(train_df["author_id"])
    test_groups = set(test_df["author_id"])
    assert len(train_groups.intersection(test_groups)) == 0


def test_leakage_auditor():
    train_df = pd.DataFrame({
        "id": [1, 2, 3],
        "text": ["unique text 1", "duplicate text", "unique text 3"],
    })
    test_df = pd.DataFrame({
        "id": [3, 4],  # ID 3 overlaps!
        "text": ["duplicate text", "unique test text"],  # 'duplicate text' overlaps!
    })

    auditor = LeakageAuditor()
    findings = auditor.audit_splits(train_df, test_df, text_col="text", id_col="id")
    
    assert findings["text_leakage_detected"] is True
    assert findings["duplicate_text_count"] == 1
    assert findings["id_leakage_detected"] is True
    assert findings["overlapping_id_count"] == 1
