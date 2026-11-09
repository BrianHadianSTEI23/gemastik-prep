"""
Tests for group and time split strategies, 3-way split leakage auditing,
leakage policy enforcement, and RetrievalDatasetBundle splitting.
"""

import pytest
import pandas as pd
import numpy as np
from nlp_workbench.config import WorkbenchConfig
from nlp_workbench.splitting import DatasetSplitter, LeakageAuditor
from nlp_workbench.data import DataLoader, RetrievalDatasetBundle


def test_group_strategy_requires_group_column():
    """Config with split_strategy='group' requires a group_column."""
    cfg = WorkbenchConfig(task="classification", split_strategy="group", group_column=None)
    with pytest.raises(ValueError, match="split_strategy='group' requires 'group_column'"):
        cfg.validate()


def test_time_strategy_requires_time_column():
    """Config with split_strategy='time' requires a time_column."""
    cfg = WorkbenchConfig(task="classification", split_strategy="time", time_column=None)
    with pytest.raises(ValueError, match="split_strategy='time' requires 'time_column'"):
        cfg.validate()


def test_three_way_leakage_audit():
    """Leakage auditor must audit all 3 pairs: Train/Val, Train/Test, and Val/Test."""
    train_df = pd.DataFrame({"id": [1, 2], "text": ["unique train 1", "unique train 2"]})
    val_df = pd.DataFrame({"id": [3, 4], "text": ["unique val 1", "leaking text between val and test"]})
    test_df = pd.DataFrame({"id": [5, 6], "text": ["leaking text between val and test", "unique test 2"]})

    rep = LeakageAuditor.audit_all_splits(train_df, val_df, test_df, text_columns=["text"], id_column="id")

    assert rep["leakage_detected"] is True
    # Train/Val has no leakage
    assert rep["train_val"]["leakage_detected"] is False
    # Train/Test has no leakage
    assert rep["train_test"]["leakage_detected"] is False
    # Val/Test has duplicate text leakage!
    assert rep["val_test"]["leakage_detected"] is True
    assert rep["val_test"]["duplicate_text_count"] == 1


def test_leakage_policy_enforcement_raises_error():
    """Leakage policy enforcement raises ValueError when leakage is present and deduplication is disabled."""
    train_df = pd.DataFrame({"id": [1, 2], "text": ["shared duplicate", "train text"]})
    val_df = pd.DataFrame({"id": [3, 4], "text": ["shared duplicate", "val text"]})
    test_df = pd.DataFrame({"id": [5, 6], "text": ["test text 1", "test text 2"]})

    with pytest.raises(ValueError, match="Data leakage detected across dataset splits"):
        LeakageAuditor.enforce_leakage_policy(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            text_columns=["text"],
            deduplicate_texts=False,
            raise_on_leakage=True,
        )


def test_leakage_deduplication_policy_purges_leaks():
    """When deduplicate_texts=True, leaking rows in val and test are purged."""
    train_df = pd.DataFrame({"id": [1, 2], "text": ["shared duplicate", "train text"]})
    val_df = pd.DataFrame({"id": [3, 4], "text": ["shared duplicate", "val text"]})
    test_df = pd.DataFrame({"id": [5, 6], "text": ["test text 1", "test text 2"]})

    tr, va, te, audit = LeakageAuditor.enforce_leakage_policy(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        text_columns=["text"],
        deduplicate_texts=True,
        raise_on_leakage=True,
    )
    assert len(tr) == 2
    assert len(va) == 1  # 'shared duplicate' removed
    assert audit["leakage_detected"] is False


def test_retrieval_bundle_splitting():
    """DatasetSplitter.split on a RetrievalDatasetBundle partitions queries while preserving corpus."""
    bundle = DataLoader.generate_synthetic_dataset("retrieval", n_samples=30, random_state=42)
    splitter = DatasetSplitter(test_size=0.2, val_size=0.1, random_state=42)

    tr_b, va_b, te_b = splitter.split(bundle)

    assert isinstance(tr_b, RetrievalDatasetBundle)
    assert isinstance(va_b, RetrievalDatasetBundle)
    assert isinstance(te_b, RetrievalDatasetBundle)

    # Corpus is preserved across all bundles
    assert len(tr_b.corpus_df) == len(bundle.corpus_df)
    assert len(va_b.corpus_df) == len(bundle.corpus_df)
    assert len(te_b.corpus_df) == len(bundle.corpus_df)

    # Queries are split
    total_queries = len(tr_b.queries_df) + len(va_b.queries_df) + len(te_b.queries_df)
    assert total_queries == len(bundle.queries_df)
