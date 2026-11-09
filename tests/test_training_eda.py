"""
Unit tests for EDA DataQualityAuditor, Preprocessing, and Cross-Validation Training.
"""

import pytest
import pandas as pd
import numpy as np
from nlp_workbench.eda import DataQualityAuditor
from nlp_workbench.preprocessing import TextCleaner, recommend_preprocessing
from nlp_workbench.training import run_cross_validation, select_best_model
from nlp_workbench.registry import registry
from nlp_workbench.data import generate_synthetic_classification_dataset
from nlp_workbench.hardware import HardwareManager


def test_hardware_manager_profile():
    hw = HardwareManager(target_device="cpu")
    profile = hw.profile
    assert profile.device == "cpu"
    rec = hw.recommend_batch_size()
    assert "batch_size" in rec
    assert "grad_accum_steps" in rec


def test_text_cleaner_negation_preservation():
    cleaner = TextCleaner(language="indonesian", remove_stopwords=True)
    text = "Pelayanan ini tidak bagus dan bukan pilihan tepat"
    cleaned = cleaner.clean_text(text)
    # Ensure negations "tidak" and "bukan" are preserved
    assert "tidak" in cleaned
    assert "bukan" in cleaned


def test_eda_data_quality_auditor(sample_classification_df):
    auditor = DataQualityAuditor()
    audit_report = auditor.audit(sample_classification_df, text_columns=["text"], target_column="target")
    
    assert "findings" in audit_report
    assert "summary_stats" in audit_report
    assert len(audit_report["findings"]) > 0

    # Test explainability: 5-point report structure
    finding = audit_report["findings"][0]
    assert "issue" in finding
    assert "evidence" in finding
    assert "why_it_matters" in finding
    assert "recommendations" in finding
    assert "potential_drawbacks" in finding
    assert "action_keys" in finding

    # Test summary table generation
    df_summary = auditor.to_dataframe(audit_report)
    assert isinstance(df_summary, pd.DataFrame)
    assert len(df_summary) == len(audit_report["findings"])


def test_cross_validation_and_model_selection():
    df = generate_synthetic_classification_dataset(n_samples=40, random_state=42)
    adapters = [
        registry.create_adapter("classical_nb", ngram_range=(1, 1), max_features=100),
        registry.create_adapter("classical_logistic", ngram_range=(1, 1), max_features=100),
    ]

    leaderboard = []
    for adapter in adapters:
        cv_result = run_cross_validation(
            adapter=adapter,
            df=df,
            task="classification",
            text_columns=["text"],
            target_column="target",
            n_splits=3,
            random_state=42,
        )
        assert "mean_cv_score" in cv_result
        assert "fold_results" in cv_result
        assert len(cv_result["fold_results"]) == 3
        leaderboard.append(cv_result)

    best_model_id, best_score = select_best_model(leaderboard, primary_metric="mean_cv_score")
    assert best_model_id in ["classical_nb", "classical_logistic"]
    assert 0.0 <= best_score <= 1.0
