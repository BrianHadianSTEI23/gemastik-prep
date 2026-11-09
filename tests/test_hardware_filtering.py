"""
Tests for hardware-aware model filtering and profiling.
"""

import pytest
from nlp_workbench.registry import registry
from nlp_workbench.hardware import HardwareManager


def test_hardware_profiling():
    """HardwareManager profile returns valid hardware attributes."""
    profile = HardwareManager.inspect("cpu")
    assert profile.device == "cpu"
    assert profile.recommended_batch_size >= 1
    assert "HARDWARE DIAGNOSTIC REPORT" in profile.summary()


def test_model_filtering_by_vram_budget():
    """get_compatible_models filters models exceeding max_vram_gb."""
    # Under 2GB budget, CPU models are retained, but high VRAM tier GPU models are excluded
    low_vram_models = registry.get_compatible_models(
        task="classification",
        language="indonesian",
        max_vram_gb=1.0,
    )
    low_ids = [m.model_id for m in low_vram_models]
    assert "classical_nb" in low_ids
    assert "classical_logistic" in low_ids
    # Transformer models requiring >= 4GB VRAM must be filtered out under 1GB budget
    assert "transformer_bert" not in low_ids

    # With 8GB budget, transformer models are included
    high_vram_models = registry.get_compatible_models(
        task="classification",
        language="indonesian",
        max_vram_gb=8.0,
    )
    high_ids = [m.model_id for m in high_vram_models]
    assert "transformer_bert" in high_ids
    assert "classical_nb" in high_ids
