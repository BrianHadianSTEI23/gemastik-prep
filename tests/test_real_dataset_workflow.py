"""
End-to-end Integration Tests on Realistic CSV-based Datasets.
Tests complete pipeline on non-trivial datasets with imperfect separability,
class imbalance, variable text lengths, HTML markup, and URLs for classification and regression.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from nlp_workbench.config import WorkbenchConfig
from nlp_workbench.orchestration import ExperimentRunner
from nlp_workbench.export import ModelExporter
from nlp_workbench.inference import predict_new


@pytest.fixture
def realistic_classification_csv(tmp_path: Path) -> Path:
    """Generates a realistic Indonesian text classification dataset with class imbalance and noise."""
    np.random.seed(42)
    classes = ["Positif", "Netral", "Negatif"]
    # 55% Positif, 30% Netral, 15% Negatif (class imbalance)
    class_probs = [0.55, 0.30, 0.15]
    
    samples = []
    vocab_pos = ["bagus", "hebat", "memuaskan", "senang", "terbaik", "ramah", "cepat", "juara", "rekomendasi"]
    vocab_neu = ["biasa", "cukup", "standar", "rata-rata", "umum", "informasi", "berita", "laporan"]
    vocab_neg = ["buruk", "rusak", "kecewa", "lambat", "mahal", "tidak puas", "parah", "rugi"]
    
    for i in range(90):
        cls = np.random.choice(classes, p=class_probs)
        if cls == "Positif":
            words = list(np.random.choice(vocab_pos, size=np.random.randint(4, 12)))
        elif cls == "Netral":
            words = list(np.random.choice(vocab_neu, size=np.random.randint(3, 8)))
        else:
            words = list(np.random.choice(vocab_neg, size=np.random.randint(5, 14)))
        
        # Add variable length filler words
        filler = ["layanan", "produk", "aplikasi", "sistem", "transaksi", "pengalaman", "hari", "ini"]
        extra = list(np.random.choice(filler, size=np.random.randint(3, 15)))
        words.extend(extra)
        np.random.shuffle(words)
        text = " ".join(words)
        
        # Inject HTML tags or URLs occasionally
        if i % 7 == 0:
            text = f"<b>Pemberitahuan:</b> {text} Silakan baca di https://review-portal.id/{i}"
        elif i % 11 == 0:
            text = f"<p>{text}</p>"
            
        samples.append({
            "review_id": f"REV_{i:04d}",
            "review_text": text,
            "sentiment": cls,
        })
    
    df = pd.DataFrame(samples)
    csv_file = tmp_path / "realistic_reviews.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def realistic_regression_csv(tmp_path: Path) -> Path:
    """Generates a realistic text regression dataset with continuous engagement scores."""
    np.random.seed(123)
    samples = []
    
    keywords = {
        "viral": 25.0,
        "terkini": 15.0,
        "heboh": 20.0,
        "penting": 10.0,
        "ekonomi": 5.0,
        "teknologi": 8.0,
        "update": 4.0,
    }
    
    for i in range(80):
        chosen_keys = np.random.choice(list(keywords.keys()), size=np.random.randint(1, 4), replace=False)
        base_score = sum(keywords[k] for k in chosen_keys)
        noise = np.random.normal(loc=0.0, scale=3.0)
        score = max(1.0, float(base_score + 15.0 + noise))
        
        text = f"Berita {' '.join(chosen_keys)} mengenai perkembangan sektor industri nomor {i} dengan analisis mendalam."
        if i % 5 == 0:
            text = f"<div>{text} Kunjungi https://news.id/{i}</div>"
            
        samples.append({
            "doc_id": f"DOC_{i:04d}",
            "article_content": text,
            "engagement": round(score, 2),
        })
        
    df = pd.DataFrame(samples)
    csv_file = tmp_path / "realistic_articles.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


def test_real_dataset_classification_workflow(realistic_classification_csv: Path, tmp_path: Path):
    """Executes full end-to-end ExperimentRunner pipeline on real imbalanced classification CSV."""
    cfg = WorkbenchConfig(
        task="classification",
        dataset_path=str(realistic_classification_csv),
        text_column="review_text",
        target_column="sentiment",
        id_column="review_id",
        model_ids=["classical_nb", "classical_logistic"],
        output_dir=str(tmp_path / "output_clf"),
        device="cpu",
    )
    
    runner = ExperimentRunner(cfg)
    result = runner.run()
    
    # 1. Validation Leaderboard
    assert not result.leaderboard.empty
    assert len(result.leaderboard) == 2
    assert "Macro-F1" in result.leaderboard.columns
    assert "Validation Score" in result.leaderboard.columns
    
    # 2. Winner Selection
    assert result.best_model_id in ["classical_nb", "classical_logistic"]
    assert result.best_adapter is not None
    
    # 3. Held-out Test Evaluation
    assert "Macro-F1" in result.test_metrics
    assert "Accuracy" in result.test_metrics
    assert "Confusion Matrix" in result.test_metrics
    assert result.test_metrics["Macro-F1"] > 0.0
    
    # 4. Export & Reload Integrity
    assert result.exported_path is not None
    assert result.exported_path.exists()
    
    bundle = ModelExporter.load(result.exported_path)
    pred_res = predict_new(
        texts=["Layanan ini sangat hebat, terbaik dan memuaskan!", "Sangat kecewa buruk sekali rusak"],
        adapter=bundle,
    )
    assert len(pred_res.predictions) == 2
    assert pred_res.predictions[0] == "Positif"


def test_real_dataset_regression_workflow(realistic_regression_csv: Path, tmp_path: Path):
    """Executes full end-to-end ExperimentRunner pipeline on real noisy regression CSV."""
    cfg = WorkbenchConfig(
        task="regression",
        dataset_path=str(realistic_regression_csv),
        text_column="article_content",
        target_column="engagement",
        id_column="doc_id",
        model_ids=["regression_tfidf_ridge"],
        output_dir=str(tmp_path / "output_reg"),
        device="cpu",
    )
    
    runner = ExperimentRunner(cfg)
    result = runner.run()
    
    # 1. Leaderboard minimizes MAE
    assert not result.leaderboard.empty
    assert "MAE" in result.leaderboard.columns
    assert result.leaderboard.iloc[0]["Primary Metric"] == "MAE"
    
    # 2. Winner Selection
    assert result.best_model_id == "regression_tfidf_ridge"
    assert result.best_adapter is not None
    
    # 3. Held-out Test Evaluation
    assert "MAE" in result.test_metrics
    assert "RMSE" in result.test_metrics
    assert result.test_metrics["MAE"] is not None
    assert result.test_metrics["MAE"] > 0.0
    
    # 4. Standalone Reload
    bundle = ModelExporter.load(result.exported_path)
    preds = bundle.predict(pd.DataFrame({"article_content": ["Berita viral heboh terbaru"]}))
    assert len(preds) == 1
    assert isinstance(preds[0], (int, float, np.floating))
