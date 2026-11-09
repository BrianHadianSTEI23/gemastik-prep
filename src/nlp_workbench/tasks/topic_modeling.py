"""
Topic Modeling Adapter.
Supports LDA, NMF, and c-TF-IDF Embedding Clustering with Topic Diversity metrics.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans

from .base import BaseTaskAdapter
from ..preprocessing import TextCleaner
from ..evaluation import evaluate_topic_modeling
from ..hardware import HardwareManager

class TopicModelingAdapter(BaseTaskAdapter):
    """
    Topic Modeling Adapter implementing probabilistic LDA, matrix factorization NMF,
    and c-TF-IDF Embedding Clustering.
    """
    def __init__(
        self,
        model_id: str,
        approach: str = "nmf",
        n_topics: int = 5,
        top_words_count: int = 10,
        random_state: int = 42,
        embedding_checkpoint: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        language: str = "indonesian",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.approach = approach.lower().strip()
        self.n_topics = n_topics
        self.top_words_count = top_words_count
        self.random_state = random_state
        self.embedding_checkpoint = embedding_checkpoint
        self.language = language.lower()
        self.text_columns: List[str] = kwargs.get("text_columns", ["content"])
        self.cleaner = kwargs.get("cleaner") or TextCleaner()
        self.vectorizer = None
        self.model = None
        self.topic_keywords: List[List[str]] = []
        self.is_trained = False

    def _prepare_text(self, df: Union[pd.DataFrame, pd.Series, List[str], np.ndarray], **kwargs) -> List[str]:
        if isinstance(df, (list, tuple, np.ndarray, pd.Series)):
            return [str(x) for x in df]

        text_col = kwargs.get("text_column") or kwargs.get("text_col")
        if text_col and text_col in df.columns:
            return self.cleaner.clean_series(df[text_col].fillna("").astype(str)).tolist()

        valid_cols = [c for c in self.text_columns if c in df.columns]
        if not valid_cols:
            if "cleaned_text" in df.columns:
                return df["cleaned_text"].tolist()
            if "content" in df.columns:
                return df["content"].tolist()
            if "text" in df.columns:
                return self.cleaner.clean_series(df["text"].fillna("").astype(str)).tolist()
            raise KeyError(f"None of {self.text_columns} exist in DataFrame.")
        combined = df[valid_cols].fillna("").astype(str).agg(" ".join, axis=1)
        return self.cleaner.clean_series(combined).tolist()

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        t0 = time.time()
        texts = self._prepare_text(train_df, **kwargs)

        sw = "english" if self.language in ["english", "en"] else None
        if "lda" in self.approach:
            self.vectorizer = CountVectorizer(max_features=5000, stop_words=sw, min_df=1)
            X = self.vectorizer.fit_transform(texts)
            self.model = LatentDirichletAllocation(n_components=self.n_topics, random_state=self.random_state)
            self.model.fit(X)
            feature_names = self.vectorizer.get_feature_names_out()
            self.topic_keywords = []
            for topic in self.model.components_:
                top_indices = topic.argsort()[:-self.top_words_count - 1:-1]
                self.topic_keywords.append([feature_names[i] for i in top_indices])

        elif "ctfidf" in self.approach or "bertopic" in self.approach:
            # c-TF-IDF Embedding Clustering
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(self.embedding_checkpoint)
            emb = embedder.encode(texts, show_progress_bar=False)
            kmeans = KMeans(n_clusters=self.n_topics, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(emb)
            self.model = kmeans

            # Class-based TF-IDF
            docs_per_cluster = pd.DataFrame({"text": texts, "cluster": cluster_labels}).groupby("cluster")["text"].apply(" ".join)
            c_tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words=sw)
            c_tfidf_mat = c_tfidf_vec.fit_transform(docs_per_cluster)
            c_features = c_tfidf_vec.get_feature_names_out()
            self.topic_keywords = []
            for i in range(len(docs_per_cluster)):
                row_scores = c_tfidf_mat[i].toarray().ravel()
                top_indices = row_scores.argsort()[:-self.top_words_count - 1:-1]
                self.topic_keywords.append([c_features[idx] for idx in top_indices])

        else:  # Default to NMF
            self.vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True, min_df=1, stop_words=sw)
            X = self.vectorizer.fit_transform(texts)
            self.model = NMF(n_components=self.n_topics, random_state=self.random_state)
            self.model.fit(X)
            feature_names = self.vectorizer.get_feature_names_out()
            self.topic_keywords = []
            for topic in self.model.components_:
                top_indices = topic.argsort()[:-self.top_words_count - 1:-1]
                self.topic_keywords.append([feature_names[i] for i in top_indices])

        self.is_trained = True
        train_duration = time.time() - t0

        return {
            "model_id": self.model_id,
            "approach": self.approach,
            "train_duration_s": round(train_duration, 3),
            "n_topics": self.n_topics,
            "topic_keywords": self.topic_keywords,
            "top_words_per_topic": self.topic_keywords,
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Topic model is not fitted yet.")
        texts = self._prepare_text(df)
        if hasattr(self.model, "transform") and self.vectorizer:
            X = self.vectorizer.transform(texts)
            doc_topics = self.model.transform(X)
            return np.argmax(doc_topics, axis=1)
        elif hasattr(self.model, "predict"):
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(self.embedding_checkpoint)
            emb = embedder.encode(texts, show_progress_bar=False)
            return self.model.predict(emb)
        return np.zeros(len(df), dtype=int)

    def evaluate(self, df: pd.DataFrame, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """Evaluates topic diversity and empirical topic coherence on reference documents."""
        texts = None
        if df is not None and len(df) > 0:
            try:
                texts = self._prepare_text(df, **kwargs)
            except Exception:
                texts = None
        metrics = evaluate_topic_modeling(
            self.topic_keywords,
            reference_corpus=texts,
            top_n=self.top_words_count,
        )
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "approach": self.approach,
            "n_topics": self.n_topics,
            "embedding_checkpoint": self.embedding_checkpoint,
            "language": self.language,
            "vectorizer": self.vectorizer,
            "model": self.model,
            "topic_keywords": self.topic_keywords,
            "text_columns": self.text_columns,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "TopicModelingAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(
            model_id=bundle["model_id"],
            approach=bundle["approach"],
            n_topics=bundle["n_topics"],
            embedding_checkpoint=bundle.get("embedding_checkpoint", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            language=bundle.get("language", "indonesian"),
            text_columns=bundle["text_columns"],
        )
        adapter.vectorizer = bundle["vectorizer"]
        adapter.model = bundle["model"]
        adapter.topic_keywords = bundle["topic_keywords"]
        adapter.is_trained = True
        return adapter
