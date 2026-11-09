"""
Information Retrieval (IR) Adapter.
Supports BM25 lexical search, Dense Bi-Encoder semantic search, and Hybrid reranking with MRR and NDCG@K metrics.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import joblib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .base import BaseTaskAdapter
from ..evaluation import evaluate_retrieval
from ..hardware import HardwareManager

class RetrievalAdapter(BaseTaskAdapter):
    """
    Information Retrieval Adapter comparing BM25, Dense Semantic Retrieval, and Hybrid Reranking.
    """
    def __init__(
        self,
        model_id: str,
        approach: str = "bm25",
        dense_checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_checkpoint: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
        top_k: int = 5,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.approach = approach.lower().strip()
        self.dense_checkpoint = dense_checkpoint
        self.reranker_checkpoint = reranker_checkpoint
        self.device = HardwareManager.inspect(device).get_torch_device()
        self.top_k = top_k
        self.corpus_df: Optional[pd.DataFrame] = None
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.dense_encoder = None
        self.reranker = None
        self.is_trained = False

    def index_corpus(
        self,
        corpus_df: Union[pd.DataFrame, List[str]],
        doc_id_col: str = "doc_id",
        text_col: str = "document",
    ):
        """Indexes corpus documents for retrieval."""
        if isinstance(corpus_df, list):
            self.corpus_df = pd.DataFrame({
                doc_id_col: [f"doc_{i}" for i in range(len(corpus_df))],
                text_col: [str(x) for x in corpus_df],
                "corpus_id": list(range(len(corpus_df))),
            })
        elif isinstance(corpus_df, pd.DataFrame):
            self.corpus_df = corpus_df.copy().reset_index(drop=True)
            if "corpus_id" not in self.corpus_df.columns:
                self.corpus_df["corpus_id"] = list(range(len(self.corpus_df)))
        else:
            raise TypeError(f"Unsupported corpus type: {type(corpus_df)}")

        self.doc_id_col = doc_id_col
        self.text_col = text_col

        corpus_texts = self.corpus_df[self.text_col].astype(str).tolist()

        if self.approach in ["bm25", "hybrid"]:
            tokenized = [doc.lower().split() for doc in corpus_texts]
            self.bm25 = BM25Okapi(tokenized)

        if self.approach in ["dense", "hybrid"]:
            from sentence_transformers import SentenceTransformer
            self.dense_encoder = SentenceTransformer(self.dense_checkpoint, device=str(self.device))
            self.corpus_embeddings = self.dense_encoder.encode(
                corpus_texts, batch_size=16, show_progress_bar=False, device=str(self.device)
            )

        self.is_trained = True

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """Indexes corpus from train_df if provided."""
        doc_col = kwargs.get("corpus_text_column") or kwargs.get("text_col") or "document"
        id_col = kwargs.get("corpus_id_column") or kwargs.get("id_col") or "doc_id"
        if doc_col in train_df.columns:
            self.index_corpus(train_df, doc_id_col=id_col, text_col=doc_col)
        return {
            "model_id": self.model_id,
            "approach": self.approach,
            "status": f"Retrieval adapter initialized with approach='{self.approach}'.",
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Returns list of top_k retrieved document records for a single query."""
        if not self.is_trained or self.corpus_df is None:
            raise RuntimeError("Corpus is not indexed yet.")

        if self.approach == "bm25":
            tokens = query.lower().split()
            scores = self.bm25.get_scores(tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]
        elif self.approach == "dense":
            q_emb = self.dense_encoder.encode([query], show_progress_bar=False, device=str(self.device))
            norm_c = self.corpus_embeddings / (np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True) + 1e-9)
            norm_q = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
            scores = (norm_c * norm_q).sum(axis=1)
            top_indices = np.argsort(scores)[::-1][:top_k]
        else:  # Hybrid + Cross-Encoder Reranker
            tokens = query.lower().split()
            bm25_raw = np.array(self.bm25.get_scores(tokens))
            bm25_norm = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-9)

            q_emb = self.dense_encoder.encode([query], show_progress_bar=False, device=str(self.device))
            norm_c = self.corpus_embeddings / (np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True) + 1e-9)
            norm_q = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
            dense_norm = (norm_c * norm_q).sum(axis=1)

            hybrid_scores = 0.5 * bm25_norm + 0.5 * dense_norm
            candidate_k = min(len(self.corpus_df), max(top_k * 4, 20))
            candidate_indices = np.argsort(hybrid_scores)[::-1][:candidate_k]

            if self.reranker_checkpoint:
                if self.reranker is None:
                    try:
                        from sentence_transformers import CrossEncoder
                        self.reranker = CrossEncoder(self.reranker_checkpoint, device=str(self.device))
                    except Exception:
                        self.reranker = False
                if self.reranker:
                    pairs = [[query, str(self.corpus_df.iloc[idx][self.text_col])] for idx in candidate_indices]
                    rerank_scores = self.reranker.predict(pairs)
                    reranked_order = np.argsort(rerank_scores)[::-1]
                    top_indices = [candidate_indices[i] for i in reranked_order[:top_k]]
                    score_map = {candidate_indices[i]: float(rerank_scores[i]) for i in range(len(candidate_indices))}
                else:
                    top_indices = candidate_indices[:top_k]
                    score_map = {idx: float(hybrid_scores[idx]) for idx in top_indices}
            else:
                top_indices = candidate_indices[:top_k]
                score_map = {idx: float(hybrid_scores[idx]) for idx in top_indices}

        results = []
        for idx in top_indices:
            row = self.corpus_df.iloc[idx]
            sc = score_map[idx] if self.approach not in ["bm25", "dense"] else float(scores[idx])
            results.append({
                "doc_id": row[self.doc_id_col],
                "corpus_id": int(row.get("corpus_id", idx)),
                "document": str(row[self.text_col]),
                "score": round(sc, 4),
            })
        return results

    def retrieve_ids(self, query: str, top_k: int = 5) -> List[str]:
        return [r["doc_id"] for r in self.retrieve(query, top_k=top_k)]

    def predict(self, df: Any, **kwargs) -> Dict[str, List[str]]:
        """Retrieves top documents for all queries in df or RetrievalDatasetBundle."""
        from ..data import RetrievalDatasetBundle
        if isinstance(df, RetrievalDatasetBundle):
            q_df = df.queries_df
            q_col = kwargs.get("query_column", df.query_text_col)
            qid_col = kwargs.get("query_id_column", df.query_id_col)
        elif isinstance(df, pd.DataFrame):
            q_df = df
            q_col = kwargs.get("query_column", "query")
            qid_col = kwargs.get("query_id_column", "query_id")
        else:
            raise TypeError(f"Unsupported input type for retrieval predict: {type(df)}")

        # Deduplicate queries to avoid redundant retrieval / encoding
        query_text_to_ids = {}
        id_to_text = {}
        for idx, row in q_df.iterrows():
            qid = str(row.get(qid_col, f"Q_{idx}"))
            q_text = str(row.get(q_col, row.iloc[0]))
            id_to_text[qid] = q_text
            if q_text not in query_text_to_ids:
                query_text_to_ids[q_text] = []
            query_text_to_ids[q_text].append(qid)

        unique_results = {}
        for q_text in query_text_to_ids.keys():
            unique_results[q_text] = self.retrieve_ids(q_text, top_k=self.top_k)

        ranked_results = {}
        for qid, q_text in id_to_text.items():
            ranked_results[qid] = unique_results[q_text]
        return ranked_results

    def evaluate(self, df: Any, y_true: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """Evaluates retrieval against ground truth relevance judgements."""
        from ..data import RetrievalDatasetBundle
        t0 = time.time()
        ranked = self.predict(df, **kwargs)
        n_queries = len(ranked)
        latency_ms = ((time.time() - t0) / max(1, n_queries)) * 1000

        # Resolve qrels
        qrels = None
        if isinstance(y_true, dict):
            qrels = y_true
        elif "qrels" in kwargs and isinstance(kwargs["qrels"], dict):
            qrels = kwargs["qrels"]
        elif isinstance(df, RetrievalDatasetBundle) and df.qrels:
            qrels = df.qrels
        elif isinstance(df, (pd.DataFrame, RetrievalDatasetBundle)):
            q_df = df.queries_df if isinstance(df, RetrievalDatasetBundle) else df
            qid_col = kwargs.get("query_id_column", getattr(df, "query_id_col", "query_id"))
            did_col = kwargs.get("corpus_id_column", getattr(df, "doc_id_col", "doc_id"))
            rel_col = kwargs.get("relevance_column", getattr(df, "relevance_col", "relevance"))
            if qid_col in q_df.columns and did_col in q_df.columns:
                qrels = {}
                for _, row in q_df.iterrows():
                    qid = str(row[qid_col])
                    did = str(row[did_col])
                    try:
                        rel = int(float(row.get(rel_col, 1))) if rel_col in q_df.columns else 1
                    except (ValueError, TypeError):
                        rel = 1
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][did] = rel

        if not qrels:
            return {"Status": "Retrieval completed. Ground truth relevance judgments (qrels) absent."}

        metrics = evaluate_retrieval(qrels, ranked, k=self.top_k)
        metrics["Latency/Sample (ms)"] = round(latency_ms, 3)
        return metrics

    def export(self, export_path: Union[str, Path]) -> Path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": self.model_id,
            "approach": self.approach,
            "dense_checkpoint": self.dense_checkpoint,
            "reranker_checkpoint": self.reranker_checkpoint,
            "top_k": self.top_k,
            "corpus_df": self.corpus_df,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "RetrievalAdapter":
        bundle = joblib.load(artifact_path)
        adapter = cls(
            model_id=bundle["model_id"],
            approach=bundle["approach"],
            dense_checkpoint=bundle["dense_checkpoint"],
            reranker_checkpoint=bundle.get("reranker_checkpoint"),
            top_k=bundle["top_k"],
        )
        if bundle["corpus_df"] is not None:
            adapter.index_corpus(bundle["corpus_df"])
        return adapter
