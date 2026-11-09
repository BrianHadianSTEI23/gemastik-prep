"""
Tests for multi-dataset Information Retrieval architecture.
Verifies separate corpus, queries, and qrels loading, custom column names,
query-level splitting, qrels preservation, and query deduplication.
"""

import pytest
import pandas as pd
from pathlib import Path

from nlp_workbench.data import DataLoader, RetrievalDatasetBundle
from nlp_workbench.splitting import DatasetSplitter
from nlp_workbench.tasks.retrieval import RetrievalAdapter


def test_load_retrieval_bundle_from_separate_files(tmp_path):
    """DataLoader.load_retrieval_bundle must load and link 3 separate files with custom column names."""
    corpus_df = pd.DataFrame({
        "my_doc_id": ["D1", "D2", "D3", "D4"],
        "my_doc_text": ["First document on AI", "Second document on NLP", "Third doc on databases", "Fourth doc on UI"],
    })
    queries_df = pd.DataFrame({
        "my_query_id": ["Q1", "Q2"],
        "my_query_text": ["machine learning algorithms", "relational database design"],
    })
    qrels_df = pd.DataFrame({
        "my_query_id": ["Q1", "Q1", "Q2"],
        "my_doc_id": ["D1", "D2", "D3"],
        "my_relevance": [2, 1, 3],
    })

    c_path = tmp_path / "corpus.csv"
    q_path = tmp_path / "queries.csv"
    r_path = tmp_path / "qrels.csv"

    corpus_df.to_csv(c_path, index=False)
    queries_df.to_csv(q_path, index=False)
    qrels_df.to_csv(r_path, index=False)

    bundle = DataLoader.load_retrieval_bundle(
        corpus_path=c_path,
        queries_path=q_path,
        qrels_path=r_path,
        doc_id_col="my_doc_id",
        doc_text_col="my_doc_text",
        query_id_col="my_query_id",
        query_text_col="my_query_text",
        relevance_col="my_relevance",
    )

    assert isinstance(bundle, RetrievalDatasetBundle)
    assert len(bundle.corpus_df) == 4
    assert len(bundle.queries_df) == 2
    assert bundle.qrels is not None
    assert bundle.qrels["Q1"]["D1"] == 2
    assert bundle.qrels["Q1"]["D2"] == 1
    assert bundle.qrels["Q2"]["D3"] == 3


def test_retrieval_query_level_splitting():
    """Splitting a RetrievalDatasetBundle partitions queries and qrels while keeping corpus shared."""
    corpus_df = pd.DataFrame({
        "doc_id": [f"D{i}" for i in range(10)],
        "document": [f"Document text {i}" for i in range(10)],
    })
    queries_df = pd.DataFrame({
        "query_id": [f"Q{i}" for i in range(10)],
        "query": [f"Query search {i}" for i in range(10)],
    })
    qrels = {f"Q{i}": {f"D{i}": 1} for i in range(10)}

    bundle = RetrievalDatasetBundle(
        corpus_df=corpus_df,
        queries_df=queries_df,
        qrels=qrels,
    )

    splitter = DatasetSplitter(test_size=0.2, val_size=0.2, random_state=42)
    train_b, val_b, test_b = splitter.split(bundle)

    # Corpus must be intact across all splits
    assert len(train_b.corpus_df) == 10
    assert len(val_b.corpus_df) == 10
    assert len(test_b.corpus_df) == 10

    # Queries partitioned with zero leakage
    train_qids = set(train_b.queries_df["query_id"])
    val_qids = set(val_b.queries_df["query_id"])
    test_qids = set(test_b.queries_df["query_id"])

    assert len(train_qids.intersection(val_qids)) == 0
    assert len(train_qids.intersection(test_qids)) == 0
    assert len(val_qids.intersection(test_qids)) == 0
    assert len(train_qids) + len(val_qids) + len(test_qids) == 10

    # Qrels correctly partitioned to match queries
    assert set(train_b.qrels.keys()) == train_qids
    assert set(val_b.qrels.keys()) == val_qids
    assert set(test_b.qrels.keys()) == test_qids


def test_retrieval_query_deduplication():
    """RetrievalAdapter.predict must deduplicate queries so identical queries are encoded once."""
    corpus_df = pd.DataFrame({
        "doc_id": ["D1", "D2"],
        "document": ["artificial intelligence", "database normalization"],
    })
    # Same query appearing multiple times (e.g., paired with different docs)
    queries_df = pd.DataFrame({
        "query_id": ["Q1_a", "Q1_b", "Q2_a"],
        "query": ["artificial intelligence", "artificial intelligence", "database normalization"],
    })

    adapter = RetrievalAdapter(model_id="ir_bm25", approach="bm25")
    adapter.index_corpus(corpus_df)

    ranked = adapter.predict(queries_df)
    assert len(ranked) == 3
    assert ranked["Q1_a"] == ranked["Q1_b"]  # Identical results for identical queries
