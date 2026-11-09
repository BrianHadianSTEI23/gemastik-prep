"""
Dataset Splitting and Data Leakage Prevention module.
Supports Stratified, Group, Time-based, and K-Fold splits with comprehensive 3-way leakage auditing.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Iterable
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    GroupKFold,
    GroupShuffleSplit,
)

from .data import RetrievalDatasetBundle


class DatasetSplitter:
    """
    Splitter supporting Stratified, Group, Time-based, and K-Fold splits.
    Usable both as an instance with configured strategy or via static helper methods.
    """

    def __init__(
        self,
        strategy: str = "stratified",
        n_splits: int = 5,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        self.strategy = strategy.lower().strip()
        self.n_splits = n_splits
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(
        self,
        df: Union[pd.DataFrame, RetrievalDatasetBundle],
        target_col: Optional[str] = None,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
        task: str = "classification",
    ) -> Tuple[Union[pd.DataFrame, RetrievalDatasetBundle], Union[pd.DataFrame, RetrievalDatasetBundle], Union[pd.DataFrame, RetrievalDatasetBundle]]:
        """
        Splits dataset into 3 partitions: train, val, and held-out test.
        Ensures strict row preservation, zero leakage, and support for RetrievalDatasetBundle.
        """
        if isinstance(df, RetrievalDatasetBundle):
            # Query-level splitting to prevent query leakage across partitions
            unique_qids = df.queries_df[df.query_id_col].unique()
            if len(unique_qids) >= 3:
                qid_df = pd.DataFrame({df.query_id_col: unique_qids})
                train_qids_df, val_qids_df, test_qids_df = self.split(
                    df=qid_df,
                    task="regression",
                )
                train_set = set(train_qids_df[df.query_id_col])
                val_set = set(val_qids_df[df.query_id_col])
                test_set = set(test_qids_df[df.query_id_col])

                q_train = df.queries_df[df.queries_df[df.query_id_col].isin(train_set)].reset_index(drop=True)
                q_val = df.queries_df[df.queries_df[df.query_id_col].isin(val_set)].reset_index(drop=True)
                q_test = df.queries_df[df.queries_df[df.query_id_col].isin(test_set)].reset_index(drop=True)
            else:
                q_train, q_val, q_test = self.split(
                    df=df.queries_df,
                    target_col=target_col if target_col in df.queries_df.columns else None,
                    group_col=group_col,
                    time_col=time_col,
                    task="classification" if (target_col and target_col in df.queries_df.columns) else "regression",
                )

            # Partition qrels dict if present
            qrels_train = None
            qrels_val = None
            qrels_test = None
            if df.qrels:
                tr_ids = set(q_train[df.query_id_col].astype(str))
                vl_ids = set(q_val[df.query_id_col].astype(str))
                ts_ids = set(q_test[df.query_id_col].astype(str))
                qrels_train = {qid: rels for qid, rels in df.qrels.items() if qid in tr_ids}
                qrels_val = {qid: rels for qid, rels in df.qrels.items() if qid in vl_ids}
                qrels_test = {qid: rels for qid, rels in df.qrels.items() if qid in ts_ids}

            train_bundle = RetrievalDatasetBundle(
                corpus_df=df.corpus_df.copy(),
                queries_df=q_train,
                qrels=qrels_train,
                doc_id_col=df.doc_id_col,
                doc_text_col=df.doc_text_col,
                query_id_col=df.query_id_col,
                query_text_col=df.query_text_col,
                relevance_col=df.relevance_col,
            )
            val_bundle = RetrievalDatasetBundle(
                corpus_df=df.corpus_df.copy(),
                queries_df=q_val,
                qrels=qrels_val,
                doc_id_col=df.doc_id_col,
                doc_text_col=df.doc_text_col,
                query_id_col=df.query_id_col,
                query_text_col=df.query_text_col,
                relevance_col=df.relevance_col,
            )
            test_bundle = RetrievalDatasetBundle(
                corpus_df=df.corpus_df.copy(),
                queries_df=q_test,
                qrels=qrels_test,
                doc_id_col=df.doc_id_col,
                doc_text_col=df.doc_text_col,
                query_id_col=df.query_id_col,
                query_text_col=df.query_text_col,
                relevance_col=df.relevance_col,
            )
            return train_bundle, val_bundle, test_bundle

        if len(df) < 3:
            raise ValueError(f"Dataset has {len(df)} rows, cannot split into train/val/test.")

        # First split into train_val and test
        train_val_df, test_df = self.split_train_test(
            df=df,
            task=task,
            target_column=target_col,
            test_size=self.test_size,
            random_state=self.random_state,
            group_column=group_col,
            time_column=time_col,
        )

        # Second split train_val into train and val
        rel_val_size = self.val_size / max(1e-5, (1.0 - self.test_size))
        train_df, val_df = self.split_train_test(
            df=train_val_df,
            task=task,
            target_column=target_col,
            test_size=rel_val_size,
            random_state=self.random_state,
            group_column=group_col,
            time_column=time_col,
        )

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def get_cv_folds(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """Yields train and validation indices for each cross-validation fold."""
        splitter = self.get_cv_splitter(
            task="classification" if self.strategy == "stratified" else "regression",
            n_splits=self.n_splits,
            random_state=self.random_state,
            group_column=group_col,
            time_column=time_col,
            strategy=self.strategy,
        )
        if self.strategy == "time" or time_col:
            # Chronological expanding window without shuffling
            if time_col and time_col in df.columns:
                sorted_df = df.sort_values(by=time_col).reset_index()
                orig_indices = sorted_df["index"].values
                for train_idx, val_idx in splitter.split(sorted_df):
                    yield orig_indices[train_idx], orig_indices[val_idx]
            else:
                for train_idx, val_idx in splitter.split(df):
                    yield train_idx, val_idx
            return

        y = df[target_col].values if (target_col and target_col in df.columns) else None
        groups = df[group_col].values if (group_col and group_col in df.columns) else None

        if group_col and groups is not None:
            yield from splitter.split(df, y, groups=groups)
        elif self.strategy == "stratified" and y is not None:
            yield from splitter.split(df, y)
        else:
            yield from splitter.split(df)

    @staticmethod
    def split_train_test(
        df: pd.DataFrame,
        task: str = "classification",
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        group_column: Optional[str] = None,
        time_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into untouched train and test partitions.
        Prioritizes time-based or group-based splitting when configured.
        """
        if len(df) < 2:
            raise ValueError(f"Dataset has {len(df)} rows, cannot split.")

        # 1. Time-based split (temporal ordering)
        if time_column:
            if time_column not in df.columns:
                raise ValueError(f"Configured time_column '{time_column}' not found in dataset.")
            sorted_df = df.sort_values(by=time_column).reset_index(drop=True)
            split_idx = int(len(sorted_df) * (1.0 - test_size))
            if split_idx <= 0 or split_idx >= len(sorted_df):
                split_idx = max(1, len(sorted_df) - 1)
            train_df = sorted_df.iloc[:split_idx].copy().reset_index(drop=True)
            test_df = sorted_df.iloc[split_idx:].copy().reset_index(drop=True)
            return train_df, test_df

        # 2. Group-based split (prevents user/document contamination)
        if group_column:
            if group_column not in df.columns:
                raise ValueError(f"Configured group_column '{group_column}' not found in dataset.")
            unique_groups = df[group_column].unique()
            if len(unique_groups) < 2:
                raise ValueError(
                    f"Configured group_column '{group_column}' has only {len(unique_groups)} group. "
                    f"Group splitting requires at least 2 distinct groups."
                )
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(df, groups=df[group_column]))
            train_df = df.iloc[train_idx].copy().reset_index(drop=True)
            test_df = df.iloc[test_idx].copy().reset_index(drop=True)
            return train_df, test_df

        # 3. Stratified split for classification
        stratify_arr = None
        if task == "classification" and target_column and target_column in df.columns:
            y = df[target_column]
            vc = y.value_counts()
            if vc.min() >= 2:
                stratify_arr = y

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arr,
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    @staticmethod
    def get_cv_splitter(
        task: str = "classification",
        n_splits: int = 5,
        random_state: int = 42,
        group_column: Optional[str] = None,
        time_column: Optional[str] = None,
        strategy: Optional[str] = None,
    ):
        """Returns the appropriate scikit-learn cross-validation splitter."""
        strat = (strategy or "").lower().strip()
        if strat == "time" or time_column:
            from sklearn.model_selection import TimeSeriesSplit
            return TimeSeriesSplit(n_splits=n_splits)
        if group_column or strat == "group":
            return GroupKFold(n_splits=n_splits)
        if task == "classification":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


class LeakageAuditor:
    """
    Audits partitions for data leakage across all split combinations:
    duplicate texts, ID overlap, and group overlap across Train/Val, Train/Test, and Val/Test.
    """

    def audit(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_columns: Optional[List[str]] = None,
        text_col: Optional[str] = None,
        id_column: Optional[str] = None,
        id_col: Optional[str] = None,
        group_column: Optional[str] = None,
        group_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.audit_splits(
            train_df=train_df,
            test_df=test_df,
            text_columns=text_columns,
            text_col=text_col,
            id_column=id_column,
            id_col=id_col,
            group_column=group_column,
            group_col=group_col,
        )

    @staticmethod
    def audit_splits(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_columns: Optional[List[str]] = None,
        text_col: Optional[str] = None,
        id_column: Optional[str] = None,
        id_col: Optional[str] = None,
        group_column: Optional[str] = None,
        group_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Investigates exact duplicate texts, ID overlap, and group overlap between two splits.
        """
        cols = text_columns or ([text_col] if text_col else [])
        valid_text_cols = [c for c in cols if c in train_df.columns and c in test_df.columns]

        if valid_text_cols:
            train_texts = set(train_df[valid_text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip())
            test_texts = set(test_df[valid_text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip())
            text_overlap = train_texts.intersection(test_texts)
        else:
            text_overlap = set()

        eff_id_col = id_column or id_col
        id_overlap = set()
        if eff_id_col and eff_id_col in train_df.columns and eff_id_col in test_df.columns:
            id_overlap = set(train_df[eff_id_col].dropna()).intersection(set(test_df[eff_id_col].dropna()))

        eff_group_col = group_column or group_col
        group_overlap = set()
        if eff_group_col and eff_group_col in train_df.columns and eff_group_col in test_df.columns:
            group_overlap = set(train_df[eff_group_col].dropna()).intersection(set(test_df[eff_group_col].dropna()))

        text_leakage = len(text_overlap) > 0
        id_leakage = len(id_overlap) > 0
        group_leakage = len(group_overlap) > 0
        leakage_detected = text_leakage or id_leakage or group_leakage

        return {
            "train_size": len(train_df),
            "test_size": len(test_df),
            "duplicate_text_count": len(text_overlap),
            "duplicate_text_pct": round(len(text_overlap) / max(1, len(test_df)) * 100, 2),
            "text_leakage_detected": text_leakage,
            "id_overlap_count": len(id_overlap),
            "overlapping_id_count": len(id_overlap),
            "id_leakage_detected": id_leakage,
            "group_overlap_count": len(group_overlap),
            "overlapping_group_count": len(group_overlap),
            "group_leakage_detected": group_leakage,
            "leakage_detected": leakage_detected,
        }

    @classmethod
    def audit_all_splits(
        cls,
        train_df: Union[pd.DataFrame, RetrievalDatasetBundle],
        val_df: Union[pd.DataFrame, RetrievalDatasetBundle],
        test_df: Union[pd.DataFrame, RetrievalDatasetBundle],
        text_columns: Optional[List[str]] = None,
        text_col: Optional[str] = None,
        id_column: Optional[str] = None,
        group_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Audits all 3 split pairs: Train/Val, Train/Test, and Val/Test.
        """
        # Unwrap RetrievalDatasetBundle to queries_df if needed
        tr = train_df.queries_df if isinstance(train_df, RetrievalDatasetBundle) else train_df
        va = val_df.queries_df if isinstance(val_df, RetrievalDatasetBundle) else val_df
        te = test_df.queries_df if isinstance(test_df, RetrievalDatasetBundle) else test_df

        train_val = cls.audit_splits(
            train_df=tr,
            test_df=va,
            text_columns=text_columns,
            text_col=text_col,
            id_column=id_column,
            group_column=group_column,
        )
        train_test = cls.audit_splits(
            train_df=tr,
            test_df=te,
            text_columns=text_columns,
            text_col=text_col,
            id_column=id_column,
            group_column=group_column,
        )
        val_test = cls.audit_splits(
            train_df=va,
            test_df=te,
            text_columns=text_columns,
            text_col=text_col,
            id_column=id_column,
            group_column=group_column,
        )

        leakage_detected = (
            train_val["leakage_detected"]
            or train_test["leakage_detected"]
            or val_test["leakage_detected"]
        )

        total_dup = (
            train_val["duplicate_text_count"]
            + train_test["duplicate_text_count"]
            + val_test["duplicate_text_count"]
        )
        total_id_overlap = (
            train_val["id_overlap_count"]
            + train_test["id_overlap_count"]
            + val_test["id_overlap_count"]
        )
        total_group_overlap = (
            train_val["group_overlap_count"]
            + train_test["group_overlap_count"]
            + val_test["group_overlap_count"]
        )

        return {
            "leakage_detected": leakage_detected,
            "train_val": train_val,
            "train_test": train_test,
            "val_test": val_test,
            "total_duplicate_text_count": total_dup,
            "total_id_overlap_count": total_id_overlap,
            "total_group_overlap_count": total_group_overlap,
            "text_leakage_detected": (
                train_val["text_leakage_detected"]
                or train_test["text_leakage_detected"]
                or val_test["text_leakage_detected"]
            ),
            "id_leakage_detected": (
                train_val["id_leakage_detected"]
                or train_test["id_leakage_detected"]
                or val_test["id_leakage_detected"]
            ),
            "group_leakage_detected": (
                train_val["group_leakage_detected"]
                or train_test["group_leakage_detected"]
                or val_test["group_leakage_detected"]
            ),
        }

    @classmethod
    def enforce_leakage_policy(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_columns: Optional[List[str]] = None,
        id_column: Optional[str] = None,
        group_column: Optional[str] = None,
        deduplicate_texts: bool = False,
        raise_on_leakage: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Enforces leakage policy across train, val, and test partitions.
        If deduplicate_texts is True, removes leaking rows from val and test.
        If prohibited leakage remains and raise_on_leakage is True, raises ValueError.
        Never executes a silent pass!
        """
        audit_res = cls.audit_all_splits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            text_columns=text_columns,
            id_column=id_column,
            group_column=group_column,
        )

        if not audit_res["leakage_detected"]:
            return train_df, val_df, test_df, audit_res

        # If deduplication requested, purge duplicates from val and test
        if deduplicate_texts and audit_res["text_leakage_detected"]:
            cols = [c for c in (text_columns or []) if c in train_df.columns]
            if cols:
                tr_texts = set(train_df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip())
                val_mask = ~val_df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip().isin(tr_texts)
                val_df = val_df[val_mask].reset_index(drop=True)

                all_seen = tr_texts.union(set(val_df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()))
                test_mask = ~test_df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip().isin(all_seen)
                test_df = test_df[test_mask].reset_index(drop=True)

                # Re-audit after deduplication
                audit_res = cls.audit_all_splits(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    text_columns=text_columns,
                    id_column=id_column,
                    group_column=group_column,
                )

        if audit_res["leakage_detected"] and raise_on_leakage:
            details = []
            if audit_res["train_val"]["leakage_detected"]:
                details.append(f"Train/Val: {audit_res['train_val']['duplicate_text_count']} duplicate texts, {audit_res['train_val']['id_overlap_count']} overlapping IDs, {audit_res['train_val']['group_overlap_count']} overlapping groups")
            if audit_res["train_test"]["leakage_detected"]:
                details.append(f"Train/Test: {audit_res['train_test']['duplicate_text_count']} duplicate texts, {audit_res['train_test']['id_overlap_count']} overlapping IDs, {audit_res['train_test']['group_overlap_count']} overlapping groups")
            if audit_res["val_test"]["leakage_detected"]:
                details.append(f"Val/Test: {audit_res['val_test']['duplicate_text_count']} duplicate texts, {audit_res['val_test']['id_overlap_count']} overlapping IDs, {audit_res['val_test']['group_overlap_count']} overlapping groups")
            raise ValueError(
                f"Data leakage detected across dataset splits violating experimental integrity policy: "
                f"{'; '.join(details)}. Set deduplicate_texts=True to purge or fix data partitioning."
            )

        return train_df, val_df, test_df, audit_res
