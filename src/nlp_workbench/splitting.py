"""
Dataset Splitting and Data Leakage Prevention module.
Supports Stratified, Group, Time-based, and K-Fold splits with comprehensive leakage auditing.
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
        self.strategy = strategy.lower()
        self.n_splits = n_splits
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
        task: str = "classification",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into 3 partitions: train_df, val_df, test_df.
        Ensures strict row preservation and zero leakage.
        """
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
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """Yields train and validation indices for each cross-validation fold."""
        splitter = self.get_cv_splitter(
            task="classification" if self.strategy == "stratified" else "regression",
            n_splits=self.n_splits,
            random_state=self.random_state,
            group_column=group_col,
        )
        y = df[target_col].values if (target_col and target_col in df.columns) else None
        groups = df[group_col].values if (group_col and group_col in df.columns) else None

        if group_col and groups is not None:
            return splitter.split(df, y, groups=groups)
        elif self.strategy == "stratified" and y is not None:
            return splitter.split(df, y)
        else:
            return splitter.split(df)

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
        if time_column and time_column in df.columns:
            sorted_df = df.sort_values(by=time_column).reset_index(drop=True)
            split_idx = int(len(sorted_df) * (1.0 - test_size))
            train_df = sorted_df.iloc[:split_idx].copy().reset_index(drop=True)
            test_df = sorted_df.iloc[split_idx:].copy().reset_index(drop=True)
            return train_df, test_df

        # 2. Group-based split (prevents user/document contamination)
        if group_column and group_column in df.columns:
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
    ):
        """Returns the appropriate scikit-learn cross-validation splitter."""
        if group_column:
            return GroupKFold(n_splits=n_splits)
        if task == "classification":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


class LeakageAuditor:
    """
    Audits partitions for data leakage: duplicate texts, ID overlap, and group overlap.
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
        Investigates exact duplicate texts, ID overlap, and group overlap across splits.
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
