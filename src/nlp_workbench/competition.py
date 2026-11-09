"""
Gemastik Data Mining Competition Runner
Handles unlabelled test set loading, inference, validation, and submission file generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class GemastikRunner:
    """
    Runner for Gemastik Data Mining and competitive NLP tasks.
    Guarantees strict row preservation, ID alignment, prediction formatting,
    and submission file verification.
    """

    def __init__(
        self,
        id_column: str = "id",
        target_column: str = "target",
        task: str = "classification",
    ):
        self.id_column = id_column
        self.target_column = target_column
        self.task = task

    def prepare_test_dataset(
        self,
        test_source: Union[str, Path, pd.DataFrame],
        expected_id_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Loads and validates the unlabelled competition test set.
        """
        id_col = expected_id_col or self.id_column

        if isinstance(test_source, (str, Path)):
            path = Path(test_source)
            if not path.exists():
                raise FileNotFoundError(f"Competition test file not found: {path.resolve()}")
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix in [".tsv", ".txt"]:
                df = pd.read_csv(path, sep="\t")
            elif path.suffix in [".jsonl", ".ndjson"]:
                df = pd.read_json(path, lines=True)
            else:
                df = pd.read_csv(path)
        elif isinstance(test_source, pd.DataFrame):
            df = test_source.copy()
        else:
            raise TypeError(f"Unsupported test source type: {type(test_source)}")

        if len(df) == 0:
            raise ValueError("Competition test dataset is empty.")

        # Ensure ID column exists; if not present, create standard 0-indexed ID
        if id_col not in df.columns:
            df[id_col] = np.arange(len(df))

        return df

    def generate_predictions(
        self,
        adapter: Any,
        test_df: pd.DataFrame,
        text_column: str = "text",
        text_b_column: Optional[str] = None,
        ensemble_adapters: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Generates predictions for the test dataset using single adapter or ensemble.
        Passes full test_df to preserve multi-column features, and aligns class universes for OOF ensembling.
        """
        adapters = ensemble_adapters if ensemble_adapters else [adapter]

        if text_column not in test_df.columns:
            raise KeyError(f"Text column '{text_column}' not found in test dataframe.")

        all_preds = []
        all_probs = []

        for model in adapters:
            # 1. Attempt predict_proba for classification
            if hasattr(model, "predict_proba") and self.task == "classification":
                try:
                    probs = model.predict_proba(test_df)
                    if probs is not None:
                        all_probs.append(probs)
                        continue
                except Exception:
                    try:
                        probs = model.predict_proba(test_df[text_column].tolist())
                        if probs is not None:
                            all_probs.append(probs)
                            continue
                    except Exception:
                        pass

            # 2. Standard predict
            try:
                preds = model.predict(test_df)
            except Exception:
                if text_b_column and text_b_column in test_df.columns:
                    preds = model.predict(test_df[text_column].tolist(), test_df[text_b_column].tolist())
                else:
                    preds = model.predict(test_df[text_column].tolist())
            all_preds.append(preds)

        # Ensembling classification via aligned probability averaging
        if all_probs and self.task == "classification":
            all_classes_set = set()
            for m in adapters:
                if hasattr(m, "classes_") and m.classes_ is not None:
                    all_classes_set.update(list(m.classes_))

            if all_classes_set:
                global_classes = np.array(sorted(list(all_classes_set)))
                global_class_to_idx = {c: i for i, c in enumerate(global_classes)}
                aligned_probs_list = []
                for m, p in zip(adapters, all_probs):
                    if hasattr(m, "classes_") and m.classes_ is not None and len(m.classes_) == p.shape[1]:
                        aligned = np.zeros((len(test_df), len(global_classes)), dtype=float)
                        for local_idx, c in enumerate(m.classes_):
                            aligned[:, global_class_to_idx[c]] = p[:, local_idx]
                        aligned_probs_list.append(aligned)
                    else:
                        aligned_probs_list.append(p)
                mean_probs = np.mean(aligned_probs_list, axis=0)
                class_indices = np.argmax(mean_probs, axis=1)
                return global_classes[class_indices]
            else:
                mean_probs = np.mean(all_probs, axis=0)
                return np.argmax(mean_probs, axis=1)

        if len(all_preds) == 1:
            return np.array(all_preds[0])

        # Ensembling multiple hard predictions
        if self.task == "regression":
            return np.mean(all_preds, axis=0)
        else:
            # Majority vote
            arr = np.array(all_preds)  # shape (n_models, n_samples)
            from scipy import stats
            mode_result = stats.mode(arr, axis=0, keepdims=False)
            return mode_result[0]

    def create_submission(
        self,
        test_df: pd.DataFrame,
        predictions: Union[np.ndarray, List[Any]],
        output_path: Union[str, Path] = "submission.csv",
        id_column: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates and strictly validates the final competition submission DataFrame and CSV.
        """
        id_col = id_column or self.id_column
        target_col = target_column or self.target_column

        if len(test_df) != len(predictions):
            raise ValueError(
                f"Prediction count mismatch: test set has {len(test_df)} rows, "
                f"but received {len(predictions)} predictions."
            )

        sub_df = pd.DataFrame({
            id_col: test_df[id_col].values,
            target_col: predictions,
        })

        # Strict Validation Checks
        # 1. No NaNs or nulls in predictions
        nan_count = sub_df[target_col].isna().sum()
        if nan_count > 0:
            raise ValueError(f"Submission contains {nan_count} NaN/null values in '{target_col}'.")

        # 2. Row count preservation
        if len(sub_df) != len(test_df):
            raise ValueError(f"Row count altered: original {len(test_df)} vs submission {len(sub_df)}")

        # 3. ID match check
        if not (sub_df[id_col].values == test_df[id_col].values).all():
            raise ValueError("ID alignment mismatch between test set and submission file.")

        # Save to CSV
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(out_p, index=False)

        summary = {
            "submission_path": str(out_p.resolve()),
            "row_count": len(sub_df),
            "columns": list(sub_df.columns),
            "id_column": id_col,
            "target_column": target_col,
            "sample_head": sub_df.head(5).to_dict(orient="records"),
            "target_distribution": (
                sub_df[target_col].value_counts().to_dict()
                if self.task == "classification"
                else {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions)),
                }
            ),
        }
        return summary
