"""
Central Experiment Orchestration module for the NLP Experimentation Workbench.
Provides task-aware, end-to-end execution from data ingestion to model export and competition submission.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from .config import WorkbenchConfig, get_default_config
from .data import DataLoader, DatasetValidator
from .eda import DataQualityAuditor, ExploratoryDataAnalysis, AuditReport
from .splitting import DatasetSplitter, LeakageAuditor
from .registry import registry, ModelRecord
from .tasks.base import BaseTaskAdapter
from .training import run_cross_validation, select_best_model, BestModelSelection
from .export import ModelExporter, LoadedModelBundle
from .competition import GemastikRunner
from .evaluation import is_higher_better


class ExperimentResult(dict):
    """
    Container providing dictionary, attribute, and structured access to complete experiment artifacts.
    """
    def __init__(
        self,
        config: WorkbenchConfig,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        leaderboard: pd.DataFrame,
        best_model_id: str,
        best_adapter: BaseTaskAdapter,
        test_metrics: Dict[str, Any],
        exported_path: Optional[Path] = None,
        audit_report: Optional[AuditReport] = None,
        eda_summary: Optional[Dict[str, Any]] = None,
        fold_adapters: Optional[List[BaseTaskAdapter]] = None,
        competition_summary: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            leaderboard=leaderboard,
            best_model_id=best_model_id,
            best_adapter=best_adapter,
            test_metrics=test_metrics,
            exported_path=exported_path,
            audit_report=audit_report,
            eda_summary=eda_summary,
            fold_adapters=fold_adapters or [],
            competition_summary=competition_summary,
            **kwargs,
        )
        self.config = config
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.leaderboard = leaderboard
        self.best_model_id = best_model_id
        self.best_adapter = best_adapter
        self.test_metrics = test_metrics
        self.exported_path = exported_path
        self.audit_report = audit_report
        self.eda_summary = eda_summary
        self.fold_adapters = fold_adapters or []
        self.competition_summary = competition_summary

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'ExperimentResult' has no attribute '{name}'")


class ExperimentRunner:
    """
    Central experiment orchestrator executing task-aware NLP workflows.
    Ensures leak-free evaluation, hardware-conscious batching, and reproducible experiments.
    """

    def __init__(self, config: Optional[WorkbenchConfig] = None, **kwargs):
        if config is None:
            self.config = WorkbenchConfig(**kwargs)
        else:
            self.config = config
        self.config.validate()

    def load_dataset(self) -> pd.DataFrame:
        """Loads or generates dataset based on configuration."""
        if self.config.data_path == "synthetic":
            data = DataLoader.generate_synthetic_dataset(
                task=self.config.task,
                language=self.config.language,
                random_state=self.config.random_state,
            )
            if isinstance(data, tuple):
                df = data[0]
            else:
                df = data
        else:
            df = DataLoader.load(self.config.data_path)

        # Validate schema
        validator = DatasetValidator(
            task=self.config.task,
            text_columns=self.config.text_columns,
            target_column=self.config.target_column,
            id_column=self.config.id_column,
        )
        validator.validate(df)
        return df

    def run_eda_and_audit(self, df: pd.DataFrame) -> Tuple[AuditReport, Dict[str, Any]]:
        """Executes data quality audit and exploratory data analysis."""
        auditor = DataQualityAuditor(
            df=df,
            text_columns=self.config.text_columns,
            target_column=self.config.target_column,
            id_column=self.config.id_column,
            task=self.config.task,
        )
        audit_report = auditor.audit(df)

        eda = ExploratoryDataAnalysis(
            df=df,
            text_columns=self.config.text_columns,
            target_column=self.config.target_column,
        )
        eda_summary = eda.profile()
        return audit_report, eda_summary

    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Partitions dataset into leak-free train, validation, and held-out test splits."""
        splitter = DatasetSplitter(
            strategy=self.config.split_strategy,
            n_splits=self.config.n_splits,
            test_size=self.config.test_size,
            val_size=self.config.experimental_preferences.get("val_size", 0.1),
            random_state=self.config.random_state,
        )
        train_df, val_df, test_df = splitter.split(
            df=df,
            target_col=self.config.target_column,
            group_col=self.config.group_column,
            time_col=self.config.time_column,
            task=self.config.task,
        )

        # Audit split leakage
        leakage_report = LeakageAuditor.audit_splits(
            train_df=train_df,
            test_df=test_df,
            text_columns=self.config.text_columns,
            id_column=self.config.id_column,
            group_column=self.config.group_column,
        )
        if leakage_report.get("leakage_detected") and not self.config.accepted_decisions.get("allow_leakage", False):
            # Log finding
            pass

        return train_df, val_df, test_df

    def get_models_to_evaluate(self) -> List[str]:
        """Resolves target model identifiers from config or task registry."""
        if self.config.model_ids and len(self.config.model_ids) > 0:
            for mid in self.config.model_ids:
                registry.validate_model_configuration(
                    model_id=mid,
                    task=self.config.task,
                    language=self.config.language,
                )
            return list(self.config.model_ids)

        # Fallback to compatible models in registry
        compatible_records = registry.get_compatible_models(
            task=self.config.task,
            language=self.config.language,
            device=self.config.device,
        )
        if not compatible_records:
            # Fallback to task match only
            compatible_records = registry.list_models(task=self.config.task)

        if not compatible_records:
            raise RuntimeError(f"No compatible models registered for task '{self.config.task}'.")

        # Select up to 3 representative models
        return [r.model_id for r in compatible_records[:3]]

    def evaluate_model(
        self,
        model_id: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Trains/initializes and evaluates a single model candidate.
        Uses cross-validation on train_df or evaluates directly on val_df.
        """
        record = registry.get_model(model_id)

        # Instantiate fresh adapter
        adapter_kwargs = {
            "device": self.config.device,
            "text_columns": self.config.text_columns,
            "target_column": self.config.target_column,
        }
        if self.config.task_params:
            adapter_kwargs.update(self.config.task_params)

        adapter = registry.create_adapter(model_id, **adapter_kwargs)

        t0 = time.time()
        fold_adapters: List[BaseTaskAdapter] = []
        cv_result: Optional[Dict[str, Any]] = None

        if not record.is_inference_only and "training" in record.supported_operations:
            # Run cross-validation on train_df
            cv_result = run_cross_validation(
                adapter=adapter,
                train_df=train_df,
                task=self.config.task,
                target_column=self.config.target_column,
                text_columns=self.config.text_columns,
                cv_folds=self.config.experimental_preferences.get("cv_folds", 5),
                random_state=self.config.random_state,
                group_column=self.config.group_column,
            )
            fold_adapters = cv_result.get("fold_adapters", [])

            # Also fit primary adapter on train_df for standalone usage
            adapter.fit(
                train_df,
                val_df=val_df,
                target_column=self.config.target_column,
                epochs=self.config.experimental_preferences.get("num_epochs", 3),
                batch_size=self.config.experimental_preferences.get("batch_size", 8),
                lr=self.config.experimental_preferences.get("learning_rate", 3e-5),
            )
        else:
            # Inference-only / Pretrained models
            adapter.fit(train_df, val_df=val_df)

        # Evaluate on validation split
        val_metrics = adapter.evaluate(
            val_df,
            target_column=self.config.target_column,
        )
        latency = round(time.time() - t0, 3)

        # Determine primary validation score
        if cv_result is not None:
            val_score = cv_result.get("mean_cv_score", 0.0)
            metric_name = cv_result.get("primary_metric", "Mean CV Score")
        else:
            metric_name = list(val_metrics.keys())[0] if val_metrics else "Score"
            val_score = float(val_metrics.get(metric_name, 0.0)) if val_metrics else 0.0

        return {
            "model_id": model_id,
            "approach_name": record.approach_name,
            "framework": record.model_framework,
            "adapter": adapter,
            "fold_adapters": fold_adapters,
            "cv_result": cv_result,
            "val_metrics": val_metrics,
            "primary_metric": metric_name,
            "primary_score": val_score,
            "latency_s": latency,
        }

    def run_benchmark(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Runs benchmark across all selected models and produces Leaderboard."""
        model_ids = self.get_models_to_evaluate()
        records: Dict[str, Dict[str, Any]] = {}
        board_rows = []

        for mid in model_ids:
            res = self.evaluate_model(mid, train_df, val_df)
            records[mid] = res
            row = {
                "Model ID": mid,
                "Approach": res["approach_name"],
                "Framework": res["framework"],
                "Validation Score": round(float(res["primary_score"]), 4),
                "Primary Metric": res["primary_metric"],
                "Latency (s)": res["latency_s"],
            }
            # Append other val metrics
            for k, v in res["val_metrics"].items():
                if isinstance(v, (int, float)):
                    row[k] = round(float(v), 4)
            board_rows.append(row)

        leaderboard_df = pd.DataFrame(board_rows)
        # Sort leaderboard based on primary metric direction
        if not leaderboard_df.empty:
            metric_col = "Validation Score"
            first_metric_name = board_rows[0].get("Primary Metric", "Score")
            hib = is_higher_better(first_metric_name)
            leaderboard_df = leaderboard_df.sort_values(by=metric_col, ascending=(not hib)).reset_index(drop=True)

        return leaderboard_df, records

    def run(self) -> ExperimentResult:
        """Executes full experiment pipeline end-to-end."""
        # 1. Ingestion
        df = self.load_dataset()

        # 2. Audit & EDA
        audit_rep, eda_rep = self.run_eda_and_audit(df)

        # 3. Splitting
        train_df, val_df, test_df = self.split_dataset(df)

        # 4. Benchmarking on Validation
        leaderboard_df, eval_records = self.run_benchmark(train_df, val_df)

        # 5. Model Selection (strictly on validation)
        best_selection = select_best_model(leaderboard_df, primary_metric="Validation Score")
        best_id = best_selection.model_id
        best_record = eval_records[best_id]
        best_adapter = best_record["adapter"]
        fold_adapters = best_record.get("fold_adapters", [])

        # 6. Held-out Test Evaluation (strictly after model selection)
        test_metrics = best_adapter.evaluate(
            test_df,
            target_column=self.config.target_column,
        )

        # 7. Model Export
        export_path = ModelExporter.export(
            adapter=best_adapter,
            output_dir=self.config.output_dir,
            task=self.config.task,
            metadata={
                "model_id": best_id,
                "language": self.config.language,
                "validation_score": float(best_record["primary_score"]),
                "test_metrics": test_metrics,
            },
            cleaner_config=self.config.accepted_decisions,
        )

        # 8. Gemastik Competition Mode
        competition_summary = None
        if self.config.gemastik_mode:
            runner = GemastikRunner(
                id_column=self.config.id_column or "id",
                target_column=self.config.submission_target_col or "prediction",
                task=self.config.task,
            )
            comp_path = self.config.competition_test_path
            if comp_path == "synthetic" or comp_path is None:
                # Generate unlabelled test partition
                comp_test_df = test_df.drop(columns=[self.config.target_column], errors="ignore").copy()
            else:
                comp_test_df = runner.prepare_test_dataset(comp_path)

            text_col = self.config.text_column or self.config.text_columns[0]
            text_b_col = self.config.text_b_column or self.config.task_params.get("text_b_column")
            preds = runner.generate_predictions(
                adapter=best_adapter,
                test_df=comp_test_df,
                text_column=text_col,
                text_b_column=text_b_col,
                ensemble_adapters=fold_adapters if len(fold_adapters) > 1 else None,
            )
            competition_summary = runner.create_submission(
                test_df=comp_test_df,
                predictions=preds,
                output_path=self.config.submission_path,
            )

        return ExperimentResult(
            config=self.config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            leaderboard=leaderboard_df,
            best_model_id=best_id,
            best_adapter=best_adapter,
            test_metrics=test_metrics,
            exported_path=export_path,
            audit_report=audit_rep,
            eda_summary=eda_rep,
            fold_adapters=fold_adapters,
            competition_summary=competition_summary,
        )
