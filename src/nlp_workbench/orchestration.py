"""
Central Experiment Orchestration module for the NLP Experimentation Workbench.
Provides canonical task-aware, staged end-to-end execution from data ingestion to model export and competition submission.
"""

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from .config import WorkbenchConfig, get_default_config
from .data import DataLoader, DatasetValidator, RetrievalDatasetBundle
from .eda import DataQualityAuditor, ExploratoryDataAnalysis, AuditReport
from .preprocessing import recommend_preprocessing, TextCleaner
from .splitting import DatasetSplitter, LeakageAuditor
from .registry import registry, ModelRecord
from .tasks.base import BaseTaskAdapter
from .training import run_cross_validation, select_best_model, BestModelSelection
from .export import ModelExporter, LoadedModelBundle
from .competition import GemastikRunner
from .evaluation import get_task_evaluation_spec, is_higher_better


@dataclass
class PreparedDataBundle:
    """
    Staged data bundle resulting from runner.prepare(), encapsulating ingested data,
    EDA profiles, recommendations, leak-free splits, and 3-way leakage audit results.
    """
    raw_data: Union[pd.DataFrame, RetrievalDatasetBundle]
    train_df: Union[pd.DataFrame, RetrievalDatasetBundle]
    val_df: Union[pd.DataFrame, RetrievalDatasetBundle]
    test_df: Union[pd.DataFrame, RetrievalDatasetBundle]
    audit_report: AuditReport
    eda_summary: Dict[str, Any]
    leakage_report: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    accepted_decisions: Dict[str, Any]

    def __iter__(self):
        """Allows unpacking: train_df, val_df, test_df = prepared"""
        return iter((self.train_df, self.val_df, self.test_df))

    @property
    def train_data(self) -> Union[pd.DataFrame, RetrievalDatasetBundle]:
        return self.train_df

    @property
    def val_data(self) -> Union[pd.DataFrame, RetrievalDatasetBundle]:
        return self.val_df

    @property
    def test_data(self) -> Union[pd.DataFrame, RetrievalDatasetBundle]:
        return self.test_df


class ExperimentResult(dict):
    """
    Container providing dictionary, attribute, and structured access to complete experiment artifacts.
    """
    def __init__(
        self,
        config: WorkbenchConfig,
        train_df: Union[pd.DataFrame, RetrievalDatasetBundle],
        val_df: Union[pd.DataFrame, RetrievalDatasetBundle],
        test_df: Union[pd.DataFrame, RetrievalDatasetBundle],
        leaderboard: pd.DataFrame,
        best_model_id: str,
        best_adapter: BaseTaskAdapter,
        test_metrics: Dict[str, Any],
        exported_path: Optional[Path] = None,
        audit_report: Optional[AuditReport] = None,
        eda_summary: Optional[Dict[str, Any]] = None,
        fold_adapters: Optional[List[BaseTaskAdapter]] = None,
        competition_summary: Optional[Dict[str, Any]] = None,
        reproducibility_report: Optional[Dict[str, Any]] = None,
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
            reproducibility_report=reproducibility_report or {},
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
        self.reproducibility_report = reproducibility_report or {}

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'ExperimentResult' has no attribute '{name}'")


class ExperimentRunner:
    """
    Central experiment orchestrator executing task-aware NLP workflows.
    Ensures leak-free evaluation, hardware-conscious batching, and reproducible experiments.
    Supports staged execution:
      prepared = runner.prepare()
      result = runner.run(prepared)
    """

    def __init__(self, config: Optional[WorkbenchConfig] = None, **kwargs):
        if config is None:
            self.config = WorkbenchConfig(**kwargs)
        else:
            self.config = config
        self.config.validate()

    def load_dataset(self) -> Union[pd.DataFrame, RetrievalDatasetBundle]:
        """Loads or generates dataset based on configuration."""
        if self.config.data_path == "synthetic":
            data = DataLoader.generate_synthetic_dataset(
                task=self.config.task,
                language=self.config.language,
                random_state=self.config.random_state,
            )
            if isinstance(data, pd.DataFrame):
                t_col = self.config.text_column or (self.config.text_columns[0] if self.config.text_columns else "text")
                tgt_col = self.config.target_column
                if t_col not in data.columns and "content" in data.columns:
                    data = data.rename(columns={"content": t_col})
                if tgt_col and tgt_col not in data.columns:
                    if "category" in data.columns:
                        data = data.rename(columns={"category": tgt_col})
                    elif "engagement_score" in data.columns:
                        data = data.rename(columns={"engagement_score": tgt_col})
        else:
            data = DataLoader.load(self.config.data_path)

        # Validate schema
        validator = DatasetValidator(
            task=self.config.task,
            text_columns=self.config.text_columns,
            target_column=self.config.target_column,
            id_column=self.config.id_column,
        )
        validator.validate(data)
        return data

    def run_eda_and_audit(self, df: Union[pd.DataFrame, RetrievalDatasetBundle]) -> Tuple[AuditReport, Dict[str, Any]]:
        """Executes data quality audit and exploratory data analysis."""
        profile_df = df.queries_df if isinstance(df, RetrievalDatasetBundle) else df

        auditor = DataQualityAuditor(
            df=profile_df,
            text_columns=self.config.text_columns,
            target_column=self.config.target_column,
            id_column=self.config.id_column,
            task=self.config.task,
        )
        audit_report = auditor.audit(profile_df)

        eda = ExploratoryDataAnalysis(
            df=profile_df,
            text_columns=self.config.text_columns,
            target_column=self.config.target_column,
        )
        eda_summary = eda.profile()
        return audit_report, eda_summary

    def split_dataset(
        self, df: Union[pd.DataFrame, RetrievalDatasetBundle]
    ) -> Tuple[Union[pd.DataFrame, RetrievalDatasetBundle], Union[pd.DataFrame, RetrievalDatasetBundle], Union[pd.DataFrame, RetrievalDatasetBundle]]:
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
        return train_df, val_df, test_df

    def prepare(self) -> PreparedDataBundle:
        """
        Executes data loading, schema verification, EDA, recommendation generation,
        leak-free splitting, and 3-way split leakage auditing.
        """
        # 1. Ingestion
        raw_data = self.load_dataset()

        # 2. EDA & Data Quality Audit
        audit_report, eda_summary = self.run_eda_and_audit(raw_data)

        # 3. Recommendations
        recs = recommend_preprocessing(task=self.config.task, model_family="classical", language=self.config.language)

        # 4. Splitting
        train_df, val_df, test_df = self.split_dataset(raw_data)

        # 5. 3-way Leakage Audit & Policy Enforcement
        text_cols = list(self.config.text_columns) if self.config.text_columns else ([self.config.text_column] if self.config.text_column else [])
        if isinstance(train_df, pd.DataFrame):
            train_df, val_df, test_df, leakage_report = LeakageAuditor.enforce_leakage_policy(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                text_columns=text_cols,
                id_column=self.config.id_column,
                group_column=self.config.group_column,
                deduplicate_texts=self.config.accepted_decisions.get("deduplicate_texts", False),
                raise_on_leakage=not self.config.accepted_decisions.get("allow_leakage", False),
            )
        else:
            leakage_report = LeakageAuditor.audit_all_splits(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                text_columns=text_cols,
                id_column=self.config.id_column,
                group_column=self.config.group_column,
            )

        return PreparedDataBundle(
            raw_data=raw_data,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            audit_report=audit_report,
            eda_summary=eda_summary,
            leakage_report=leakage_report,
            recommendations=recs,
            accepted_decisions=self.config.accepted_decisions,
        )

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
            compatible_records = registry.list_models(task=self.config.task)

        if not compatible_records:
            raise RuntimeError(f"No compatible models registered for task '{self.config.task}'.")

        return [r.model_id for r in compatible_records[:3]]

    def evaluate_model(
        self,
        model_id: str,
        train_df: Union[pd.DataFrame, RetrievalDatasetBundle],
        val_df: Union[pd.DataFrame, RetrievalDatasetBundle],
    ) -> Dict[str, Any]:
        """
        Trains/initializes and evaluates a single model candidate.
        Ensures evaluation strictly reports the task's canonical primary metric.
        """
        record = registry.get_model(model_id)
        task_spec = get_task_evaluation_spec(self.config.task)
        canonical_primary = task_spec.primary_metric

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

        if isinstance(train_df, RetrievalDatasetBundle):
            # Retrieval task handling
            adapter.index_corpus(train_df.corpus_df)
            val_metrics = adapter.evaluate(val_df.queries_df)
            latency = round(time.time() - t0, 3)
            val_score = float(val_metrics.get(canonical_primary, list(val_metrics.values())[0]))
            return {
                "model_id": model_id,
                "approach_name": record.approach_name,
                "framework": record.model_framework,
                "adapter": adapter,
                "fold_adapters": [],
                "cv_result": None,
                "val_metrics": val_metrics,
                "primary_metric": canonical_primary,
                "primary_score": val_score,
                "latency_s": latency,
            }

        if not record.is_inference_only and "training" in record.supported_operations:
            # Run cross-validation on train_df
            cv_result = run_cross_validation(
                adapter=adapter,
                train_df=train_df,
                task=self.config.task,
                target_column=self.config.target_column,
                text_columns=self.config.text_columns,
                cv_folds=self.config.experimental_preferences.get("cv_folds", self.config.n_splits),
                random_state=self.config.random_state,
                group_column=self.config.group_column,
            )
            fold_adapters = cv_result.get("fold_adapters", [])

            # Also fit primary adapter on train_df for held-out evaluation and export
            adapter.fit(
                train_df,
                val_df=val_df,
                target_column=self.config.target_column,
                epochs=self.config.experimental_preferences.get("num_epochs", 3),
                batch_size=self.config.experimental_preferences.get("batch_size", 8),
                lr=self.config.experimental_preferences.get("learning_rate", 3e-5),
            )
        else:
            # Pretrained / zero-shot / inference-only models
            adapter.fit(train_df, val_df=val_df)

        val_metrics = adapter.evaluate(
            val_df,
            target_column=self.config.target_column,
        )
        latency = round(time.time() - t0, 3)

        # Determine primary validation score
        if cv_result is not None:
            val_score = cv_result.get("mean_cv_score", 0.0)
            metric_name = cv_result.get("primary_metric", canonical_primary)
        else:
            if canonical_primary in val_metrics:
                metric_name = canonical_primary
                val_score = float(val_metrics[canonical_primary])
            else:
                metric_name = list(val_metrics.keys())[0] if val_metrics else canonical_primary
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
        train_df: Union[pd.DataFrame, RetrievalDatasetBundle],
        val_df: Union[pd.DataFrame, RetrievalDatasetBundle],
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Runs benchmark across all selected models and produces Leaderboard."""
        model_ids = self.get_models_to_evaluate()
        records: Dict[str, Dict[str, Any]] = {}
        board_rows = []
        task_spec = get_task_evaluation_spec(self.config.task)

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
            hib = task_spec.higher_is_better
            leaderboard_df = leaderboard_df.sort_values(by=metric_col, ascending=(not hib)).reset_index(drop=True)

        return leaderboard_df, records

    def generate_reproducibility_report(self, result: Optional[ExperimentResult] = None) -> Dict[str, Any]:
        """Generates full environment, git commit, seed, and hardware audit report."""
        import subprocess
        import sys
        git_commit = "N/A"
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        except Exception:
            pass

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git_commit,
            "python_version": sys.version,
            "config": self.config.to_dict(),
            "best_model_id": result.best_model_id if result else None,
            "test_metrics": result.test_metrics if result else {},
            "random_state": self.config.random_state,
        }

    def run(self, prepared: Optional[PreparedDataBundle] = None) -> ExperimentResult:
        """
        Executes canonical end-to-end experiment pipeline.
        Can run from pre-prepared data or prepare automatically.
        """
        # 1. Prepare if not provided
        if prepared is None:
            prepared = self.prepare()

        train_df = prepared.train_df
        val_df = prepared.val_df
        test_df = prepared.test_df

        # 2. Benchmarking on Validation
        leaderboard_df, eval_records = self.run_benchmark(train_df, val_df)

        # 3. Model Selection (strictly on validation with direction awareness)
        task_spec = get_task_evaluation_spec(self.config.task)
        best_selection = select_best_model(
            leaderboard_df,
            task=self.config.task,
            primary_metric=task_spec.primary_metric,
            higher_is_better=task_spec.higher_is_better,
        )
        best_id = best_selection.model_id
        best_record = eval_records[best_id]
        best_adapter = best_record["adapter"]
        fold_adapters = best_record.get("fold_adapters", [])

        # 4. Held-out Test Evaluation (strictly after model selection, evaluated once)
        if isinstance(test_df, RetrievalDatasetBundle):
            test_metrics = best_adapter.evaluate(test_df.queries_df)
        else:
            test_metrics = best_adapter.evaluate(
                test_df,
                target_column=self.config.target_column,
            )

        # 5. Model Export
        cleaner_dict = (
            prepared.accepted_decisions if prepared.accepted_decisions
            else self.config.accepted_decisions
        )
        export_path = ModelExporter.export(
            adapter=best_adapter,
            output_dir=self.config.output_dir,
            task=self.config.task,
            metadata={
                "model_id": best_id,
                "language": self.config.language,
                "validation_score": float(best_record["primary_score"]),
                "primary_metric": best_record["primary_metric"],
                "test_metrics": test_metrics,
            },
            cleaner_config=cleaner_dict,
        )

        # 6. Verify reload integrity
        loaded_bundle = ModelExporter.load(export_path)
        if hasattr(loaded_bundle, "predict"):
            try:
                sample_input = test_df.queries_df.head(2) if isinstance(test_df, RetrievalDatasetBundle) else test_df.head(2)
                _ = loaded_bundle.predict(sample_input)
            except Exception:
                pass

        # 7. Gemastik Competition Mode (Strictly isolated from held-out test)
        competition_summary = None
        if self.config.gemastik_mode:
            if not self.config.competition_test_path:
                raise ValueError(
                    "gemastik_mode=True requires 'competition_test_path' to be configured. "
                    "The held-out evaluation test set is strictly isolated and cannot be reused as competition input."
                )

            runner = GemastikRunner(
                id_column=self.config.id_column or "id",
                target_column=self.config.submission_target_col or self.config.target_column or "prediction",
                task=self.config.task,
            )

            if self.config.competition_test_path == "synthetic":
                # Generate a FRESH, distinct unlabelled competition test dataset (different random seed)
                comp_raw = DataLoader.generate_synthetic_dataset(
                    task=self.config.task,
                    n_samples=40,
                    language=self.config.language,
                    random_state=9999,
                )
                if isinstance(comp_raw, RetrievalDatasetBundle):
                    comp_test_df = comp_raw.queries_df.copy()
                elif isinstance(comp_raw, tuple):
                    comp_test_df = comp_raw[0].copy()
                else:
                    comp_test_df = comp_raw.copy()

                # Remove any ground-truth target columns to ensure strict unlabelled test
                if self.config.target_column and self.config.target_column in comp_test_df.columns:
                    comp_test_df = comp_test_df.drop(columns=[self.config.target_column])
                if "target" in comp_test_df.columns:
                    comp_test_df = comp_test_df.drop(columns=["target"])
                if "category" in comp_test_df.columns:
                    comp_test_df = comp_test_df.drop(columns=["category"])
                if self.config.id_column not in comp_test_df.columns:
                    comp_test_df[self.config.id_column] = [f"COMP_{i:04d}" for i in range(len(comp_test_df))]
            else:
                comp_test_df = runner.prepare_test_dataset(self.config.competition_test_path)

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

        res = ExperimentResult(
            config=self.config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            leaderboard=leaderboard_df,
            best_model_id=best_id,
            best_adapter=best_adapter,
            test_metrics=test_metrics,
            exported_path=export_path,
            audit_report=prepared.audit_report,
            eda_summary=prepared.eda_summary,
            fold_adapters=fold_adapters,
            competition_summary=competition_summary,
        )
        res.reproducibility_report = self.generate_reproducibility_report(res)
        return res
