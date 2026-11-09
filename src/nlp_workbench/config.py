"""
Configuration module for the NLP Experimentation Workbench.
Provides task-aware schema validation and centralized experiment control.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml

VALID_TASKS = [
    "classification",
    "regression",
    "multilabel",
    "ner",
    "qa",
    "summarization",
    "translation",
    "generation",
    "similarity",
    "retrieval",
    "topic_modeling",
    "pair_classification",
    "zeroshot",
]

VALID_LANGUAGES = ["indonesian", "english", "multilingual"]

@dataclass
class WorkbenchConfig:
    # 1. Core Dataset & Task Configuration
    task: str = "classification"
    language: str = "indonesian"
    data_path: str = "synthetic"
    dataset_path: Optional[str] = None
    test_path: Optional[str] = None
    text_columns: List[str] = field(default_factory=lambda: ["title", "content"])
    text_column: Optional[str] = None
    text_b_column: Optional[str] = None
    target_column: Optional[str] = "category"
    id_column: Optional[str] = "id"
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    model_ids: Optional[List[str]] = None
    
    # 2. Hardware & Environment
    device: str = "auto"  # "auto", "cuda", "cpu"
    random_state: int = 42
    random_seed: Optional[int] = None
    output_dir: str = "output_model"
    
    # Splitting parameters
    split_strategy: str = "stratified"
    n_splits: int = 5
    test_size: float = 0.2
    
    # 3. Competition Mode (Gemastik)
    gemastik_mode: bool = False
    competition_test_path: Optional[str] = None
    submission_path: str = "submission.csv"
    submission_target_col: str = "prediction"
    
    # 4. Task-Specific Parameters
    task_params: Dict[str, Any] = field(default_factory=dict)
    
    # 5. User-Accepted Preprocessing & Data Quality Decisions
    accepted_decisions: Dict[str, bool] = field(default_factory=lambda: {
        "enable_html_cleaning": True,
        "handle_urls": True,
        "enable_class_weights": True,
        "stratified_splitting": True,
        "deduplicate_texts": False,
        "log_transform_target": False,
    })
    
    # 6. Experimental Hyperparameters
    experimental_preferences: Dict[str, Any] = field(default_factory=lambda: {
        "test_size": 0.2,
        "cv_folds": 5,
        "max_seq_length": 128,
        "batch_size": 8,
        "grad_accum_steps": 4,
        "use_fp16": True,
        "learning_rate": 3e-5,
        "num_epochs": 3,
        "max_tfidf_features": 15000,
        "ngram_range": (1, 2),
    })

    def __post_init__(self):
        # Resolve aliases
        if self.dataset_path is not None:
            self.data_path = self.dataset_path
        if self.text_column is not None:
            self.text_columns = [self.text_column]
        elif self.text_columns:
            self.text_column = self.text_columns[0]
        if self.random_seed is not None:
            self.random_state = self.random_seed
        if self.text_b_column is not None:
            self.task_params["text_b_column"] = self.text_b_column

        # Validate task
        self.task = self.task.lower().strip()
        if self.task not in VALID_TASKS:
            raise ValueError(
                f"Unsupported NLP task '{self.task}'. Must be one of: {', '.join(VALID_TASKS)}"
            )

        self.language = self.language.lower().strip()
        if self.language not in VALID_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{self.language}'. Must be one of: {', '.join(VALID_LANGUAGES)}"
            )

    def validate(self) -> None:
        """Validates configuration parameters, paths, and task requirements."""
        self.__post_init__()

        # Validate data_path existence (if not synthetic)
        if self.data_path != "synthetic":
            p = Path(self.data_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"Configured dataset file '{self.data_path}' was not found. "
                    f"Please verify the path or set data_path='synthetic' for demo mode."
                )

        if self.test_path and self.test_path != "synthetic":
            p_test = Path(self.test_path)
            if not p_test.exists():
                raise FileNotFoundError(f"Configured test set '{self.test_path}' was not found.")

        if self.gemastik_mode:
            if not self.competition_test_path and self.data_path != "synthetic":
                raise ValueError(
                    "gemastik_mode=True requires 'competition_test_path' to be set."
                )
            if self.competition_test_path and self.competition_test_path != "synthetic":
                p_comp = Path(self.competition_test_path)
                if not p_comp.exists():
                    raise FileNotFoundError(
                        f"Competition test file '{self.competition_test_path}' not found."
                    )

        # Task-specific schema checks
        if self.task in ["classification", "regression"] and not self.target_column:
            raise ValueError(f"target_column is required for task '{self.task}'.")

        if self.task in ["pair_classification", "similarity"]:
            text_b = self.task_params.get("text_b_column") or self.text_b_column
            if not text_b and len(self.text_columns) < 2:
                raise ValueError(
                    f"Task '{self.task}' requires either 'text_b_column' "
                    f"or at least two columns in 'text_columns'."
                )

        if self.task == "zeroshot":
            candidates = self.task_params.get("candidate_labels")
            if not candidates or not isinstance(candidates, list) or len(candidates) < 2:
                raise ValueError(
                    "Zero-shot classification requires 'task_params[\"candidate_labels\"]' "
                    "with a list of at least 2 label names."
                )

    def validate_for_execution(self) -> None:
        """Alias for validate()"""
        self.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "language": self.language,
            "data_path": self.data_path,
            "test_path": self.test_path,
            "text_columns": list(self.text_columns),
            "text_column": self.text_column,
            "target_column": self.target_column,
            "id_column": self.id_column,
            "group_column": self.group_column,
            "time_column": self.time_column,
            "model_ids": self.model_ids,
            "device": self.device,
            "random_state": self.random_state,
            "output_dir": self.output_dir,
            "split_strategy": self.split_strategy,
            "n_splits": self.n_splits,
            "gemastik_mode": self.gemastik_mode,
            "competition_test_path": self.competition_test_path,
            "submission_path": self.submission_path,
            "submission_target_col": self.submission_target_col,
            "task_params": dict(self.task_params),
            "accepted_decisions": dict(self.accepted_decisions),
            "experimental_preferences": dict(self.experimental_preferences),
        }

    def to_yaml(self) -> str:
        """Exports configuration to a YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkbenchConfig":
        return cls(**data)


def get_default_config(task: str = "classification", language: str = "indonesian") -> WorkbenchConfig:
    """Returns a ready-to-run default configuration for a given task and language."""
    cfg = WorkbenchConfig(task=task, language=language)
    if task == "classification":
        cfg.text_columns = ["text"]
        cfg.text_column = "text"
        cfg.target_column = "target"
    elif task == "pair_classification":
        cfg.text_columns = ["text_a"]
        cfg.text_column = "text_a"
        cfg.text_b_column = "text_b"
        cfg.task_params["text_b_column"] = "text_b"
        cfg.target_column = "label"
    elif task == "similarity":
        cfg.text_columns = ["text_a"]
        cfg.text_column = "text_a"
        cfg.text_b_column = "text_b"
        cfg.task_params["text_b_column"] = "text_b"
        cfg.target_column = "similarity_score"
    elif task == "qa":
        cfg.text_columns = ["question"]
        cfg.text_column = "question"
        cfg.task_params["context_column"] = "context"
        cfg.task_params["answer_column"] = "answer_text"
    elif task == "ner":
        cfg.text_columns = ["tokens"]
        cfg.text_column = "tokens"
        cfg.target_column = "ner_tags"
    elif task == "summarization":
        cfg.text_columns = ["document"]
        cfg.text_column = "document"
        cfg.target_column = "summary"
    elif task == "translation":
        cfg.text_columns = ["source_text"]
        cfg.text_column = "source_text"
        cfg.target_column = "target_text"
    elif task == "retrieval":
        cfg.text_columns = ["query"]
        cfg.text_column = "query"
        cfg.task_params["corpus_text_column"] = "text"
        cfg.target_column = "relevance"
    return cfg
