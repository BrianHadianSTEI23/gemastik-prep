"""
Model Packaging and Serialization module.
Provides model-specific serialization preserving tokenizers, vectorizers, and configuration.
"""

from pathlib import Path
from typing import Dict, Optional, Any, Union
import joblib

from .tasks.base import BaseTaskAdapter
from .registry import registry


class LoadedModelBundle(dict):
    """
    Wrapper for a loaded model bundle supporting both dictionary and attribute access.
    """
    def __init__(self, adapter: BaseTaskAdapter, **kwargs):
        super().__init__(adapter=adapter, **kwargs)
        self.adapter = adapter
        for k, v in kwargs.items():
            setattr(self, k, v)


class ModelExporter:
    @staticmethod
    def export(
        adapter: BaseTaskAdapter,
        export_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        artifact_name: str = "nlp_model_bundle.joblib",
        task: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        cleaner_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Path:
        """
        Exports the selected model adapter to disk.
        Delegates to adapter's model-specific serialization.
        """
        target_dir = output_dir or export_dir or "output_model"
        output_path = Path(target_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        artifact_file = output_path / artifact_name

        # Invoke adapter's model-specific export
        saved_path = adapter.export(artifact_file)

        # Update bundle with metadata, task, and cleaner_config
        existing_bundle = joblib.load(saved_path)
        if isinstance(existing_bundle, dict):
            if task:
                existing_bundle["task"] = task
            if metadata:
                existing_bundle["metadata"] = metadata
            if extra_metadata:
                existing_bundle["extra_metadata"] = extra_metadata
            if cleaner_config:
                existing_bundle["cleaner_config"] = cleaner_config
            joblib.dump(existing_bundle, saved_path)

        # Write overall manifest
        manifest = {
            "model_id": adapter.model_id,
            "adapter_class": adapter.__class__.__name__,
            "artifact_file": str(saved_path.name),
            "task": task or getattr(adapter, "task", "classification"),
            "extra_metadata": extra_metadata or {},
        }
        joblib.dump(manifest, output_path / "manifest.joblib")

        return saved_path

    @staticmethod
    def load(export_path: Union[str, Path]) -> LoadedModelBundle:
        """Loads serialized model bundle using model-specific loader."""
        p = Path(export_path)
        if p.is_dir():
            p = p / "nlp_model_bundle.joblib"

        bundle = joblib.load(p)
        model_id = bundle.get("model_id")
        record = registry.get_model(model_id)

        # Delegate to adapter class load method
        adapter = record.adapter_cls.load(p)

        bundle_kwargs = {k: v for k, v in bundle.items() if k != "adapter"}
        return LoadedModelBundle(adapter=adapter, **bundle_kwargs)
