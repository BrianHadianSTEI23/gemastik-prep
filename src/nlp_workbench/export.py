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
    Wrapper for a loaded model bundle supporting both dictionary and attribute access,
    and delegating inference methods directly to the underlying adapter.
    """
    def __init__(self, adapter: BaseTaskAdapter, **kwargs):
        super().__init__(adapter=adapter, **kwargs)
        self.adapter = adapter
        for k, v in kwargs.items():
            setattr(self, k, v)

    def predict(self, *args, **kwargs):
        return self.adapter.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        if hasattr(self.adapter, "predict_proba"):
            return self.adapter.predict_proba(*args, **kwargs)
        raise NotImplementedError(f"Underlying adapter '{type(self.adapter).__name__}' does not support predict_proba.")

    def evaluate(self, *args, **kwargs):
        return self.adapter.evaluate(*args, **kwargs)


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
        Handles both directory-based (e.g. HuggingFace checkpoints) and single-file (joblib) artifacts.
        """
        target_dir = output_dir or export_dir or "output_model"
        output_path = Path(target_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        artifact_target = output_path / artifact_name

        # Invoke adapter's model-specific export
        saved_path = Path(adapter.export(artifact_target))

        # Update bundle metadata in file or directory
        meta_file = (saved_path / "metadata.joblib") if saved_path.is_dir() else saved_path
        if meta_file.exists() and meta_file.is_file():
            try:
                existing_bundle = joblib.load(meta_file)
                if isinstance(existing_bundle, dict):
                    if task:
                        existing_bundle["task"] = task
                    if metadata:
                        existing_bundle["metadata"] = metadata
                    if extra_metadata:
                        existing_bundle["extra_metadata"] = extra_metadata
                    if cleaner_config:
                        existing_bundle["cleaner_config"] = cleaner_config
                    joblib.dump(existing_bundle, meta_file)
            except Exception:
                pass

        # Write overall manifest
        manifest = {
            "model_id": adapter.model_id,
            "adapter_class": adapter.__class__.__name__,
            "artifact_file": saved_path.name,
            "is_directory": saved_path.is_dir(),
            "task": task or getattr(adapter, "task", "classification"),
            "metadata": metadata or {},
            "extra_metadata": extra_metadata or {},
            "cleaner_config": cleaner_config or {},
        }
        joblib.dump(manifest, output_path / "manifest.joblib")

        return saved_path

    @staticmethod
    def load(export_path: Union[str, Path]) -> LoadedModelBundle:
        """Loads serialized model bundle using model-specific loader."""
        p = Path(export_path)
        manifest_path = p / "manifest.joblib" if p.is_dir() else p.parent / "manifest.joblib"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = joblib.load(manifest_path)
            except Exception:
                pass

        if p.is_dir():
            if (p / "metadata.joblib").exists():
                artifact_path = p
                meta_file = p / "metadata.joblib"
            elif (p / "nlp_model_bundle.joblib").exists():
                candidate = p / "nlp_model_bundle.joblib"
                artifact_path = candidate
                meta_file = (candidate / "metadata.joblib") if candidate.is_dir() else candidate
            else:
                artifact_path = p
                meta_file = p / "metadata.joblib"
        else:
            artifact_path = p
            meta_file = p

        bundle_data = {}
        if meta_file.exists() and meta_file.is_file():
            try:
                bundle_data = joblib.load(meta_file)
            except Exception:
                pass

        model_id = bundle_data.get("model_id") or manifest.get("model_id")
        if not model_id:
            raise ValueError(f"Could not determine model_id from {export_path}")

        record = registry.get_model(model_id)
        adapter = record.adapter_cls.load(artifact_path)

        combined_info = dict(bundle_data)
        combined_info.update(manifest)
        bundle_kwargs = {k: v for k, v in combined_info.items() if k != "adapter"}
        return LoadedModelBundle(adapter=adapter, **bundle_kwargs)
