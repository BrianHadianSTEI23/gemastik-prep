"""
Hardware diagnostics and optimization module for the NLP Experimentation Workbench.
Optimized for NVIDIA GeForce RTX 3050 (4GB/6GB VRAM) and CPU-only systems.
"""

import os
import gc
import platform
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import torch

@dataclass
class HardwareProfile:
    device: str
    cuda_available: bool
    gpu_name: Optional[str]
    vram_total_mb: int
    vram_allocated_mb: int
    vram_reserved_mb: int
    vram_free_mb: int
    device_capability: Optional[Tuple[int, int]]
    supports_fp16: bool
    supports_bf16: bool
    recommended_batch_size: int
    recommended_grad_accum: int
    notes: str

    def get_torch_device(self) -> torch.device:
        return torch.device(self.device)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "HARDWARE DIAGNOSTIC REPORT",
            "=" * 60,
            f"Platform:              {platform.system()} {platform.release()} ({platform.machine()})",
            f"PyTorch Version:       {torch.__version__}",
            f"CUDA Available:        {self.cuda_available}",
            f"Active Device:         {self.device.upper()}",
        ]
        if self.cuda_available and self.gpu_name:
            lines.extend([
                f"GPU Model:             {self.gpu_name}",
                f"Compute Capability:    {self.device_capability}",
                f"Total VRAM:            {self.vram_total_mb:,} MB (~{self.vram_total_mb/1024:.2f} GB)",
                f"Allocated VRAM:        {self.vram_allocated_mb} MB",
                f"Free VRAM:             {self.vram_free_mb:,} MB",
                f"Supports FP16 / BF16:  {self.supports_fp16} / {self.supports_bf16}",
            ])
        lines.extend([
            f"Recommended Batch Size:{self.recommended_batch_size} (with {self.recommended_grad_accum} grad accum steps)",
            f"Guidance:              {self.notes}",
            "=" * 60,
        ])
        return "\n".join(lines)


class HardwareManager:
    def __init__(self, target_device: str = "auto"):
        self.profile = self.inspect(target_device)

    def recommend_batch_size(self) -> Dict[str, int]:
        return {
            "batch_size": self.profile.recommended_batch_size,
            "grad_accum_steps": self.profile.recommended_grad_accum,
        }

    @staticmethod
    def inspect(device_preference: str = "auto") -> HardwareProfile:
        """Inspects host hardware and returns a hardware profile with memory-safe recommendations."""
        cuda_available = torch.cuda.is_available()
        pref = device_preference.lower().strip()

        if pref == "cpu":
            target_device = "cpu"
        elif pref == "cuda":
            if not cuda_available:
                raise RuntimeError("CUDA was explicitly requested, but no CUDA-enabled GPU is detected.")
            target_device = "cuda"
        else:  # "auto"
            target_device = "cuda" if cuda_available else "cpu"

        profile = HardwareProfile(
            device=target_device,
            cuda_available=cuda_available,
            gpu_name=None,
            vram_total_mb=0,
            vram_allocated_mb=0,
            vram_reserved_mb=0,
            vram_free_mb=0,
            device_capability=None,
            supports_fp16=False,
            supports_bf16=False,
            recommended_batch_size=16,
            recommended_grad_accum=1,
            notes="",
        )

        if target_device == "cuda" and cuda_available:
            props = torch.cuda.get_device_properties(0)
            profile.gpu_name = props.name
            profile.vram_total_mb = props.total_memory // (1024 * 1024)
            profile.vram_allocated_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)
            profile.vram_reserved_mb = torch.cuda.memory_reserved(0) // (1024 * 1024)
            profile.vram_free_mb = profile.vram_total_mb - profile.vram_reserved_mb
            cap = torch.cuda.get_device_capability(0)
            profile.device_capability = cap
            profile.supports_fp16 = True
            profile.supports_bf16 = (cap[0] >= 8)

            if profile.vram_total_mb <= 4500:
                profile.recommended_batch_size = 8
                profile.recommended_grad_accum = 4
                profile.notes = (
                    "RTX 3050 (4GB VRAM) detected: Recommended batch_size=8, grad_accum=4, "
                    "mixed precision=FP16, and dynamic padding to prevent CUDA OOM."
                )
            elif profile.vram_total_mb <= 6500:
                profile.recommended_batch_size = 16
                profile.recommended_grad_accum = 2
                profile.notes = (
                    "RTX 3050 (6GB VRAM) detected: Recommended batch_size=16, grad_accum=2, "
                    "mixed precision=FP16."
                )
            else:
                profile.recommended_batch_size = 32
                profile.recommended_grad_accum = 1
                profile.notes = "High-VRAM GPU (>=8GB) detected: Standard batch_size=32 enabled."
        else:
            profile.notes = (
                "Running in CPU-only mode. Classical scikit-learn models execute quickly. "
                "For neural models, use compact checkpoints (e.g. DistilBERT, MiniLM)."
            )

        return profile

    @staticmethod
    def clear_cuda_cache() -> None:
        """Flushes cached CUDA memory and triggers garbage collection to avoid fragmentation."""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    @staticmethod
    def get_autocast_context(device: str = "cuda", enabled: bool = True):
        """Returns modern PyTorch autocast context for mixed precision."""
        if device == "cuda" and torch.cuda.is_available() and enabled:
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return torch.amp.autocast(device_type="cpu", enabled=False)
