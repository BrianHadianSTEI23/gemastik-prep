#!/usr/bin/env python3
"""
Hardware Benchmark Utility for NLP Experimentation Workbench.
Evaluates model throughput, latency, sequence scaling, and peak VRAM allocation
on host hardware (RTX 3050 4GB/6GB/8GB or CPU).
"""

import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nlp_workbench.hardware import HardwareManager


def run_benchmark(
    checkpoint: str = "distilbert-base-uncased",
    device_pref: str = "auto",
    batch_sizes: Optional[List[int]] = None,
    seq_lengths: Optional[List[int]] = None,
    precision: str = "fp16",
    num_warmup: int = 3,
    num_iters: int = 10,
) -> Dict[str, Any]:
    """
    Benchmarks transformer forward pass across varying batch sizes and sequence lengths.
    Measures latency, throughput, and peak memory.
    """
    profile = HardwareManager.inspect(device_pref)
    device = profile.get_torch_device()
    is_cuda = (device.type == "cuda")

    batch_sizes = batch_sizes or ([1, 4, 8, 16] if is_cuda else [1, 2, 4])
    seq_lengths = seq_lengths or [64, 128, 256]

    print(f"\n{profile.summary()}\n")
    print(f"Benchmarking checkpoint: {checkpoint}")
    print(f"Precision: {precision.upper()}, Device: {device}")
    print("-" * 75)
    print(f"{'Batch':<8}{'SeqLen':<10}{'Throughput (samp/s)':<22}{'Mean Lat (ms)':<16}{'Peak Mem (MB)':<15}")
    print("-" * 75)

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device)
    model.eval()

    use_fp16 = is_cuda and precision.lower() in ["fp16", "half", "mixed"]

    results = []

    for bs in batch_sizes:
        for seq_len in seq_lengths:
            # Dummy tokens
            dummy_input_ids = torch.randint(100, 2000, (bs, seq_len), dtype=torch.long, device=device)
            dummy_attn = torch.ones((bs, seq_len), dtype=torch.long, device=device)

            if is_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=device)

            # Warmup
            try:
                for _ in range(num_warmup):
                    with torch.no_grad():
                        if use_fp16:
                            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                                _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attn)
                        else:
                            _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attn)
                if is_cuda:
                    torch.cuda.synchronize(device=device)

                # Timed iterations
                latencies = []
                for _ in range(num_iters):
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        if use_fp16:
                            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                                _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attn)
                        else:
                            _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attn)
                    if is_cuda:
                        torch.cuda.synchronize(device=device)
                    latencies.append((time.perf_counter() - t0) * 1000)

                mean_lat = float(np.mean(latencies))
                p50_lat = float(np.percentile(latencies, 50))
                p95_lat = float(np.percentile(latencies, 95))
                throughput = float((bs * 1000.0) / max(1e-5, mean_lat))

                peak_mem_mb = 0.0
                if is_cuda:
                    peak_mem_mb = float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024))

                print(f"{bs:<8}{seq_len:<10}{throughput:<22.1f}{mean_lat:<16.2f}{peak_mem_mb:<15.1f}")

                results.append({
                    "batch_size": bs,
                    "seq_length": seq_len,
                    "throughput_samples_per_sec": round(throughput, 2),
                    "mean_latency_ms": round(mean_lat, 2),
                    "p50_latency_ms": round(p50_lat, 2),
                    "p95_latency_ms": round(p95_lat, 2),
                    "peak_memory_mb": round(peak_mem_mb, 2),
                    "status": "SUCCESS",
                })
            except torch.cuda.OutOfMemoryError:
                print(f"{bs:<8}{seq_len:<10}{'CUDA OOM':<22}{'N/A':<16}{'OOM':<15}")
                results.append({
                    "batch_size": bs,
                    "seq_length": seq_len,
                    "status": "OOM",
                })
                if is_cuda:
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"{bs:<8}{seq_len:<10}{'ERROR':<22}{str(e)[:16]:<16}{'N/A':<15}")
                results.append({
                    "batch_size": bs,
                    "seq_length": seq_len,
                    "status": f"ERROR: {str(e)}",
                })

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": checkpoint,
        "device": str(device),
        "precision": precision,
        "hardware_profile": {
            "gpu_name": profile.gpu_name,
            "vram_total_mb": profile.vram_total_mb,
            "supports_fp16": profile.supports_fp16,
            "supports_bf16": profile.supports_bf16,
        },
        "benchmarks": results,
    }
    return report


def run_adapter_benchmark(
    model_id: str,
    task: Optional[str] = None,
    device_pref: str = "auto",
    n_samples: int = 32,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Benchmarks real workbench task adapter with representative inputs.
    Measures training optimizer step (if trainable), inference latency, throughput, and peak VRAM.
    """
    from nlp_workbench.registry import registry
    from nlp_workbench.data import DataLoader, RetrievalDatasetBundle

    rec = registry.get_model(model_id)
    eval_task = task or rec.supported_tasks[0]

    profile = HardwareManager.inspect(device_pref)
    device = profile.get_torch_device()
    is_cuda = (device.type == "cuda")

    print(f"\n{profile.summary()}\n")
    print(f"Benchmarking Task Adapter: {model_id} ({rec.approach_name})")
    print(f"Task: {eval_task}, Framework: {rec.model_framework}, Device: {device}")
    print("-" * 75)

    data = DataLoader.generate_synthetic_dataset(task=eval_task, n_samples=n_samples)
    adapter = registry.create_adapter(model_id, device=device_pref)

    train_report = {}
    if rec.is_trainable:
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=device)

        t_start = time.perf_counter()
        if isinstance(data, RetrievalDatasetBundle):
            adapter.index_corpus(data.corpus_df)
        else:
            adapter.fit(data, epochs=1, batch_size=batch_size)
        if is_cuda:
            torch.cuda.synchronize(device=device)
        train_duration = time.perf_counter() - t_start

        peak_vram_train = 0.0
        if is_cuda:
            peak_vram_train = float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024))

        throughput_train = float(n_samples / max(1e-5, train_duration))
        print(f"Training / Step Phase:")
        print(f"  Duration:           {train_duration:.3f} s")
        print(f"  Throughput:         {throughput_train:.1f} samples/s")
        print(f"  Peak VRAM (Train):  {peak_vram_train:.1f} MB")
        train_report = {
            "is_trainable": True,
            "train_duration_s": round(train_duration, 3),
            "train_throughput_samples_per_sec": round(throughput_train, 2),
            "peak_vram_train_mb": round(peak_vram_train, 1),
        }
    else:
        print(f"Model '{model_id}' is non-trainable / inference-only.")
        train_report = {"is_trainable": False}

    # Inference Benchmark
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)

    infer_input = data.queries_df.head(batch_size) if isinstance(data, RetrievalDatasetBundle) else data.head(batch_size)
    num_infer = len(infer_input)

    # Warmup
    try:
        _ = adapter.predict(infer_input)
    except Exception:
        pass
    if is_cuda:
        torch.cuda.synchronize(device=device)

    # Timed inference
    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = adapter.predict(infer_input)
        if is_cuda:
            torch.cuda.synchronize(device=device)
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_lat_batch = float(np.mean(latencies))
    mean_lat_sample = float(mean_lat_batch / max(1, num_infer))
    throughput_infer = float((num_infer * 1000.0) / max(1e-5, mean_lat_batch))

    peak_vram_infer = 0.0
    if is_cuda:
        peak_vram_infer = float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024))

    print(f"\nInference Phase (batch_size={num_infer}):")
    print(f"  Mean Latency/Sample:{mean_lat_sample:.2f} ms")
    print(f"  Throughput:         {throughput_infer:.1f} samples/s")
    print(f"  Peak VRAM (Infer):  {peak_vram_infer:.1f} MB")
    print("-" * 75)

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "approach_name": rec.approach_name,
        "task": eval_task,
        "device": str(device),
        "hardware_profile": {
            "gpu_name": profile.gpu_name,
            "vram_total_mb": profile.vram_total_mb,
            "supports_fp16": profile.supports_fp16,
        },
        "training_benchmark": train_report,
        "inference_benchmark": {
            "batch_size": num_infer,
            "mean_latency_per_sample_ms": round(mean_lat_sample, 2),
            "throughput_samples_per_sec": round(throughput_infer, 2),
            "peak_vram_infer_mb": round(peak_vram_infer, 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark NLP model throughput & memory on host hardware.")
    parser.add_argument("--model_id", type=str, default=None, help="Workbench model ID from architecture registry")
    parser.add_argument("--task", type=str, default=None, help="Task name for adapter benchmark")
    parser.add_argument("--checkpoint", type=str, default="distilbert-base-uncased", help="HuggingFace model checkpoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Hardware target")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"], help="Floating point precision")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=None, help="Batch sizes to evaluate")
    parser.add_argument("--seq_lengths", nargs="+", type=int, default=None, help="Sequence lengths to evaluate")
    parser.add_argument("--output_json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    if args.model_id:
        report = run_adapter_benchmark(
            model_id=args.model_id,
            task=args.task,
            device_pref=args.device,
        )
    else:
        report = run_benchmark(
            checkpoint=args.checkpoint,
            device_pref=args.device,
            batch_sizes=args.batch_sizes,
            seq_lengths=args.seq_lengths,
            precision=args.precision,
        )

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved benchmark report to: {out_p.resolve()}")


if __name__ == "__main__":
    main()
