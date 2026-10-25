#!/usr/bin/env python3
"""Measure warm optimized-kernel latency for one Ornith linear-attention layer."""

from __future__ import annotations

import json
import statistics
import time
import warnings
from collections.abc import Callable

import causal_conv1d
import fla
import torch
import transformers
import triton
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def benchmark(
    call: Callable[[], object], warmup: int, iterations: int
) -> dict[str, float | int]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return {
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "samples": len(samples),
    }


def main() -> int:
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    length = 2048

    x = torch.randn(1, 8192, length, device=device, dtype=torch.bfloat16)
    conv_weight = torch.randn(8192, 4, device=device, dtype=torch.bfloat16) * 0.05
    conv_bias = torch.randn(8192, device=device, dtype=torch.bfloat16) * 0.05
    conv = benchmark(
        lambda: causal_conv1d_fn(x, conv_weight, conv_bias, activation="silu"),
        warmup=3,
        iterations=20,
    )

    query = torch.randn(1, length, 16, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, length, 16, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, length, 32, 128, device=device, dtype=torch.bfloat16)
    decay = -torch.rand(1, length, 32, device=device, dtype=torch.float32)
    beta = torch.rand(1, length, 32, device=device, dtype=torch.float32)

    def delta() -> object:
        return chunk_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            scale=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )

    delta_bench = benchmark(delta, warmup=5, iterations=30)

    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    config = Qwen3_5MoeTextConfig(
        vocab_size=4096,
        hidden_size=2048,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=1,
        num_experts_per_tok=1,
        max_position_embeddings=4096,
        pad_token_id=0,
        layer_types=["linear_attention"],
    )
    model = Qwen3_5MoeForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()
    token_ids = torch.randint(0, 4096, (1, length), device=device)
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer = benchmark(
            lambda: model(input_ids=token_ids, use_cache=False).logits,
            warmup=3,
            iterations=20,
        )
        fallbacks = [
            str(item.message)
            for item in caught
            if "falling back" in str(item.message).casefold()
        ]

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "versions": {
                    "torch": torch.__version__,
                    "triton": triton.__version__,
                    "transformers": transformers.__version__,
                    "fla": fla.__version__,
                    "causal_conv1d": causal_conv1d.__version__,
                    "triton_target": str(
                        triton.runtime.driver.active.get_current_target()
                    ),
                },
                "sequence_length": length,
                "causal_conv": conv,
                "gated_delta_rule": delta_bench,
                "one_linear_layer_model": layer,
                "fallback_warnings": fallbacks,
                "peak_gpu_mib": round(
                    torch.cuda.max_memory_allocated(device) / 1024**2, 1
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
