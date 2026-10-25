#!/usr/bin/env python3
"""Verify the optimized Ornith kernels on the visible CUDA device."""

from __future__ import annotations

import json
import sys
import time
import warnings

import causal_conv1d
import fla
import torch
import transformers
import triton
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def main() -> int:
    device = torch.device("cuda:0")
    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "triton_target": str(triton.runtime.driver.active.get_current_target()),
        "transformers": transformers.__version__,
        "fla": fla.__version__,
        "causal_conv1d": causal_conv1d.__version__,
    }

    x = torch.randn(1, 8192, 2048, device=device, dtype=torch.bfloat16)
    weight = torch.randn(8192, 4, device=device, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(8192, device=device, dtype=torch.bfloat16) * 0.05
    started = time.perf_counter()
    output = causal_conv1d_fn(x, weight, bias, activation="silu")
    torch.cuda.synchronize()
    info["causal_conv"] = {
        "shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all().item()),
        "cold_seconds": round(time.perf_counter() - started, 6),
    }

    length = 2048
    query = torch.randn(1, length, 16, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, length, 16, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, length, 32, 128, device=device, dtype=torch.bfloat16)
    decay = -torch.rand(1, length, 32, device=device, dtype=torch.float32)
    beta = torch.rand(1, length, 32, device=device, dtype=torch.float32)
    started = time.perf_counter()
    attention, final_state = chunk_gated_delta_rule(
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
    torch.cuda.synchronize()
    info["gated_delta"] = {
        "output_shape": list(attention.shape),
        "final_state_shape": list(final_state.shape),
        "finite": bool(
            torch.isfinite(attention).all().item()
            and torch.isfinite(final_state).all().item()
        ),
        "cold_seconds": round(time.perf_counter() - started, 6),
    }

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
        started = time.perf_counter()
        logits = model(input_ids=token_ids, use_cache=False).logits
        torch.cuda.synchronize()
    info["transformers_linear_layer"] = {
        "logits_shape": list(logits.shape),
        "finite": bool(torch.isfinite(logits).all().item()),
        "cold_seconds": round(time.perf_counter() - started, 6),
        "fallback_warnings": [
            str(item.message)
            for item in caught
            if "falling back" in str(item.message).casefold()
        ],
    }
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
