#!/usr/bin/env python3
"""Benchmark routed-expert chunks on one Ornith-shaped CUDA layer."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time

import torch
from mlx_gptq.ornith_qat import FusedExpertsQAT, plan_expert_chunks


class SourceExperts(torch.nn.Module):
    """Synthetic source with the concatenated Ornith expert layout."""

    def __init__(
        self, experts: int, hidden: int, intermediate: int, device: torch.device
    ) -> None:
        super().__init__()
        factory: dict[str, object] = {
            "device": device,
            "dtype": torch.bfloat16,
        }
        self.gate_up_proj = torch.nn.Parameter(
            torch.randn(experts, 2 * intermediate, hidden, **factory) * 0.02
        )
        self.down_proj = torch.nn.Parameter(
            torch.randn(experts, hidden, intermediate, **factory) * 0.02
        )
        self.act_fn = torch.nn.SiLU()
        self.num_experts = experts
        self.hidden_dim = hidden
        self.intermediate_dim = intermediate
        self.has_gate = True
        self.has_bias = False
        self.is_concatenated = True
        self.is_transposed = False


def routing(
    tokens: int, top_k: int, experts: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(experts, (tokens, top_k), generator=generator)
    for expert in range(experts):
        indices[expert % tokens, 0] = expert
    weights = torch.rand(tokens, top_k, generator=generator)
    return indices.to(device), weights.to(device, torch.bfloat16)


def stats(samples: list[float]) -> dict[str, float | int]:
    return {
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "samples": len(samples),
    }


def measure_arm(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Time one forward and one training step of an interleaved arm."""
    module.expert_chunk_size = chunk_size
    module.zero_grad(set_to_none=True)
    hidden_states.grad = None

    with torch.no_grad():
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        module(hidden_states, top_k_index, top_k_weights)
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started

    module.zero_grad(set_to_none=True)
    hidden_states.grad = None
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = module(hidden_states, top_k_index, top_k_weights)
    output.float().square().sum().backward()
    torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - started

    module.zero_grad(set_to_none=True)
    hidden_states.grad = None
    return {
        "forward_seconds": forward_seconds,
        "step_seconds": step_seconds,
    }


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    difference = actual.float() - reference.float()
    denominator = reference.float().norm()
    return float((difference.norm() / denominator.clamp_min(1e-12)).item())


def tensor_mib(tensors: tuple[torch.Tensor, ...]) -> float:
    bytes_ = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    return round(bytes_ / 2**20, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--chunk-sizes", default="0,8,16,32", help="comma-separated sizes"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--no-checkpoint-experts",
        action="store_true",
        help="measure eager expert graphs",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    source = SourceExperts(
        args.experts, args.hidden, args.intermediate, device
    )
    module = FusedExpertsQAT(
        source,
        rank=args.rank,
        alpha=args.alpha,
        quantize=True,
        checkpoint_experts=not args.no_checkpoint_experts,
        expert_chunk_size=0,
    ).to(device=device)
    with torch.no_grad():
        module.gate_up_lora.lora_b.normal_(0.0, 0.02)
        module.down_lora.lora_b.normal_(0.0, 0.02)
    hidden_states = torch.randn(
        args.tokens, args.hidden, device=device, dtype=torch.bfloat16,
        requires_grad=True,
    )
    top_k_index, top_k_weights = routing(
        args.tokens, args.top_k, args.experts, args.seed + 1, device
    )
    routed = torch.bincount(
        top_k_index.reshape(-1), minlength=args.experts
    ).tolist()
    chunk_sizes = [
        int(value)
        for value in args.chunk_sizes.split(",")
        if value.strip()
    ]

    def clear_gradients() -> None:
        module.zero_grad(set_to_none=True)
        hidden_states.grad = None

    def release_memory() -> None:
        gc.collect()
        torch.cuda.empty_cache()

    module.expert_chunk_size = 0
    module.zero_grad(set_to_none=True)
    reference = module(hidden_states, top_k_index, top_k_weights)
    reference_loss = reference.float().square().sum()
    reference_loss.backward()
    reference_output = reference.detach().clone()
    reference_gate_grad = module.gate_up_lora.lora_b.grad.detach().clone()
    reference_down_grad = module.down_lora.lora_b.grad.detach().clone()
    reference_hidden_grad = hidden_states.grad.detach().clone()
    del reference, reference_loss
    release_memory()

    parity: dict[int, dict[str, float]] = {}
    for chunk_size in chunk_sizes:
        module.expert_chunk_size = chunk_size
        clear_gradients()
        release_memory()
        output = module(hidden_states, top_k_index, top_k_weights)
        output.float().square().sum().backward()
        parity[chunk_size] = {
            "output_relative_l2": relative_l2(output.detach(), reference_output),
            "gate_lora_gradient_relative_l2": relative_l2(
                module.gate_up_lora.lora_b.grad, reference_gate_grad
            ),
            "down_lora_gradient_relative_l2": relative_l2(
                module.down_lora.lora_b.grad, reference_down_grad
            ),
            "hidden_gradient_relative_l2": relative_l2(
                hidden_states.grad, reference_hidden_grad
            ),
        }
        del output
        clear_gradients()
        release_memory()

    for _ in range(args.warmup):
        for chunk_size in chunk_sizes:
            measure_arm(
                module,
                hidden_states,
                top_k_index,
                top_k_weights,
                chunk_size,
                device,
            )

    forward_samples: dict[int, list[float]] = {c: [] for c in chunk_sizes}
    step_samples: dict[int, list[float]] = {c: [] for c in chunk_sizes}
    for _ in range(args.rounds):
        for chunk_size in chunk_sizes:
            measurement = measure_arm(
                module,
                hidden_states,
                top_k_index,
                top_k_weights,
                chunk_size,
                device,
            )
            forward_samples[chunk_size].append(measurement["forward_seconds"])
            step_samples[chunk_size].append(measurement["step_seconds"])

    baseline = chunk_sizes[0]
    rows: list[dict[str, object]] = []
    for chunk_size in chunk_sizes:
        forward_delta = [
            value - base
            for value, base in zip(
                forward_samples[chunk_size], forward_samples[baseline]
            )
        ]
        step_delta = [
            value - base
            for value, base in zip(
                step_samples[chunk_size], step_samples[baseline]
            )
        ]
        if chunk_size:
            plans = plan_expert_chunks(routed, chunk_size)
            chunks = len(plans)
            padded_rows = sum(plan.padded_rows for plan in plans)
        else:
            chunks = args.experts
            padded_rows = sum(routed)
        rows.append(
            {
                "chunk_size": chunk_size,
                "chunks": chunks,
                "padded_rows": padded_rows,
                "routed_rows": sum(routed),
                "forward": stats(forward_samples[chunk_size]),
                "training_step": stats(step_samples[chunk_size]),
                "paired_forward_delta_seconds": statistics.median(forward_delta),
                "paired_training_step_delta_seconds": statistics.median(
                    step_delta
                ),
                "paired_forward_delta_samples": forward_delta,
                "paired_training_step_delta_samples": step_delta,
                **parity[chunk_size],
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    frozen = (
        module.gate_up_proj.detach(),
        module.down_proj.detach(),
    )
    adapters = (
        module.gate_up_lora.lora_a.detach(),
        module.gate_up_lora.lora_b.detach(),
        module.down_lora.lora_a.detach(),
        module.down_lora.lora_b.detach(),
    )
    result = {
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "shape": {
            "experts": args.experts,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "tokens": args.tokens,
            "top_k": args.top_k,
            "rank": args.rank,
        },
        "checkpoint_experts": not args.no_checkpoint_experts,
        "frozen_weight_mib": tensor_mib(frozen),
        "lora_factor_mib": tensor_mib(adapters),
        "rows": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
