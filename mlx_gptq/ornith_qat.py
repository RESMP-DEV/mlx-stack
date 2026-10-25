"""MXFP4 quantization-aware training for Qwen3_5Moe fused routed experts.

The Ornith-class target (Qwen3_5Moe, 40 layers, 192 routed experts, top-8
routing) stores every routed expert as two fused 3D tensors inside the
decoder layer:

    model.layers.<i>.mlp.experts.gate_up_proj   [E, 2 * I, H]
    model.layers.<i>.mlp.experts.down_proj       [E, H, I]

A 2D LoRA cannot address those tensors, so this module adds a *per-expert*
LoRA: a rank-r update for every expert of every layer, evaluated on the
effective weight ``W + scaling * B[e] @ A[e]`` and passed through the MXFP4
fake quantizer with a straight-through estimator (STE).

``mlx_gptq.qat.fake_quantize_mxfp4`` is reused unchanged; it already accepts
3D tensors and groups along the last axis, which is the reduction axis for
both fused expert weights.

The module owns four things:

* ``FusedExpertsQAT`` - a drop-in replacement for ``Qwen3_5MoeExperts`` that
  keeps the upstream parameter names so a merged export reproduces the
  original checkpoint key layout. It routes tokens either one expert at a
  time (the sequential reference path) or in vectorized chunks of experts
  (the default path).
* ``install_expert_qat_lora`` - the monkey-patch that swaps every fused expert
  module in a loaded model and freezes everything else.
* a token-window dataset, an AdamW training loop with checkpoint/resume, and
  an adapter/merge export path.
* ``preflight`` - a dry run that reports the expert inventory and the LoRA
  budget from a config alone, without loading or downloading any weight.

Non-goals: this trainer does not claim model quality, does not select
routing experts, and does not train the shared expert or the router.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import accumulate
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint

from .grid import MX_GROUP_SIZE, choose_mxfp_scale, quantize_mxfp_with_scale
from .qat import fake_quantize_mxfp4

ORNITH_EXPERTS = 192
ORNITH_TOP_K = 8
ORNITH_LAYERS = 40
ORNITH_SEQ_LEN = 2048
ORNITH_RANK = 16
ORNITH_ALPHA = 32.0
ORNITH_EXPERT_CHUNK_SIZE = 16

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _is_adapter_key(name: str) -> bool:
    """Return True for LoRA factor parameter names."""
    return name.endswith("lora_a") or name.endswith("lora_b")


class ExpertLoRA(torch.nn.Module):
    """Rank-r LoRA update applied independently to every expert.

    The fused expert tensor has shape ``[experts, rows, cols]``. The update
    for expert ``e`` is ``scaling * B[e] @ A[e]`` with ``A`` of shape
    ``[experts, rank, cols]`` and ``B`` of shape ``[experts, rows, rank]``.
    ``B`` starts at zero, so an untrained adapter reproduces the base model.
    """

    def __init__(
        self,
        experts: int,
        rows: int,
        cols: int,
        rank: int,
        alpha: float,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if rank > min(rows, cols):
            raise ValueError(
                f"LoRA rank {rank} exceeds the expert matrix rank {min(rows, cols)}"
            )
        self.experts = int(experts)
        self.rows = int(rows)
        self.cols = int(cols)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        factory: dict[str, Any] = {}
        if device is not None:
            factory["device"] = device
        self.lora_a = torch.nn.Parameter(
            torch.empty(self.experts, self.rank, self.cols, **factory)
        )
        self.lora_b = torch.nn.Parameter(
            torch.zeros(self.experts, self.rows, self.rank, **factory)
        )
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def delta(self, index: int) -> torch.Tensor:
        """Return the scaled update for one expert, shaped ``[rows, cols]``."""
        return (self.lora_b[index] @ self.lora_a[index]) * self.scaling

    def extra_repr(self) -> str:
        return (
            f"experts={self.experts}, rows={self.rows}, cols={self.cols}, "
            f"rank={self.rank}, scaling={self.scaling}"
        )


@dataclass
class ExpertChunkPlan:
    """One group of routed experts in the vectorized chunk path.

    ``experts`` holds the expert ids of the group in ascending order and
    ``token_counts`` holds the number of routed tokens of each of them.
    ``max_tokens`` is the padded token count of the group, so the batched
    matmul of the group evaluates ``len(experts) * max_tokens`` token rows
    while only ``routed_rows`` of them carry a routed token.
    """

    experts: list[int]
    token_counts: list[int]
    max_tokens: int

    @property
    def padded_rows(self) -> int:
        """Return the token rows the batched matmul of the group evaluates."""
        return len(self.experts) * self.max_tokens

    @property
    def routed_rows(self) -> int:
        """Return the token rows the routing of the group actually needs."""
        return sum(self.token_counts)


def plan_expert_chunks(
    counts: Sequence[int], chunk_size: int
) -> list[ExpertChunkPlan]:
    """Group the routed experts into chunks of at most ``chunk_size`` experts.

    Experts without a routed token are dropped: an empty expert would only
    add padded rows. The remaining experts keep their ascending id order, so
    the plan of a routing is deterministic.

    A larger chunk needs fewer batched matmuls but pads every expert of the
    chunk to the largest token count of the chunk, so the two effects trade
    against each other. ``plan.padded_rows`` and ``plan.routed_rows`` make
    that tradeoff measurable without a GPU.
    """
    if chunk_size <= 0:
        raise ValueError("the expert chunk size must be positive")
    hits = [
        (expert, int(count))
        for expert, count in enumerate(counts)
        if int(count) > 0
    ]
    plans: list[ExpertChunkPlan] = []
    for start in range(0, len(hits), chunk_size):
        group = hits[start : start + chunk_size]
        plans.append(
            ExpertChunkPlan(
                experts=[expert for expert, _ in group],
                token_counts=[count for _, count in group],
                max_tokens=max(count for _, count in group),
            )
        )
    return plans


class FusedExpertsQAT(torch.nn.Module):
    """Drop-in replacement for ``Qwen3_5MoeExperts`` with per-expert LoRA QAT.

    The base tensors keep their upstream names (``gate_up_proj`` and
    ``down_proj``) and are reused, not copied, so installing the adapter does
    not duplicate the frozen weights and a merged export reproduces the
    original checkpoint key layout. The LoRA factors are added as extra
    parameters.

    Every forward differs from the upstream eager expert loop in two ways:
    the effective weight of a hit expert carries a LoRA update and goes
    through the MXFP4 fake quantizer, and the expert computation can be
    wrapped in gradient checkpointing so a 192-expert layer does not retain
    one quantized weight per hit expert until backward.

    Two routing paths exist. The sequential path walks the routed experts
    one at a time and is the reference and compatibility mode. The chunked
    path groups the routed experts into groups of at most
    ``expert_chunk_size`` experts, pads every group to the largest token
    count of the group and evaluates the group with batched matmuls, so a
    192-expert layer needs a handful of kernel launches instead of 192.
    ``expert_chunk_size=0`` selects the sequential path.
    """

    def __init__(
        self,
        source: torch.nn.Module,
        rank: int = ORNITH_RANK,
        alpha: float = ORNITH_ALPHA,
        quantize: bool = True,
        checkpoint_experts: bool = False,
        expert_chunk_size: int = ORNITH_EXPERT_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        if int(expert_chunk_size) < 0:
            raise ValueError(
                "the expert chunk size must be zero for the sequential path "
                "or positive for the vectorized path"
            )
        gate_up = getattr(source, "gate_up_proj", None)
        down = getattr(source, "down_proj", None)
        if not isinstance(gate_up, torch.Tensor) or gate_up.ndim != 3:
            raise ValueError("gate_up_proj must be a 3D tensor")
        if not isinstance(down, torch.Tensor) or down.ndim != 3:
            raise ValueError("down_proj must be a 3D tensor")
        if gate_up.shape[0] != down.shape[0]:
            raise ValueError("gate_up_proj and down_proj disagree on expert count")
        experts, rows, cols = gate_up.shape
        if bool(getattr(source, "has_gate", True)) is False:
            raise ValueError("the trainer needs a gated gate/up expert weight")
        if bool(getattr(source, "is_concatenated", True)) is False:
            raise ValueError(
                "the trainer needs the concatenated [gate; up] expert layout"
            )
        if bool(getattr(source, "is_transposed", False)):
            raise ValueError("the trainer needs a non-transposed expert layout")
        if rows != 2 * down.shape[2]:
            raise ValueError(
                f"gate_up_proj has {rows} rows but the concatenated [gate; up] "
                f"layout needs 2 * {down.shape[2]}"
            )
        if cols % MX_GROUP_SIZE or down.shape[2] % MX_GROUP_SIZE:
            raise ValueError(
                "MXFP4 QAT needs the expert reduction axis to be a multiple of "
                f"{MX_GROUP_SIZE}"
            )
        self.num_experts = int(getattr(source, "num_experts", experts))
        self.hidden_dim = int(getattr(source, "hidden_dim", cols))
        self.intermediate_dim = int(getattr(source, "intermediate_dim", rows // 2))
        self.act_fn = source.act_fn
        self.config = getattr(source, "config", None)
        self.has_gate = bool(getattr(source, "has_gate", True))
        self.has_bias = bool(getattr(source, "has_bias", False))
        self.is_transposed = bool(getattr(source, "is_transposed", False))
        self.is_concatenated = bool(getattr(source, "is_concatenated", True))
        self.quantize = bool(quantize)
        self.checkpoint_experts = bool(checkpoint_experts)
        self.expert_chunk_size = int(expert_chunk_size)
        self._delta_mode: str | None = None
        self.merged = False
        # Reuse the frozen parameters instead of copying them.
        self.gate_up_proj = gate_up
        self.down_proj = down
        self.gate_up_proj.requires_grad_(False)
        self.down_proj.requires_grad_(False)
        device = gate_up.device
        self.gate_up_lora = ExpertLoRA(experts, rows, cols, rank, alpha, device)
        self.down_lora = ExpertLoRA(
            experts,
            down.shape[1],
            down.shape[2],
            rank,
            alpha,
            device,
        )

    def effective_gate_up(self, index: int) -> torch.Tensor:
        """Return the quantized effective gate/up weight of one expert."""
        return self._effective(self.gate_up_proj, self.gate_up_lora, index)

    def effective_down(self, index: int) -> torch.Tensor:
        """Return the quantized effective down weight of one expert."""
        return self._effective(self.down_proj, self.down_lora, index)

    def _effective(
        self,
        base: torch.Tensor,
        lora: ExpertLoRA,
        index: int,
    ) -> torch.Tensor:
        if self.merged:
            return base[index]
        weight = base[index].float() + lora.delta(index)
        if self.quantize:
            weight = fake_quantize_mxfp4(weight)
        return weight.to(base.dtype)

    def _effective_chunk(
        self,
        base: torch.Tensor,
        lora: ExpertLoRA,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the quantized effective weights of a group of experts.

        The MXFP4 group runs along the last axis, so quantizing a stacked
        group of experts gives each expert exactly the codes the sequential
        path computes for it.
        """
        if self.merged:
            return base.index_select(0, expert_ids)
        weight = base.index_select(0, expert_ids).float() + self._delta_stack(
            lora, expert_ids
        )
        if self.quantize:
            weight = fake_quantize_mxfp4(weight)
        return weight.to(base.dtype)

    def _delta_stack(
        self, lora: ExpertLoRA, expert_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return the scaled LoRA update of every expert of a group.

        The batched product is used only when it is bit-identical to the
        per-expert product on this device, which the probe establishes once
        per module. Otherwise the exact per-expert product is kept, so the
        effective weight of an expert never moves off the sequential value.
        """
        if self._delta_mode == "exact":
            return torch.stack(
                [lora.delta(int(expert)) for expert in expert_ids.tolist()]
            )
        return (
            torch.bmm(
                lora.lora_b.index_select(0, expert_ids),
                lora.lora_a.index_select(0, expert_ids),
            )
            * lora.scaling
        )

    def _probe_delta_mode(self, expert_ids: torch.Tensor) -> str:
        """Return the LoRA delta mode this device can use.

        ``bmm`` is accepted only when both LoRA shapes produce their exact
        per-expert products bit for bit, so a quantized expert weight never
        moves off the sequential value. The probe runs once per module.
        """
        if self.merged or self.gate_up_lora is None:
            return "bmm"
        for lora in (self.gate_up_lora, self.down_lora):
            if lora is None:
                continue
            batched = (
                torch.bmm(
                    lora.lora_b.index_select(0, expert_ids),
                    lora.lora_a.index_select(0, expert_ids),
                )
                * lora.scaling
            )
            exact = torch.stack(
                [lora.delta(int(expert)) for expert in expert_ids.tolist()]
            )
            if not torch.equal(batched, exact):
                return "exact"
        return "bmm"

    def freeze_merged(self) -> None:
        """Drop the adapter once its update is merged into the base weight.

        The merged module then holds exactly the upstream parameters, so the
        model state dict matches an unadapted model again.
        """
        self.gate_up_lora = None
        self.down_lora = None
        self.merged = True

    def _expert_forward(
        self,
        expert_index: torch.Tensor,
        current_state: torch.Tensor,
        top_k_weights: torch.Tensor,
        token_index: torch.Tensor,
        top_k_position: torch.Tensor,
    ) -> torch.Tensor:
        gate_up = self.effective_gate_up(expert_index)
        gate, up = F.linear(current_state, gate_up).chunk(2, dim=-1)
        hidden = self.act_fn(gate) * up
        down = self.effective_down(expert_index)
        output = F.linear(hidden, down)
        return output * top_k_weights[token_index, top_k_position, None]

    def _sequential_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Route tokens to experts one expert at a time.

        This is the reference path: it keeps the upstream eager loop and is
        the compatibility mode of the trainer.
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(
                top_k_index, num_classes=self.num_experts + 1
            ).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_row in expert_hit:
            expert_index = expert_row[0]
            if expert_index == self.num_experts:
                continue
            top_k_position, token_index = torch.where(expert_mask[expert_index])
            current_state = hidden_states[token_index]
            if self.checkpoint_experts and torch.is_grad_enabled():
                output = checkpoint(
                    self._expert_forward,
                    expert_index,
                    current_state,
                    top_k_weights,
                    token_index,
                    top_k_position,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                output = self._expert_forward(
                    expert_index,
                    current_state,
                    top_k_weights,
                    token_index,
                    top_k_position,
                )
            final_hidden_states.index_add_(
                0, token_index, output.to(final_hidden_states.dtype)
            )
        return final_hidden_states

    def _chunked_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Route tokens to experts in vectorized chunks of experts.

        The routing is flattened, sorted by expert and grouped into chunks of
        at most ``expert_chunk_size`` experts. Every chunk is padded to the
        largest token count of the chunk and evaluated with batched
        matmuls, so the number of kernel launches per layer drops from one
        per routed expert to one per chunk.
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        tokens = hidden_states.shape[0]
        top_k = top_k_index.shape[1]
        if tokens == 0 or top_k == 0:
            return final_hidden_states
        device = hidden_states.device
        flat_expert = top_k_index.reshape(-1)
        flat_token = torch.arange(tokens, device=device).repeat_interleave(top_k)
        flat_weight = top_k_weights.reshape(-1)
        in_range = (flat_expert >= 0) & (flat_expert < self.num_experts)
        safe_expert = torch.where(
            in_range,
            flat_expert,
            torch.full_like(flat_expert, self.num_experts),
        )
        sorted_expert, order = torch.sort(safe_expert, stable=True)
        sorted_token = flat_token[order]
        sorted_weight = flat_weight[order]
        counts = torch.bincount(sorted_expert, minlength=self.num_experts + 1)
        counts_host = counts.tolist()
        routed_counts = counts_host[: self.num_experts]
        plans = plan_expert_chunks(routed_counts, self.expert_chunk_size)
        if not plans:
            return final_hidden_states
        offsets = [0] + list(accumulate(routed_counts))
        for plan in plans:
            expert_ids = torch.tensor(
                plan.experts, dtype=torch.long, device=device
            )
            if self._delta_mode is None:
                self._delta_mode = self._probe_delta_mode(expert_ids)
            starts = torch.tensor(
                [offsets[expert] for expert in plan.experts],
                dtype=torch.long,
                device=device,
            )
            counts_tensor = torch.tensor(
                plan.token_counts, dtype=torch.long, device=device
            )
            positions = torch.arange(plan.max_tokens, device=device)
            gather = starts.unsqueeze(1) + positions.unsqueeze(0)
            keep = positions.unsqueeze(0) < counts_tensor.unsqueeze(1)
            gather = torch.where(keep, gather, torch.zeros_like(gather))
            token_index = sorted_token[gather]
            token_weight = sorted_weight[gather]
            mask = keep.to(hidden_states.dtype)
            output = self._run_chunk(
                hidden_states, expert_ids, token_index, token_weight, mask
            )
            final_hidden_states.index_add_(
                0,
                token_index.reshape(-1),
                output.reshape(-1, hidden_states.shape[-1]),
            )
        return final_hidden_states

    def _run_chunk(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        token_index: torch.Tensor,
        token_weight: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate one chunk of experts, with or without checkpointing."""
        if self.checkpoint_experts and torch.is_grad_enabled():
            return checkpoint(
                self._chunk_forward,
                hidden_states,
                expert_ids,
                token_index,
                token_weight,
                mask,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._chunk_forward(
            hidden_states, expert_ids, token_index, token_weight, mask
        )

    def _chunk_forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        token_index: torch.Tensor,
        token_weight: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the experts of one chunk with batched matmuls.

        ``token_index`` holds the routed token of every padded row and
        ``mask`` marks the rows that carry a routed token, so the padded rows
        contribute nothing to the routed output.
        """
        state = hidden_states[token_index]
        gate_up = self._effective_chunk(
            self.gate_up_proj, self.gate_up_lora, expert_ids
        )
        intermediate = gate_up.shape[1] // 2
        gate = torch.bmm(state, gate_up[:, :intermediate, :].transpose(1, 2))
        up = torch.bmm(state, gate_up[:, intermediate:, :].transpose(1, 2))
        hidden = self.act_fn(gate) * up
        down = self._effective_chunk(self.down_proj, self.down_lora, expert_ids)
        output = torch.bmm(hidden, down.transpose(1, 2))
        scale = token_weight.unsqueeze(-1) * mask.unsqueeze(-1)
        return (output * scale).to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Route tokens to experts with quantized per-expert LoRA weights."""
        if self.expert_chunk_size > 0:
            return self._chunked_forward(
                hidden_states, top_k_index, top_k_weights
            )
        return self._sequential_forward(
            hidden_states, top_k_index, top_k_weights
        )

    def extra_repr(self) -> str:
        return (
            f"experts={self.num_experts}, hidden={self.hidden_dim}, "
            f"intermediate={self.intermediate_dim}, quantize={self.quantize}, "
            f"checkpoint_experts={self.checkpoint_experts}, "
            f"expert_chunk_size={self.expert_chunk_size}"
        )


def _module_tensor(module: torch.nn.Module, name: str) -> torch.Tensor | None:
    value = getattr(module, name, None)
    return value if isinstance(value, torch.Tensor) else None


def is_fused_expert_module(module: torch.nn.Module) -> bool:
    """Return True for a module holding fused 3D routed-expert weights."""
    gate_up = _module_tensor(module, "gate_up_proj")
    down = _module_tensor(module, "down_proj")
    if gate_up is None or down is None:
        return False
    if gate_up.ndim != 3 or down.ndim != 3:
        return False
    return gate_up.shape[0] == down.shape[0]


def find_expert_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """Return the ``(path, module)`` pairs of every fused expert module."""
    found: list[tuple[str, torch.nn.Module]] = []
    for path, module in model.named_modules():
        if is_fused_expert_module(module):
            found.append((path, module))
    return found


@dataclass
class InstalledAdapter:
    """Record of one replaced fused-expert module."""

    module_path: str
    experts: int
    gate_up_shape: tuple[int, int, int]
    down_shape: tuple[int, int, int]
    trainable_parameters: int


def install_expert_qat_lora(
    model: torch.nn.Module,
    rank: int = ORNITH_RANK,
    alpha: float = ORNITH_ALPHA,
    quantize: bool = True,
    checkpoint_experts: bool = False,
    expert_chunk_size: int = ORNITH_EXPERT_CHUNK_SIZE,
) -> list[InstalledAdapter]:
    """Swap every fused routed-expert module for a LoRA QAT adapter.

    Every other parameter is frozen, so the optimizer only ever sees the
    per-expert LoRA factors. The base tensors are reused in place, so no
    weight is duplicated.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets: list[tuple[torch.nn.Module, str, torch.nn.Module, str]] = []
    for parent_path, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not is_fused_expert_module(child):
                continue
            path = f"{parent_path}.{child_name}" if parent_path else child_name
            targets.append((parent, child_name, child, path))
    if not targets:
        raise RuntimeError(
            "no fused routed-expert module was found; the model does not expose "
            "3D gate_up_proj/down_proj expert weights"
        )
    installed: list[InstalledAdapter] = []
    for parent, child_name, child, path in targets:
        adapter = FusedExpertsQAT(
            child,
            rank=rank,
            alpha=alpha,
            quantize=quantize,
            checkpoint_experts=checkpoint_experts,
            expert_chunk_size=expert_chunk_size,
        )
        setattr(parent, child_name, adapter)
        trainable = sum(
            parameter.numel()
            for parameter in adapter.parameters()
            if parameter.requires_grad
        )
        installed.append(
            InstalledAdapter(
                module_path=path,
                experts=adapter.num_experts,
                gate_up_shape=tuple(adapter.gate_up_proj.shape),
                down_shape=tuple(adapter.down_proj.shape),
                trainable_parameters=trainable,
            )
        )
    return installed


def adapter_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Return the trainable LoRA parameters in a stable order."""
    return [
        parameter
        for name, parameter in model.named_parameters()
        if _is_adapter_key(name) and parameter.requires_grad
    ]


def adapter_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return the LoRA factors of a model as a CPU state dict."""
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if _is_adapter_key(name)
    }


def load_adapter_state_dict(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    """Load LoRA factors back into a model.

    Only the adapter tensors are compared: the frozen base weights are not
    part of an adapter checkpoint, so they are never reported as missing.
    """
    expected = {name for name in model.state_dict() if _is_adapter_key(name)}
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - expected)
    if missing:
        raise RuntimeError(
            f"the adapter checkpoint is missing {missing[:4]} of "
            f"{len(expected)} adapter tensors"
        )
    if unexpected:
        raise RuntimeError(
            f"the adapter checkpoint has unexpected keys {unexpected[:4]}"
        )
    model.load_state_dict(state, strict=False)


def text_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the text-model config of a multimodal or text-only config."""
    text = payload.get("text_config")
    if isinstance(text, dict):
        return text
    return payload


def lora_parameter_budget(
    experts: int,
    hidden: int,
    intermediate: int,
    layers: int,
    rank: int,
) -> dict[str, Any]:
    """Return the per-expert LoRA parameter budget of a fused-expert model.

    Per layer the adapter holds ``experts * rank * (2 * hidden + 3 * intermediate)``
    values: ``gate_up`` needs ``A`` of ``[E, r, H]`` plus ``B`` of
    ``[E, 2I, r]`` and ``down`` needs ``A`` of ``[E, r, I]`` plus ``B`` of
    ``[E, H, r]``.
    """
    gate_up = experts * rank * (hidden + 2 * intermediate)
    down = experts * rank * (intermediate + hidden)
    per_layer = gate_up + down
    total = per_layer * layers
    byte_total = total * 4
    return {
        "per_layer": per_layer,
        "total": total,
        "gate_up_per_layer": gate_up,
        "down_per_layer": down,
        "parameter_bytes_fp32": byte_total,
        "gradient_bytes_fp32": byte_total,
        "optimizer_bytes_fp32": 2 * byte_total,
        "trainer_bytes_fp32": 4 * byte_total,
        "trainer_gib_fp32": 4 * byte_total / (1024**3),
    }


def preflight(
    payload: dict[str, Any],
    rank: int = ORNITH_RANK,
    alpha: float = ORNITH_ALPHA,
    quantize: bool = True,
) -> dict[str, Any]:
    """Report the expert inventory and the LoRA budget from a config alone.

    No weight is loaded and no download happens, so this is safe to run on a
    queue host before the model is staged.
    """
    text = text_config_payload(payload)
    experts = int(text.get("num_experts", 0) or 0)
    top_k = int(text.get("num_experts_per_tok", 0) or 0)
    layers = int(text.get("num_hidden_layers", 0) or 0)
    hidden = int(text.get("hidden_size", 0) or 0)
    intermediate = int(text.get("moe_intermediate_size", 0) or 0)
    shared = int(text.get("shared_expert_intermediate_size", 0) or 0)
    vocab = int(text.get("vocab_size", 0) or 0)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check("config_has_experts", experts > 0, f"num_experts={experts}")
    check("config_has_layers", layers > 0, f"num_hidden_layers={layers}")
    check(
        "mx_group_hidden",
        hidden > 0 and hidden % MX_GROUP_SIZE == 0,
        f"hidden_size={hidden} must be a multiple of {MX_GROUP_SIZE}",
    )
    check(
        "mx_group_intermediate",
        intermediate > 0 and intermediate % MX_GROUP_SIZE == 0,
        f"moe_intermediate_size={intermediate} must be a multiple of "
        f"{MX_GROUP_SIZE}",
    )
    check(
        "rank_fits_gate_up",
        0 < rank <= min(hidden, 2 * intermediate),
        f"rank={rank} must fit [2I, H] = [{2 * intermediate}, {hidden}]",
    )
    check(
        "rank_fits_down",
        0 < rank <= min(hidden, intermediate),
        f"rank={rank} must fit [H, I] = [{hidden}, {intermediate}]",
    )
    check(
        "top_k_within_experts",
        0 < top_k <= experts,
        f"num_experts_per_tok={top_k} within num_experts={experts}",
    )
    budget = lora_parameter_budget(experts, hidden, intermediate, layers, rank)
    return {
        "model_type": text.get("model_type"),
        "experts": experts,
        "experts_per_token": top_k,
        "layers": layers,
        "hidden_size": hidden,
        "moe_intermediate_size": intermediate,
        "shared_expert_intermediate_size": shared,
        "vocab_size": vocab,
        "rank": rank,
        "alpha": alpha,
        "scaling": alpha / rank if rank else 0.0,
        "quantize_mxfp4": quantize,
        "mx_group_size": MX_GROUP_SIZE,
        "expected_expert_modules": layers,
        "expected_expert_path": "model.layers.<i>.mlp.experts",
        "expected_expert_path_multimodal": (
            "model.language_model.layers.<i>.mlp.experts"
        ),
        "budget": budget,
        "checks": checks,
        "passed": all(entry["ok"] for entry in checks),
        "non_claims": [
            "Preflight reads a config only; no weight is loaded or downloaded.",
            "Budget numbers are parameter counts, not a measured memory peak.",
        ],
    }


def load_token_ids(
    path: str | Path,
    tokenizer: str | None = None,
    max_tokens: int = 0,
    add_special_tokens: bool = True,
) -> list[int]:
    """Load a flat token-id stream from a ``.npy`` file or a text corpus.

    A ``.npy`` input holds one token id per element. A text input needs a
    tokenizer, because the token ids of the calibration corpus have to match
    the vocabulary the model was trained with.
    """
    source = Path(path)
    if source.suffix == ".npy":
        array = np.load(source)
        if array.ndim != 1:
            raise ValueError(f"expected a 1D token array, got shape {array.shape}")
        token_ids = [int(value) for value in array.tolist()]
    else:
        if not tokenizer:
            raise ValueError(
                f"{source} is not a .npy file; pass a tokenizer so the text can "
                "be encoded with the model vocabulary"
            )
        from transformers import AutoTokenizer

        encoder = AutoTokenizer.from_pretrained(tokenizer)
        text = source.read_text(encoding="utf-8")
        encoded = encoder(text, add_special_tokens=add_special_tokens)
        token_ids = [int(value) for value in encoded["input_ids"]]
    if max_tokens:
        token_ids = token_ids[:max_tokens]
    if not token_ids:
        raise ValueError(f"no tokens were read from {source}")
    return token_ids


class TokenWindowDataset(torch.utils.data.Dataset):
    """Causal training windows over a flat token-id stream.

    Window ``i`` covers tokens ``[i * stride, i * stride + seq_len + 1)``; the
    first token is the context and the remaining ones are the labels.
    """

    def __init__(
        self,
        token_ids: list[int],
        seq_len: int,
        stride: int | None = None,
        limit: int = 0,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        self.token_ids = list(token_ids)
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride else int(seq_len)
        if self.stride < 1:
            raise ValueError("stride must be positive")
        span = self.seq_len + 1
        count = 0
        if len(self.token_ids) >= span:
            count = (len(self.token_ids) - span) // self.stride + 1
        if limit:
            count = min(count, int(limit))
        self.starts = [index * self.stride for index in range(count)]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.starts[index]
        window = torch.tensor(
            self.token_ids[start : start + self.seq_len + 1], dtype=torch.long
        )
        return {"input_ids": window[:-1], "labels": window[1:]}

    def digest(self) -> str:
        """Return a stable digest of the window layout and the token stream."""
        hasher = hashlib.sha256()
        hasher.update(str(self.seq_len).encode())
        hasher.update(str(self.stride).encode())
        hasher.update(str(len(self.token_ids)).encode())
        hasher.update(np.asarray(self.token_ids, dtype=np.int64).tobytes())
        return hasher.hexdigest()


@dataclass
class TrainConfig:
    """Everything one Ornith MXFP4 QAT run needs."""

    model_path: str = ""
    model_class: str = "causal_lm"
    output_dir: str = "build/ornith-qat"
    data_path: str = ""
    tokenizer: str | None = None
    rank: int = ORNITH_RANK
    alpha: float = ORNITH_ALPHA
    seq_len: int = ORNITH_SEQ_LEN
    stride: int = 0
    epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 0.0
    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    loss_chunk: int = 512
    seed: int = 17
    quantize: bool = True
    checkpoint_experts: bool = True
    expert_chunk_size: int = ORNITH_EXPERT_CHUNK_SIZE
    gradient_checkpointing: bool = False
    device: str = "cuda"
    dtype: str = "bfloat16"
    log_every: int = 10
    max_steps: int = 0
    limit_windows: int = 0
    max_tokens: int = 0
    resume: bool = True
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def training_fingerprint(config: TrainConfig, dataset_digest: str) -> str:
    """Return a digest that ties a checkpoint to the run that produced it."""
    payload = {
        "model_path": config.model_path,
        "model_class": config.model_class,
        "rank": config.rank,
        "alpha": config.alpha,
        "seq_len": config.seq_len,
        "stride": config.stride,
        "quantize": config.quantize,
        "dataset": dataset_digest,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_base_model(config: TrainConfig) -> torch.nn.Module:
    """Load the frozen base model in the requested dtype."""
    from transformers import (
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
    )

    if config.dtype not in _DTYPES:
        raise ValueError(f"unsupported dtype: {config.dtype}")
    dtype = _DTYPES[config.dtype]
    if config.model_class == "causal_lm":
        loader = Qwen3_5MoeForCausalLM
    elif config.model_class == "conditional_generation":
        loader = Qwen3_5MoeForConditionalGeneration
    else:
        raise ValueError(
            f"unsupported model class: {config.model_class}; use causal_lm or "
            "conditional_generation"
        )
    model = loader.from_pretrained(config.model_path, dtype=dtype)
    model.config.use_cache = False
    return model


def resolve_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """Return the module that maps token ids to hidden states."""
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("the model does not expose a backbone module")
    return backbone


def _chunk_loss(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return the summed cross-entropy of one hidden-state chunk."""
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("the model does not expose an lm_head")

    def compute(hidden_chunk: torch.Tensor, target_chunk: torch.Tensor) -> torch.Tensor:
        logits = lm_head(hidden_chunk).float()
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_chunk.reshape(-1),
            reduction="sum",
        )

    if torch.is_grad_enabled():
        return checkpoint(
            compute,
            hidden,
            targets,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return compute(hidden, targets)


def causal_lm_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    loss_chunk: int = 0,
) -> torch.Tensor:
    """Return the mean next-token cross-entropy of one causal window.

    The vocabulary projection is evaluated in chunks so the logits of a
    2048-token window against a 248k vocabulary never materialize at once.
    """
    backbone = resolve_backbone(model)
    outputs = backbone(input_ids=input_ids, use_cache=False)
    hidden = outputs.last_hidden_state
    if labels is None:
        labels = input_ids[:, 1:]
        hidden = hidden[:, :-1, :]
    else:
        hidden = hidden[:, : labels.shape[1], :]
    width = labels.shape[1]
    if width == 0:
        raise ValueError("the causal window holds no label token")
    if loss_chunk and loss_chunk < width:
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        count = 0
        for start in range(0, width, loss_chunk):
            stop = min(start + loss_chunk, width)
            total = total + _chunk_loss(
                model, hidden[:, start:stop], labels[:, start:stop]
            )
            count += labels[:, start:stop].numel()
        return total / max(count, 1)
    return _chunk_loss(model, hidden, labels) / max(labels.numel(), 1)


def save_checkpoint(
    path: Path,
    fingerprint: str,
    epoch: int,
    next_epoch: int,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, Any]],
) -> None:
    """Write the adapter, optimizer state, and RNG state atomically.

    ``next_epoch`` is the epoch a resume starts from. An epoch that was cut
    short by the step cap is replayed instead of being counted as done.
    """
    payload = {
        "fingerprint": fingerprint,
        "epoch": epoch,
        "next_epoch": next_epoch,
        "step": step,
        "adapter": adapter_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "rng_state": torch.get_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def merged_expert_weight(
    base: torch.Tensor,
    lora: ExpertLoRA,
    quantize: bool = True,
) -> torch.Tensor:
    """Return the merged weight of a fused expert tensor.

    The expert loop is used so the peak transient memory stays at one expert
    instead of the whole fused tensor.
    """
    merged = torch.empty_like(base)
    for index in range(base.shape[0]):
        weight = base[index].float() + lora.delta(index)
        if quantize:
            weight = fake_quantize_mxfp4(weight)
        merged[index] = weight.to(base.dtype)
    return merged


def merged_state_dict(
    model: torch.nn.Module, quantize: bool = True
) -> dict[str, torch.Tensor]:
    """Return the model state dict with every adapter merged into its base.

    The returned keys are exactly the upstream checkpoint keys: the base
    tensors keep their names and the LoRA factors are dropped.
    """
    merged: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if _is_adapter_key(name):
            continue
        merged[name] = tensor.detach().cpu().clone()
    for module_path, module in model.named_modules():
        if not isinstance(module, FusedExpertsQAT):
            continue
        if module.merged:
            raise RuntimeError(
                "the adapter was already merged into the base weights"
            )
        pairs = (
            ("gate_up_proj", module.gate_up_lora),
            ("down_proj", module.down_lora),
        )
        for attribute, lora in pairs:
            key = f"{module_path}.{attribute}" if module_path else attribute
            base = getattr(module, attribute).detach().cpu()
            merged[key] = merged_expert_weight(base, lora, quantize)
    return merged


def mxfp4_codes(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the MXFP4 codes and E8M0 scale bytes of a fused expert tensor."""
    grouped = weight.detach().float().reshape(
        *weight.shape[:-1], -1, MX_GROUP_SIZE
    )
    scale_bytes, scales = choose_mxfp_scale(grouped, "mxfp4")
    codes, _effective = quantize_mxfp_with_scale(grouped, scales, "mxfp4")
    return (
        codes.reshape(weight.shape).contiguous(),
        scale_bytes.reshape(*weight.shape[:-1], -1).contiguous(),
    )


def export_adapter(
    model: torch.nn.Module,
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> Path:
    """Write the LoRA factors of a model as a safetensors adapter."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tensors = adapter_state_dict(model)
    if not tensors:
        raise RuntimeError(
            "the model holds no adapter tensor; it was merged or never "
            "adapted"
        )
    path = output / "adapter.safetensors"
    save_file(tensors, str(path))
    payload = dict(metadata)
    payload["tensors"] = sorted(tensors)
    (output / "adapter.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def merge_adapters_into_base(
    model: torch.nn.Module, quantize: bool = True
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Merge every adapter into its base weight in place.

    Returns the MXFP4 codes and scale bytes of the merged weights, keyed by
    module path and attribute. The codes are computed from the stored value,
    so they describe the exported bytes exactly.
    """
    merged: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for module_path, module in model.named_modules():
        if not isinstance(module, FusedExpertsQAT):
            continue
        entries: dict[str, dict[str, torch.Tensor]] = {}
        pairs = (
            ("gate_up_proj", module.gate_up_lora),
            ("down_proj", module.down_lora),
        )
        for attribute, lora in pairs:
            base = getattr(module, attribute)
            value = merged_expert_weight(base.detach(), lora, quantize)
            value = value.to(base.dtype)
            with torch.no_grad():
                base.copy_(value)
            if quantize:
                codes, scales = mxfp4_codes(value)
                entries[attribute] = {"q": codes, "scales": scales}
        merged[module_path] = entries
        module.freeze_merged()
    if not merged:
        raise RuntimeError("no adapter was found to merge")
    return merged


def serialized_checkpoint_keys(output_dir: Path) -> set[str]:
    """Return the tensor names of a saved model directory."""
    index = output_dir / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        return set(payload.get("weight_map", {}))
    single = output_dir / "model.safetensors"
    if single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as handle:
            return set(handle.keys())
    return set()


def expert_code_sidecar(
    merged: dict[str, dict[str, dict[str, torch.Tensor]]],
    serialized: set[str],
) -> dict[str, torch.Tensor]:
    """Return the per-expert MXFP4 codes keyed like the saved checkpoint.

    The fused ``gate_up`` tensor is split into its gate and up halves, which
    is exact: the MX group runs along the reduction axis, so a row of codes
    belongs to one half.
    """
    sidecar: dict[str, torch.Tensor] = {}
    for module_path, entries in merged.items():
        gate_up = entries.get("gate_up_proj")
        down = entries.get("down_proj")
        if gate_up is None or down is None:
            continue
        experts, rows, _columns = gate_up["q"].shape
        intermediate = rows // 2
        for expert in range(experts):
            halves = (
                ("gate_proj", slice(0, intermediate)),
                ("up_proj", slice(intermediate, rows)),
            )
            for part, rows_slice in halves:
                key = f"{module_path}.{expert}.{part}.weight"
                if key not in serialized:
                    continue
                sidecar[f"{key}.q"] = gate_up["q"][expert, rows_slice].contiguous()
                sidecar[f"{key}.scales"] = gate_up["scales"][
                    expert, rows_slice
                ].contiguous()
            key = f"{module_path}.{expert}.down_proj.weight"
            if key not in serialized:
                continue
            sidecar[f"{key}.q"] = down["q"][expert].contiguous()
            sidecar[f"{key}.scales"] = down["scales"][expert].contiguous()
    return sidecar


def export_merged(
    model: torch.nn.Module,
    output_dir: str | Path,
    metadata: dict[str, Any],
    quantize: bool = True,
    with_codes: bool = True,
) -> dict[str, Any]:
    """Merge every adapter into its base weight and write a model directory.

    The export goes through ``save_pretrained``, so the directory holds the
    stock checkpoint layout of the model class, including the per-expert
    tensors that the upstream checkpoint uses and the shard index of a large
    model. The model is moved to host memory first, so a 35B bf16 export
    needs roughly 70 GiB of RAM.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged = merge_adapters_into_base(model, quantize)
    model.to("cpu")
    model.save_pretrained(output, safe_serialization=True)
    serialized = serialized_checkpoint_keys(output)
    codes = expert_code_sidecar(merged, serialized) if with_codes else {}
    if codes:
        save_file(codes, str(output / "mxfp4.safetensors"))
    summary = {
        **metadata,
        "serialized_tensors": len(serialized),
        "mxfp4_code_tensors": len(codes),
        "merged_modules": len(merged),
        "quantized": quantize,
        "non_claims": [
            "The merged weights are an MXFP4 fake-quantized export, not a "
            "measured deployment artifact.",
            "No runtime kernel parity or serving throughput is proved here.",
        ],
    }
    (output / "export.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def train(config: TrainConfig) -> dict[str, Any]:
    """Run the Ornith MXFP4 QAT loop and return a run summary."""
    started = time.time()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    token_ids = load_token_ids(
        config.data_path, config.tokenizer, config.max_tokens
    )
    dataset = TokenWindowDataset(
        token_ids,
        config.seq_len,
        config.stride or None,
        config.limit_windows,
    )
    if not len(dataset):
        raise RuntimeError("the token stream holds no complete training window")
    fingerprint = training_fingerprint(config, dataset.digest())
    device = torch.device(config.device)
    model = load_base_model(config)
    model.to(device)
    if config.gradient_checkpointing and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()
    adapters = install_expert_qat_lora(
        model,
        rank=config.rank,
        alpha=config.alpha,
        quantize=config.quantize,
        checkpoint_experts=config.checkpoint_experts,
        expert_chunk_size=config.expert_chunk_size,
    )
    parameters = adapter_parameters(model)
    if not parameters:
        raise RuntimeError("no trainable adapter parameter was installed")
    plan = {
        "fingerprint": fingerprint,
        "adapters": [asdict(adapter) for adapter in adapters],
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters
        ),
        "windows": len(dataset),
        "seq_len": config.seq_len,
        "epochs": config.epochs,
        "device": str(device),
        "dtype": config.dtype,
        "quantize_mxfp4": config.quantize,
        "checkpoint_experts": config.checkpoint_experts,
        "expert_chunk_size": config.expert_chunk_size,
    }
    if config.dry_run:
        summary = {
            **plan,
            "dry_run": True,
            "elapsed_seconds": time.time() - started,
            "non_claims": [
                "A dry run loads the base model and installs the adapter; it "
                "performs no optimizer step and measures no throughput.",
            ],
        }
        (output / "dry_run.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    optimizer = torch.optim.AdamW(
        parameters, lr=config.lr, weight_decay=config.weight_decay
    )
    checkpoint_path = output / "checkpoint.pt"
    metrics_path = output / "metrics.jsonl"
    start_epoch = 0
    step = 0
    history: list[dict[str, Any]] = []
    if config.resume and checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "the checkpoint fingerprint does not match this run; the model, "
                "adapter shape, or dataset changed since the checkpoint"
            )
        load_adapter_state_dict(model, payload["adapter"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("next_epoch", int(payload["epoch"]) + 1))
        step = int(payload["step"])
        history = list(payload.get("history", []))
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state.to(torch.uint8))

    micro = max(1, int(config.micro_batch_size))
    accum = max(1, int(config.grad_accum_steps))
    stopped = False
    for epoch in range(start_epoch, config.epochs):
        model.train()
        order = list(range(len(dataset)))
        random.Random(config.seed + epoch).shuffle(order)
        epoch_loss = 0.0
        epoch_batches = 0
        for group_start in range(0, len(order), micro * accum):
            group = order[group_start : group_start + micro * accum]
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            micro_batches = 0
            for offset in range(0, len(group), micro):
                chunk = group[offset : offset + micro]
                input_ids = torch.stack(
                    [dataset[index]["input_ids"] for index in chunk]
                ).to(device)
                labels = torch.stack(
                    [dataset[index]["labels"] for index in chunk]
                ).to(device)
                loss = causal_lm_loss(
                    model, input_ids, labels, config.loss_chunk
                )
                (loss / accum).backward()
                group_loss += float(loss.detach().item())
                micro_batches += 1
            group_loss = group_loss / max(micro_batches, 1)
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()
            step += 1
            epoch_loss += group_loss
            epoch_batches += 1
            record = {
                "epoch": epoch,
                "step": step,
                "loss": group_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if config.log_every and step % config.log_every == 0:
                print(json.dumps(record, sort_keys=True), flush=True)
            if config.max_steps and step >= config.max_steps:
                stopped = True
                break
        epoch_record = {
            "epoch": epoch,
            "step": step,
            "mean_loss": epoch_loss / max(epoch_batches, 1),
            "batches": epoch_batches,
        }
        history.append(epoch_record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, sort_keys=True) + "\n")
        save_checkpoint(
            checkpoint_path,
            fingerprint,
            epoch,
            epoch if stopped else epoch + 1,
            step,
            model,
            optimizer,
            history,
        )
        if stopped:
            break
    summary = {
        **plan,
        "dry_run": False,
        "steps": step,
        "epochs_completed": start_epoch + (0 if stopped else 1),
        "history": history,
        "elapsed_seconds": time.time() - started,
        "non_claims": [
            "The run is a calibration smoke test; the loss trace is not a "
            "model-quality result.",
            "No GPU throughput, memory peak, or serving parity is measured.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    export_adapter(
        model,
        output,
        {
            "fingerprint": fingerprint,
            "rank": config.rank,
            "alpha": config.alpha,
            "quantize_mxfp4": config.quantize,
            "model_path": config.model_path,
            "steps": step,
        },
    )
    return summary


def read_config_payload(path: str) -> dict[str, Any]:
    """Read a config.json file or the config.json of a model directory."""
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "config.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"no config file at {candidate}")
    return json.loads(candidate.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of the Ornith QAT trainer."""
    parser = argparse.ArgumentParser(
        description="MXFP4 QAT trainer for Qwen3_5Moe fused routed experts."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser(
        "preflight", help="report the expert inventory and the LoRA budget"
    )
    plan.add_argument("--config", help="path to a config.json or model directory")
    plan.add_argument("--config-json", help="inline config JSON")
    plan.add_argument("--rank", type=int, default=ORNITH_RANK)
    plan.add_argument("--alpha", type=float, default=ORNITH_ALPHA)
    plan.add_argument("--no-quantize", action="store_true")

    trainer = commands.add_parser("train", help="run the MXFP4 QAT loop")
    trainer.add_argument("--model", required=True)
    trainer.add_argument("--data", required=True, help=".npy tokens or a text file")
    trainer.add_argument("--tokenizer", help="tokenizer for a text input")
    trainer.add_argument("--output", default="build/ornith-qat")
    trainer.add_argument(
        "--model-class",
        choices=("causal_lm", "conditional_generation"),
        default="causal_lm",
    )
    trainer.add_argument("--rank", type=int, default=ORNITH_RANK)
    trainer.add_argument("--alpha", type=float, default=ORNITH_ALPHA)
    trainer.add_argument("--seq-len", type=int, default=ORNITH_SEQ_LEN)
    trainer.add_argument("--stride", type=int, default=0)
    trainer.add_argument("--epochs", type=int, default=3)
    trainer.add_argument("--lr", type=float, default=1e-4)
    trainer.add_argument("--weight-decay", type=float, default=0.0)
    trainer.add_argument("--micro-batch-size", type=int, default=1)
    trainer.add_argument("--grad-accum-steps", type=int, default=1)
    trainer.add_argument("--max-grad-norm", type=float, default=1.0)
    trainer.add_argument("--loss-chunk", type=int, default=512)
    trainer.add_argument("--seed", type=int, default=17)
    trainer.add_argument("--device", default="cuda")
    trainer.add_argument("--dtype", default="bfloat16")
    trainer.add_argument("--log-every", type=int, default=10)
    trainer.add_argument("--max-steps", type=int, default=0)
    trainer.add_argument("--limit-windows", type=int, default=0)
    trainer.add_argument("--max-tokens", type=int, default=0)
    trainer.add_argument("--no-quantize", action="store_true")
    trainer.add_argument("--no-checkpoint-experts", action="store_true")
    trainer.add_argument(
        "--expert-chunk-size",
        type=int,
        default=ORNITH_EXPERT_CHUNK_SIZE,
        help=(
            "experts per vectorized chunk; 0 selects the sequential "
            "reference path"
        ),
    )
    trainer.add_argument("--gradient-checkpointing", action="store_true")
    trainer.add_argument("--no-resume", action="store_false", dest="resume")
    trainer.add_argument("--dry-run", action="store_true")

    export = commands.add_parser(
        "export", help="merge an adapter into the base weights"
    )
    export.add_argument("--model", required=True)
    export.add_argument("--adapter", required=True, help="adapter.safetensors")
    export.add_argument("--output", required=True)
    export.add_argument(
        "--model-class",
        choices=("causal_lm", "conditional_generation"),
        default="causal_lm",
    )
    export.add_argument("--rank", type=int, default=ORNITH_RANK)
    export.add_argument("--alpha", type=float, default=ORNITH_ALPHA)
    export.add_argument("--no-quantize", action="store_true")
    export.add_argument("--no-codes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Ornith MXFP4 QAT command line."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "preflight":
        if args.config_json:
            payload = json.loads(args.config_json)
        elif args.config:
            payload = read_config_payload(args.config)
        else:
            parser.error("preflight needs --config or --config-json")
        report = preflight(
            payload, args.rank, args.alpha, not args.no_quantize
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    if args.command == "train":
        config = TrainConfig(
            model_path=args.model,
            model_class=args.model_class,
            output_dir=args.output,
            data_path=args.data,
            tokenizer=args.tokenizer,
            rank=args.rank,
            alpha=args.alpha,
            seq_len=args.seq_len,
            stride=args.stride,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            micro_batch_size=args.micro_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            loss_chunk=args.loss_chunk,
            seed=args.seed,
            quantize=not args.no_quantize,
            checkpoint_experts=not args.no_checkpoint_experts,
            expert_chunk_size=args.expert_chunk_size,
            gradient_checkpointing=args.gradient_checkpointing,
            device=args.device,
            dtype=args.dtype,
            log_every=args.log_every,
            max_steps=args.max_steps,
            limit_windows=args.limit_windows,
            max_tokens=args.max_tokens,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        summary = train(config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "export":
        from safetensors.torch import load_file

        adapter_path = Path(args.adapter)
        metadata_path = adapter_path.with_name("adapter.json")
        rank = args.rank
        alpha = args.alpha
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            rank = int(metadata.get("rank", rank))
            alpha = float(metadata.get("alpha", alpha))
        config = TrainConfig(
            model_path=args.model,
            model_class=args.model_class,
            rank=rank,
            alpha=alpha,
            quantize=not args.no_quantize,
        )
        model = load_base_model(config)
        install_expert_qat_lora(
            model,
            rank=rank,
            alpha=alpha,
            quantize=config.quantize,
        )
        load_adapter_state_dict(model, load_file(str(adapter_path)))
        summary = export_merged(
            model,
            args.output,
            {"model_path": args.model, "rank": rank, "alpha": alpha},
            quantize=config.quantize,
            with_codes=not args.no_codes,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
