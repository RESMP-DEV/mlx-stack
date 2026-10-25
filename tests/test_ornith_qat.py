"""Tests for the Ornith MXFP4 QAT trainer.

Every test uses a tiny synthetic Qwen3_5Moe model. Nothing here downloads a
weight, claims GPU execution, or measures model quality.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from mlx_gptq import ornith_qat
from mlx_gptq.grid import dequantize_mxfp
from mlx_gptq.ornith_qat import (
    ExpertLoRA,
    FusedExpertsQAT,
    TokenWindowDataset,
    TrainConfig,
    adapter_parameters,
    adapter_state_dict,
    causal_lm_loss,
    export_merged,
    find_expert_modules,
    install_expert_qat_lora,
    lora_parameter_budget,
    merged_state_dict,
    mxfp4_codes,
    plan_expert_chunks,
    preflight,
    train,
    training_fingerprint,
)
from mlx_gptq.qat import fake_quantize_mxfp4
from safetensors.torch import load_file

HIDDEN = 64
INTERMEDIATE = 32
EXPERTS = 4
LAYERS = 2


def _tiny_config(layers: int = LAYERS):
    from transformers import Qwen3_5MoeTextConfig

    return Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_intermediate_size=INTERMEDIATE,
        shared_expert_intermediate_size=INTERMEDIATE,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        pad_token_id=0,
        layer_types=["full_attention"] * layers,
    )


def _tiny_model(layers: int = LAYERS):
    from transformers import Qwen3_5MoeForCausalLM

    torch.manual_seed(11)
    return Qwen3_5MoeForCausalLM(_tiny_config(layers))


def _routing(tokens: int, top_k: int, experts: int = EXPERTS):
    """Return a routing assignment that visits every expert at least once."""
    generator = torch.Generator().manual_seed(5)
    indices = []
    for expert in range(experts):
        indices.append(expert)
    while len(indices) < tokens * top_k:
        indices.append(
            int(torch.randint(experts, (1,), generator=generator).item())
        )
    stacked = torch.tensor(indices[: tokens * top_k], dtype=torch.long)
    return stacked.reshape(tokens, top_k)


def _reference_expert_forward(module, hidden_states, top_k_index, top_k_weights):
    """Recompute the expert output straight from the MXFP4 grid."""
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=module.num_experts + 1
        ).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for row in hit:
        expert = row[0]
        if expert == module.num_experts:
            continue
        position, token = torch.where(mask[expert])
        state = hidden_states[token]
        gate_up = fake_quantize_mxfp4(
            module.gate_up_proj[expert].float() + module.gate_up_lora.delta(expert)
        )
        gate, up = torch.nn.functional.linear(
            state, gate_up.to(module.gate_up_proj.dtype)
        ).chunk(2, dim=-1)
        hidden = module.act_fn(gate) * up
        down = fake_quantize_mxfp4(
            module.down_proj[expert].float() + module.down_lora.delta(expert)
        )
        output = torch.nn.functional.linear(
            hidden, down.to(module.down_proj.dtype)
        )
        final = final.index_add(
            0, token, output * top_k_weights[token, position, None]
        )
    return final


def test_fake_quantize_mxfp4_is_preserved_and_shared():
    assert ornith_qat.fake_quantize_mxfp4 is fake_quantize_mxfp4
    weight = torch.randn(3, 32, 64, requires_grad=True)
    quantized = fake_quantize_mxfp4(weight)
    quantized.sum().backward()
    assert torch.allclose(weight.grad, torch.ones_like(weight))
    assert quantized.shape == weight.shape


def test_expert_lora_starts_at_zero_with_per_expert_shapes():
    lora = ExpertLoRA(EXPERTS, 64, 32, rank=4, alpha=8.0)
    assert lora.lora_a.shape == (EXPERTS, 4, 32)
    assert lora.lora_b.shape == (EXPERTS, 64, 4)
    assert lora.scaling == pytest.approx(2.0)
    for expert in range(EXPERTS):
        assert torch.count_nonzero(lora.delta(expert)) == 0


def test_install_freezes_the_base_and_keeps_the_checkpoint_keys():
    model = _tiny_model()
    original = set(model.state_dict())
    adapters = install_expert_qat_lora(model, rank=4, alpha=8.0)
    assert len(adapters) == LAYERS
    assert adapters[0].module_path == "model.layers.0.mlp.experts"
    assert adapters[0].gate_up_shape == (EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    assert adapters[0].down_shape == (EXPERTS, HIDDEN, INTERMEDIATE)
    keys = set(model.state_dict())
    assert original.issubset(keys)
    assert {key for key in keys if not key.endswith(("lora_a", "lora_b"))} == original
    for name, parameter in model.named_parameters():
        if name.endswith(("lora_a", "lora_b")):
            assert parameter.requires_grad
        else:
            assert not parameter.requires_grad
    assert find_expert_modules(model)
    assert all(
        isinstance(module, FusedExpertsQAT) for _, module in find_expert_modules(model)
    )


def test_effective_weight_sits_on_the_mxfp4_grid():
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    module = model.model.layers[0].mlp.experts
    with torch.no_grad():
        module.gate_up_lora.lora_b.normal_(0.0, 0.02)
    effective = module.effective_gate_up(0)
    reference = fake_quantize_mxfp4(
        module.gate_up_proj[0].float() + module.gate_up_lora.delta(0)
    ).to(module.gate_up_proj.dtype)
    assert torch.allclose(effective, reference)
    codes, scale_bytes = mxfp4_codes(module.gate_up_proj[0].float())
    assert codes.shape == module.gate_up_proj[0].shape
    assert scale_bytes.shape == (HIDDEN, HIDDEN // 32)
    restored = dequantize_mxfp(codes, scale_bytes, "mxfp4").reshape(HIDDEN, HIDDEN)
    assert restored.shape == module.gate_up_proj[0].shape
    quantized = fake_quantize_mxfp4(module.gate_up_proj[0].float())
    assert torch.allclose(restored, quantized, atol=1e-6)


def test_adapter_forward_matches_the_manual_reference():
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    module = model.model.layers[0].mlp.experts
    with torch.no_grad():
        module.gate_up_lora.lora_b.normal_(0.0, 0.02)
        module.down_lora.lora_b.normal_(0.0, 0.02)
    tokens = 8
    hidden = torch.randn(tokens, HIDDEN)
    top_k_index = _routing(tokens, 2)
    top_k_weights = torch.rand(tokens, 2)
    produced = module(hidden, top_k_index, top_k_weights)
    expected = _reference_expert_forward(module, hidden, top_k_index, top_k_weights)
    assert torch.allclose(produced, expected, atol=1e-5, rtol=1e-4)


def test_checkpointed_experts_match_the_plain_forward():
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    module = model.model.layers[0].mlp.experts
    with torch.no_grad():
        module.gate_up_lora.lora_b.normal_(0.0, 0.02)
    tokens = 8
    hidden = torch.randn(tokens, HIDDEN)
    top_k_index = _routing(tokens, 2)
    top_k_weights = torch.rand(tokens, 2)
    plain = module(hidden, top_k_index, top_k_weights)
    module.checkpoint_experts = True
    checkpointed = module(hidden, top_k_index, top_k_weights)
    assert torch.allclose(plain, checkpointed, atol=1e-5, rtol=1e-4)
    plain.sum().backward()
    plain_grad = module.gate_up_lora.lora_b.grad.clone()
    module.zero_grad(set_to_none=True)
    checkpointed.sum().backward()
    assert torch.allclose(plain_grad, module.gate_up_lora.lora_b.grad, atol=1e-6)


def test_gradient_reaches_every_expert():
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    module = model.model.layers[0].mlp.experts
    tokens = EXPERTS
    hidden = torch.randn(tokens, HIDDEN, requires_grad=True)
    top_k_index = _routing(tokens, 2)
    top_k_weights = torch.rand(tokens, 2)
    module(hidden, top_k_index, top_k_weights).sum().backward()
    grad = module.gate_up_lora.lora_b.grad
    assert grad is not None
    assert grad.shape == module.gate_up_lora.lora_b.shape
    for expert in range(EXPERTS):
        assert float(grad[expert].abs().sum()) > 0.0
    assert module.down_lora.lora_b.grad is not None
    for expert in range(EXPERTS):
        assert float(module.down_lora.lora_b.grad[expert].abs().sum()) > 0.0


def _random_routing(tokens: int, top_k: int, experts: int, seed: int = 0):
    """Return a random routing that visits every expert at least once."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(experts, (tokens, top_k), generator=generator)
    for expert in range(experts):
        indices[expert % tokens, 0] = expert
    return indices


def _synthetic_experts(experts: int, hidden: int, intermediate: int):
    """Return a fused-expert source module with the Ornith weight layout."""

    class _Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(
                torch.randn(experts, 2 * intermediate, hidden) * 0.02
            )
            self.down_proj = torch.nn.Parameter(
                torch.randn(experts, hidden, intermediate) * 0.02
            )
            self.act_fn = torch.nn.SiLU()
            self.num_experts = experts
            self.hidden_dim = hidden
            self.intermediate_dim = intermediate
            self.has_gate = True
            self.has_bias = False
            self.is_concatenated = True
            self.is_transposed = False

    return _Source()


def test_expert_chunk_plan_exposes_the_padding_tradeoff():
    counts = [12, 0, 30, 28, 31, 0, 9]
    small = plan_expert_chunks(counts, 2)
    large = plan_expert_chunks(counts, 4)
    assert [plan.experts for plan in small] == [[0, 2], [3, 4], [6]]
    assert [plan.token_counts for plan in small] == [[12, 30], [28, 31], [9]]
    assert [plan.max_tokens for plan in small] == [30, 31, 9]
    assert small[0].routed_rows == 42
    assert small[0].padded_rows == 60
    # A bigger chunk needs fewer batched matmuls but pads more rows.
    assert len(large) < len(small)
    assert sum(plan.padded_rows for plan in large) > sum(
        plan.padded_rows for plan in small
    )
    assert sum(plan.routed_rows for plan in large) == sum(
        plan.routed_rows for plan in small
    )
    with pytest.raises(ValueError, match="chunk size"):
        plan_expert_chunks(counts, 0)


def test_chunked_forward_matches_the_sequential_forward():
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0, expert_chunk_size=0)
    module = model.model.layers[0].mlp.experts
    with torch.no_grad():
        module.gate_up_lora.lora_b.normal_(0.0, 0.02)
        module.down_lora.lora_b.normal_(0.0, 0.02)
    tokens = 16
    generator = torch.Generator().manual_seed(4)
    hidden = torch.randn(tokens, HIDDEN, generator=generator)
    top_k_index = _random_routing(tokens, 2, EXPERTS, seed=9)
    top_k_weights = torch.rand(tokens, 2, generator=generator)

    module.expert_chunk_size = 0
    sequential = module(hidden, top_k_index, top_k_weights)
    for chunk_size in (1, 2, 3, EXPERTS, 16):
        module.expert_chunk_size = chunk_size
        chunked = module(hidden, top_k_index, top_k_weights)
        assert torch.allclose(sequential, chunked, atol=1e-5, rtol=1e-4)
        module.checkpoint_experts = True
        checkpointed = module(hidden, top_k_index, top_k_weights)
        module.checkpoint_experts = False
        assert torch.allclose(sequential, checkpointed, atol=1e-5, rtol=1e-4)


def test_chunked_effective_weights_are_bit_identical():
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    module = model.model.layers[0].mlp.experts
    with torch.no_grad():
        module.gate_up_lora.lora_b.normal_(0.0, 0.02)
        module.down_lora.lora_b.normal_(0.0, 0.02)
    expert_ids = torch.arange(EXPERTS)
    per_expert = torch.stack(
        [module.effective_gate_up(int(expert)) for expert in expert_ids]
    )
    chunked = module._effective_chunk(
        module.gate_up_proj, module.gate_up_lora, expert_ids
    )
    assert torch.equal(per_expert, chunked)
    per_expert_down = torch.stack(
        [module.effective_down(int(expert)) for expert in expert_ids]
    )
    chunked_down = module._effective_chunk(
        module.down_proj, module.down_lora, expert_ids
    )
    assert torch.equal(per_expert_down, chunked_down)
    # The exact fallback keeps the same values when the batched product is
    # not bit-identical on the device.
    module._delta_mode = "exact"
    fallback = module._effective_chunk(
        module.gate_up_proj, module.gate_up_lora, expert_ids
    )
    assert torch.equal(per_expert, fallback)


def test_chunked_gradient_reaches_every_selected_expert():
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    module = model.model.layers[0].mlp.experts
    tokens = 12
    generator = torch.Generator().manual_seed(6)
    hidden = torch.randn(tokens, HIDDEN, generator=generator)
    top_k_index = _random_routing(tokens, 2, EXPERTS, seed=2)
    top_k_weights = torch.rand(tokens, 2, generator=generator)
    module.expert_chunk_size = 2
    module(hidden, top_k_index, top_k_weights).sum().backward()
    selected = sorted({int(expert) for expert in top_k_index.reshape(-1)})
    assert selected
    for expert in selected:
        assert float(module.gate_up_lora.lora_b.grad[expert].abs().sum()) > 0.0
        assert float(module.down_lora.lora_b.grad[expert].abs().sum()) > 0.0
    for expert in range(EXPERTS):
        if expert in selected:
            continue
        assert float(module.gate_up_lora.lora_b.grad[expert].abs().sum()) == 0.0
        assert float(module.down_lora.lora_b.grad[expert].abs().sum()) == 0.0


def test_chunked_and_sequential_gradients_agree():
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    module = model.model.layers[0].mlp.experts
    with torch.no_grad():
        module.gate_up_lora.lora_b.normal_(0.0, 0.02)
        module.down_lora.lora_b.normal_(0.0, 0.02)
    tokens = 16
    generator = torch.Generator().manual_seed(8)
    hidden = torch.randn(tokens, HIDDEN, generator=generator)
    top_k_index = _random_routing(tokens, 2, EXPERTS, seed=5)
    top_k_weights = torch.rand(tokens, 2, generator=generator)
    grads = []
    for chunk_size in (0, 2, 4):
        module.expert_chunk_size = chunk_size
        module.zero_grad(set_to_none=True)
        module(hidden, top_k_index, top_k_weights).pow(2).sum().backward()
        grads.append(
            (
                module.gate_up_lora.lora_b.grad.clone(),
                module.down_lora.lora_b.grad.clone(),
            )
        )
    for gate_up_grad, down_grad in grads[1:]:
        assert torch.allclose(grads[0][0], gate_up_grad, atol=1e-5, rtol=1e-4)
        assert torch.allclose(grads[0][1], down_grad, atol=1e-5, rtol=1e-4)


def test_chunked_forward_ignores_the_drop_bucket():
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    module = model.model.layers[0].mlp.experts
    tokens = 8
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(tokens, HIDDEN, generator=generator)
    top_k_index = _random_routing(tokens, 2, EXPERTS, seed=7)
    top_k_index[0, 0] = EXPERTS
    top_k_weights = torch.rand(tokens, 2, generator=generator)
    module.expert_chunk_size = 2
    chunked = module(hidden, top_k_index, top_k_weights)
    module.expert_chunk_size = 0
    sequential = module(hidden, top_k_index, top_k_weights)
    assert torch.allclose(chunked, sequential, atol=1e-5, rtol=1e-4)
def test_expert_chunk_size_knob_reaches_the_module_and_the_cli():
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0, expert_chunk_size=8)
    module = model.model.layers[0].mlp.experts
    assert module.expert_chunk_size == 8
    assert "expert_chunk_size=8" in module.extra_repr()
    parser = ornith_qat.build_parser()
    parsed = parser.parse_args(
        ["train", "--model", "m", "--data", "d", "--expert-chunk-size", "8"]
    )
    assert parsed.expert_chunk_size == 8
    default = parser.parse_args(["train", "--model", "m", "--data", "d"])
    assert default.expert_chunk_size == ornith_qat.ORNITH_EXPERT_CHUNK_SIZE
    assert TrainConfig().expert_chunk_size == ornith_qat.ORNITH_EXPERT_CHUNK_SIZE
    with pytest.raises(ValueError, match="chunk size"):
        FusedExpertsQAT(
            model.model.layers[0].mlp.experts, rank=4, alpha=8.0,
            expert_chunk_size=-1,
        )


def test_train_reports_the_expert_chunk_size(tmp_path):
    config, _ = _prepare_run(tmp_path)
    dry = TrainConfig(
        **{**config.as_dict(), "expert_chunk_size": 4, "dry_run": True}
    )
    summary = train(dry)
    assert summary["expert_chunk_size"] == 4


def test_chunk_size_tradeoff_benchmark():
    """Expose the chunk-size tradeoff on a synthetic layer, without a GPU.

    The shape keeps the structure of the Ornith layer (many experts, a
    grouped token count) but is small enough to run on CPU. The printed
    table is the measurement; the assertions only pin the numerics.
    """
    experts, hidden, intermediate = 64, 256, 64
    tokens, top_k = 512, 8
    torch.manual_seed(21)
    source = _synthetic_experts(experts, hidden, intermediate)
    generator = torch.Generator().manual_seed(22)
    hidden_states = torch.randn(tokens, hidden, generator=generator)
    top_k_index = torch.randint(
        experts, (tokens, top_k), generator=generator
    )
    top_k_weights = torch.rand(tokens, top_k, generator=generator)
    routed = torch.bincount(top_k_index.reshape(-1), minlength=experts).tolist()
    module = FusedExpertsQAT(
        source,
        rank=8,
        alpha=16.0,
        quantize=True,
        checkpoint_experts=False,
        expert_chunk_size=0,
    )
    with torch.no_grad():
        module.gate_up_lora.lora_b.normal_(0.0, 0.02)
        module.down_lora.lora_b.normal_(0.0, 0.02)
    rows = []
    reference = None
    for chunk_size in (0, 8, 16, 32, 64):
        module.expert_chunk_size = chunk_size
        module.zero_grad(set_to_none=True)
        start = time.perf_counter()
        output = module(hidden_states, top_k_index, top_k_weights)
        output.pow(2).sum().backward()
        elapsed = time.perf_counter() - start
        if reference is None:
            reference = output.detach()
        else:
            assert torch.allclose(reference, output.detach(), atol=1e-4, rtol=1e-3)
        if chunk_size:
            plans = plan_expert_chunks(routed, chunk_size)
            chunks = len(plans)
            padded = sum(plan.padded_rows for plan in plans)
        else:
            chunks, padded = experts, sum(routed)
        rows.append((chunk_size, chunks, padded, elapsed))
    print("\nchunk_size chunks padded_rows seconds")
    for chunk_size, chunks, padded, elapsed in rows:
        print(f"{chunk_size:>10} {chunks:>6} {padded:>11} {elapsed:>7.4f}")
    assert rows[0][1] == experts
    assert [row[1] for row in rows[1:]] == sorted(
        [row[1] for row in rows[1:]], reverse=True
    )
    assert rows[-1][1] == 1
    assert all(row[2] >= sum(routed) for row in rows)
def test_adapter_rejects_a_mismatched_expert_layout():
    class _FakeExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.zeros(2, 10, 32))
            self.down_proj = torch.nn.Parameter(torch.zeros(2, 32, 4))
            self.act_fn = torch.nn.SiLU()

    with pytest.raises(ValueError, match="concatenated"):
        FusedExpertsQAT(_FakeExperts(), rank=4, alpha=8.0)


def test_token_window_dataset_shifts_the_labels():
    token_ids = list(range(20))
    dataset = TokenWindowDataset(token_ids, seq_len=4, stride=2)
    assert len(dataset) == 8
    window = dataset[0]
    assert window["input_ids"].tolist() == [0, 1, 2, 3]
    assert window["labels"].tolist() == [1, 2, 3, 4]
    assert dataset[1]["input_ids"].tolist() == [2, 3, 4, 5]
    assert dataset.digest() == TokenWindowDataset(token_ids, 4, 2).digest()
    assert dataset.digest() != TokenWindowDataset(token_ids + [99], 4, 2).digest()
    assert len(TokenWindowDataset(token_ids, seq_len=4, limit=3)) == 3
    with pytest.raises(ValueError, match="seq_len"):
        TokenWindowDataset(token_ids, seq_len=1)


def test_preflight_reports_the_ornith_budget():
    payload = {
        "num_hidden_layers": 40,
        "num_experts": 192,
        "num_experts_per_tok": 8,
        "hidden_size": 2048,
        "moe_intermediate_size": 512,
        "vocab_size": 248320,
    }
    report = preflight(payload, rank=16, alpha=32.0)
    assert report["passed"]
    assert report["experts"] == 192
    assert report["experts_per_token"] == 8
    assert report["layers"] == 40
    assert report["budget"]["total"] == 692060160
    assert report["budget"]["per_layer"] == 17301504
    assert report["scaling"] == pytest.approx(2.0)
    assert report["non_claims"]
    nested = preflight({"text_config": payload}, rank=16, alpha=32.0)
    assert nested["budget"] == report["budget"]


def test_preflight_rejects_a_config_that_breaks_the_mx_group():
    payload = {
        "num_hidden_layers": 40,
        "num_experts": 192,
        "num_experts_per_tok": 8,
        "hidden_size": 2000,
        "moe_intermediate_size": 512,
    }
    report = preflight(payload, rank=16, alpha=32.0)
    assert not report["passed"]
    failed = {entry["name"] for entry in report["checks"] if not entry["ok"]}
    assert "mx_group_hidden" in failed


def test_lora_budget_matches_the_installed_adapter():
    budget = lora_parameter_budget(EXPERTS, HIDDEN, INTERMEDIATE, LAYERS, 4)
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    installed = sum(
        parameter.numel() for parameter in adapter_parameters(model)
    )
    assert installed == budget["total"]


def _prepare_run(tmp_path: Path, tokens: int = 256):
    from transformers import Qwen3_5MoeForCausalLM

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _tiny_model().save_pretrained(model_dir)
    data_path = tmp_path / "tokens.npy"
    np.save(data_path, (np.arange(tokens) % 64).astype(np.int64))
    config = TrainConfig(
        model_path=str(model_dir),
        output_dir=str(tmp_path / "run"),
        data_path=str(data_path),
        rank=4,
        alpha=8.0,
        seq_len=8,
        epochs=1,
        micro_batch_size=1,
        grad_accum_steps=1,
        loss_chunk=4,
        device="cpu",
        dtype="float32",
        log_every=0,
        max_steps=0,
        limit_windows=2,
        resume=True,
    )
    return config, Qwen3_5MoeForCausalLM


def test_train_runs_writes_a_checkpoint_and_resumes(tmp_path):
    config, _ = _prepare_run(tmp_path)
    summary = train(config)
    assert summary["steps"] == 2
    assert summary["epochs_completed"] == 1
    assert summary["trainable_parameters"] == 7168
    run = Path(config.output_dir)
    assert (run / "checkpoint.pt").is_file()
    assert (run / "adapter.safetensors").is_file()
    assert (run / "adapter.json").is_file()
    assert (run / "summary.json").is_file()
    metrics = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len([record for record in metrics if "loss" in record]) == 2
    assert len([record for record in metrics if "mean_loss" in record]) == 1
    assert metrics[-1]["step"] == 2
    assert metrics[-1]["batches"] == 2

    resumed = TrainConfig(**{**config.as_dict(), "epochs": 2})
    summary = train(resumed)
    assert summary["steps"] == 4
    assert summary["epochs_completed"] == 2
    metrics = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len([record for record in metrics if "loss" in record]) == 4
    assert len([record for record in metrics if "mean_loss" in record]) == 2


def test_train_refuses_a_checkpoint_from_another_fingerprint(tmp_path):
    config, _ = _prepare_run(tmp_path)
    train(config)
    mismatched = TrainConfig(**{**config.as_dict(), "rank": 8})
    assert mismatched.resume
    with pytest.raises(RuntimeError, match="fingerprint"):
        train(mismatched)


def test_interrupted_epoch_is_replayed_on_resume(tmp_path):
    config, _ = _prepare_run(tmp_path)
    capped = TrainConfig(**{**config.as_dict(), "max_steps": 1})
    summary = train(capped)
    assert summary["steps"] == 1
    assert summary["epochs_completed"] == 0
    resumed = TrainConfig(**{**config.as_dict(), "max_steps": 0})
    summary = train(resumed)
    assert summary["steps"] == 3
    assert summary["epochs_completed"] == 1


def test_dry_run_reports_the_plan_without_training(tmp_path):
    config, _ = _prepare_run(tmp_path)
    dry = TrainConfig(**{**config.as_dict(), "dry_run": True})
    summary = train(dry)
    assert summary["dry_run"]
    assert summary["trainable_parameters"] == 7168
    run = Path(config.output_dir)
    assert (run / "dry_run.json").is_file()
    assert not (run / "checkpoint.pt").exists()


def test_export_merged_keeps_the_upstream_keys_and_reloads(tmp_path):
    from transformers import Qwen3_5MoeForCausalLM

    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    module = model.model.layers[0].mlp.experts
    with torch.no_grad():
        module.gate_up_lora.lora_b.normal_(0.0, 0.02)
    merged = merged_state_dict(model, quantize=True)
    assert set(merged) == set(_tiny_model().state_dict())
    expected = fake_quantize_mxfp4(
        module.gate_up_proj[0].float() + module.gate_up_lora.delta(0)
    ).to(module.gate_up_proj.dtype)
    merged_gate_up = merged["model.layers.0.mlp.experts.gate_up_proj"]
    assert torch.allclose(merged_gate_up[0], expected)

    export_dir = tmp_path / "merged"
    summary = export_merged(model, export_dir, {"rank": 4, "alpha": 8.0})
    assert summary["merged_modules"] == LAYERS
    assert (export_dir / "model.safetensors").is_file()
    assert (export_dir / "config.json").is_file()
    assert (export_dir / "mxfp4.safetensors").is_file()

    reference = _tiny_model()
    reference.save_pretrained(tmp_path / "reference")
    exported = load_file(str(export_dir / "model.safetensors"))
    stock = load_file(str(tmp_path / "reference" / "model.safetensors"))
    assert set(exported) == set(stock)

    reloaded = Qwen3_5MoeForCausalLM.from_pretrained(export_dir)
    assert torch.allclose(
        reloaded.model.layers[0].mlp.experts.gate_up_proj[0],
        expected,
    )

    codes = load_file(str(export_dir / "mxfp4.safetensors"))
    expert_keys = {key for key in stock if ".experts." in key}
    assert expert_keys
    for key in expert_keys:
        assert f"{key}.q" in codes
        assert f"{key}.scales" in codes


def test_merge_without_an_update_keeps_the_base_weight():
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    merged = merged_state_dict(model, quantize=False)
    base = _tiny_model().state_dict()
    for name, tensor in base.items():
        assert torch.allclose(merged[name], tensor, atol=1e-6)


def test_adapter_state_dict_round_trips(tmp_path):
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    state = adapter_state_dict(model)
    assert state and all(key.endswith(("lora_a", "lora_b")) for key in state)
    target = _tiny_model()
    install_expert_qat_lora(target, rank=4, alpha=8.0)
    ornith_qat.load_adapter_state_dict(target, state)
    for name, tensor in target.state_dict().items():
        if name.endswith(("lora_a", "lora_b")):
            assert torch.allclose(tensor, state[name])


def test_chunked_and_unchunked_losses_agree():
    model = _tiny_model()
    install_expert_qat_lora(model, rank=4, alpha=8.0)
    input_ids = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        whole = causal_lm_loss(model, input_ids, loss_chunk=0)
        chunked = causal_lm_loss(model, input_ids, loss_chunk=5)
    assert torch.isfinite(whole)
    assert torch.allclose(whole, chunked, atol=1e-5, rtol=1e-4)
    shifted = causal_lm_loss(model, input_ids, labels=input_ids[:, 1:], loss_chunk=4)
    assert torch.allclose(whole, shifted, atol=1e-5, rtol=1e-4)


def test_training_fingerprint_tracks_the_run_shape():
    config = TrainConfig(model_path="model", rank=16, alpha=32.0, seq_len=2048)
    digest = "abc"
    assert training_fingerprint(config, digest) == training_fingerprint(
        config, digest
    )
    other = TrainConfig(model_path="model", rank=8, alpha=32.0, seq_len=2048)
    assert training_fingerprint(config, digest) != training_fingerprint(other, digest)
    assert training_fingerprint(config, digest) != training_fingerprint(config, "def")
