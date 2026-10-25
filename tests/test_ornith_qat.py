"""Tests for the Ornith MXFP4 QAT trainer.

Every test uses a tiny synthetic Qwen3_5Moe model. Nothing here downloads a
weight, claims GPU execution, or measures model quality.
"""

import json
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
