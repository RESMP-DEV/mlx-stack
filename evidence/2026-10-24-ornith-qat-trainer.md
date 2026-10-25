# Ornith 192-expert MXFP4 QAT trainer

Date: 2026-10-24
Agent: Codex
Source revision: `b4cc7c6d10419ddc6c6abf5ef9a90b47ed50caa3` (worktree `work/ornith-qat-trainer`)
Environment used for the runs below: Python 3.13, torch 2.13.0, transformers 5.18.0, ruff 0.17.1

## Scope

A queueable quantization-aware trainer for the Qwen3_5Moe fused routed
experts, so a 192-expert model can be adapted to the MXFP4 grid with a
per-expert LoRA. This work adds code and tests only. It does not download the
35B model, does not run on a GPU, and makes no model-quality claim.

## Target shape, read from the installed library

The layout was read from `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`
(5.18.0), not from documentation:

- `Qwen3_5MoeExperts` holds two fused 3D parameters per layer:
  `gate_up_proj` of shape `[E, 2 * I, H]` and `down_proj` of shape `[E, H, I]`.
- `gate_up_proj` is consumed as `nn.functional.linear(x, gate_up_proj[e])`,
  so the reduction axis is the last axis, which is where the MXFP4 group of 32
  belongs for both tensors.
- The expert module is reached at `model.layers.<i>.mlp.experts` for
  `Qwen3_5MoeForCausalLM` and at `model.language_model.layers.<i>.mlp.experts`
  for the multimodal `Qwen3_5MoeModel` wrapper.
- The shared expert and the router are separate 2D modules and are not
  touched.

```bash
/Users/kearm/mlx/.worktrees/mlx-gptq-mxfp/.venv/bin/python -c "
from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
..."
```

Observed on a tiny synthetic config (4 experts, 2 layers):
`model.layers.0.mlp.experts.gate_up_proj (4, 32, 32)` and
`model.layers.0.mlp.experts.down_proj (4, 32, 16)`.

## Design

`mlx_gptq/ornith_qat.py` (1402 lines) is the whole trainer.

- `ExpertLoRA` holds one rank-r update per expert: `A` of `[E, r, C]` and
  `B` of `[E, R, r]`, with `B` zero-initialised so an untrained adapter
  reproduces the base weight exactly.
- `FusedExpertsQAT` replaces `Qwen3_5MoeExperts`. It reuses the upstream
  parameter objects (no copy of the frozen weights) and keeps their names, so
  the base keys of a merged export are exactly the upstream checkpoint keys.
  The effective weight of a hit expert is
  `fake_quantize_mxfp4(W + scaling * B[e] @ A[e])`, evaluated per hit expert
  inside the upstream eager routing loop, never as a whole fused tensor.
- `mlx_gptq.qat.fake_quantize_mxfp4` is reused unchanged and is the only
  quantizer in the path; the straight-through estimator is its identity
  gradient on the effective weight.
- `install_expert_qat_lora` discovers expert modules structurally (3D
  `gate_up_proj` and `down_proj` with a matching expert count), so both the
  text and the multimodal wrapper are handled without a hard-coded path. It
  freezes every other parameter, so the optimizer only sees LoRA factors.
- `checkpoint_experts` wraps the per-expert computation in
  `torch.utils.checkpoint` (`use_reentrant=False`). Without it, one quantized
  weight per hit expert per layer is retained until backward, which is about
  2.4 GB per layer for the Ornith shape.
- `TokenWindowDataset` builds causal windows over a flat token-id stream read
  from a `.npy` file, or from a text corpus when a tokenizer is supplied.
- `causal_lm_loss` evaluates the vocabulary projection in chunks, so the
  logits of a 2048-token window against a 248k vocabulary never materialize
  at once. The chunk is itself checkpointed.
- Checkpoint and resume store the adapter, the optimizer state, the RNG state
  and a fingerprint of the model path, adapter shape and dataset digest. A
  mismatched fingerprint is refused. An epoch cut short by the step cap is
  replayed on resume rather than counted as done.
- `export_merged` merges the adapter into the base weight in place, drops the
  adapter, and writes the directory through `save_pretrained`. The result
  therefore holds the stock checkpoint layout of the model class: the
  per-expert tensors that `save_pretrained` writes for a MoE model, the shard
  index of a large model, and the config. A `mxfp4.safetensors` sidecar holds
  the E2M1 codes and E8M0 scale bytes of every per-expert tensor, keyed like
  the checkpoint. The codes are computed from the stored value, so they
  describe the exported bytes exactly.

## Preflight result for the Ornith shape

```bash
PYTHONPATH=. python -m mlx_gptq.ornith_qat preflight \
  --config-json '{"num_hidden_layers":40,"num_experts":192,
    "num_experts_per_tok":8,"hidden_size":2048,
    "moe_intermediate_size":512,"vocab_size":248320}' \
  --rank 16 --alpha 32
```

```json
{
  "experts": 192,
  "experts_per_token": 8,
  "layers": 40,
  "hidden_size": 2048,
  "moe_intermediate_size": 512,
  "rank": 16,
  "alpha": 32.0,
  "scaling": 2.0,
  "passed": true,
  "budget": {
    "per_layer": 17301504,
    "total": 692060160,
    "trainer_gib_fp32": 10.3125
  }
}
```

The budget is 692,060,160 trainable LoRA values, 10.31 GiB of fp32
parameters, gradients and Adam moments. It is a parameter count, not a
measured memory peak.

## Commands and results

Run from the worktree root
`/Users/kearm/mlx/.worktrees/mlx-gptq-mxfp/.worktrees/ornith-qat-trainer`.

```bash
PYTHONPATH=. /Users/kearm/mlx/.worktrees/mlx-gptq-mxfp/.venv/bin/python \
  -m pytest tests/test_ornith_qat.py -q
# 21 passed in 1.70s

RUFF_CACHE_DIR=/tmp/ruff-cache ~/.local/bin/ruff check \
  mlx_gptq/ornith_qat.py tests/test_ornith_qat.py
# All checks passed!  (exit 0)

PYTHONPATH=. /Users/kearm/mlx/.worktrees/mlx-gptq-mxfp/.venv/bin/python \
  -m pytest tests scripts/tests -q \
  --ignore=tests/test_awq_lfm.py \
  --ignore=tests/test_flint_torch.py \
  --ignore=tests/test_grid_and_packing.py \
  --ignore=tests/test_mxfp_artifacts.py
# 144 passed, 3 skipped in 3.23s
```

The command line was also exercised end to end on a tiny synthetic model
staged under `/tmp`, so the parent starts from a known-good path shape:

```bash
python -m mlx_gptq.ornith_qat preflight --config-json '{...}' --rank 16 --alpha 32
python -m mlx_gptq.ornith_qat train --model <dir> --data <tokens.npy> \
  --output <run> --rank 4 --alpha 8 --seq-len 8 --epochs 1 --max-steps 2 \
  --device cpu --dtype float32 --dry-run
python -m mlx_gptq.ornith_qat train ... --max-steps 2
python -m mlx_gptq.ornith_qat export --model <dir> \
  --adapter <run>/adapter.safetensors --output <merged>
```

The dry run reported `trainable_parameters: 7168` for the 2-layer, 4-expert
toy model and wrote no checkpoint. The training run wrote
`checkpoint.pt`, `adapter.safetensors`, `adapter.json`, `metrics.jsonl` and
`summary.json`, and the loss fell over the steps. The export reported
`serialized_tensors: 53` and `mxfp4_code_tensors: 48`, and the exported key
set was identical to the key set of a stock `save_pretrained` of the same
model, with a code and a scale tensor for each of the 24 per-expert tensors.
The exported directory reloaded into a stock `Qwen3_5MoeForCausalLM`.

The four ignored files import `mlx.core`, which aborts in this sandbox with
`[metal::load_device] No Metal device available`. That failure is
pre-existing and unrelated to this change; it was reproduced on the
unmodified worktree.

The frozen environment needs the project's `dev` extra for pytest. The parent
acceptance commands are:

```bash
uv run --frozen --extra dev python -m pytest -q tests/test_ornith_qat.py
uv run --frozen ruff check mlx_gptq/ornith_qat.py tests/test_ornith_qat.py
```

Parent result at 2026-10-24 19:04 Pacific:

```text
21 passed in 7.10s
All checks passed!
```

The parent also ran `just quality`. Its first run failed only on two
pre-existing lint errors in `scripts/local27_metrics_watcher.py`; after
removing the unused `os` import and one excess blank line in a separate
cleanup, `just quality` passed all five checks and wrote
`build/reports/quality/receipt.json` with `passed: true`. The root test,
DFlash runtime test, campaign test, root lint, and CUDA compile checks all
exited 0.

Ruff configuration note: `ruff check` on these paths resolves its config by
walking up the directory tree and finds `[tool.ruff.lint] select = ["E", "F",
"I"]` in the ancestor `/Users/kearm/mlx/pyproject.toml`, with the default line
length of 88. The owned files satisfy that stricter set, not only ruff's
default `E4,E7,E9,F`.

## What the tests cover

`tests/test_ornith_qat.py` (484 lines, 21 tests) uses a tiny synthetic
`Qwen3_5Moe` model of 2 layers, 4 experts, hidden 64, intermediate 32. It
never downloads a weight.

- The MXFP4 fake quantizer is still the shared `mlx_gptq.qat` function and
  still has an identity gradient.
- The adapter starts at exactly zero and has the per-expert shapes.
- Installing the adapter freezes the base, keeps the upstream checkpoint
  keys, and adds only LoRA keys.
- The effective weight equals `fake_quantize_mxfp4(W + delta)` and the codes
  round-trip through the repo's MXFP4 grid helpers.
- The adapter forward equals a hand-written reference that recomputes the
  expert output from the MXFP4 grid.
- The checkpointed expert path matches the plain path in value and gradient.
- The gradient reaches every expert individually, in both `gate_up` and
  `down`.
- The chunked and unchunked losses agree.
- The dataset shifts the labels by one and has a stable digest.
- Preflight reports the Ornith budget and rejects a hidden size that breaks
  the MX group of 32.
- A training run writes a checkpoint, an adapter, and a metrics trace; a
  resume continues from the checkpoint; a mismatched fingerprint is refused;
  an epoch cut short by the step cap is replayed.
- The merged export holds the stock checkpoint keys, covers every per-expert
  tensor of that checkpoint with a code and a scale tensor, and reloads into
  a stock `Qwen3_5MoeForCausalLM`.

## Non-claims

- No 35B weight was downloaded, loaded, or run. Every test uses a tiny
  synthetic model on CPU.
- No GPU execution, throughput, memory peak, or kernel parity is claimed.
  The trainer has never seen a CUDA device.
- No model-quality result is claimed. The loss trace of a calibration run is
  not a quality measurement, and this work produced no loss trace at all.
- The MXFP4 path is a fake-quantization simulation in torch. It reproduces
  the grid arithmetic of `mlx_gptq.qat`, not an MXFP4 kernel, and it does
  not prove that the exported codes load into mlx-lm.
- The merged export is a stock-layout checkpoint directory plus a code
  sidecar. It is not a measured deployment artifact, the codes are not
  proven to load into an mlx-lm runtime, and a 35B bf16 export needs roughly
  70 GiB of host RAM.
- The adapter covers the routed experts only. The shared expert, the router,
  the attention, and the embeddings stay frozen and unquantized.
- The preflight budget is arithmetic over the config, not a memory
  measurement of the 35B model.

## Remaining GPU blockers

The parent must run these on a GPU host; none of them can be settled here.

1. Stage the model and a calibration corpus, then run the smoke step:
   `python -m mlx_gptq.ornith_qat train --model <dir> --data <tokens.npy>
   --output <run> --rank 16 --alpha 32 --seq-len 2048 --epochs 1
   --max-steps 1 --device cuda`. Watch for the peak memory of one layer with
   `checkpoint_experts` on, and confirm the loss is finite.
2. Confirm the checkpointed expert path on the real shape. The 192-expert
   eager loop runs 192 iterations per layer per forward, and the masked
   expert tensor alone is about 25 MB per layer.
3. Only then run the 3-epoch calibration and the export, and check that the
   merged state dict loads back into the stock model class on that host.
4. Decide whether the export needs to be repacked into the mlx-lm layout
   used by the Flint lineage. That step is not implemented here.
