# Ornith 192-expert MXFP4 QAT on RTX PRO 6000

Date: 2026-10-24, 19:10 to 21:05 Pacific
Agent: Codex
Trainer revision: `dec3514` on `work/ornith-qat-trainer`

## Status

**Measured:** the new routed-expert MXFP4 QAT trainer executed real optimizer
steps against the 35.95B-parameter Ornith checkpoint materialized to 192
routed experts. A 2,048-token one-step gate passed, followed by a bounded
100-step tuning pass. Both runs exited 0 and produced finite loss.

**Not claimed:** this is not a quality-recovery result. No held-out perplexity
or task evaluation ran, no merged deployment artifact was exported, and the
loss trace is noisy across different calibration windows.

## Hardware and lifecycle

| Property | Value |
| --- | --- |
| Provider | Packet on demand |
| Instance | `i-6074151b-8a96-49c1-bc9d-ba04fe845a12` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| VRAM | 97,249 MiB |
| Compute capability | 12.0 |
| Driver | 580.126.09 |
| CPU/RAM/disk | 6 vCPU, 96 GiB RAM, 134 GB local disk |
| Price | 66 cents per hour |

The instance was created after a live capacity check found one free
`us-west-2` slot. It was deleted immediately after artifact transfer. The
final Packet instance list was empty. The wallet began this task at 253 cents
and ended at 121 cents; provider accounting can lag, so that is an observed
delta of 132 cents, not yet an invoice-attributed cost.

## Model and data provenance

- Source: `ornith-ai/Ornith-1.5-35B-A3B`, revision
  `10fbf86fed7ecee4a061f8b499a618f46001cac1`.
- The remote host downloaded all 33 source files in 49 seconds through
  Hugging Face `snapshot_download`.
- The 192-expert materialization used mask SHA-256
  `137efe6c7df7b61f3bf5d4949b3c6a075089fe96f723ef91ac7cdb812e044255`.
- Remote materialization sliced 120 tensors and produced
  55,787,258,440 payload bytes. All 16 shard SHA-256 values matched the local
  materialization receipt. The task-owned 67 GB source copy was removed after
  parity verification.
- Expert-ID map SHA-256:
  `2d32386385d915a0159f1f8e6b2fa8f2c4aee733874a51955e8565e5ffc40471`.
- Token stream: 2,396,160 Ornith token IDs derived from 1,170 contiguous
  disjoint 2,048-token windows. Flat stream SHA-256:
  `58b9c2fb701277c155c9c883d73f43a1651e49af9402b570054418d57b6a3123`.
- The heldout agentic corpus was not included.

## Verification before tuning

The remote environment used Python 3.12.3, torch 2.13.0, transformers 5.18.0,
and the locked `uv sync --frozen --extra dev` environment.

```text
uv run --frozen --extra dev python -m pytest -q tests/test_ornith_qat.py
21 passed in 11.21s
```

Config-only preflight reported 192 experts, top-8, 40 layers, hidden 2048,
intermediate 512, rank 16, alpha 32, and 692,060,160 trainable parameters.

## One-step gates

| Run | Sequence | Elapsed including load | Loss | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| First CUDA step | 64 | 30.494 s | 3.159979 | 65,438 MiB |
| Target-length step | 2,048 | 52.485 s | 1.595706 | 65,582 MiB |

Both runs used the multimodal `Qwen3_5MoeForConditionalGeneration` loader,
BF16 base weights, MXFP4 fake quantization, rank 16, alpha 32, expert
checkpointing, chunked vocabulary loss, and one optimizer update.

The target run showed all 40 expected expert modules under
`model.language_model.layers.<i>.mlp.experts`. The tensors had the expected
fused shapes:

- `gate_up_proj`: `[192, 1024, 2048]`
- `down_proj`: `[192, 2048, 512]`

## 100-step bounded tune

Command shape:

```text
python -m mlx_gptq.ornith_qat train \
  --model /home/ubuntu/models/ornith-192experts \
  --model-class conditional_generation \
  --data /home/ubuntu/ornith/token_stream_2396160.npy \
  --rank 16 --alpha 32 --seq-len 2048 --limit-windows 100 \
  --epochs 1 --max-steps 100 --loss-chunk 512 \
  --device cuda --dtype bfloat16 --gradient-checkpointing
```

Results:

| Metric | Value |
| --- | ---: |
| Optimizer steps | 100 |
| Windows | 100 |
| Context tokens processed | 204,800 |
| Elapsed | 3,913.072 s |
| Average including load | 39.131 s / step |
| Peak sampled VRAM | 66,558 MiB |
| GPU samples | 769 |
| Mean batch loss | 1.432461 |
| Median batch loss | 1.263351 |
| Min/max batch loss | 0.129747 / 3.700701 |
| First logged batch loss (step 5) | 0.410384 |
| Final batch loss (step 100) | 0.954244 |
| Exit | 0 |

At the measured 39.13 seconds per step, a full three-epoch pass over all
1,170 windows would take about 38.2 hours on this lane. That is much slower
than the earlier planning estimate because this implementation quantizes every
routed expert selected across the window and uses the reference linear
attention kernels. The run explicitly warned that `causal-conv1d` and
`flash-linear-attention` were absent.

## Local artifacts

The full resumable step-100 run is preserved under ignored build storage:

```text
/Users/kearm/mlx-stack/build/ornith-qat/seq2048-step100
```

Important files:

| File | Size | SHA-256 |
| --- | ---: | --- |
| `adapter.safetensors` | 2.6 GB | `d88e2204a70b400d9b44286a84fb92888b5502d742fc4155ff553ce84e4cbc2b` |
| `checkpoint.pt` | 7.8 GB | `de851d59b6b6d5dcdcfc80640ab60cac4499ada6dc971b4b4792350262f25979` |
| `summary.json` | 22 KB | `f3db5ce07072bf7a1e3b361b1d32f0059489ad10ef3c09bdbdf93d7426df7964` |
| `metrics.jsonl` | 6.6 KB | `f61201b7315f157e1c7d28d30d0da0f557511596c30c775c113766d1dd6b0b3a` |
| `train.log` | 24 KB | `25a6f7742b8f6e9dcb5f55fd1ebd564cc7aaae6da88b6f8e3102cd980e6f435d` |
| `gpu.csv` | 32 KB | `fb791eebb1d83389a3ee4277c8a8baac016fce0c1fd977bf83d952415d850ecd` |
| `exit` | 2 B | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |

Artifact integrity checks passed locally:

- Summary fingerprint equals checkpoint fingerprint:
  `713b572f5638be028468a9a763a2108ddf9c911b217caceb3b2c920d4434cd24`.
- Checkpoint step is 100.
- Adapter file and checkpoint each contain 160 LoRA tensors, with identical
  key sets.
- All 80 `lora_a` tensors and all 80 `lora_b` tensors are nonzero after
  training.
- The optimizer checkpoint has one parameter group.

The one-step logs and summaries are alongside this run in
`build/ornith-qat/seq64-step1` and `build/ornith-qat/seq2048-step1`.

## Non-claims

- No held-out evaluation or perplexity pass ran after tuning.
- The 100 batch losses are not comparable to a perplexity measurement because
  each value is one shuffled 2,048-token window.
- No merged BF16/MXFP4 checkpoint was exported; the durable result is the
  adapter and optimizer checkpoint.
- No MLX runtime load, kernel parity, serving throughput, or deployment claim
  is made.
- Installing and measuring `causal-conv1d` plus `flash-linear-attention` is
  still open; the measured 39.13 seconds per step is for the reference-kernel
  path.
- B550 remained unreachable over ZeroTier and LAN during this work, so no
  B550 job was queued or disturbed.
