# Ornith QAT SM120 offline image and expert-chunk measurement

Date: 2026-10-25
Agent: Codex
Source revision: `1794125` plus this worktree's uncommitted Docker and SM120
memory changes.

## Status

**Implemented, built, and measured:** Strix built the Ornith QAT SM120 image
from a 43 MiB offline context with Docker `--network=none`, and a real
one-layer Ornith-shaped routed-expert benchmark ran beside the resident
embedding service. The existing GPU owner was PID `121013`
(`/home/kearm/.probe-mxfp4-sm120/bin/python`); it remained alive at 4572 MiB
before and after every run. It was not stopped or restarted.

**Not claimed:** this is not full-model QAT, not a quality-recovery result, and
not a clean single-tenant GPU benchmark. Strix has an 8151 MiB GPU and the
resident service owned 4572 MiB during the measurements.

## Host and image

Authenticated identity:

```text
hostname: strix
machine-id: b71b3a116ec74b7a8797f38a0af8168a
GPU: NVIDIA GeForce RTX 5070 Laptop GPU, SM 12.0
driver: 595.99.02
```

Offline build inputs:

```text
context: 43 MiB, 30 Linux CPython 3.12 wheels
causal-conv1d: v1.7.0, commit cd81f0413cad2fc1e6f17e785ac39f59aae690cd
source tar SHA-256: d6320e63c8e5739e64bcd3202f8418d4c7708ea5903f35b32c9ae05ff4a22a3d
```

Build command:

```bash
DOCKER_BUILDKIT=1 docker build --network=none \
  --build-arg BASE_IMAGE=tune-env:blackwell-sm120-cuda13.0.1-ubuntu24.04 \
  --build-arg CUDA_ARCH=12.0 \
  --build-arg COMPUTE_CAP=120 \
  --build-arg MAX_JOBS=4 \
  --build-arg NVCC_THREADS=2 \
  -t mlx-stack:ornith-qat-sm120-cuda13.0-torch2.14 .
```

Resulting image identifiers from the BuildKit log:

```text
tag: mlx-stack:ornith-qat-sm120-cuda13.0-torch2.14
config: sha256:207be499fca46d999705b36546070b7a9e952f1ca28507ec92ed632c8df96766
manifest list: sha256:c303eeaf410aff7da783948d7bd24bb8da9ebf221e45f2991bbb4474d6155d8
```

Validated image versions:

```text
torch 2.14.0+cu130
triton 3.8.0
transformers 5.18.0
flash-linear-attention 0.6.0
causal-conv1d 1.7.0
CUDA capability (12, 0)
```

The first online-style dependency resolution tried to pull a redundant Torch
and CUDA stack through `fla-core[cuda]`. The offline lock instead uses
`fla-core==0.6.0`, `flash-linear-attention==0.6.0`, and the Torch/Triton/CUDA
packages already present in the architecture-specific base image.

## Kernel and layer checks

The image checks were run with GPU access. Cold calls included Triton
compilation and are not performance evidence.

Warm medians at sequence length 2048:

| Check | Median | Peak |
| --- | ---: | ---: |
| `causal_conv1d_fn`, shape `[1, 8192, 2048]` | 0.000209 s | n/a |
| `chunk_gated_delta_rule`, real GDN shape | 0.000972 s | n/a |
| synthetic one-linear-layer Transformers model | 0.008640 s | 710.8 MiB |

There were no Transformers fallback warnings.

## Routed-expert chunk sweep

The real synthetic routed-expert shape was:

```text
experts: 192
hidden: 2048
intermediate: 512
tokens: 2048
top_k: 8
routed rows: 16384
LoRA rank: 16
dtype: bfloat16
gradient checkpointing: enabled
frozen weights: 1152.0 MiB
LoRA factors: 66.0 MiB
```

The timing method interleaved arms in five paired rounds, so the same GPU
contention from the resident service affects the sequential and chunked paths.
Paired delta is chunk time minus sequential time in the same round; negative is
an improvement.

| Chunk | Forward median | Training-step median | Paired forward delta | Paired step delta |
| ---: | ---: | ---: | ---: | ---: |
| 0 sequential | 3.708 s | 7.293 s | baseline | baseline |
| 8 | 0.999 s | 2.065 s | -2.703 s | -5.235 s |
| 16 | 0.955 s | 2.034 s | -2.754 s | -5.259 s |

Minimum paired samples show the same direction: sequential forward minimum was
3.679 s versus 0.365 s at chunk 8 and 0.391 s at chunk 16; sequential training
step minimum was 4.983 s versus 0.824 s at chunk 8 and 0.872 s at chunk 16.

Peak allocated memory from the same fixed-path run was 1602.1 MiB sequential,
1846.3 MiB at chunk 8, and 2188.9 MiB at chunk 16. The first chunked attempt
before the grid repair failed while trying to allocate a 512 MiB nested
`torch.where` temporary. `_e2m1_codes` now accumulates thresholds in uint8;
CPU tests compare that form bit-for-bit with the old nested-threshold form at
random values and every signed boundary.

Observed BF16 relative L2 differences against one sequential reference were:

| Chunk | Output | Gate LoRA grad | Down LoRA grad | Hidden grad |
| ---: | ---: | ---: | ---: | ---: |
| 0 same path | 5.32e-4 | 9.83e-4 | 8.57e-4 | 1.32e-3 |
| 8 | 6.01e-3 | 6.78e-3 | 7.47e-3 | 7.50e-3 |
| 16 | 6.01e-3 | 6.78e-3 | 7.47e-3 | 7.55e-3 |

The chunk-0 row measures same-path CUDA/BF16 nondeterminism. The chunked rows
add batched-matmul reduction-order differences. Full-shape CPU float32 parity
previously measured output relative error near 5.3e-8; no model-quality or
token-agreement claim follows from this synthetic layer.

## Commands and validation

Local source checks:

```bash
uv run --frozen --extra dev ruff check \
  mlx_gptq/grid.py tests/test_grid_and_packing.py docker/ornith-qat
# All checks passed

uv run --frozen --extra dev python -m pytest -q \
  tests/test_grid_and_packing.py tests/test_ornith_qat.py
# 35 passed, 11 skipped

just quality
# passed: root_lint, root_tests, runtime_tests, campaign_tests, cuda_compile
# build/reports/quality/receipt.json, recorded_at 2026-10-25T23:40:17Z
```

One exact-head rerun exposed a one-off timeout in
`TestL2Lifecycle.test_evict_promotes_to_l2_disk`; that single test then passed
five consecutive targeted runs, and the final full gate above passed at the
same clean revision.

Remote logs preserved on Strix:

```text
/home/kearm/ornith-qat-offline/build-ornith-qat-sm120.log
/home/kearm/ornith-qat-offline/version.txt
/home/kearm/ornith-qat-offline/ornith-env-preflight-sm120.json
/home/kearm/ornith-qat-offline/ornith-env-bench-sm120.json
/home/kearm/ornith-qat-offline/ornith-expert-chunk-fixed.log
/home/kearm/ornith-qat-offline/ornith-expert-chunk-interleaved.log
```

An independent read-only Qoder review was launched for the Docker and benchmark
diff, but it timed out after 900 seconds with no completed tool commands or
final report. It contributed no finding and made no changes.
