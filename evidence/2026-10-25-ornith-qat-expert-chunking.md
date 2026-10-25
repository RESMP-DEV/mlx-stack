# Ornith routed-expert MXFP4 QAT: vectorized expert chunks

Date: 2026-10-25
Agent: Codex (CCR evidence worker)
Source revision: `095c439877417f344d4c113c27b4e1c7d8b1e2f6` (worktree `work/ornith-qat-trainer`)
Owned files: `mlx_gptq/ornith_qat.py`, `tests/test_ornith_qat.py`
Environment: Python 3.13.9, torch 2.13.0, transformers 5.18.0, ruff 0.17.1, CPU only

## Status

**Implemented and measured on CPU:** the routed-expert forward path of the
Ornith MXFP4 QAT trainer now has a vectorized chunk path that groups the
routed experts, pads the group to its largest token count and evaluates the
group with batched matmuls. The previous one-expert-at-a-time loop is kept
unchanged as the sequential reference and compatibility mode.

**Not claimed:** no 35B weight was downloaded, no CUDA device was reached,
and no GPU step time is measured here. The timings below are CPU timings of
one synthetic layer at the Ornith layer shape. The expected GPU gain is a
mechanism argument plus the measured launch-count reduction, not a measured
RTX PRO 6000 step time.

## Design

### Two routing paths in `FusedExpertsQAT`

`expert_chunk_size` selects the path:

- `expert_chunk_size = 0`: the sequential reference path. It is the previous
  implementation, unchanged, including the per-expert gradient checkpoint.
- `expert_chunk_size > 0` (default `ORNITH_EXPERT_CHUNK_SIZE = 16`): the
  vectorized chunk path.

`FusedExpertsQAT.forward` dispatches to `_sequential_forward` or
`_chunked_forward`. The sequential path is byte-for-byte the previous
forward, so it stays the reference the chunk path is measured against.

### The chunk path

1. The routing is flattened: `top_k_index.reshape(-1)` gives one expert id
   per `(token, position)` pair, `torch.arange(tokens).repeat_interleave(top_k)`
   gives the token of each pair and `top_k_weights.reshape(-1)` the weight.
   An expert id outside `[0, num_experts)` is mapped to the drop bucket
   `num_experts`, exactly as the sequential path skips it.
2. The pairs are sorted by expert id with a stable sort. The valid region
   stays a prefix of the sorted array, so the offset of every expert is the
   running sum of the per-expert counts.
3. `plan_expert_chunks(counts, chunk_size)` groups the experts that have at
   least one routed token into groups of at most `chunk_size` experts, in
   ascending id order. It is a pure function of the counts, so the plan of a
   routing is deterministic and testable without a GPU.
4. A group is padded to the largest token count of the group. The gather
   index, the token weights and the mask are built with tensor operations
   from the group offsets, so no per-expert Python work remains inside the
   group.
5. The group is evaluated with batched matmuls: `torch.bmm` over the group
   axis for the gate half, the up half and the down projection. The padded
   rows carry a mask of zero and contribute nothing to the routed output.
6. The group result is scattered back with `index_add_` into the same output
   buffer the sequential path fills.

`checkpoint_experts` now wraps a group instead of a single expert, so one
recompute per group replaces one recompute per routed expert.

### The effective weight is unchanged

The MXFP4 group runs along the last axis of the weight, which is the
reduction axis of both fused tensors. Quantizing a stacked group of experts
therefore gives each expert exactly the codes the sequential path computes
for it. The test `test_chunked_effective_weights_are_bit_identical` pins
that with `torch.equal` on the quantized tensors.

The LoRA update is the one place where a batched product could differ from
the per-expert product in the last bit, which could move a weight across a
quantization boundary. The module probes this once per module: the batched
`bmm` product is compared against the exact per-expert product, and the
exact per-expert product is used when the two differ. On this machine the
probe selected `bmm`, meaning the batched product is bit-identical here.

### Configuration surface

| Surface | Knob |
| --- | --- |
| `TrainConfig` | `expert_chunk_size: int = ORNITH_EXPERT_CHUNK_SIZE` |
| CLI | `--expert-chunk-size N` on `train`, `0` selects the sequential path |
| `install_expert_qat_lora` | `expert_chunk_size: int = ORNITH_EXPERT_CHUNK_SIZE` |
| `FusedExpertsQAT` | `expert_chunk_size: int = ORNITH_EXPERT_CHUNK_SIZE` |

The value is recorded in the training plan and in `summary.json` as
`expert_chunk_size`. The training fingerprint is unchanged, so an existing
checkpoint still resumes; the chunk size does not change the adapter
format.

## Files changed

| File | Change |
| --- | --- |
| `mlx_gptq/ornith_qat.py` | chunk path, `plan_expert_chunks`, `ExpertChunkPlan`, the `expert_chunk_size` knob, the delta probe |
| `tests/test_ornith_qat.py` | 9 new tests: plan tradeoff, chunk/sequential agreement, bit-identical effective weights, gradient coverage, gradient agreement, drop bucket, the knob, the summary field, the benchmark |
| `evidence/2026-10-25-ornith-qat-expert-chunking.md` | this receipt |

No Docker file, production configuration, live service, routing table or
remote was touched.

## Commands and results

Run from the worktree root
`/Users/kearm/mlx/.worktrees/mlx-gptq-mxfp/.worktrees/ornith-qat-trainer`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --frozen --extra dev python -m pytest -q tests/test_ornith_qat.py
# 30 passed in 2.03s

RUFF_CACHE_DIR=/tmp/ruff-cache UV_CACHE_DIR=/tmp/uv-cache uv run --frozen ruff check mlx_gptq/ornith_qat.py tests/test_ornith_qat.py
# All checks passed!  (exit 0)

PYTHONPATH=. .venv/bin/python -m pytest tests scripts/tests -q \
  --ignore=tests/test_awq_lfm.py \
  --ignore=tests/test_flint_torch.py \
  --ignore=tests/test_grid_and_packing.py \
  --ignore=tests/test_mxfp_artifacts.py
# 153 passed, 3 skipped in 3.62s
```

`UV_CACHE_DIR` and `RUFF_CACHE_DIR` point at `/tmp` because the default
cache directories are outside the writable roots of this worker sandbox.
They do not change what is checked.

`just quality` was run as well:

```bash
RUFF_CACHE_DIR=/tmp/ruff-cache UV_CACHE_DIR=/tmp/uv-cache just quality
# root_lint 0, root_tests 0 (160 passed, 17 skipped), runtime_tests 2,
# campaign_tests 0, cuda_compile 0, aggregate passed: false
```

The single failure is `runtime_tests`, which runs
`uv run --no-project --with mlx==0.32.1 ...` and failed to reach PyPI from
this sandbox (`dns error: failed to lookup address information`). It fails
before any test runs and is unrelated to this change. The receipt is at
`build/reports/quality/receipt.json` with `passed: false`; the parent should
re-run `just quality` on a networked host.

Parent acceptance after review:

```bash
uv run --frozen --extra dev python -m pytest -q tests/test_ornith_qat.py
# 30 passed in 2.11s

uv run --frozen ruff check mlx_gptq/ornith_qat.py tests/test_ornith_qat.py \
  docker/ornith-qat/preflight.py docker/ornith-qat/benchmark.py
# All checks passed!
```

The parent also tightened the one-time device probe so both the gate/up and
down LoRA shapes must match their exact per-expert products before `bmm` is
selected. A later networked parent `just quality` attempt still failed only
while fetching `mlx==0.32.1` from PyPI after a timeout; root lint, root tests,
campaign tests, and CUDA compile passed. That network fetch is unrelated to
this source change.

### Routing-shape analysis at the Ornith shape

192 experts, 2048 tokens, top-8, 16384 routed pairs, a random routing with
all 192 experts hit:

```text
chunk  chunks  batched_matmuls  padded_rows  padding_overhead
    1     192             384        16384             0.00%
    8      24              48        18992             15.92%
   16      12              24        19664             20.02%
   32       6              12        20448             24.80%
   64       3               6        21632             32.03%
  192       1               2        22272             35.94%
```

The padding only affects the token matmul. The per-layer weight
quantization (604M elements) and the LoRA product (192 products) are
unaffected by the chunk size, so at chunk 16 the padded token work grows by
20% while the number of batched matmuls per layer falls from 384 to 24.

### CPU timing of one Ornith-shaped layer

192 experts, hidden 2048, intermediate 512, 2048 tokens, top-8, float32,
random base weights, nonzero LoRA adapter, forward only:

```text
chunk=  0 forward=  1.569s checksum=1128338.875000
chunk=  8 forward=  1.067s checksum=1128338.875000
chunk= 16 forward=  1.015s checksum=1128338.875000
chunk= 32 forward=  1.133s checksum=1128338.875000
chunk= 64 forward=  1.252s checksum=1128338.875000
```

The checksum of the absolute output sum is identical for every chunk size.
Chunk 16 is the fastest on this shape, and it is the default.

### Sequential versus chunked agreement at the Ornith layer shape

Same shape, same adapter, forward and backward:

```text
chunk=  8 out_maxdiff=5.960e-08 out_rel=2.669e-08 gu_grad_maxdiff=5.341e-05 gu_rel=7.709e-07 dn_rel=4.020e-07
chunk= 16 out_maxdiff=1.192e-07 out_rel=5.338e-08 gu_grad_maxdiff=5.341e-05 gu_rel=7.709e-07 dn_rel=4.020e-07
chunk= 32 out_maxdiff=5.960e-08 out_rel=2.669e-08 gu_grad_maxdiff=5.341e-05 gu_rel=7.709e-07 dn_rel=4.020e-07
```

The outputs and the LoRA gradients agree at the float32 rounding level.

The same comparison with `checkpoint_experts = True` (the configuration of
the measured run) gives the same numbers, so the checkpointed chunk path
recomputes the same values:

```text
chunk=  0 out_maxdiff=0.000e+00 out_rel=0.000e+00 gu_grad_maxdiff=0.000e+00 gu_rel=0.000e+00 dn_rel=0.000e+00
chunk= 16 out_maxdiff=1.192e-07 out_rel=5.338e-08 gu_grad_maxdiff=5.341e-05 gu_rel=7.709e-07 dn_rel=4.020e-07
```

## Expected optimization mechanism

The 39.13 s/step of the RTX PRO 6000 run was measured with the sequential
path, which walks 192 routed experts per layer per forward, 40 layers, and
repeats the walk when the checkpoint recomputes it. Per routed expert the
path issues a one-hot mask build, a `where`, a weight gather, two LoRA
products, two MXFP4 quantizations, two linear layers, an activation and an
`index_add_`: on the order of ten kernel launches per expert, so about 77k
launches per forward and the same again in the backward recompute.

The chunk path keeps every arithmetic operation and removes the per-expert
Python and launch overhead:

- 24 batched matmuls per layer instead of 384 linear layers at chunk 16.
- 12 checkpoint segments per layer instead of 192, so 12 recomputes instead
  of 192 in the backward.
- The `[num_experts + 1, top_k, tokens]` one-hot mask of the sequential path
  (about 25 MB per layer at the Ornith shape) is not built at all.
- The MXFP4 quantization of a group is one kernel over
  `[chunk, rows, columns]` instead of one kernel per expert, with the same
  element count and the same codes.
- The cost paid for this is the padding of the token axis: 20% more token
  rows at chunk 16, on the token matmul only.

The layer cost is dominated by the weight quantization and the LoRA product,
which the chunk path does not change, so the expected gain is bounded by the
share of the layer time spent in the expert loop overhead, not by the
padding. The CPU measurement above puts that share at roughly 35% of the
layer forward at this shape. A GPU host is launch-bound far more than a CPU
host is, so the expected GPU gain is larger than the CPU ratio, but that is
a hypothesis, not a measurement.

## Remaining GPU benchmarks

The parent must run these on the RTX PRO 6000 lane; none can be settled
here.

1. One step at the target shape with the default chunk size and with
   `--expert-chunk-size 0`, same seed, same window:
   `python -m mlx_gptq.ornith_qat train --model <dir> --data <tokens.npy>
   --output <run> --rank 16 --alpha 32 --seq-len 2048 --epochs 1
   --max-steps 1 --device cuda --dtype bfloat16 --gradient-checkpointing`.
   Compare the step time and the peak VRAM.
2. A chunk-size sweep at the target shape, `--expert-chunk-size 8`, `16`,
   `32` and `0`, one step each, to find the GPU optimum. The padding
   analysis above says the CPU optimum is 16, but the GPU optimum can sit
   elsewhere because the launch saving grows with the chunk size while the
   padding cost grows with it too.
3. A loss-parity check: the same window through the sequential path and the
   chunk path must give the same loss to the printed precision and the same
   adapter bytes.
4. Check the probe decision on the GPU host. If the CUDA `bmm` differs from
   the per-expert product in the last bit, the module falls back to the
   exact per-expert product and the chunk path keeps the same speedup from
   the batched matmuls alone.

## Non-claims

- No GPU execution, throughput, memory peak or kernel parity is claimed.
  Every number above is a CPU number on this machine.
- No 35B weight was downloaded, loaded or run. The synthetic layer uses
  random weights of the Ornith shape, not the Ornith checkpoint.
- The agreement between the two paths is at the float32 rounding level
  (about 1e-7 relative on the output). It is not a bit-exact match of the
  output tensors, because a batched matmul and a per-expert matmul do not
  have to accumulate in the same order. The quantized weights themselves
  are bit-identical, which the test pins.
- The CPU speedup of the synthetic layer does not transfer to the RTX PRO
  6000 as a number. Only the mechanism and the launch counts do.
- The step time of the 39.13 s/run is not re-measured, and the reference
  causal-conv1d and flash-linear-attention kernels are still absent, so
  part of that step time is unrelated to the expert path.
- No model-quality claim is made. The chunk path changes the floating-point
  accumulation order of the expert matmuls, which is not a training result.
- The exported adapter format is unchanged, and the merged export path was
  not re-run against a checkpoint in this work.
