# MLX Stack architecture

## Philosophy

Correctness precedes speed: bit-exact layouts and explicit numerical gates come
first, then bounded-memory preparation, then measured kernel throughput, and
finally maintainability of the consolidated stack. A performance result without
provenance and non-claims is not accepted.

## Ownership

| Boundary | Owner | Current position |
|---|---|---|
| Calibration engine | Root `mlx_gptq/` and `tools/` | Active |
| DFlash MLX runtime | `runtime/dflash-mlx/` | Subtree-merged lineage |
| Qwen campaigns | `campaigns/mlx-gptq-calibration/` | Subtree-merged lineage |
| CUDA MXFP probes | `cuda/mlx-cuda-mxfp/` | Subtree-merged lineage |
| Upstream MLX fork | `RESMP-DEV/mlx` | Pure fork; not dumped into here |
| MLX runtime contract | `RESMP-DEV/mlx-lm` | External contract |

## Open choices

| Topic | Current position | Decision gate |
|---|---|---|
| Default branch promotion | `work/dflash2-calibration` is complete; `main` is older | `just quality` green after subtree merges |
| Old repository retirement | Historical repositories retained read-only as subtree sources; redundant local clones removed | New work targets `mlx-stack` only |
| Full projection matrix | Dominant activation/GEMM cases qualified; all weights qualified | Extend only when a missing shape owns measured time |
| Local 27B CCR fallback | Calibrated MXFP4 loopback server qualified at 32 short-context requests | Promote only through source-policy review and idle listener restart |
| Batched MTP serving | Portable confidence-row, admission, acceptance, and rollback contracts are implemented; live batching is not wired | Exact serial parity, row independence, rollback equality, prefix-cache equality, and paired concurrency gates |
| Target-only serving mode | Default-off `--target-batch-only` passed three live 32 x 128 acceptance runs at 128 to 140 aggregate tok/s with zero errors and zero retries, a paired serial-lane run produced an identical token-ID digest, and 64-way and 128-way bursts now complete cleanly after the backlog hardening | Promote only after a decision on whether the isolated service replaces or complements the resident serial service |

## 2026-10-20 work log

- Renamed the owned engine repository from `mlx-gptq-mxfp` to `mlx-stack`,
  preserving GitHub redirects and updating local remotes.
- Subtree-merged `dflash-mlx`, `mlx-gptq-calibration`, and `mlx-cuda-mxfp`.
- Added fused SM120 activation MXFP4 packing, direct uint32 E2M1 weight
  repacking, layout regression coverage, and the Tetra evidence receipt
  `evidence/2026-10-20-tetra-fused-mxfp4-activation-weight-packing.md`.
- Added the stack quality dispatcher and component ownership map.
- The full quality dispatcher passed after an M4-specific DFlash verify-kernel
  deviation ceiling was pinned; receipt summary:
  `evidence/2026-10-20-mlx-stack-consolidation.md`.
- Imported the campaign evidence formerly tracked in `RESMP-DEV/mlx`, merged
  the pushed Qwen parity-tooling branch, and removed the three verified-clean
  redundant local clones. `/Users/kearm/mlx-stack` is the convenience entry to
  the canonical checkout.
- Measured the local calibrated Qwen3.8-27B MXFP4 artifact through an isolated
  CCR OpenAI route: 142 aggregate tok/s at 32 short-context requests, but only
  about two useful concurrent 8k prefills. Receipt:
  `evidence/2026-10-20-local-27b-ccr-fallback-capacity.md`.
- Consolidated four bounded EAGLE-4 audits into the correctness-first Qwen
  speculative-decoding plan. The pinned draft, official inference source,
  paper, target-vocabulary contract, low-rank/full-head behavior, and Qwen GDN
  verification constraint are recorded in
  `evidence/2026-10-20-eagle4-qwen38-27b-mlx-port-research.md`.
- Added the first EAGLE-4 MLX draft model and strict loader/schema gate. The
  real Qwen checkpoint lazy-loads all 19 BF16 tensors and passes a one-token
  Metal full-head/candidate-head smoke; the existing DFlash engine backend is
  explicitly gated until the EAGLE adapter is built. See the implementation
  section of the same evidence receipt.

## 2026-10-21 work log

- Added the correctness-first EAGLE-4 chain backend to the DFlash engine,
  including absolute target positions, exact candidate-ID rescoring, greedy
  suppression, and explicit rejection of unsupported tree/sample paths.
- Parent review caught and repaired the initial zero-based RoPE positions;
  official DraftExtend uses `seq_len + j`. An independent review approved the
  pairing and mapping, and its sparse-context and greedy-path repairs are
  included.
- Measured corrected EAGLE-4 b2 at 8.618 tok/s on BF16 and 33.568 tok/s on
  the owned calibrated MXFP4 target, with exact native greedy IDs. The
  incumbent MXFP4 DFlash2 stack remains best at 51.028 tok/s. Full details,
  commands, and non-claims:
  `evidence/2026-10-21-eagle4-bf16-mxfp4-serving-comparison.md`.
- Added selective W4 EAGLE-4 draft-body quantization while preserving its
  embedding, exact full head, and rank-128 selector. Warm median improved to
  34.522 tok/s with exact native IDs; the default remains explicit
  `--draft-quant w4`.
- Audited the seven new upstream MLX commits and rejected the two QMM changes
  for serving: split-K precision failed on M4 and real model throughput did
  not improve. Synced the two safe MLX-LM upstream fixes into
  `RESMP-DEV/mlx-lm`.
- Pulled and verified the official DSpark BF16 head, quantized the pinned
  local MTP BF16 head ourselves, and measured both with upstream `mlx-vlm`:
  local MTP BF16 b2 reached 26.705 tok/s, local MTP W4 b2 reached 25.481
  tok/s, and DSpark reached 24.680 tok/s. A briefly downloaded community MTP
  quant was removed from the decision set and replaced by the local artifact.
  The compute-plus-bandwidth model and NVIDIA comparison are recorded in
  `evidence/2026-10-21-prediction-head-upstream-and-cost-audit.md`.
- Stood up a resident loopback local27 service using only the owned calibrated
  MXFP4 target and campaign-owned GPTQ-MXFP4 DFlash2 drafter. The service is
  launchd-managed on `127.0.0.1:8081`, holds about 19.4 GiB active, serves a
  real request at 54.19 decode tok/s, and supports explicit reversible
  eviction. The Strata/DeepSeek scheduler analysis and staged concurrency
  experiments are recorded in
  `evidence/2026-10-21-resident-local27-mxfp4-serving.md`.
- Added phase-one bounded admission on loopback port 8080. It removes burst
  connection resets, but DFlash remains serial because MLX-LM marks loaded
  drafters non-batchable. A temporary plain-MLX batch control did not
  reproduce the earlier 142 tok/s operating point and was stopped; the next
  concurrency experiment must replay that receipt's exact workload and arrival
  pattern before changing the resident engine.
- Audited FreeToken `cc4aa085`, Strata `66067e30`, MLX-LM 0.31.3, and the
  DFlash seams with four bounded workers. FreeToken has no serving-loop MTP
  and drops MTP checkpoints; its reusable contribution is multi-currency
  admission and capability-declared cache management. Strata's pure
  confidence-varlen planner is the portable batched-MTP policy. Added the
  backend-neutral confidence-row planner, short-first aging admission,
  per-row acceptance contract, rollback grouping, and JSON interop in
  `runtime/dflash-mlx/dflash_mlx/server/serving_contracts.py`. See
  `evidence/2026-10-21-freetoken-serving-and-batched-mtp-architecture.md`.
- Added default-off target-only continuous batching and repaired the target
  hook boundary after a real BatchKVCache failure. The isolated mode reached
  138.95 aggregate completion tok/s for 32 requests x 128 tokens with exact
  visible greedy-content agreement to the resident serial lane. Added a
  single-call heterogeneous GDN `rollback_batch` primitive and a durable
  target-batch health/matrix probe. See
  `evidence/2026-10-21-target-batch-hardening.md`.

- Hardened the target-batch probe into a post-deploy acceptance tool. It
  samples `/metrics` on a background thread during the matrix, hashes the
  exact visible content of every response, offers an optional streaming
  health request, and captures generated token IDs only from the field the
  pinned MLX-LM 0.31.3 server actually returns. The token-ID path was
  verified by executing the pinned server's own response builder.
- Ran the probe live against an isolated target-batch service on port 8082
  using the owned calibrated artifact. Three consecutive 32-way runs of 128
  tokens completed 32 of 32 requests with zero errors, one identical token-ID
  digest across all rows, a passing streaming health request, and 130.11 to
  142.29 aggregate completion tok/s. The live runs exposed two defects that
  are now fixed: transient connection resets under a 32-way burst, which the
  probe now retries, and the metrics sampler perturbing that burst, so
  sampling is now opt-in.
- Paired the batch lane against the resident serial DFlash lane on the same
  32-way, 128-token workload with token-ID capture. Both lanes produced the
  same token-ID digest and the same content digest, so batching changed
  throughput and not output. The batch lane finished in 31.48 s at 130.11
  aggregate tok/s; the serial lane took 238.67 s at 17.16 aggregate tok/s.
- Measured long prompts on the batch lane with about 9,025 prompt tokens and
  distinct prefixes. Prefill reached 212.3 tok/s at one way, 382.0 at two,
  and 261.6 at four, so long-prompt concurrency stays low and the lane
  should not take a high-concurrency long-prompt workload. See
  `evidence/2026-10-21-target-batch-probe-hardening.md`.
- Hardened the HTTP server listen backlog. The stdlib server listened with a
  queue of 5 pending connections, so a burst of concurrent clients had its
  excess connections reset by the kernel while the server logged 200 for
  every request. The runtime now serves with a backlog of 1024. A 32-way
  run that previously needed 13 and 12 client retries now needs none, and
  64-way and 128-way bursts complete with zero errors. The token-ID digest
  is unchanged, so the fix changed connection behavior and not output. See
  `evidence/2026-10-21-http-server-backlog-hardening.md`.

## 2026-10-22 work log

- Recovered the Flint state after the Mac's button-sequence reset. The MLX-LM
  adapter and stack are committed and remote-matched; B550 completed all 49
  real-checkpoint GPTQ-MXFP4 calibration layers, the 20 GB artifacts were
  copied and hash-matched, and the full five-shard 21 GiB model was packed
  through its load-and-generate verification path. Post-restart adapter and
  calibration tests passed. Quality evaluation and publication remain open.
  The reset diagnostic proves a forced reset path but not the original stall
  cause. See `evidence/2026-10-22-flint-post-restart-status.md`.

## 2026-10-23 work log

- Tested the real Flint checkpoint in MLX, found and repaired a routed-expert
  gate/up activation-order defect in `RESMP-DEV/mlx-lm`, and invalidated the
  earlier incoherent adapter measurements. On the real held-out coding corpus,
  the repaired owned MXFP4 checkpoint reached PPL 7.4038, BF16 reached 8.5360,
  and the official Apple Q4_1 control reached 6.1512. The owned artifact kept
  77.86% next-token top-1 agreement with BF16 and decoded at 131.5 tok/s with
  35.5 GiB peak memory. It is published publicly as 25 verified roughly 1 GB
  shards under `RESMP-DEV/Flint-40B-A2.5B-MLX-GPTQ-MXFP4`; every remote LFS
  SHA-256 and size matched locally. See
  `evidence/2026-10-23-flint-mlx-eval-publication.md`.
- Benchmarked the repaired public Flint checkpoint on the M4 Max. Direct
  greedy generation decoded at 126.98 tok/s at short context, 111.97 tok/s
  after 8K prompt tokens, and 91.89 tok/s after 32K prompt tokens. A
  high-backlog temporary MLX-LM server reached its practical best at 128
  concurrent short-prompt requests: 663.99 aggregate completion tok/s, 801.20
  aggregate decode tok/s, 2.93 s median TTFT, and a 54 GB footprint. The
  192-way run is not recommended: although decode reached 840.05 tok/s,
  completion throughput and TTFT regressed and the footprint reached 92 GB.
  See `evidence/2026-10-23-flint-m4-performance.md`.
- Added an independent WikiText-2 raw-validation correctness gate. Over sixteen
  fixed 2,048-token windows and identical source tokenization, MXFP4 reached
  PPL 21.2884 versus BF16 22.3666, with mean NLL lower by 0.04940 nats/token
  and 73.20% exact top-1 agreement with BF16. The upstream RecursiveSI source
  revision is unchanged, upstream MLX-LM has no Flint implementation, and the
  Flint cache is standard hybrid attention: 14 full `KVCache` layers plus 35
  513-token `RotatingKVCache` layers with learnable per-head attention-sink
  offsets. See
  `evidence/2026-10-23-flint-independent-ppl-and-upstream.md`.
- Published the independent WikiText-2 metrics and caveat on the public Flint
  Hugging Face model card at `edc81733`. Promoted the high-backlog server
  wrapper and synchronized streaming benchmark into supported scripts, with a
  real two-request smoke and report provenance. A live vLLM Metal 0.32 probe
  showed that macOS paged attention with GQA and sinks is real and numerically
  close to MLX SDPA, but Flint currently fails before load because the stable
  plugin does not recognize the `flint` Transformers config and its MLX-LM
  dependency lacks the repaired adapter. See
  `evidence/2026-10-23-flint-public-metrics-and-serving.md`.
- Repaired the eager Flint calibration skeleton to use NeoX-style half-split
  RoPE, matching the repaired MLX runtime's `traditional=False` behavior. A
  direct MLX parity regression now pins the layout. Future RCO-mask search,
  materialized 192-expert recalibration, and any GSQ lane must run after this
  repair. See `evidence/2026-10-23-flint-calibration-rope-repair.md`.
- Ported the omni-use top-20 logprob Shannon-entropy controller to an opt-in
  Flint benchmark. MLX-LM now supports 20 top logprobs, and mlx-stack calibrates
  a real-prefix cut/continue policy with fail-closed missing-entropy behavior.
  A two-task live smoke exercised the full wire path and reduced the capped run
  from 64 to 32 mean tokens without changing the measured hit count. This is
  transport proof, not general quality or savings. See
  `evidence/2026-10-23-flint-reasoning-entropy.md`.

## 2026-10-24 work log

- Added a 3D-aware routed-expert MXFP4 QAT trainer for Qwen3.5-MoE fused
  expert tensors, with per-expert LoRA, straight-through MXFP4 fake
  quantization, token-window training, resume fingerprints, stock-layout
  merge export, and 21 synthetic tests. A real one-step and bounded 100-step
  run on a Packet RTX PRO 6000 96GB completed at sequence length 2,048,
  processing 204,800 context tokens with 66,558 MiB peak VRAM and preserving
  a local adapter plus optimizer checkpoint. This is execution and memory
  proof, not a quality-recovery result. See
  `evidence/2026-10-24-ornith-qat-rtxpro6000-100step.md`.

## Contracts to preserve

- E2M1 uint32 words use MLX's LSB-first `4*i` layout; byte operands use low
  nibble first.
- Scale layout changes must be byte-compared, not inferred from throughput.
- GPU claims require live device receipts and explicit source provenance.
- The root, runtime, campaign, and CUDA checks are one quality surface.
- Community-quantized targets or drafters are never acceptance baselines;
  quantized comparisons must come from the pinned source and this stack.
- The resident local27 MXFP4 service is loaded by default. Explicitly evict it
  with its service script before a substantially larger local job and restore
  it afterward; do not kill the worker and race launchd.

- Built and sealed a physical 224-expert Flint MXFP4 artifact from the
  calibrated full-256 checkpoint, preserving q/scales and adding a manifest,
  checksum set, expert map, and mask. The artifact matches full 256 on 32
  spread agentic-coding windows: PPL 6.20609 versus 6.20711, NLL delta
  -0.00016, CI [-0.04943, 0.04477], and saves 2.35 GiB. Added the resident
  `dev.resmp.flint224-mxfp4` LaunchAgent on `127.0.0.1:8084`, with clean
  process environment, four-way decode admission, health/status commands, and
  live completion, streaming, top-20 logprob, and four-request burst checks.
  See `evidence/2026-10-23-flint-224-mxfp4-serving.md`.
