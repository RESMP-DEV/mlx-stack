# Ornith QAT tuning image

`Dockerfile` adds the Ornith training stack to an existing
architecture-specific `tune-env` image. It is the fast path on B550, where the
CUDA 13.0.1 Torch 2.14 base images are already built.

`Dockerfile.standalone` starts from `nvidia/cuda:13.0.1-devel-ubuntu24.04`
and installs every dependency itself. Use it on a fresh host.

`Dockerfile.offline` adds the same Python stack to an existing `tune-env`
base from a prepared wheelhouse. It is for hosts with working Docker and GPU
access but unreliable DNS or package-index routes.

## B550 SM86

```bash
docker build -f docker/ornith-qat/Dockerfile \
  --build-arg BASE_IMAGE=tune-env:ampere-sm86-cuda13.0.1-ubuntu24.04 \
  --build-arg CUDA_ARCH=8.6 \
  --build-arg COMPUTE_CAP=86 \
  -t mlx-stack:ornith-qat-sm86-cuda13.0-torch2.14 .
```

## RTX PRO 6000 SM120

```bash
docker build -f docker/ornith-qat/Dockerfile.standalone \
  --build-arg CUDA_ARCH=12.0 \
  --build-arg COMPUTE_CAP=120 \
  -t mlx-stack:ornith-qat-sm120-cuda13.0-torch2.14 .
```

For a local `tune-env:blackwell-sm120-cuda13.0.1-ubuntu24.04` base, substitute
the smaller `Dockerfile` and pass that base image.

## Offline SM120

Prepare `requirements.txt`, a CPython 3.12 Linux wheelhouse, and a source
tarball for `causal-conv1d` commit
`cd81f0413cad2fc1e6f17e785ac39f59aae690cd`. Copy this file and the runtime
checks into that context, then build:

```bash
DOCKER_BUILDKIT=1 docker build --network=none -f Dockerfile.offline \
  --build-arg BASE_IMAGE=tune-env:blackwell-sm120-cuda13.0.1-ubuntu24.04 \
  --build-arg CUDA_ARCH=12.0 \
  --build-arg COMPUTE_CAP=120 \
  -t mlx-stack:ornith-qat-sm120-cuda13.0-torch2.14 .
```

## Runtime checks

Mount this repository at `/workspace/mlx-stack` and run:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace/mlx-stack" \
  -w /workspace/mlx-stack \
  mlx-stack:ornith-qat-sm86-cuda13.0-torch2.14 \
  /opt/tuning/bin/python docker/ornith-qat/preflight.py
```

For warm latency measurements, replace `preflight.py` with `benchmark.py`.
For routed-expert chunk sizing, replace it with `expert_chunk_benchmark.py`.
The routed-expert benchmark imports `mlx_gptq` from this repository mount.
The first call may compile Triton kernels; only the benchmark's repeated warm
samples are performance evidence.

The image pins Torch 2.14.0+cu130, Triton 3.8.0, Transformers 5.18.0,
Flash-Linear-Attention 0.6.0, and `causal-conv1d` commit
`cd81f0413cad2fc1e6f17e785ac39f59aae690cd`. The causal-convolution build is
restricted to the requested compute capability so an image build does not
compile the upstream's full architecture matrix.
