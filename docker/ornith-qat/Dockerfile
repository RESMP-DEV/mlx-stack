# Standard Ornith QAT environment. The base image supplies CUDA, PyTorch,
# Triton, and the architecture-specific FlashAttention build.
ARG BASE_IMAGE=tune-env:ampere-sm86-cuda13.0.1-ubuntu24.04
FROM ${BASE_IMAGE}

ARG USERNAME=ubuntu
ARG MAX_JOBS=8
ARG NVCC_THREADS=4
ARG CUDA_ARCH=8.6
ARG COMPUTE_CAP=86
ARG CAUSAL_CONV1D_REF=cd81f0413cad2fc1e6f17e785ac39f59aae690cd

ENV PYTORCH_CUDA_ARCH_LIST=${CUDA_ARCH} \
    CUDAARCHES=${COMPUTE_CAP} \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS}

USER root
RUN --mount=type=cache,target=/home/${USERNAME}/.cache/uv \
    uv pip install --python /opt/tuning/bin/python \
      "numpy==2.5.1" "transformers==5.18.0" "safetensors==0.8.0" \
      "accelerate==1.15.0" "tqdm==4.70.0" "pytest==9.1.1" \
      "fla-core==0.6.0" "flash-linear-attention==0.6.0"

# Build only the target architecture. The upstream setup otherwise emits many
# architectures and turns a focused image build into a long compile job.
RUN --mount=type=cache,target=/home/${USERNAME}/.cache/git \
    set -eu; \
    repo=/home/${USERNAME}/.cache/git/causal-conv1d; \
    if [ -d "$repo/.git" ]; then git -C "$repo" fetch --tags --force origin; \
    else git clone https://github.com/Dao-AILab/causal-conv1d.git "$repo"; fi; \
    git -C "$repo" checkout --detach "${CAUSAL_CONV1D_REF}"; \
    git clone --no-hardlinks "$repo" /tmp/causal-conv1d-build; \
    cd /tmp/causal-conv1d-build; \
    sed -i '/cc_flag.append("-gencode")/,/cc_flag.append("arch=compute_121,code=sm_121")/d' setup.py; \
    sed -i "/cc_flag = \[\]/a\\    cc_flag.append(\"-gencode\")\n    cc_flag.append(\"arch=compute_${COMPUTE_CAP},code=sm_${COMPUTE_CAP}\")" setup.py; \
    uv pip install --python /opt/tuning/bin/python --no-build-isolation .

RUN mkdir -p /home/${USERNAME}/.cache/{inductor,triton,torch,uv} \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.cache

USER ${USERNAME}
ENV PATH=/opt/tuning/bin:${PATH}
WORKDIR /workspace
CMD ["/opt/tuning/bin/python", "-c", "import torch, transformers, fla, causal_conv1d; print(torch.__version__, transformers.__version__, fla.__version__, causal_conv1d.__version__)"]
