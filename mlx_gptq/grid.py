"""MLX affine quantization grid, implemented in torch.

MLX affine format (verified against mlx.core 0.31 on random tensors):
  - groups of ``group_size`` consecutive elements along the LAST axis (in_features)
  - dequant: w = q * scale + bias, with integer q in [0, 2**bits - 1]
  - scales/biases stored in the model dtype (bf16/fp16), shape [..., in/group]
  - q packed LSB-first into uint32 words (32/bits values per word)

mx.dequantize accepts arbitrary float scales/biases, so parameters chosen here
(by calibration) load as a completely standard MLX quantized model. Scales and
biases are rounded to the storage dtype *before* q is computed so that the
error feedback in GPTQ accounts for exactly what inference will see.
"""

from __future__ import annotations

import torch

SUPPORTED_BITS = (2, 4, 8)  # bit-packed MLX formats
SUPPORTED_MODES = ("affine", "mxfp4", "mxfp8")
MX_GROUP_SIZE = 32

_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def mode_defaults(mode: str) -> tuple[int, int]:
    if mode == "mxfp4":
        return MX_GROUP_SIZE, 4
    if mode == "mxfp8":
        return MX_GROUP_SIZE, 8
    if mode == "affine":
        return 64, 4
    raise ValueError(f"unsupported quantization mode: {mode}")


def _mxfp_max(mode: str) -> float:
    if mode == "mxfp4":
        return 6.0
    if mode == "mxfp8":
        return 448.0
    raise ValueError(f"not a microscaling mode: {mode}")


def choose_mxfp_scale(w: torch.Tensor, mode: str):
    """Choose MLX-compatible E8M0 scales for 32-value groups.

    MLX rounds the base-2 exponent of ``max(abs(w)) / format_max`` rather
    than always rounding upward. This deliberately permits endpoint clipping
    and must be reproduced exactly for native RTN parity.
    """
    max_value = _mxfp_max(mode)
    max_abs = w.abs().amax(dim=-1)
    safe = torch.where(max_abs == 0, torch.full_like(max_abs, max_value), max_abs)
    exponent = torch.round(torch.log2(safe / max_value)).to(torch.int32)
    exponent = exponent.clamp(-127, 127)
    exponent = torch.where(max_abs == 0, torch.zeros_like(exponent), exponent)
    scale_bytes = (exponent + 127).to(torch.uint8)
    scales = torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=w.device),
        exponent.to(torch.float32),
    )
    return scale_bytes, scales


def _e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    mag = torch.zeros_like(ax, dtype=torch.uint8)
    # Accumulate thresholds in uint8. Nested where creates large int64 temporaries.
    mag += (ax > 0.25).to(torch.uint8)
    mag += (ax >= 0.75).to(torch.uint8)
    mag += (ax > 1.25).to(torch.uint8)
    mag += (ax >= 1.75).to(torch.uint8)
    mag += (ax > 2.5).to(torch.uint8)
    mag += (ax >= 3.5).to(torch.uint8)
    mag += (ax > 5.0).to(torch.uint8)
    return mag | ((x < 0).to(torch.uint8) << 3)


def quantize_mxfp_with_scale(w: torch.Tensor, scales: torch.Tensor, mode: str):
    """Project values onto an MLX MXFP grid using fixed per-group scales."""
    normalized = w / scales.unsqueeze(-1)
    if mode == "mxfp4":
        q = _e2m1_codes(normalized)
        lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=w.device)
        dq = lut[q.to(torch.long)] * scales.unsqueeze(-1)
    elif mode == "mxfp8":
        # OCP MXFP8 E4M3 uses finite endpoints at +/-448. PyTorch's E4M3FN
        # conversion emits NaN above that range, so clamp before encoding.
        bounded = normalized.clamp(-448.0, 448.0).contiguous()
        encoded = bounded.to(torch.float8_e4m3fn).contiguous()
        q = encoded.view(torch.uint8)
        dq = encoded.float() * scales.unsqueeze(-1)
    else:
        raise ValueError(f"not a microscaling mode: {mode}")
    return q, dq


def dequantize_mxfp(q: torch.Tensor, scale_bytes: torch.Tensor, mode: str):
    exponent = scale_bytes.to(torch.float32) - 127.0
    scales = torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=q.device), exponent
    )
    if mode == "mxfp4":
        lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=q.device)
        values = lut[q.to(torch.long)]
    elif mode == "mxfp8":
        values = q.contiguous().view(torch.float8_e4m3fn).float()
    else:
        raise ValueError(f"not a microscaling mode: {mode}")
    return values.reshape(*q.shape[:-1], -1, MX_GROUP_SIZE) * scales.unsqueeze(-1)


@torch.no_grad()
def rtn_quantize_mxfp(W: torch.Tensor, mode: str):
    """Native MLX MXFP round-to-nearest reference in torch."""
    *lead, rows, columns = W.shape
    if columns % MX_GROUP_SIZE:
        raise ValueError(f"columns {columns} not divisible by {MX_GROUP_SIZE}")
    grouped = W.reshape(*lead, rows, columns // MX_GROUP_SIZE, MX_GROUP_SIZE).float()
    scale_bytes, scales = choose_mxfp_scale(grouped, mode)
    q, dq = quantize_mxfp_with_scale(grouped, scales, mode)
    return (
        q.reshape(*lead, rows, columns),
        scale_bytes,
        None,
        dq.reshape(*lead, rows, columns),
    )


def dequantize_artifact(
    q: torch.Tensor,
    scales: torch.Tensor,
    biases,
    mode: str,
    group_size: int,
):
    """Dequantize one Stage-A artifact for layer-to-layer propagation."""
    if mode == "affine":
        return (
            q.to(torch.float32).reshape(*q.shape[:-1], -1, group_size)
            * scales.float().unsqueeze(-1)
            + biases.float().unsqueeze(-1)
        ).reshape(q.shape)
    if group_size != MX_GROUP_SIZE:
        raise ValueError(f"{mode} requires group size {MX_GROUP_SIZE}")
    return dequantize_mxfp(q, scales, mode).reshape(q.shape)


def choose_qparams(
    w: torch.Tensor,
    bits: int,
    method: str = "mse",
    storage_dtype: torch.dtype = torch.bfloat16,
    grid_steps: int = 20,
    max_shrink: float = 0.2,
):
    """Choose per-group scale/bias for groups given as the last axis.

    Args:
        w: [..., group_size] float32 tensor of (error-compensated) weights.
        method: "minmax" or "mse" (range-shrink search, GPTQ style).

    Returns:
        (scales_stored, biases_stored, scales_f32, biases_f32) with shape [...].
        The f32 values are exact float32 copies of the stored (rounded) values.
    """
    maxq = float(2**bits - 1)
    wmin = w.amin(dim=-1)
    wmax = w.amax(dim=-1)

    if method == "minmax":
        shrinks = [1.0]
    else:
        shrinks = [1.0 - max_shrink * i / grid_steps for i in range(grid_steps + 1)]

    best_err = None
    best_s = best_b = None
    for p in shrinks:
        # Shrink the range toward zero (standard GPTQ clip search).
        s = ((wmax - wmin) * (p / maxq)).to(storage_dtype).float()
        b = (wmin * p).to(storage_dtype).float()
        s_safe = torch.where(s.abs() < 1e-12, torch.ones_like(s), s)
        q = ((w - b.unsqueeze(-1)) / s_safe.unsqueeze(-1)).round_().clamp_(0.0, maxq)
        dq = q * s.unsqueeze(-1) + b.unsqueeze(-1)
        err = (dq - w).pow(2).sum(dim=-1)
        if best_err is None:
            best_err, best_s, best_b = err, s, b
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_s = torch.where(better, s, best_s)
            best_b = torch.where(better, b, best_b)

    return (
        best_s.to(storage_dtype),
        best_b.to(storage_dtype),
        best_s.float(),
        best_b.float(),
    )


def quantize_with_params(w, s32, b32, bits):
    """Quantize [..., G] values with given f32 params. Returns (q float, dq float)."""
    maxq = float(2**bits - 1)
    s_safe = torch.where(s32.abs() < 1e-12, torch.ones_like(s32), s32)
    q = ((w - b32.unsqueeze(-1)) / s_safe.unsqueeze(-1)).round_().clamp_(0.0, maxq)
    dq = q * s32.unsqueeze(-1) + b32.unsqueeze(-1)
    return q, dq


@torch.no_grad()
def rtn_quantize(
    W: torch.Tensor,
    bits: int,
    group_size: int,
    method: str = "mse",
    storage_dtype: torch.dtype = torch.bfloat16,
):
    """Round-to-nearest on the MLX grid (no Hessian). W: [..., rows, cols] f32.

    Returns (q uint8 [..., rows, cols], scales, biases [..., rows, cols/G], W_dq).
    """
    *lead, R, C = W.shape
    assert C % group_size == 0, f"cols {C} not divisible by group size {group_size}"
    wg = W.reshape(*lead, R, C // group_size, group_size).float()
    s_st, b_st, s32, b32 = choose_qparams(wg, bits, method, storage_dtype)
    q, dq = quantize_with_params(wg, s32, b32, bits)
    return (
        q.to(torch.uint8).reshape(*lead, R, C),
        s_st,
        b_st,
        dq.reshape(*lead, R, C),
    )
