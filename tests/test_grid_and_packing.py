"""Bit-exactness of our torch grid + numpy packer against mlx.core."""

import numpy as np
import pytest
import torch
from mlx_gptq.artifacts import pack_nibbles, unpack_nibbles
from mlx_gptq.grid import _e2m1_codes, rtn_quantize, rtn_quantize_mxfp
from mlx_gptq.packing import pack_q, unpack_q


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64])
def test_pack_matches_mx_dequantize(bits, group_size):
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(0)
    R, C = 8, 256
    q = rng.integers(0, 2**bits, (R, C)).astype(np.uint8)
    s = (rng.random((R, C // group_size)).astype(np.float32) + 0.25)
    b = rng.standard_normal((R, C // group_size)).astype(np.float32)

    words = pack_q(q, bits)
    ref = np.array(
        mx.dequantize(mx.array(words), mx.array(s), mx.array(b),
                      group_size=group_size, bits=bits)
    )
    manual = q * np.repeat(s, group_size, 1) + np.repeat(b, group_size, 1)
    np.testing.assert_allclose(ref, manual, rtol=1e-6, atol=1e-4)
    np.testing.assert_array_equal(unpack_q(words, bits, C), q)


@pytest.mark.parametrize("bits", [4, 8])
def test_rtn_grid_roundtrips_through_mlx(bits):
    """Weights RTN'd by our torch grid must reload exactly via mx.dequantize."""
    mx = pytest.importorskip("mlx.core")
    torch.manual_seed(0)
    W = torch.randn(16, 128)
    q, s, b, W_dq = rtn_quantize(W.float(), bits, 64, method="mse",
                                 storage_dtype=torch.float32)
    words = pack_q(q.numpy(), bits)
    ref = np.array(
        mx.dequantize(mx.array(words), mx.array(s.numpy()), mx.array(b.numpy()),
                      group_size=64, bits=bits)
    )
    np.testing.assert_allclose(ref, W_dq.numpy(), rtol=0, atol=1e-5)


def test_rtn_reduces_error_vs_minmax():
    torch.manual_seed(1)
    W = torch.randn(64, 256) * torch.rand(64, 1) * 3
    W[0, 0] = 12.0  # outlier
    _, _, _, dq_mse = rtn_quantize(W.float(), 4, 64, method="mse")
    _, _, _, dq_mm = rtn_quantize(W.float(), 4, 64, method="minmax")
    assert (dq_mse - W).norm() <= (dq_mm - W).norm() * 1.001


def _reference_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Return the E2M1 codes with the nested threshold form."""
    ax = x.abs()
    mag = torch.where(
        ax > 5.0,
        7,
        torch.where(
            ax >= 3.5,
            6,
            torch.where(
                ax > 2.5,
                5,
                torch.where(
                    ax >= 1.75,
                    4,
                    torch.where(
                        ax > 1.25,
                        3,
                        torch.where(ax >= 0.75, 2, torch.where(ax > 0.25, 1, 0)),
                    ),
                ),
            ),
        ),
    ).to(torch.uint8)
    return mag | ((x < 0).to(torch.uint8) << 3)


def test_e2m1_codes_cover_both_rounding_boundaries():
    magnitudes = (0.0, 0.25, 0.26, 0.74, 0.75, 1.25, 1.26, 1.74, 1.75)
    magnitudes += (2.50, 2.51, 3.49, 3.50, 4.99, 5.01, 6.0)
    codes = _e2m1_codes(torch.tensor(magnitudes, dtype=torch.float32))
    expected = (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7)
    assert torch.equal(codes, torch.tensor(expected, dtype=torch.uint8))


def test_e2m1_codes_match_the_nested_threshold_form():
    torch.manual_seed(31)
    values = torch.randn(4096) * 3
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
    )
    samples = torch.cat([values, boundaries, -boundaries])
    assert torch.equal(_e2m1_codes(samples), _reference_e2m1_codes(samples))


def test_nibble_roundtrip():
    q = torch.randint(0, 16, (5, 64), dtype=torch.uint8)
    assert torch.equal(unpack_nibbles(pack_nibbles(q)), q)


def test_harness_e2m1_layout_matches_mlx_packer():
    from tools.mxfp4_kernel.e2m1_packing import (
        pack_e2m1,
        pack_e2m1_bytes,
        unpack_e2m1,
    )

    q = torch.randint(0, 16, (5, 64), dtype=torch.uint8)
    words = pack_e2m1(q)
    np.testing.assert_array_equal(words.numpy(), pack_q(q.numpy(), 4))
    assert torch.equal(unpack_e2m1(words, q.shape[1]), q)
    assert torch.equal(pack_e2m1_bytes(q), pack_nibbles(q))


def test_stacked_3d_packing():
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(2)
    q = rng.integers(0, 16, (3, 8, 64)).astype(np.uint8)  # [E, out, in]
    s = rng.random((3, 8, 1)).astype(np.float32) + 0.5
    b = rng.standard_normal((3, 8, 1)).astype(np.float32)
    words = pack_q(q, 4)
    ref = np.array(mx.dequantize(mx.array(words), mx.array(s), mx.array(b),
                                 group_size=64, bits=4))
    manual = q * np.repeat(s, 64, -1) + np.repeat(b, 64, -1)
    np.testing.assert_allclose(ref, manual, rtol=0, atol=1e-5)


@pytest.mark.parametrize("mode,bits", [("mxfp4", 4), ("mxfp8", 8)])
def test_mxfp_rtn_matches_mlx(mode, bits):
    """Torch emits the exact native MLX MXFP codes and E8M0 scales."""
    mx = pytest.importorskip("mlx.core")
    torch.manual_seed(23)
    weight = torch.randn(17, 96) * torch.linspace(0.01, 8.0, 17).unsqueeze(1)
    q, scales, biases, dq = rtn_quantize_mxfp(weight, mode)
    words = pack_q(q.numpy(), bits)
    native = mx.quantize(
        mx.array(weight.numpy()), group_size=32, bits=bits, mode=mode
    )
    np.testing.assert_array_equal(words, np.array(native[0]))
    np.testing.assert_array_equal(scales.numpy(), np.array(native[1]))
    assert biases is None
    np.testing.assert_array_equal(
        dq.numpy(),
        np.array(
            mx.dequantize(*native, group_size=32, bits=bits, mode=mode).astype(
                mx.float32
            )
        ),
    )
