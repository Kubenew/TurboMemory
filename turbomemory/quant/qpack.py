"""TurboMemory v3.1 Quantization - q4/q6/q8 packing."""

import numpy as np


def _scale_to_uint(vec: np.ndarray, bits: int):
    """Map [-1,1] float to unsigned int range [0..2^bits-1]."""
    levels = (1 << bits) - 1
    v = np.clip(vec, -1.0, 1.0)
    u = ((v + 1.0) * 0.5 * levels).round().astype(np.uint8)
    return u


def _uint_to_scale(u: np.ndarray, bits: int):
    """Map unsigned int back to [-1,1] float."""
    levels = (1 << bits) - 1
    v = (u.astype(np.float32) / levels) * 2.0 - 1.0
    return v


def pack_bits(u: np.ndarray, bits: int) -> bytes:
    """Pack uint values into bitstream."""
    assert bits in (4, 6, 8)
    if bits == 8:
        return u.tobytes()

    total_bits = u.size * bits
    out_bytes = (total_bits + 7) // 8
    out = np.zeros(out_bytes, dtype=np.uint8)

    bitpos = 0
    for val in u:
        byte_idx = bitpos // 8
        shift = bitpos % 8
        out[byte_idx] |= (val << shift) & 0xFF

        overflow = (shift + bits) - 8
        if overflow > 0:
            out[byte_idx + 1] |= (val >> (bits - overflow)) & 0xFF

        bitpos += bits

    return out.tobytes()


def unpack_bits(blob: bytes, count: int, bits: int) -> np.ndarray:
    """Unpack bitstream to uint array."""
    assert bits in (4, 6, 8)
    if bits == 8:
        return np.frombuffer(blob, dtype=np.uint8, count=count)

    data = np.frombuffer(blob, dtype=np.uint8)
    out = np.zeros(count, dtype=np.uint8)

    bitpos = 0
    mask = (1 << bits) - 1

    for i in range(count):
        byte_idx = bitpos // 8
        shift = bitpos % 8

        val = (data[byte_idx] >> shift) & 0xFF

        overflow = (shift + bits) - 8
        if overflow > 0:
            val |= (data[byte_idx + 1] << (8 - shift)) & 0xFF

        out[i] = val & mask
        bitpos += bits

    return out


def pack_q(vec: np.ndarray, bits: int) -> bytes:
    """Pack vector to specified bit depth."""
    u = _scale_to_uint(vec, bits)
    return pack_bits(u, bits)


def unpack_q(blob: bytes, dim: int, bits: int) -> np.ndarray:
    """Unpack vector from specified bit depth."""
    u = unpack_bits(blob, dim, bits)
    return _uint_to_scale(u, bits)


def pack_q4(vec: np.ndarray) -> bytes:
    """Pack to 4-bit."""
    return pack_q(vec, 4)


def pack_q6(vec: np.ndarray) -> bytes:
    """Pack to 6-bit."""
    return pack_q(vec, 6)


def pack_q8(vec: np.ndarray) -> bytes:
    """Pack to 8-bit."""
    return pack_q(vec, 8)


def unpack_q4(blob: bytes, dim: int) -> np.ndarray:
    """Unpack from 4-bit."""
    return unpack_q(blob, dim, 4)


def unpack_q6(blob: bytes, dim: int) -> np.ndarray:
    """Unpack from 6-bit."""
    return unpack_q(blob, dim, 6)


def unpack_q8(blob: bytes, dim: int) -> np.ndarray:
    """Unpack from 8-bit."""
    return unpack_q(blob, dim, 8)