"""Quantization module for packed embedding storage."""

import base64
from typing import Dict, Any, Tuple, Optional
import numpy as np


def pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    """Packs uint values into bitstream."""
    values = values.astype(np.uint32).ravel()
    out = bytearray()
    buf = 0
    buf_bits = 0
    mask = (1 << bits) - 1

    for v in values:
        buf |= (int(v) & mask) << buf_bits
        buf_bits += bits
        while buf_bits >= 8:
            out.append(buf & 0xFF)
            buf >>= 8
            buf_bits -= 8

    if buf_bits > 0:
        out.append(buf & 0xFF)

    return bytes(out)


def unpack_unsigned(data: bytes, bits: int, n_values: int) -> np.ndarray:
    """Unpacks bitstream into uint array."""
    out = np.zeros(n_values, dtype=np.uint16)
    buf = 0
    buf_bits = 0
    idx = 0
    mask = (1 << bits) - 1

    for b in data:
        buf |= int(b) << buf_bits
        buf_bits += 8
        while buf_bits >= bits and idx < n_values:
            out[idx] = buf & mask
            buf >>= bits
            buf_bits -= bits
            idx += 1
        if idx >= n_values:
            break

    return out


def quantize_packed(vec: np.ndarray, bits: int = 6) -> Dict[str, Any]:
    """Symmetric uniform quantization with packed storage."""
    if bits not in (4, 6, 8):
        raise ValueError("bits must be 4, 6, or 8")

    vec = vec.astype(np.float32)
    qmax = (2 ** (bits - 1)) - 1

    vmax = float(np.max(np.abs(vec)) + 1e-9)
    scale = vmax / qmax

    q = np.round(vec / scale).astype(np.int16)
    q = np.clip(q, -qmax, qmax).astype(np.int16)

    uq = (q + qmax).astype(np.uint16)
    packed = pack_unsigned(uq, bits)

    return {
        "bits": bits,
        "scale": float(scale),
        "qmax": int(qmax),
        "shape": list(vec.shape),
        "data": base64.b64encode(packed).decode("utf-8")
    }


def dequantize_packed(qobj: Dict[str, Any]) -> np.ndarray:
    """Dequantize a packed quantization object back to float32 vector."""
    bits = int(qobj["bits"])
    scale = float(qobj["scale"])
    qmax = int(qobj["qmax"])
    shape = tuple(qobj["shape"])

    raw = base64.b64decode(qobj["data"])
    uq = unpack_unsigned(raw, bits, int(np.prod(shape))).astype(np.int16)

    q = (uq - qmax).astype(np.float32)
    return (q * scale).reshape(shape)


class Quantizer:
    """Quantizer for embedding storage with multiple bit depths."""

    SUPPORTED_BITS = (4, 6, 8)

    def __init__(self, default_bits: int = 6) -> None:
        if default_bits not in self.SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {self.SUPPORTED_BITS}")
        self.default_bits = default_bits

    def quantize(self, vec: np.ndarray, bits: Optional[int] = None) -> Dict[str, Any]:
        """Quantize a vector with optional custom bit depth."""
        return quantize_packed(vec, bits or self.default_bits)

    def dequantize(self, qobj: Dict[str, Any]) -> np.ndarray:
        """Dequantize a quantized vector."""
        return dequantize_packed(qobj)

    def quantize_batch(self, vectors: np.ndarray, bits: Optional[int] = None) -> List[Dict[str, Any]]:
        """Quantize multiple vectors at once."""
        return [self.quantize(v, bits) for v in vectors]

    def dequantize_batch(self, qobjs: List[Dict[str, Any]]) -> np.ndarray:
        """Dequantize multiple vectors at once."""
        return np.array([self.dequantize(q) for q in qobjs])

    @staticmethod
    def get_compression_ratio(bits: int, original_dtype: np.dtype = np.float32) -> float:
        """Calculate compression ratio for given bit depth."""
        bits_per_float = original_dtype().itemsize * 8
        return bits_per_float / bits