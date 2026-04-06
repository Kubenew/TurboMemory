"""Tests for the quantization module."""

import pytest
import numpy as np

from turbomemory.quantization import (
    quantize_packed,
    dequantize_packed,
    pack_unsigned,
    unpack_unsigned,
    Quantizer,
)


class TestQuantizePacked:
    def test_quantize_4bit_roundtrip(self):
        vec = np.random.randn(384).astype(np.float32)
        q = quantize_packed(vec, bits=4)
        reconstructed = dequantize_packed(q)
        
        assert reconstructed.shape == vec.shape
        similarity = np.dot(vec, reconstructed) / (np.linalg.norm(vec) * np.linalg.norm(reconstructed) + 1e-9)
        assert similarity > 0.90

    def test_quantize_6bit_roundtrip(self):
        vec = np.random.randn(384).astype(np.float32)
        q = quantize_packed(vec, bits=6)
        reconstructed = dequantize_packed(q)
        
        assert reconstructed.shape == vec.shape
        similarity = np.dot(vec, reconstructed) / (np.linalg.norm(vec) * np.linalg.norm(reconstructed) + 1e-9)
        assert similarity > 0.95

    def test_quantize_8bit_roundtrip(self):
        vec = np.random.randn(384).astype(np.float32)
        q = quantize_packed(vec, bits=8)
        reconstructed = dequantize_packed(q)
        
        assert reconstructed.shape == vec.shape
        similarity = np.dot(vec, reconstructed) / (np.linalg.norm(vec) * np.linalg.norm(reconstructed) + 1e-9)
        assert similarity > 0.99

    def test_invalid_bits(self):
        vec = np.random.randn(10)
        with pytest.raises(ValueError):
            quantize_packed(vec, bits=5)
        with pytest.raises(ValueError):
            quantize_packed(vec, bits=2)

    def test_zero_vector(self):
        vec = np.zeros(384)
        q = quantize_packed(vec, bits=6)
        reconstructed = dequantize_packed(q)
        np.testing.assert_array_almost_equal(vec, reconstructed)


class TestPackUnsigned:
    def test_4bit_roundtrip(self):
        values = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint16)
        packed = pack_unsigned(values, bits=4)
        unpacked = unpack_unsigned(packed, bits=4, n_values=8)
        np.testing.assert_array_equal(values, unpacked)

    def test_6bit_roundtrip(self):
        values = np.array([0, 10, 20, 30, 40, 50, 60, 63], dtype=np.uint16)
        packed = pack_unsigned(values, bits=6)
        unpacked = unpack_unsigned(packed, bits=6, n_values=8)
        np.testing.assert_array_equal(values, unpacked)

    def test_8bit_roundtrip(self):
        values = np.array([0, 50, 100, 150, 200, 220, 250, 255], dtype=np.uint16)
        packed = pack_unsigned(values, bits=8)
        unpacked = unpack_unsigned(packed, bits=8, n_values=8)
        np.testing.assert_array_equal(values, unpacked)

    def test_compression_ratio(self):
        import math
        
        vec = np.random.randn(384).astype(np.float32)
        
        for bits in [4, 6, 8]:
            q = quantize_packed(vec, bits=bits)
            import base64
            packed_size = len(base64.b64decode(q["data"]))
            original_size = vec.nbytes
            ratio = original_size / packed_size
            assert ratio > 1.0


class TestQuantizer:
    def test_default_bits(self):
        q = Quantizer(default_bits=6)
        vec = np.random.randn(10)
        result = q.quantize(vec)
        assert result["bits"] == 6

    def test_custom_bits(self):
        q = Quantizer(default_bits=8)
        vec = np.random.randn(10)
        result = q.quantize(vec, bits=4)
        assert result["bits"] == 4

    def test_batch_quantize(self):
        q = Quantizer(default_bits=6)
        vectors = [np.random.randn(10) for _ in range(5)]
        results = q.quantize_batch(vectors)
        assert len(results) == 5

    def test_compression_ratio_calculation(self):
        ratio = Quantizer.get_compression_ratio(4, np.float32)
        assert ratio == 8.0
        
        ratio = Quantizer.get_compression_ratio(8, np.float32)
        assert ratio == 4.0