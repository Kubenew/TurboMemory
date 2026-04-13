"""Test quantization pack/unpack roundtrip."""

import numpy as np
from turbomemory.quant.qpack import pack_q4, pack_q6, pack_q8, unpack_q4, unpack_q6, unpack_q8


def test_qpack_roundtrip():
    """Test q4/q6/q8 pack and unpack produce reasonable vectors."""
    np.random.seed(42)
    
    # Test multiple dimensions
    for dim in [128, 384, 768]:
        vec = np.random.uniform(-1, 1, dim).astype(np.float32)
        
        # Test each bit depth
        for pack_fn, unpack_fn, bits in [
            (pack_q4, unpack_q4, 4),
            (pack_q6, unpack_q6, 6),
            (pack_q8, unpack_q8, 8),
        ]:
            # Pack
            blob = pack_fn(vec)
            
            # Unpack
            v2 = unpack_fn(blob, dim)
            
            # Check shape
            assert v2.shape == vec.shape, f"Shape mismatch: {v2.shape} vs {vec.shape}"
            
            # Check range is roughly [-1, 1]
            assert v2.min() >= -1.01, f"Below -1: {v2.min()}"
            assert v2.max() <= 1.01, f"Above 1: {v2.max()}"
            
            # Check similarity (high for higher bits)
            sim = np.dot(vec, v2) / (np.linalg.norm(vec) * np.linalg.norm(v2) + 1e-9)
            
            print(f"dim={dim}, bits={bits}: similarity={sim:.4f}")
            
            # Higher bits should have higher similarity
            if bits == 8:
                assert sim > 0.99, f"q8 similarity too low: {sim}"
            elif bits == 6:
                assert sim > 0.95, f"q6 similarity too low: {sim}"
            elif bits == 4:
                assert sim > 0.85, f"q4 similarity too low: {sim}"
    
    print("✓ All quantization tests passed!")


if __name__ == "__main__":
    test_qpack_roundtrip()