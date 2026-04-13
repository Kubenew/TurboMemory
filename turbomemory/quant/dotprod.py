"""TurboMemory v3.1 dot product on packed vectors."""

import numpy as np
from .qpack import unpack_q


BITS_MAP = {"q4": 4, "q6": 6, "q8": 8}


def dot_packed(query_vec: np.ndarray, blob: bytes, dim: int, dtype: str) -> float:
    """
    Compute dot product between query vector and packed embedding.
    
    Args:
        query_vec: Query embedding (float32)
        blob: Packed embedding bytes
        dim: Embedding dimension
        dtype: Quantization type (q4, q6, q8)
    
    Returns:
        Dot product score
    """
    bits = BITS_MAP.get(dtype)
    if bits is None:
        # Fallback to float32
        emb = np.frombuffer(blob, dtype=np.float32)
        return float(np.dot(query_vec, emb))
    
    emb = unpack_q(blob, dim, bits)
    return float(np.dot(query_vec, emb))


def cosine_packed(query_vec: np.ndarray, blob: bytes, dim: int, dtype: str) -> float:
    """Compute cosine similarity on packed vectors."""
    q_norm = np.linalg.norm(query_vec)
    if q_norm < 1e-9:
        return 0.0
    
    bits = BITS_MAP.get(dtype)
    if bits is None:
        emb = np.frombuffer(blob, dtype=np.float32)
    else:
        emb = unpack_q(blob, dim, bits)
    
    emb_norm = np.linalg.norm(emb)
    if emb_norm < 1e-9:
        return 0.0
    
    return float(np.dot(query_vec, emb) / (q_norm * emb_norm))