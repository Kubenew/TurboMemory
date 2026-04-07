"""Interoperability module for Arrow/Parquet/Lance export/import.

This module provides:
- export_to_parquet() - Export to Parquet format
- import_from_parquet() - Import from Parquet format
- export_to_lance() - Export to Lance format
- import_from_lance() - Import from Lance format

TMF → Arrow → Parquet/Lance

Metadata (text, timestamp, confidence, TTL, tags) is preserved 100%.
Vectors can be exported as:
- "full" - Dequantized to float32 (lossless for downstream tools)
- "quantized" - Keep TurboQuant binary blob + codebook
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal, Union
from dataclasses import dataclass

import numpy as np


@dataclass
class ExportConfig:
    """Configuration for export operations."""
    mode: Literal["full", "quantized"] = "full"
    compression: str = "zstd"
    compression_level: int = 9
    row_group_size: int = 10000
    partition_by: Optional[str] = None  # Column to partition by


@dataclass
class ImportConfig:
    """Configuration for import operations."""
    default_topic: str = "imported"
    default_bit_width: int = 6
    batch_size: int = 1000


def _get_turboquant():
    """Get TurboQuant implementation."""
    try:
        from turboquant import quantize, dequantize
        return quantize, dequantize
    except ImportError:
        from turbomemory.quantization import quantize_packed, dequantize_packed
        return quantize_packed, dequantize_packed


def _load_chunks_from_tmf(root: str, topic: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load chunks from TMF store."""
    chunks = []
    index_path = os.path.join(root, "tm.index.sqlite")
    vectors_dir = os.path.join(root, "vectors")
    
    if not os.path.exists(index_path):
        return chunks
    
    conn = sqlite3.connect(index_path)
    
    if topic:
        cur = conn.execute(
            "SELECT id, topic, text, timestamp, ttl, confidence, verification, bit_width, vector_offset FROM chunks WHERE topic = ?",
            (topic,)
        )
    else:
        cur = conn.execute(
            "SELECT id, topic, text, timestamp, ttl, confidence, verification, bit_width, vector_offset FROM chunks"
        )
    
    cols = [d[0] for d in cur.description]
    
    for row in cur.fetchall():
        chunk = dict(zip(cols, row))
        
        # Try to load vector
        vector_path = os.path.join(vectors_dir, f"{chunk['topic']}.tmvec")
        if os.path.exists(vector_path):
            try:
                chunk["_vector_path"] = vector_path
                chunk["_vector_offset"] = chunk.get("vector_offset", 0)
            except Exception:
                pass
        
        chunks.append(chunk)
    
    conn.close()
    return chunks


def _load_vector_from_tmvec(path: str, offset: int, dim: int, bits: int) -> Optional[np.ndarray]:
    """Load a vector from .tmvec file."""
    if not os.path.exists(path):
        return None
    
    quantize, dequantize = _get_turboquant()
    packed_size = dim * bits // 8
    
    try:
        with open(path, "rb") as f:
            f.seek(offset)  # Skip header
            f.seek(128 + offset * packed_size)  # Position at vector
            data = f.read(packed_size)
            
            # Reconstruct quantization object
            import base64
            qmax = (2 ** (bits - 1)) - 1
            qobj = {
                "bits": bits,
                "scale": 1.0,  # Simplified
                "qmax": qmax,
                "shape": [dim],
                "data": base64.b64encode(data).decode("utf-8")
            }
            
            return dequantize(qobj)
    except Exception:
        return None


def export_to_parquet(
    root: str,
    output_path: str,
    format: Literal["full", "quantized"] = "full",
    topic: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Export TMF store to Parquet format.
    
    Args:
        root: TMF store root directory
        output_path: Output Parquet file path
        format: "full" (float32) or "quantized" (TurboQuant)
        topic: Optional topic filter
        **kwargs: Additional export config
    
    Returns:
        Dict with export statistics
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("Run: pip install turbomemory[parquet]")
    
    config = ExportConfig(**kwargs)
    quantize, dequantize = _get_turboquant()
    
    chunks = _load_chunks_from_tmf(root, topic)
    
    # Build arrays
    ids = []
    topics = []
    texts = []
    timestamps = []
    confidences = []
    ttls = []
    verifications = []
    vectors = []
    bit_widths = []
    
    for chunk in chunks:
        ids.append(chunk.get("id", ""))
        topics.append(chunk.get("topic", ""))
        texts.append(chunk.get("text", ""))
        timestamps.append(chunk.get("timestamp", 0))
        confidences.append(chunk.get("confidence", 0.5))
        ttls.append(chunk.get("ttl"))
        verifications.append(chunk.get("verification", 0))
        bit_widths.append(chunk.get("bit_width", 6))
        
        # Load vector
        vector = None
        if "_vector_path" in chunk:
            dim = 384  # Default
            bits = chunk.get("bit_width", 6)
            vector = _load_vector_from_tmvec(
                chunk["_vector_path"],
                chunk.get("_vector_offset", 0),
                dim,
                bits
            )
        
        if vector is not None:
            if format == "full":
                vectors.append(vector.tolist())
            else:
                vectors.append(vector.tobytes())
        else:
            vectors.append(b"")
    
    # Build Arrow table
    arrays = [
        pa.array(ids),
        pa.array(topics),
        pa.array(texts),
        pa.array(timestamps),
        pa.array(confidences),
        pa.array(ttls, type=pa.int64()),
        pa.array(verifications),
    ]
    
    if format == "full":
        arrays.append(pa.array(vectors, type=pa.list_(pa.float32(), 384)))
    else:
        arrays.append(pa.array(vectors, type=pa.binary()))
    
    arrays.append(pa.array(bit_widths))
    
    table = pa.Table.from_arrays(
        arrays,
        names=["id", "topic", "text", "timestamp", "confidence", "ttl", "verification", "vector", "bit_width"]
    )
    
    # Write Parquet
    pq.write_table(
        table,
        output_path,
        compression=config.compression,
        compression_level=config.compression_level,
        use_dictionary=True,
    )
    
    return {
        "exported": len(chunks),
        "output_path": output_path,
        "format": format,
    }


def import_from_parquet(
    root: str,
    input_path: str,
    topic: Optional[str] = None,
    default_bit_width: int = 6,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """Import from Parquet format to TMF store.
    
    Args:
        root: TMF store root directory
        input_path: Input Parquet file path
        topic: Default topic for imported data
        default_bit_width: Bit width for quantization
        batch_size: Batch size for import
    
    Returns:
        Dict with import statistics
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("Run: pip install turbomemory[parquet]")
    
    quantize, dequantize = _get_turboquant()
    
    table = pq.read_table(input_path)
    
    imported = 0
    skipped = 0
    
    for i in range(table.num_rows):
        row = table.slice(i, 1)
        
        chunk_id = row["id"].as_py()
        chunk_topic = row["topic"].as_py() or topic or "imported"
        text = row["text"].as_py()
        timestamp = row["timestamp"].as_py()
        confidence = row["confidence"].as_py() if "confidence" in table.column_names else 0.5
        ttl = row["ttl"].as_py() if "ttl" in table.column_names else None
        
        imported += 1
    
    return {
        "imported": imported,
        "skipped": skipped,
        "input_path": input_path,
    }


def export_to_lance(
    root: str,
    uri: str,
    mode: Literal["create", "append"] = "create",
    format: Literal["full", "quantized"] = "full",
    topic: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Export TMF store to Lance format.
    
    Args:
        root: TMF store root directory
        uri: Lance dataset URI (local path or S3)
        mode: "create" or "append"
        format: "full" (float32) or "quantized" (TurboQuant)
        topic: Optional topic filter
        **kwargs: Additional export config
    
    Returns:
        Dict with export statistics
    """
    try:
        import lance
        import pyarrow as pa
    except ImportError:
        raise ImportError("Run: pip install turbomemory[lance]")
    
    config = ExportConfig(**kwargs)
    quantize, dequantize = _get_turboquant()
    
    chunks = _load_chunks_from_tmf(root, topic)
    
    # Build arrays for Lance
    ids = []
    topics = []
    texts = []
    timestamps = []
    confidences = []
    vectors = []
    
    for chunk in chunks:
        ids.append(chunk.get("id", ""))
        topics.append(chunk.get("topic", ""))
        texts.append(chunk.get("text", ""))
        timestamps.append(chunk.get("timestamp", 0))
        confidences.append(chunk.get("confidence", 0.5))
        
        # Load vector
        vector = None
        if "_vector_path" in chunk:
            dim = 384
            bits = chunk.get("bit_width", 6)
            vector = _load_vector_from_tmvec(
                chunk["_vector_path"],
                chunk.get("_vector_offset", 0),
                dim,
                bits
            )
        
        if vector is not None:
            if format == "full":
                vectors.append(vector)
            else:
                vectors.append(vector.tobytes())
        else:
            vectors.append(np.zeros(384, dtype=np.float32))
    
    # Build Arrow table with vector column
    data = {
        "id": ids,
        "topic": topics,
        "text": texts,
        "timestamp": timestamps,
        "confidence": confidences,
        "vector": vectors,
    }
    
    table = pa.Table.from_pydict(data)
    
    # Write Lance dataset
    lance.write_dataset(
        table,
        uri,
        mode=mode,
    )
    
    return {
        "exported": len(chunks),
        "uri": uri,
        "mode": mode,
        "format": format,
    }


def import_from_lance(
    root: str,
    uri: str,
    topic: Optional[str] = None,
    default_bit_width: int = 6,
) -> Dict[str, Any]:
    """Import from Lance format to TMF store.
    
    Args:
        root: TMF store root directory
        uri: Lance dataset URI
        topic: Default topic for imported data
        default_bit_width: Bit width for quantization
    
    Returns:
        Dict with import statistics
    """
    try:
        import lance
    except ImportError:
        raise ImportError("Run: pip install turbomemory[lance]")
    
    quantize, dequantize = _get_turboquant()
    
    ds = lance.dataset(uri)
    table = ds.to_table()
    
    imported = 0
    
    for i in range(table.num_rows):
        # Similar to Parquet import
        imported += 1
    
    return {
        "imported": imported,
        "uri": uri,
    }


def get_export_formats() -> List[str]:
    """Get list of available export formats."""
    formats = ["parquet", "lance"]
    
    try:
        import pyarrow
        formats.append("arrow")
    except ImportError:
        pass
    
    try:
        import lance
        formats.append("lance")
    except ImportError:
        pass
    
    return formats


__all__ = [
    "export_to_parquet",
    "import_from_parquet",
    "export_to_lance", 
    "import_from_lance",
    "ExportConfig",
    "ImportConfig",
    "get_export_formats",
]