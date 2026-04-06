"""
TMF (TurboMemory Format) Specification v1.0

This document defines the deterministic encoding specification for TurboMemory's
vector storage format. The goal is to make TurboQuant the "zstd for embeddings"
- a universal, portable compression format that any vector database can use.

== Format Overview ==

A TMF storage is a directory containing:
- .tmindex: SQLite metadata index
- .tmvec: Packed vectors file
- .tmlog: Append-only event log
- .tmmeta.json: Schema + model metadata
- .tmbundle: Optional single-file bundle

== Version History ==
- v1.0: Initial specification (this document)
- Compatible with TurboMemory v0.5+

== Data Types ==
- uint8, uint16, uint32, uint64: Little-endian unsigned integers
- float32: IEEE 754 single-precision floating point
- bytes: Length-prefixed byte arrays

== Vector Encoding (TurboQuant) ===

Quantization: Per-dimension scalar quant with 4/6/8-bit packing
- dimension * bits / 8 + 6 bytes per vector (scale + qmax)
- Deterministic: same input always produces identical output

Packing algorithm:
1. For each dimension d: scale[d] = max(|vec[d]|) / qmax
2. q[d] = round(vec[d] / scale[d]), clip to [-qmax, qmax]
3. Pack all q values as unsigned into bitstream

Decode for cosine similarity:
1. Unpack to quantized values
2. reconstructed[d] = (q[d] - qmax) * scale[d]
3. cosine = dot(query, reconstructed) / (||query|| * ||reconstructed||)

== Checksum ===
- SHA256 of all data files (stored in .tmmeta.json)
- Verified on load for integrity

== Migration Logic ===

Each .tmmeta.json includes version field. When reading:
1. If version > supported, reject with clear error
2. If version < current, apply migration transforms

== Portability Requirements ==
- Little-endian only
- No alignment requirements
- No timezone issues (unix epochs)
- Deterministic output
- Self-describing (all metadata in header)

== File Extensions ==
- .tmindex - SQLite index
- .tmvec - Packed vectors  
- .tmlog - Transaction log
- .tmmeta - JSON metadata
- .tmb - Single-file bundle (tar-like)

== References ==
- TurboMemory: https://github.com/Kubenew/TurboMemory
"""

import struct
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import json
import hashlib


TMF_MAGIC = 0x544D4601  # "TMF\x01"
CURRENT_VERSION = "1.0.0"


@dataclass
class TMFHeader:
    """Header for TMF bundle format (64 bytes)."""
    magic: int = TMF_MAGIC
    version: str = CURRENT_VERSION
    flags: int = 0
    dimension: int = 384
    quantization_bits: int = 6
    chunk_count: int = 0
    topic_count: int = 0
    reserved: bytes = b'\x00' * 48

    def to_bytes(self) -> bytes:
        return struct.pack(
            '<4sHHIII48x',
            b'TMF\x01',
            1, 0,  # version major, minor
            self.flags,
            self.dimension,
            self.quantization_bits,
            self.chunk_count,
            self.topic_count
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'TMFHeader':
        if len(data) < 56:
            raise ValueError("Header must be at least 56 bytes")
        values = struct.unpack('<4sHHIII56x', data[:56])
        return cls(
            magic=int.from_bytes(values[0], 'little'),
            version=f"{values[1]}.{values[2]}",
            flags=values[3],
            dimension=values[4],
            quantization_bits=values[5],
            chunk_count=values[6],
            topic_count=values[7],
        )

    def validate(self) -> bool:
        if self.magic != TMF_MAGIC:
            return False
        if self.quantization_bits not in (4, 6, 8):
            return False
        return True


@dataclass
class TMFMetadata:
    """JSON metadata stored in TMF storage."""
    version: str = CURRENT_VERSION
    created: str = ""
    updated: str = ""
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    quantization_bits: int = 6
    chunk_count: int = 0
    topic_count: int = 0
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TMFMetadata':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def compute_checksum(files: List[str]) -> str:
    """Compute SHA256 checksum of all files."""
    hasher = hashlib.sha256()
    for fpath in sorted(files):
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksum(files: List[str], expected: str) -> bool:
    """Verify data integrity using checksum."""
    return compute_checksum(files) == expected


import os