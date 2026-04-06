"""TMF vector storage with packed embeddings."""

import struct
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from turbomemory import quantize_packed, dequantize_packed


VECTOR_HEADER_FORMAT = "4sHHI"  # magic(4), dim(2), bits(2), count(4)
VECTOR_HEADER_SIZE = struct.calcsize(VECTOR_HEADER_FORMAT)
VECTOR_MAGIC = b"TVEC"


@dataclass
class VectorHeader:
    """Header for vector storage file."""
    magic: bytes = VECTOR_MAGIC
    dimension: int = 384
    bits: int = 6
    count: int = 0


class TMFVectorStore:
    """Packed vector storage for TMF."""
    
    def __init__(self, vector_path: str, dimension: int = 384, bits: int = 6):
        self.vector_path = Path(vector_path)
        self.dimension = dimension
        self.bits = bits
        
        # Track offsets for each embedding
        self._offset_map: Dict[Tuple[str, str], int] = {}  # (topic, chunk_key) -> offset
        self._reverse_map: Dict[int, Tuple[str, str]] = {}  # offset -> (topic, chunk_key)
        self._load_index()
    
    def _load_index(self) -> None:
        """Load offset index from file."""
        index_path = self.vector_path.with_suffix(".idx")
        
        if index_path.exists():
            import json
            with open(index_path, "r") as f:
                data = json.load(f)
                self._offset_map = {(tuple(k[0]), k[1]): v for k, v in data["offsets"].items()}
                self._reverse_map = {v: tuple(k[0]) for k, v in data["offsets"].items()}
                self.dimension = data.get("dimension", self.dimension)
                self.bits = data.get("bits", self.bits)
    
    def _save_index(self) -> None:
        """Save offset index to file."""
        index_path = self.vector_path.with_suffix(".idx")
        
        import json
        data = {
            "dimension": self.dimension,
            "bits": self.bits,
            "offsets": {str(list(k)): v for k, v in self._offset_map.items()},
        }
        with open(index_path, "w") as f:
            json.dump(data, f)
    
    def _read_header(self) -> VectorHeader:
        """Read vector file header."""
        if not self.vector_path.exists():
            return VectorHeader(dimension=self.dimension, bits=self.bits)
        
        with open(self.vector_path, "rb") as f:
            header_data = f.read(VECTOR_HEADER_SIZE)
            if len(header_data) < VECTOR_HEADER_SIZE:
                return VectorHeader(dimension=self.dimension, bits=self.bits)
            
            magic, dim, bits, count = struct.unpack(VECTOR_HEADER_FORMAT, header_data)
            
            if magic != VECTOR_MAGIC:
                raise ValueError(f"Invalid vector file magic: {magic}")
            
            return VectorHeader(magic=magic, dimension=dim, bits=bits, count=count)
    
    def _write_header(self, header: VectorHeader) -> None:
        """Write vector file header."""
        with open(self.vector_path, "wb") as f:
            f.write(struct.pack(
                VECTOR_HEADER_FORMAT,
                header.magic,
                header.dimension,
                header.bits,
                header.count
            ))
    
    def add_vector(self, topic: str, chunk_key: str, vector: np.ndarray) -> int:
        """Add a vector to storage. Returns offset."""
        # Quantize the vector
        q = quantize_packed(vector, bits=self.bits)
        
        # Encode quantization data
        import base64
        packed_data = base64.b64decode(q["data"])
        
        # Get current file size as offset
        offset = self.vector_path.stat().st_size if self.vector_path.exists() else VECTOR_HEADER_SIZE
        
        # Write vector data
        with open(self.vector_path, "ab") as f:
            f.write(packed_data)
        
        # Update header
        header = self._read_header()
        header.count += 1
        self._write_header(header)
        
        # Update index
        self._offset_map[(topic, chunk_key)] = offset
        self._reverse_map[offset] = (topic, chunk_key)
        self._save_index()
        
        return offset
    
    def get_vector(self, topic: str, chunk_key: str) -> Optional[np.ndarray]:
        """Retrieve a vector by topic and chunk key."""
        if (topic, chunk_key) not in self._offset_map:
            return None
        
        offset = self._offset_map[(topic, chunk_key)]
        
        # Read packed data
        with open(self.vector_path, "rb") as f:
            f.seek(offset)
            packed_data = f.read()
        
        # Decode and dequantize
        import base64
        q = {
            "bits": self.bits,
            "scale": 1.0,  # Will need to store these separately
            "qmax": (2 ** (self.bits - 1)) - 1,
            "shape": [self.dimension],
            "data": base64.b64encode(packed_data).decode("utf-8")
        }
        
        return dequantize_packed(q)
    
    def get_vectors(self, offsets: List[int]) -> Dict[int, np.ndarray]:
        """Retrieve multiple vectors by offsets."""
        result = {}
        
        with open(self.vector_path, "rb") as f:
            for offset in offsets:
                f.seek(offset)
                packed_data = f.read()
                
                import base64
                q = {
                    "bits": self.bits,
                    "scale": 1.0,
                    "qmax": (2 ** (self.bits - 1)) - 1,
                    "shape": [self.dimension],
                    "data": base64.b64encode(packed_data).decode("utf-8")
                }
                result[offset] = dequantize_packed(q)
        
        return result
    
    def delete_vector(self, topic: str, chunk_key: str) -> bool:
        """Delete a vector (marks as deleted, doesn't reclaim space)."""
        if (topic, chunk_key) not in self._offset_map:
            return False
        
        offset = self._offset_map[(topic, chunk_key)]
        
        # Mark as deleted by removing from maps
        del self._offset_map[(topic, chunk_key)]
        del self._reverse_map[offset]
        self._save_index()
        
        # Update header count
        header = self._read_header()
        header.count = max(0, header.count - 1)
        self._write_header(header)
        
        return True
    
    def get_count(self) -> int:
        """Get total vector count."""
        return self._read_header().count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector storage statistics."""
        header = self._read_header()
        
        file_size = self.vector_path.stat().st_size if self.vector_path.exists() else 0
        data_size = file_size - VECTOR_HEADER_SIZE
        
        # Estimate uncompressed size
        uncompressed = header.count * header.dimension * 4  # float32
        
        return {
            "vector_count": header.count,
            "dimension": header.dimension,
            "bits": header.bits,
            "file_bytes": file_size,
            "data_bytes": data_size,
            "compression_ratio": uncompressed / data_size if data_size > 0 else 0,
        }
    
    def optimize(self) -> int:
        """Defragment storage and reclaim space. Returns count of removed vectors."""
        # This is a placeholder - real implementation would compact the file
        return 0
    
    def close(self):
        """Close the store."""
        self._save_index()
