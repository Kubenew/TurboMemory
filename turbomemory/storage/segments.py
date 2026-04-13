"""TurboMemory v3 Segment files for cold storage."""

import os
import struct
import zstandard
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Iterator, Tuple
import numpy as np

logger = logging.getLogger(__name__)

MAGIC = b"TMSG"
VERSION = 1
BLOCK_HEADER_SIZE = 64


class SegmentHeader:
    """Segment file header."""
    
    def __init__(
        self, 
        created_at: int, 
        block_count: int,
        min_ts: int = 0,
        max_ts: int = 0,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
    ):
        self.magic = MAGIC
        self.version = VERSION
        self.created_at = created_at
        self.block_count = block_count
        self.min_ts = min_ts
        self.max_ts = max_ts
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.reserved = b"\x00" * 32
    
    def to_bytes(self) -> bytes:
        return struct.pack(
            "<5sIHIIff32s",
            self.magic,
            self.version,
            self.created_at,
            self.block_count,
            self.min_ts,
            self.max_ts,
            self.min_confidence,
            self.max_confidence,
            self.reserved
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "SegmentHeader":
        values = struct.unpack("<5sIHIIff32s", data[:64])
        return cls(
            created_at=values[2],
            block_count=values[3],
            min_ts=values[4],
            max_ts=values[5],
            min_confidence=values[6],
            max_confidence=values[7],
        )


class BlockHeader:
    """Embedding block header."""
    
    def __init__(
        self,
        block_id: int,
        record_count: int,
        dim: int,
        dtype: str,
        centroid: Optional[bytes] = None,
        compressed_size: int = 0,
    ):
        self.block_id = block_id
        self.record_count = record_count
        self.dim = dim
        self.dtype = dtype
        self.centroid = centroid or b"\x00" * 32
        self.compressed_size = compressed_size
    
    def to_bytes(self) -> bytes:
        return struct.pack(
            "<IIII32sI",
            self.block_id,
            self.record_count,
            self.dim,
            self.dtype.encode()[:4].ljust(4, b"\x00"),
            self.centroid,
            self.compressed_size
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "BlockHeader":
        values = struct.unpack("<IIII32sI", data[:BLOCK_HEADER_SIZE])
        return cls(
            block_id=values[0],
            record_count=values[1],
            dim=values[2],
            dtype=values[3].decode().strip("\x00"),
            centroid=values[4],
            compressed_size=values[5],
        )


class SegmentWriter:
    """Write segment files."""
    
    def __init__(self, path: str):
        self.path = path
        self.blocks = []
        self.records = []
        self._file = None
    
    def add(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
    ):
        """Add a record to current block."""
        self.records.append({
            "text": text,
            "embedding": embedding.tobytes(),
            "metadata": metadata,
        })
    
    def flush(self, dtype: str = "q6") -> int:
        """Flush current block to disk."""
        if not self.records:
            return 0
        
        block_id = len(self.blocks)
        dim = len(self.records[0]["embedding"]) // 4  # float32
        
        centroids = np.mean(
            np.frombuffer(b"".join([r["embedding"] for r in self.records]), dtype=np.float32
        ).reshape(-1, dim),
            axis=0
        ).tobytes()
        
        texts = json.dumps([r["text"] for r in self.records]).encode("utf-8")
        metadata = json.dumps([r["metadata"] for r in self.records]).encode("utf-8")
        
        cctx = zstandard.ZstdCompressor()
        texts_compressed = cctx.compress(texts)
        meta_compressed = cctx.compress(metadata)
        
        embeddings_compressed = cctx.compress(b"".join([r["embedding"] for r in self.records]))
        
        block_data = texts_compressed + meta_compressed + embeddings_compressed
        
        header = BlockHeader(
            block_id=block_id,
            record_count=len(self.records),
            dim=dim,
            dtype=dtype,
            centroid=centroids,
            compressed_size=len(block_data),
        )
        
        self.blocks.append(header)
        
        if not self._file:
            self._file = open(self.path, "wb")
            now = int(datetime.now(timezone.utc).timestamp())
            seg_header = SegmentHeader(
                created_at=now,
                block_count=0,
            )
            self._file.write(seg_header.to_bytes())
        
        self._file.write(header.to_bytes())
        self._file.write(block_data)
        
        self.records = []
        return block_id
    
    def close(self):
        if self._file:
            self._file.seek(0)
            now = int(datetime.now(timezone.utc).timestamp())
            seg_header = SegmentHeader(
                created_at=now,
                block_count=len(self.blocks),
            )
            self._file.write(seg_header.to_bytes())
            self._file.close()
            self._file = None


class SegmentReader:
    """Read segment files."""
    
    def __init__(self, path: str):
        self.path = path
        self.header = None
        self.block_headers = []
        self._file = None
        self._load()
    
    def _load(self):
        if not os.path.exists(self.path):
            return
        
        self._file = open(self.path, "rb")
        
        header_data = self._file.read(BLOCK_HEADER_SIZE)
        self.header = SegmentHeader.from_bytes(header_data)
        
        for i in range(self.header.block_count):
            block_data = self._file.read(BLOCK_HEADER_SIZE)
            self.block_headers.append(BlockHeader.from_bytes(block_data))
    
    def read_block(self, block_id: int) -> List[Dict[str, Any]]:
        """Read a specific block."""
        if block_id >= len(self.block_headers):
            return []
        
        header = self.block_headers[block_id]
        self._file.seek((1 + block_id) * BLOCK_HEADER_SIZE)
        block_data = self._file.read(header.compressed_size)
        
        dctx = zstandard.ZstdDecompressor()
        
        text_size = len(block_data) // 3
        data = dctx.decompress(block_data)
        
        texts = json.loads(data[:text_size].decode("utf-8"))
        meta = json.loads(data[text_size:text_size*2].decode("utf-8"))
        embeddings = np.frombuffer(
            dctx.decompress(data[text_size*2:]),
            dtype=np.float32
        ).reshape(-1, header.dim)
        
        return [
            {
                "text": texts[i],
                "embedding": embeddings[i],
                "metadata": meta[i],
            }
            for i in range(header.record_count)
        ]
    
    def read_all(self) -> Iterator[Dict[str, Any]]:
        """Read all blocks."""
        for block_id in range(len(self.block_headers)):
            for record in self.read_block(block_id):
                yield record
    
    def close(self):
        if self._file:
            self._file.close()
            self._file = None


class SegmentManager:
    """Manage cold storage segments."""
    
    def __init__(self, segments_dir: str):
        self.segments_dir = segments_dir
        os.makedirs(segments_dir, exist_ok=True)
        
        self.active_writer = None
        self.segment_counter = 0
    
    def get_active_writer(self, max_records: int = 10000) -> SegmentWriter:
        """Get or create active writer."""
        if self.active_writer is None:
            self.segment_counter += 1
            path = os.path.join(
                self.segments_dir,
                f"segment_{self.segment_counter:06d}.tms"
            )
            self.active_writer = SegmentWriter(path)
        
        if len(self.active_writer.records) >= max_records:
            self.active_writer.flush()
            self.active_writer.close()
            self.active_writer = None
        
        return self.active_writer or self.get_active_writer(max_records)
    
    def add(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
    ) -> None:
        """Add a record to cold storage."""
        writer = self.get_active_writer()
        writer.add(text, embedding, metadata)
    
    def flush(self) -> None:
        """Flush active writer."""
        if self.active_writer:
            self.active_writer.flush()
            self.active_writer.close()
            self.active_writer = None
    
    def get_segments(self) -> List[str]:
        """List all segment files."""
        return sorted([
            f for f in os.listdir(self.segments_dir)
            if f.endswith(".tms")
        ])
    
    def read_segment(self, name: str) -> SegmentReader:
        """Read a specific segment."""
        return SegmentReader(os.path.join(self.segments_dir, name))
    
    def compact(self, merged_segment: Optional[str] = None) -> str:
        """Compact multiple segments."""
        if merged_segment is None:
            merged_segment = f"segment_compacted_{self.segment_counter:06d}.tms"
        
        writer = SegmentWriter(
            os.path.join(self.segments_dir, merged_segment)
        )
        
        for seg_name in self.get_segments():
            if "compacted" in seg_name:
                continue
            
            reader = self.read_segment(seg_name)
            for record in reader.read_all():
                writer.add(
                    record["text"],
                    np.frombuffer(record["embedding"], dtype=np.float32),
                    record["metadata"],
                )
            reader.close()
        
        writer.flush()
        writer.close()
        
        return merged_segment