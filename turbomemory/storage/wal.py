"""TurboMemory v3 WAL (Write-Ahead Log) for durability."""

import os
import json
import struct
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Iterator, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

MAGIC = b"TMLW"
VERSION = 1


class WALWriter:
    """Append-only WAL writer."""
    
    def __init__(self, log_path: str, mode: str = "a"):
        self.log_path = log_path
        self.mode = mode
        self._file = None
        self._open()
    
    def _open(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._file = open(self.log_path, self.mode + "b")
    
    def write(self, op: str, data: Dict[str, Any]) -> int:
        """Write a record to WAL."""
        now = int(datetime.now(timezone.utc).timestamp())
        
        record = {
            "op": op,
            "ts": now,
            **data
        }
        
        payload = json.dumps(record, ensure_ascii=False).encode("utf-8")
        
        self._file.write(MAGIC)
        self._file.write(struct.pack("<I", VERSION))
        self._file.write(struct.pack("<I", len(payload)))
        self._file.write(payload)
        self._file.flush()
        
        return now
    
    def close(self):
        if self._file:
            self._file.close()
            self._file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class WALReader:
    """WAL reader with recovery support."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
    
    def read(self, since: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Read records since timestamp."""
        if not os.path.exists(self.log_path):
            return
        
        with open(self.log_path, "rb") as f:
            while True:
                magic = f.read(4)
                if not magic or magic != MAGIC:
                    break
                
                version = struct.unpack("<I", f.read(4))[0]
                if version != VERSION:
                    logger.warning(f"Unknown WAL version {version}, skipping")
                    break
                
                size = struct.unpack("<I", f.read(4))[0]
                payload = f.read(size)
                
                record = json.loads(payload.decode("utf-8"))
                
                if since is None or record.get("ts", 0) > since:
                    yield record
    
    def read_all(self) -> List[Dict[str, Any]]:
        """Read all records."""
        return list(self.read())
    
    def replay(self, handler: Callable[[Dict[str, Any]], since: Optional[int] = None) -> int:
        """Replay records with handler."""
        count = 0
        for record in self.read(since):
            try:
                handler(record)
                count += 1
            except Exception as e:
                logger.error(f"Error replaying record: {e}")
        return count


class WALIndex:
    """WAL index for efficient recovery."""
    
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.offset_index = {}  # op -> last offset
        self._load()
    
    def _load(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        self.offset_index[parts[0]] = int(parts[1])
    
    def save(self):
        with open(self.index_path, "w") as f:
            for op, offset in self.offset_index.items():
                f.write(f"{op},{offset}\n")
    
    def get_last_offset(self, op: str) -> int:
        return self.offset_index.get(op, 0)
    
    def set_last_offset(self, op: str, offset: int):
        self.offset_index[op] = offset


class WALLog:
    """Complete WAL manager with compaction."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.wal_active = os.path.join(log_dir, "active.tmlog")
        self.index_path = os.path.join(log_dir, ".tmlog.idx")
        self.compacted_dir = os.path.join(log_dir, "compacted")
        os.makedirs(self.compacted_dir, exist_ok=True)
        
        self.writer = WALWriter(self.wal_active, mode="a")
        self.index = WALIndex(self.index_path)
    
    def append(self, op: str, **data) -> int:
        """Append a record."""
        return self.writer.write(op, data)
    
    def read_since(self, op: str, since: int) -> List[Dict[str, Any]]:
        """Read records since timestamp."""
        results = []
        for record in self.writer.read(since):
            if record.get("op") == op:
                results.append(record)
        return results
    
    def recover(self, handler: Callable[[Dict[str, Any]]]) -> int:
        """Recover from WAL."""
        return self.writer.replay(handler)
    
    def rotate(self) -> str:
        """Rotate WAL file."""
        self.writer.close()
        
        import time
        rotated = os.path.join(
            self.compacted_dir, 
            f"{int(time.time())}.tmlog"
        )
        
        if os.path.exists(self.wal_active):
            os.rename(self.wal_active, rotated)
        
        self.writer = WALWriter(self.wal_active, mode="w")
        return rotated
    
    def close(self):
        self.writer.close()
        self.index.save()