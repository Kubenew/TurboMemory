"""TurboMemory Format (TMF) v1 specification.

TMF is a portable storage format for semantic memory:
- .tmindex: SQLite metadata index
- .tmvec: Packed vectors file
- .tmlog: Append-only event log
- .tmmeta.json: Schema + model metadata
"""

import json
import hashlib
import struct
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


TMF_VERSION = "1.0.0"
TMF_MAGIC = b"TMF1"

HEADER_FORMAT = "4sHHI"  # magic(4), version_major(2), version_minor(2), header_size(4)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


@dataclass
class TMFMeta:
    """Metadata for TMF storage."""
    version: str = TMF_VERSION
    created: str = ""
    updated: str = ""
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    quantization_bits: int = 6
    chunk_count: int = 0
    topic_count: int = 0
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TMFMeta":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TMFFileManifest:
    """Manifest describing all TMF files."""
    index_file: str = "index.tmindex"
    vector_file: str = "vectors.tmvec"
    log_file: str = "events.tmlog"
    meta_file: str = "tmmeta.json"
    
    @property
    def all_files(self) -> List[str]:
        return [self.index_file, self.vector_file, self.log_file, self.meta_file]


class TMFFormat:
    """Main TMF format handler."""
    
    VERSION = TMF_VERSION
    
    def __init__(self, root: str):
        self.root = Path(root)
        self.manifest = TMFFileManifest()
        
    @property
    def index_path(self) -> Path:
        return self.root / self.manifest.index_file
    
    @property
    def vector_path(self) -> Path:
        return self.root / self.manifest.vector_file
    
    @property
    def log_path(self) -> Path:
        return self.root / self.manifest.log_file
    
    @property
    def meta_path(self) -> Path:
        return self.root / self.manifest.meta_file
    
    def exists(self) -> bool:
        """Check if TMF storage exists."""
        return all((self.index_path.exists(), 
                   self.vector_path.exists(),
                   self.log_path.exists(),
                   self.meta_path.exists()))
    
    def create(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384) -> "TMFMeta":
        """Initialize new TMF storage."""
        self.root.mkdir(parents=True, exist_ok=True)
        
        now = datetime.now(timezone.utc).isoformat()
        meta = TMFMeta(
            version=TMF_VERSION,
            created=now,
            updated=now,
            model_name=model_name,
            embedding_dim=embedding_dim,
        )
        
        # Create empty files
        self.index_path.touch()
        self.vector_path.touch()
        self.log_path.touch()
        
        # Write metadata
        self._write_meta(meta)
        
        # Compute initial checksum
        self._update_checksum(meta)
        
        return meta
    
    def open(self) -> TMFMeta:
        """Open existing TMF storage."""
        if not self.exists():
            raise FileNotFoundError(f"TMF storage not found at {self.root}")
        
        return self._read_meta()
    
    def _write_meta(self, meta: TMFMeta) -> None:
        """Write metadata to JSON file."""
        with open(self.meta_path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)
    
    def _read_meta(self) -> TMFMeta:
        """Read metadata from JSON file."""
        with open(self.meta_path, "r") as f:
            data = json.load(f)
        return TMFMeta.from_dict(data)
    
    def update_meta(self, **kwargs) -> TMFMeta:
        """Update metadata fields."""
        meta = self._read_meta()
        for key, value in kwargs.items():
            if hasattr(meta, key):
                setattr(meta, key, value)
        
        meta.updated = datetime.now(timezone.utc).isoformat()
        self._write_meta(meta)
        return meta
    
    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of all TMF files."""
        hasher = hashlib.sha256()
        
        # Hash in deterministic order
        for fpath in sorted([self.index_path, self.vector_path, self.log_path]):
            if fpath.exists():
                with open(fpath, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _update_checksum(self, meta: TMFMeta) -> TMFMeta:
        """Update checksum in metadata."""
        meta.checksum = self._compute_checksum()
        self._write_meta(meta)
        return meta
    
    def verify_integrity(self) -> Tuple[bool, str]:
        """Verify storage integrity using checksum."""
        meta = self._read_meta()
        
        if meta.checksum is None:
            return False, "No checksum stored"
        
        current = self._compute_checksum()
        
        if current == meta.checksum:
            return True, "Integrity verified"
        else:
            return False, f"Checksum mismatch: expected {meta.checksum[:16]}..., got {current[:16]}..."
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get version information."""
        meta = self._read_meta()
        return {
            "tmf_version": TMF_VERSION,
            "stored_version": meta.version,
            "compatible": self._is_compatible(meta.version),
        }
    
    def _is_compatible(self, stored_version: str) -> bool:
        """Check if stored version is compatible."""
        stored_major = int(stored_version.split(".")[0])
        return stored_major == int(TMF_VERSION.split(".")[0])
    
    def export(self, export_path: str) -> str:
        """Export TMF storage to directory."""
        import shutil
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all files
        for fpath in [self.index_path, self.vector_path, self.log_path, self.meta_path]:
            if fpath.exists():
                shutil.copy2(fpath, export_dir / fpath.name)
        
        return str(export_dir)
    
    def import_from(self, import_path: str) -> None:
        """Import TMF storage from directory."""
        import shutil
        
        import_dir = Path(import_path)
        
        if not import_dir.exists():
            raise FileNotFoundError(f"Import path not found: {import_path}")
        
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Copy all files
        for fname in TMFFileManifest().all_files:
            src = import_dir / fname
            if src.exists():
                shutil.copy2(src, self.root / fname)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        meta = self._read_meta()
        
        index_size = self.index_path.stat().st_size if self.index_path.exists() else 0
        vector_size = self.vector_path.stat().st_size if self.vector_path.exists() else 0
        log_size = self.log_path.stat().st_size if self.log_path.exists() else 0
        
        return {
            "version": meta.version,
            "chunk_count": meta.chunk_count,
            "topic_count": meta.topic_count,
            "index_bytes": index_size,
            "vector_bytes": vector_size,
            "log_bytes": log_size,
            "total_bytes": index_size + vector_size + log_size,
            "checksum": meta.checksum[:16] + "..." if meta.checksum else None,
        }


def get_version() -> str:
    """Get current TMF version."""
    return TMF_VERSION


def validate_format(root: str) -> Dict[str, Any]:
    """Validate TMF format and return status."""
    tmf = TMFFormat(root)
    
    result = {
        "valid": False,
        "exists": False,
        "version": None,
        "compatible": False,
        "checksum_valid": False,
        "errors": [],
    }
    
    if not tmf.exists():
        result["errors"].append("TMF storage not initialized")
        return result
    
    result["exists"] = True
    
    try:
        meta = tmf._read_meta()
        result["version"] = meta.version
        result["compatible"] = tmf._is_compatible(meta.version)
        
        valid, msg = tmf.verify_integrity()
        result["checksum_valid"] = valid
        
        if not valid:
            result["errors"].append(msg)
        
        result["valid"] = result["compatible"] and result["checksum_valid"]
        
    except Exception as e:
        result["errors"].append(str(e))
    
    return result
