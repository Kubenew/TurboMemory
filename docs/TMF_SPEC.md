# TMF v1 Specification (TurboMemory Format)

> "The SQLite + Parquet of semantic memory"

## Overview

TMF v1 is a portable, self-describing storage format for semantic memory. It combines:
- SQLite for fast metadata queries
- Packed vectors (TurboQuant) for compression
- Append-only logs for durability
- Checksums for integrity

## Design Goals

1. **Portability** - Copy directory → load instantly on any machine
2. **Durability** - Append-only log is source of truth
3. **Efficiency** - 4/6/8-bit TurboQuant vectors
4. **Self-describing** - Versioning, checksums, schema everywhere
5. **Backward-compatible** - Migration support

## Directory Layout

```
tm_data/
├── .tmmeta.json          # Global metadata + schema version
├── tm.index.sqlite       # SQLite metadata DB
├── tm.log                # Append-only event log
├── vectors/              # Packed vector files
│   ├── notes.tmvec
│   └── code.tmvec
└── centroids/            # Optional precomputed centroids
```

## File Formats

### .tmmeta.json - Global Header

```json
{
  "tmf_version": "1.0",
  "created_at": "2026-04-07T09:28:00Z",
  "last_modified": "2026-04-07T10:15:00Z",
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dim": 768,
  "default_bit_width": 6,
  "topics": ["notes", "code"],
  "checksum_algorithm": "sha256",
  "global_checksum": "abc123...",
  "migration_history": [
    {"version": "1.0", "applied_at": "2026-04-07T09:28:00Z"}
  ]
}
```

### .tmindex.sqlite - Metadata Index

```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    text TEXT,
    timestamp INTEGER,
    ttl INTEGER,
    confidence REAL,
    verification INTEGER DEFAULT 0,
    bit_width INTEGER DEFAULT 6,
    vector_offset INTEGER,
    vector_length INTEGER,
    tags TEXT,
    checksum TEXT
);

CREATE TABLE topics (
    name TEXT PRIMARY KEY,
    centroid_id TEXT,
    vector_count INTEGER DEFAULT 0,
    last_consolidated INTEGER
);

CREATE INDEX idx_topic ON chunks(topic);
CREATE INDEX idx_timestamp ON chunks(timestamp);
CREATE INDEX idx_confidence ON chunks(confidence);
```

### .tmvec - Packed Vectors

Binary format (128-byte header + packed body):

```
[Header - 128 bytes]
Magic: "TMFV1" (5 bytes)
Version: u32
Bit Width: u8 (4/6/8)
Dimension: u16
Vector Count: u32
Reserved: 32 bytes

[Body - packed vectors]
Each vector: dim * bit_width / 8 bytes
```

### .tmlog - Event Log

Binary append-only format:

```
[Entry - 32 bytes header]
Magic: "TMLOG" (5 bytes)
Entry Size: u32
Entry Type: u8 (0=ADD, 1=UPDATE, 2=DELETE)
Timestamp: u64 (unix ms)
Entry ID: 16 bytes (UUID)

[Payload]
JSON-serialized event data
```

## Integrity & Versioning

1. **Global checksum** - SHA-256 of all files (stored in .tmmeta.json)
2. **Per-chunk checksums** - In SQLite index
3. **Vector file checksums** - In footer
4. **Migration history** - Tracked in .tmmeta.json

## Usage

```python
from turbomemory.tmf import create_tmf_store, open_tmf_store, verify_tmf_store

# Create new store
store = create_tmf_store("./data")

# Open existing
store = open_tmf_store("./data")

# Verify integrity
result = verify_tmf_store("./data")
print(result)
```

## CLI Commands

```bash
# Verify integrity
turbomemory verify --path ./data

# Export to bundle
turbomemory bundle --output backup.tm

# Import from bundle
turbomemory restore --path backup.tm
```

## Compression Benchmarks

| Format | Size (10K×768) | Compression |
|--------|----------------|-------------|
| float32 | 28.8 MB | 1x |
| 8-bit | 7.3 MB | **4x** |
| 6-bit | 5.5 MB | **5.2x** |
| 4-bit | 3.7 MB | **8x** |

## Implementation

See `turbomemory/tmf/__init__.py` for the complete implementation.

## Future Extensions (v1.1+)

- Single-file .tm bundle (tar + index)
- Optional AES-GCM encryption
- Zstd compression on top
- Hybrid search with FTS5
