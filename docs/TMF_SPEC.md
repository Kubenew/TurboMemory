# TMF v1 — TurboMemory Format Specification

**Version**: 1.0 (Draft)  
**Status**: In Progress (Target: v0.5)  
**Date**: April 2026  
**Purpose**: Define a portable, verifiable, and efficient on-disk format for semantic memory that combines SQLite simplicity with TurboQuant-compressed embeddings.

## 1. Design Goals

- **Portability** — The entire memory store must be a self-contained directory that can be copied, zipped, backed up, or moved between machines/devices with zero reconfiguration.
- **Durability** — Append-only log as the single source of truth; all other files are derived/cache.
- **Efficiency** — Extreme compression via TurboQuant (4/6/8-bit packed scalar quantization) while preserving high cosine similarity.
- **Simplicity** — Leverage SQLite for rich metadata and fast filtering.
- **Verifiability** — Checksums everywhere; fast validation on load.
- **Extensibility** — Versioned, with clear migration path.
- **Topic-aware** — Support centroid-based pre-filtering and on-demand loading.

## 2. On-Disk Layout

A TurboMemory store is a directory (default: `./tm_data/` or user-specified root):
<root>/
├── .tmmeta.json                 # Global metadata and schema version
├── tm.index.sqlite              # SQLite metadata index (fast filtering)
├── tm.log                       # Append-only event/transcript log (durability)
├── vectors/                     # Topic-partitioned quantized vector files
│   ├── <topic1>.tmvec
│   ├── <topic2>.tmvec
│   └── ...
├── centroids/                   # Optional precomputed centroids per topic
│   └── <topic>.tmcent
├── snapshots/                   # Optional compressed snapshots (future)
└── attachments/                 # Optional binary blobs for multimodal (future)
textThe whole directory (or a `.tm` zip bundle) is portable.

## 3. `.tmmeta.json` — Global Header

JSON (or CBOR for smaller/faster parsing in future versions).

```json
{
  "tmf_version": "1.0",
  "created_at": "2026-04-08T09:48:00Z",
  "last_modified": "2026-04-08T10:15:00Z",
  "embedding_model": "nomic-embed-text-v1.5",
  "embedding_dim": 768,
  "default_bit_width": 6,
  "topics": ["notes", "code", "preferences"],
  "checksum_algorithm": "sha256",
  "global_checksum": "sha256-of-all-files-combined",
  "migration_history": [
    {
      "version": "1.0",
      "applied_at": "2026-04-08T09:48:00Z",
      "description": "Initial TMF v1"
    }
  ],
  "config": {
    "quantization": {},
    "consolidation": {}
  }
}
4. tm.index.sqlite — Metadata Index
Standard SQLite database (tm.index.sqlite).
Core tables (subject to evolution via migrations):
SQLCREATE TABLE chunks (
    id            TEXT PRIMARY KEY,           -- UUID or content hash
    topic         TEXT NOT NULL,
    text          TEXT,                        -- original text or reference to log
    timestamp     INTEGER NOT NULL,            -- unix timestamp in ms
    ttl           INTEGER,                     -- seconds until expiry (NULL = never)
    confidence    REAL DEFAULT 0.5,            -- quality score 0.0–1.0
    verification  INTEGER DEFAULT 0,           -- 0=unverified, 1=verified
    bit_width     INTEGER DEFAULT 6,           -- 4/6/8
    vector_offset INTEGER NOT NULL,            -- byte offset in <topic>.tmvec
    vector_length INTEGER NOT NULL,            -- number of vectors (usually 1)
    tags          TEXT,                        -- JSON array
    checksum      TEXT                         -- per-chunk checksum
);

CREATE TABLE topics (
    name               TEXT PRIMARY KEY,
    centroid_offset    INTEGER,                -- reference in centroids/<topic>.tmcent
    vector_count       INTEGER DEFAULT 0,
    last_consolidated  INTEGER
);

-- Indexes for performance
CREATE INDEX idx_topic ON chunks(topic);
CREATE INDEX idx_timestamp ON chunks(timestamp);
CREATE INDEX idx_confidence ON chunks(confidence);
5. .tmvec Files — Quantized Vector Storage (TurboQuant)
One file per topic: vectors/<topic>.tmvec
Binary Layout
Header (fixed 128 bytes):

Bytes 0–4: Magic "TMFV1"
Bytes 4–8: Version (u32, currently 1)
Byte 8: Bit width (u8: 4, 6, or 8)
Bytes 9–10: Dimension (u16)
Bytes 11–14: Number of vectors (u32)
Bytes 15–(15+dim*4): Optional centroid (float32[dim])
Remaining bytes: reserved (zero-padded to 128)

Body — Packed vectors (no padding between vectors):

4-bit: 2 values per byte (nibble packing)
6-bit: 4 values per 3 bytes (efficient bit packing)
8-bit: 1 byte per dimension

Quantization is deterministic (fixed seed + per-dimension min/max or learned codebook stored in header or separate .tmcent file).
Footer:

SHA-256 checksum of the entire body

6. tm.log — Append-only Transcript Log
Single file (or rotated as tm.log.0001, etc.).
Entry format (WAL-style):

Fixed header (32 bytes):
Magic "TMLOG"
Entry size (u32)
Entry type (u8: 0=ADD, 1=UPDATE, 2=DELETE, 3=CONSOLIDATE, ...)
Timestamp (u64 ms)
Chunk ID (16-byte UUID)
CRC32 of payload

Variable payload: CBOR or JSON-serialized event (original text, raw embedding hash, metadata)

The log is the source of truth. The consolidation daemon replays it to update vector files and the SQLite index.
7. TurboQuant Details (Core Compression Codec)

Scalar quantization to 4/6/8 bits per dimension.
Packing rules are deterministic and specified above.
Supported similarity metric: cosine (with optional normalization).
Quality targets (current benchmarks, 384-dim embeddings):
8-bit → ~4× compression, 0.997 cosine
6-bit → ~5.2× compression, 0.968 cosine
4-bit → ~8× compression, 0.912 cosine


Reference implementation lives in the turboquant/ package.
8. Integrity & Verification

Every major file ends with a SHA-256 checksum.
.tmmeta.json contains a global checksum.
On load, TurboMemory performs full or lazy verification.
CLI command: tm verify (planned).

9. Versioning & Migration

Handled via .tmmeta.json migration_history array.
Future versions (1.1, 1.2…) will document breaking changes and automatic migration steps (e.g., re-quantization when bit_width changes).

10. Interoperability

Export/Import — Full support via CLI (export, import).
Parquet / Lance — Planned bridge via Apache Arrow (see interop/ module). Vectors can be exported dequantized (float32) or as quantized binary blobs.

11. Future Extensions (v1.1+)

Single-file .tm bundle format
Built-in encryption (AES-GCM)
Native multimodal blob support
Snapshot files for faster cold starts


Implementation Note
This specification is derived from the current tmf/ module, tm.index.sqlite usage, .tmlog append-only pattern, topic-based .tmvec files, and TurboQuant quantization logic present in the codebase.
Once the tmf/ readers/writers fully match this document, mark TMF v1 as stable and update the ROADMAP.
Feedback and pull requests welcome — this spec aims to make TMF the SQLite + zstd-for-embeddings standard for local semantic memory.

