# TurboMemory API Reference

## Core Classes

### TurboMemory

Main memory system class.

```python
from turbomemory import TurboMemory

tm = TurboMemory(root="./data", model_name="all-MiniLM-L6-v2")
```

#### Constructor

```python
TurboMemory(
    root: str = "turbomemory_data",
    model_name: str = "all-MiniLM-L6-v2",
    config: Optional[TurboMemoryConfig] = None
)
```

#### Methods

##### `add_memory(topic, text, confidence=0.8, bits=None, source_ref=None, ttl_days=None)`

Add a memory chunk to a topic.

**Parameters:**
- `topic` (str): Topic name
- `text` (str): Text content
- `confidence` (float): Confidence score 0-1
- `bits` (int): Quantization bits (4, 6, or 8)
- `source_ref` (str): Source reference
- `ttl_days` (float): Time to live in days

**Returns:** `Optional[str]` - chunk_id or None if excluded

##### `query(query_text, k=5, top_topics=5, min_confidence=None, require_verification=False)`

Query memory with semantic search.

**Parameters:**
- `query_text` (str): Query text
- `k` (int): Number of results
- `top_topics` (int): Topics to search
- `min_confidence` (float): Minimum confidence filter
- `require_verification` (bool): Require verified results

**Returns:** `List[Tuple[float, str, Dict]]` - (score, topic, chunk)

##### `load_topic(topic)`

Load topic data from file.

**Parameters:**
- `topic` (str): Topic name

**Returns:** `Dict` - topic data

##### `save_topic(topic_data)`

Save topic data to file.

**Parameters:**
- `topic_data` (Dict): Topic data to save

##### `get_metrics()`

Get comprehensive memory metrics.

**Returns:** `MemoryMetrics` object

##### `stats()`

Get quick memory statistics.

**Returns:** `Dict` - statistics

##### `backup(backup_path)`

Create a backup of the memory store.

**Parameters:**
- `backup_path` (str): Path for backup

**Returns:** `str` - backup path

##### `restore(backup_path)`

Restore from a backup.

**Parameters:**
- `backup_path` (str): Path to backup

##### `bulk_import(items, topic_field="topic", text_field="text")`

Import multiple memory items.

**Parameters:**
- `items` (List[Dict]): Items to import
- `topic_field` (str): Field name for topic
- `text_field` (str): Field name for text

**Returns:** `Dict` - {"imported": n, "excluded": n, "failed": n}

##### `export_topic(topic, include_embeddings=False)`

Export a topic's data.

**Parameters:**
- `topic` (str): Topic name
- `include_embeddings` (bool): Include embeddings

**Returns:** `Dict` - topic data

##### `export_all(include_embeddings=False)`

Export all topics.

**Parameters:**
- `include_embeddings` (bool): Include embeddings

**Returns:** `List[Dict]` - all topic data

##### `close()`

Close all database connections.

---

### TurboMemoryConfig

Configuration for TurboMemory.

```python
config = TurboMemoryConfig(
    root="data",
    model_name="all-MiniLM-L6-v2",
    default_bits=6,
    pool_size=5,
    default_ttl_days=None,
    max_chunks_per_topic=300,
    contradiction_decay=0.6,
    min_confidence=0.0,
    lazy_load_model=True,
    enable_verification=True,
    verification_threshold=0.7,
    min_cross_refs=2,
    enable_exclusions=True,
    exclusion_rules=None,
    min_quality_threshold=0.3,
    quality_decay_rate=0.01,
    use_faiss=True,
    enable_encryption=False,
    encryption_key=None,
)
```

---

### ExclusionRules

Configurable rules for what NOT to store in memory.

```python
rules = ExclusionRules(
    blocked_patterns=[...],
    blocked_topics=[...],
    min_text_length=10,
    max_text_length=5000,
    block_code_snippets=True,
    block_debug_output=True,
    block_file_paths=False,
    block_secrets=True,
)
```

**Methods:**

##### `should_exclude(text, topic="")`

Check if text should be excluded.

**Returns:** `Tuple[bool, str]` - (should_exclude, reason)

---

## Quantization Functions

### `quantize_packed(vec, bits=6)`

Symmetric uniform quantization with packed storage.

**Parameters:**
- `vec` (np.ndarray): Input vector
- `bits` (int): Quantization bits (4, 6, or 8)

**Returns:** `Dict` - quantized object

### `dequantize_packed(qobj)`

Dequantize a packed quantization object back to float32.

**Parameters:**
- `qobj` (Dict): Quantized object

**Returns:** `np.ndarray` - dequantized vector

---

## Storage Classes

### StorageManager

High-level storage manager with migrations and retry logic.

```python
from turbomemory.storage import StorageManager

sm = StorageManager("path/to/db.sqlite")
```

**Methods:**

- `get_conn()` - Context manager for database connections
- `execute_with_retry(sql, params)` - Execute SQL with retry
- `close()` - Close all connections

### SQLitePool

Thread-safe SQLite connection pool with WAL mode.

```python
pool = SQLitePool("path/to/db.sqlite", pool_size=5)
conn = pool.get_connection()
```

---

## Retrieval Classes

### RetrievalEngine

Retrieval engine with pluggable index backends.

```python
from turbomemory.retrieval import RetrievalEngine

engine = RetrievalEngine(
    dimension=384,
    use_faiss=True,
    index_type="Flat"
)
```

**Methods:**

- `add_vectors(vectors, ids)` - Add vectors to index
- `search(query, k, filter_ids)` - Search for similar vectors
- `save_index(path)` - Save index to disk
- `load_index(path)` - Load index from disk
- `clear()` - Clear the index

---

## API Server

### Running the Server

```bash
turbomemory-server --host 0.0.0.0 --port 8000 --root ./data
```

Or programmatically:

```python
from turbomemory.api import run_server

run_server(host="0.0.0.0", port=8000, root="./data")
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/memory` | Add memory chunk |
| POST | `/memory/bulk` | Bulk import |
| GET | `/memory/{topic}` | Get topic |
| DELETE | `/memory/{topic}/{chunk_id}` | Delete chunk |
| POST | `/query` | Query memory |
| POST | `/query/verify` | Query with verification |
| GET | `/topics` | List all topics |
| GET | `/metrics` | Get system metrics |
| GET | `/stats` | Get quick stats |
| GET | `/audit` | Get audit logs |
| GET | `/export/topic/{topic}` | Export topic |
| GET | `/export` | Export all topics |
| POST | `/backup` | Create backup |
| POST | `/restore` | Restore from backup |
| POST | `/consolidate` | Run consolidation |
| GET | `/config` | Get configuration |
| PATCH | `/config` | Update configuration |

---

## Configuration Files

### Config File (config.json)

```json
{
  "root": "turbomemory_data",
  "model_name": "all-MiniLM-L6-v2",
  "default_bits": 6,
  "pool_size": 5,
  "default_ttl_days": null,
  "max_chunks_per_topic": 300,
  "contradiction_decay": 0.6,
  "min_confidence": 0.0,
  "lazy_load_model": true,
  "enable_verification": true,
  "verification_threshold": 0.7,
  "min_cross_refs": 2,
  "enable_exclusions": true,
  "min_quality_threshold": 0.3,
  "quality_decay_rate": 0.01,
  "use_faiss": true,
  "enable_encryption": false
}
```

### Environment Variables

- `TURBOMEMORY_ROOT` - Data directory
- `TURBOMEMORY_MODEL` - Model name
- `TURBOMEMORY_BITS` - Default quantization bits
- `TURBOMEMORY_FAISS` - Enable FAISS (true/false)
- `TURBOMEMORY_ENCRYPTION_KEY` - Encryption key