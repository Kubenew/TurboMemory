# TurboMemory Interoperability

Export and import TurboMemory data to/from Parquet and Lance formats.

## Overview

This module provides interoperability between TurboMemory and broader data ecosystems:

- **Parquet** - Analytics/ML pipeline integration (Spark, DuckDB, Polars)
- **Lance** - AI-native lakehouse with native vectors and versioning

## Installation

```bash
# For Parquet support
pip install turbomemory[parquet]

# For Lance support  
pip install turbomemory[lance]

# For both
pip install turbomemory[export]
```

## Quick Start

### Parquet Export/Import

```python
from turbomemory.interop import export_to_parquet, import_from_parquet

# Export TMF to Parquet (full = float32 vectors)
export_to_parquet(
    root="./data",
    output_path="export.parquet",
    format="full"  # or "quantized" for TurboQuant binary
)

# Import from Parquet
result = import_from_parquet(
    root="./data",
    input_path="export.parquet",
    topic="notes"  # default topic for imported data
)
```

### Lance Export/Import

```python
from turbomemory.interop import export_to_lance, import_from_lance

# Export to Lance dataset
export_to_lance(
    root="./data",
    uri="./lance_dataset",
    format="full",
    mode="create"  # or "append"
)

# Import from Lance
result = import_from_lance(
    root="./data",
    uri="./lance_dataset",
    topic="notes"
)
```

## CLI Usage

```bash
# Export to Parquet
turbomemory export-parquet --output data.parquet --format full --topic notes

# Export to Lance
turbomemory export-lance --uri ./dataset --format full --mode create

# Import from Parquet
turbomemory import-parquet --input data.parquet --topic notes

# Import from Lance
turbomemory import-lance --uri ./dataset --topic notes
```

## Format Options

### Vector Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `full` | Dequantized to float32 | Analytics, ML pipelines |
| `quantized` | TurboQuant binary blob | Preserve compression |

### Compression

Parquet uses Zstd compression level 9 by default for optimal size/speed tradeoff.

## Round-Trip Quality

Vectors maintain >0.99 cosine similarity after export/import round-trip when using "full" format.

## Schema

Exported Parquet/Lance tables include:

| Column | Type | Description |
|--------|------|-------------|
| id | string | Unique chunk identifier |
| topic | string | Topic name |
| text | string | Original text content |
| timestamp | int64 | Unix timestamp (ms) |
| confidence | float64 | Confidence score (0-1) |
| ttl | int64 | Time to live (seconds) |
| verification | int64 | Verification status |
| vector | list<float32> | Embedding vector (full) or binary (quantized) |
| bit_width | int8 | Quantization bits (4/6/8) |

## Integrations

### DuckDB
```python
import duckdb
conn = duckdb.read_parquet("export.parquet")
# Query semantic similarity with SQL
SELECT text, array_distance(vector, query_vector) as dist 
FROM export.parquet ORDER BY dist LIMIT 10
```

### Polars
```python
import polars as pl
df = pl.read_parquet("export.parquet")
```

### Lance + LangChain
```python
import lance
ds = lance.dataset("./dataset")
# Lance handles vectors natively for similarity search
```

## Requirements

- Python 3.9+
- pyarrow >= 15.0.0 (for Parquet)
- lance >= 0.10.0 (for Lance)
- pandas >= 2.0.0
