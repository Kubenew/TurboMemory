# TurboMemory Benchmarks

Performance benchmarks for TurboMemory quantization and retrieval.

## Running Benchmarks

### Default Suite

```bash
python -m turbomemory.benchmark
```

### Custom Configuration

```python
from turbomemory.benchmark import MemoryBenchmark, BenchmarkConfig

config = BenchmarkConfig(
    num_vectors=10000,
    bit_depths=[4, 6, 8],
    num_queries=100,
    k=10,
    batch_sizes=[1, 10, 50, 100]
)

bench = MemoryBenchmark(config)
results = bench.run_all()
bench.print_summary()
bench.save_results("results.json")
```

## Benchmark Categories

### 1. Compression Benchmarks

Measures compression ratios across different bit depths:

- **4-bit**: ~8x compression, 0.91 recall
- **6-bit**: ~5x compression, 0.97 recall  
- **8-bit**: ~4x compression, 0.99 recall

### 2. Throughput Benchmarks

- **Encode**: vectors/second for batch encoding
- **Decode**: vectors/second for batch decoding
- **Query**: queries/second

### 3. Latency Benchmarks

- P50, P95, P99 query latency
- Varies by dataset size

### 4. Storage Benchmarks

- Total bytes by bit depth
- Compression vs float32

## Results Format

Results are saved as JSON:

```json
{
  "timestamp": "2024-01-01T00:00:00",
  "config": {
    "num_vectors": 10000,
    "bit_depths": [4, 6, 8]
  },
  "results": [
    {
      "name": "compression_6bit",
      "duration_ms": 150.0,
      "throughput": 6666.0,
      "memory_mb": 45.2,
      "metadata": {
        "compression_ratio": 5.2
      }
    }
  ]
}
```

## Adding Custom Benchmarks

```python
class CustomBenchmark(MemoryBenchmark):
    def benchmark_custom(self):
        # Your benchmark code
        return BenchmarkResult(...)
```

## Hardware Requirements

- CPU: Any x86_64
- RAM: 4GB minimum
- Disk: 1GB for test data
