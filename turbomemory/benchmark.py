"""Comprehensive benchmarking suite for TurboMemory."""

import os
import time
import json
import tempfile
import shutil
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from turbomemory import TurboMemory, TurboMemoryConfig
from turbomemory.quantization import quantize_packed, dequantize_packed


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    num_vectors: int = 10000
    dimensions: List[int] = field(default_factory=lambda: [384, 768])
    bit_depths: List[int] = field(default_factory=lambda: [4, 6, 8])
    num_queries: int = 100
    k: int = 10
    batch_sizes: List[int] = field(default_factory=lambda: [1, 10, 100])


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    duration_ms: float
    throughput: float
    memory_mb: float
    accuracy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryBenchmark:
    """Benchmark TurboMemory performance."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self._temp_dir: Optional[str] = None
    
    def setup(self) -> str:
        """Create temporary directory for benchmarks."""
        self._temp_dir = tempfile.mkdtemp(prefix="turbomemory_bench_")
        return self._temp_dir
    
    def teardown(self) -> None:
        """Clean up temporary directory."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        self.setup()
        
        try:
            # Compression benchmarks
            self.benchmark_compression()
            
            # Encode throughput
            self.benchmark_encode_throughput()
            
            # Decode throughput
            self.benchmark_decode_throughput()
            
            # Query latency
            self.benchmark_query_latency()
            
            # Batch operations
            self.benchmark_batch_operations()
            
        # Storage size
        self.benchmark_storage_size()
        
        # Load time
        self.benchmark_load_time()
        
        # Recall quality
            self.benchmark_recall_quality()
            
        finally:
            self.teardown()
        
        return self.results
    
    def benchmark_compression(self) -> BenchmarkResult:
        """Benchmark compression ratios across bit depths."""
        print("\n=== Compression Benchmark ===")
        
        results = []
        dimension = 384
        
        for bits in self.config.bit_depths:
            vectors = np.random.randn(1000, dimension).astype(np.float32)
            
            start = time.time()
            for vec in vectors:
                q = quantize_packed(vec, bits=bits)
            duration = time.time() - start
            
            # Calculate compression
            original_size = vectors.nbytes
            import base64
            packed_size = len(base64.b64decode(q["data"]))
            ratio = original_size / (packed_size * 1000)
            
            result = BenchmarkResult(
                name=f"compression_{bits}bit",
                duration_ms=duration * 1000,
                throughput=1000 / duration,
                memory_mb=0,
                metadata={"bits": bits, "compression_ratio": ratio}
            )
            results.append(result)
            print(f"  {bits}-bit: {ratio:.1f}x compression, {duration*1000:.1f}ms for 1000 vectors")
        
        self.results.extend(results)
        return results[0]
    
    def benchmark_encode_throughput(self) -> BenchmarkResult:
        """Benchmark encoding throughput."""
        print("\n=== Encode Throughput Benchmark ===")
        
        dimension = 384
        vectors = np.random.randn(self.config.num_vectors, dimension).astype(np.float32)
        
        results = []
        for bits in [6]:  # Default
            start = time.time()
            quantized = [quantize_packed(v, bits=bits) for v in vectors]
            duration = time.time() - start
            
            result = BenchmarkResult(
                name=f"encode_{self.config.num_vectors}_vectors",
                duration_ms=duration * 1000,
                throughput=self.config.num_vectors / duration,
                memory_mb=self._get_memory_usage(),
                metadata={"bits": bits, "dimension": dimension}
            )
            results.append(result)
            print(f"  Encoded {self.config.num_vectors} vectors in {duration*1000:.1f}ms ({result.throughput:.0f} vectors/sec)")
        
        self.results.extend(results)
        return results[0]
    
    def benchmark_decode_throughput(self) -> BenchmarkResult:
        """Benchmark decoding throughput."""
        print("\n=== Decode Throughput Benchmark ===")
        
        dimension = 384
        vectors = np.random.randn(self.config.num_vectors, dimension).astype(np.float32)
        quantized = [quantize_packed(v, bits=6) for v in vectors]
        
        start = time.time()
        decoded = [dequantize_packed(q) for q in quantized]
        duration = time.time() - start
        
        result = BenchmarkResult(
            name=f"decode_{self.config.num_vectors}_vectors",
            duration_ms=duration * 1000,
            throughput=self.config.num_vectors / duration,
            memory_mb=self._get_memory_usage(),
            metadata={"dimension": dimension}
        )
        self.results.append(result)
        print(f"  Decoded {self.config.num_vectors} vectors in {duration*1000:.1f}ms ({result.throughput:.0f} vectors/sec)")
        
        return result
    
    def benchmark_query_latency(self) -> BenchmarkResult:
        """Benchmark query latency."""
        print("\n=== Query Latency Benchmark ===")
        
        root = self._temp_dir or self.setup()
        tm = TurboMemory(root=root, model_name="all-MiniLM-L6-v2")
        
        # Add test data
        for i in range(min(1000, self.config.num_vectors)):
            tm.add_memory("test", f"Test document number {i} with some content", confidence=0.8)
        
        latencies = []
        queries = [f"query {i}" for i in range(self.config.num_queries)]
        
        start_mem = self._get_memory_usage()
        
        for query in queries:
            start = time.time()
            tm.query(query, k=self.config.k)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        result = BenchmarkResult(
            name="query_latency",
            duration_ms=avg_latency,
            throughput=1000 / avg_latency,
            memory_mb=self._get_memory_usage() - start_mem,
            metadata={
                "avg_ms": avg_latency,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "num_queries": self.config.num_queries,
            }
        )
        
        print(f"  Avg: {avg_latency:.2f}ms, P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")
        
        self.results.append(result)
        tm.close()
        
        return result
    
    def benchmark_batch_operations(self) -> BenchmarkResult:
        """Benchmark batch add operations."""
        print("\n=== Batch Operations Benchmark ===")
        
        results = []
        
        for batch_size in self.config.batch_sizes:
            root = self._temp_dir or self.setup()
            tm = TurboMemory(root=root, model_name="all-MiniLM-L6-v2")
            
            texts = [f"Batch document {i}" for i in range(batch_size)]
            
            start = time.time()
            for text in texts:
                tm.add_memory("batch", text, confidence=0.8)
            duration = time.time() - start
            
            result = BenchmarkResult(
                name=f"batch_add_{batch_size}",
                duration_ms=duration * 1000,
                throughput=batch_size / duration,
                memory_mb=self._get_memory_usage(),
                metadata={"batch_size": batch_size}
            )
            results.append(result)
            print(f"  Batch {batch_size}: {duration*1000:.1f}ms ({result.throughput:.1f} docs/sec)")
            
            tm.close()
        
        self.results.extend(results)
        return results[0]
    
    def benchmark_storage_size(self) -> BenchmarkResult:
        """Benchmark storage size across configurations."""
        print("\n=== Storage Size Benchmark ===")
        
        results = []
        
        for bits in self.config.bit_depths:
            root = self._temp_dir or self.setup()
            tm = TurboMemory(root=root, model_name="all-MiniLM-L6-v2", config=TurboMemoryConfig(default_bits=bits))
            
            # Add vectors
            for i in range(1000):
                tm.add_memory("storage", f"Document {i} with content", confidence=0.8)
            
            # Measure storage
            total_size = 0
            for dirpath, _, filenames in os.walk(root):
                for f in filenames:
                    total_size += os.path.getsize(os.path.join(dirpath, f))
            
            # Calculate compression vs float32
            float32_size = 1000 * 384 * 4
            compression_ratio = float32_size / total_size
            
            result = BenchmarkResult(
                name=f"storage_{bits}bit",
                duration_ms=0,
                throughput=0,
                memory_mb=total_size / (1024 * 1024),
                metadata={
                    "bits": bits,
                    "total_bytes": total_size,
                    "compression_ratio": compression_ratio,
                    "vector_count": 1000,
                }
            )
            results.append(result)
            print(f"  {bits}-bit: {total_size/1024:.1f}KB ({compression_ratio:.1f}x vs float32)")
            
            tm.close()
        
        self.results.extend(results)
        return results[0]
    
    def benchmark_load_time(self) -> BenchmarkResult:
        """Benchmark time to load copied TMF store."""
        print("\n=== Load Time Benchmark ===")
        
        # Create and populate a store
        root = self._temp_dir or self.setup()
        tm = TurboMemory(root=root, model_name="all-MiniLM-L6-v2")
        
        for i in range(1000):
            tm.add_memory("load_test", f"Document {i}", confidence=0.8)
        
        tm.close()
        
        # Measure load time
        start = time.time()
        tm2 = TurboMemory(root=root, model_name="all-MiniLM-L6-v2")
        load_time = (time.time() - start) * 1000
        
        result = BenchmarkResult(
            name="load_time",
            duration_ms=load_time,
            throughput=0,
            memory_mb=0,
            metadata={
                "vectors": 1000,
                "load_time_ms": load_time,
            }
        )
        
        print(f"  Loaded 1000 vectors in {load_time:.1f}ms")
        
        self.results.append(result)
        tm2.close()
        
        return result
    
    def benchmark_recall_quality(self) -> BenchmarkResult:
        """Benchmark recall quality vs compression."""
        print("\n=== Recall Quality Benchmark ===")
        
        dimension = 384
        
        # Generate test vectors
        np.random.seed(42)
        base_vectors = np.random.randn(500, dimension).astype(np.float32)
        query_vectors = np.random.randn(10, dimension).astype(np.float32)
        
        results = []
        
        for bits in self.config.bit_depths:
            # Quantize base vectors
            quantized = [quantize_packed(v, bits=bits) for v in base_vectors]
            decoded = np.array([dequantize_packed(q) for q in quantized])
            
            # Calculate average cosine similarity between original and decoded
            similarities = []
            for orig, recon in zip(base_vectors, decoded):
                orig_norm = orig / (np.linalg.norm(orig) + 1e-9)
                recon_norm = recon / (np.linalg.norm(recon) + 1e-9)
                similarities.append(np.dot(orig_norm, recon_norm))
            
            avg_sim = np.mean(similarities)
            
            result = BenchmarkResult(
                name=f"recall_{bits}bit",
                duration_ms=0,
                throughput=0,
                memory_mb=0,
                accuracy=avg_sim,
                metadata={
                    "bits": bits,
                    "avg_cosine_similarity": avg_sim,
                    "dimension": dimension,
                }
            )
            results.append(result)
            print(f"  {bits}-bit: {avg_sim:.4f} avg cosine similarity")
        
        self.results.extend(results)
        return results[0]
    
    def save_results(self, path: str) -> None:
        """Save benchmark results to JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "results": [
                {
                    "name": r.name,
                    "duration_ms": r.duration_ms,
                    "throughput": r.throughput,
                    "memory_mb": r.memory_mb,
                    "accuracy": r.accuracy,
                    "metadata": r.metadata,
                }
                for r in self.results
            ]
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {path}")
    
    def print_summary(self) -> None:
        """Print summary of all benchmarks."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        for result in self.results:
            print(f"\n{result.name}:")
            print(f"  Duration: {result.duration_ms:.2f}ms")
            print(f"  Throughput: {result.throughput:.2f}")
            print(f"  Memory: {result.memory_mb:.2f}MB")
            if result.accuracy:
                print(f"  Accuracy: {result.accuracy:.4f}")
            if result.metadata:
                for k, v in result.metadata.items():
                    print(f"  {k}: {v}")


def run_default_benchmarks() -> List[BenchmarkResult]:
    """Run default benchmark suite."""
    config = BenchmarkConfig(
        num_vectors=5000,
        dimensions=[384],
        bit_depths=[4, 6, 8],
        num_queries=50,
        k=10,
        batch_sizes=[1, 10, 50],
    )
    
    bench = MemoryBenchmark(config)
    results = bench.run_all()
    bench.print_summary()
    
    return results


if __name__ == "__main__":
    results = run_default_benchmarks()