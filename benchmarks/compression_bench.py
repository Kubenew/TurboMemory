"""Compression benchmarks for TurboMemory quantization."""

import numpy as np
import base64
from typing import Dict, Any, List
from turbomemory import quantize_packed, dequantize_packed, cosine_sim


class CompressionBenchmark:
    """Benchmark quantization compression across different bit levels."""

    def __init__(self, dimensions: List[int] = None):
        self.dimensions = dimensions or [384, 768, 1024, 1536, 3072]
        self.bits_levels = [4, 6, 8]

    def benchmark_single_vector(self, dim: int, bits: int, n_trials: int = 100) -> Dict[str, float]:
        """Benchmark compression for a single vector."""
        original_size = dim * 4  # float32
        similarities = []
        compressed_sizes = []

        for _ in range(n_trials):
            vec = np.random.randn(dim).astype(np.float32)
            q = quantize_packed(vec, bits=bits)
            reconstructed = dequantize_packed(q)

            sim = cosine_sim(vec, reconstructed)
            similarities.append(sim)
            compressed_sizes.append(len(base64.b64decode(q["data"])))

        return {
            "dimension": dim,
            "bits": bits,
            "original_bytes": original_size,
            "avg_compressed_bytes": np.mean(compressed_sizes),
            "compression_ratio": original_size / np.mean(compressed_sizes),
            "avg_similarity": np.mean(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities),
            "std_similarity": np.std(similarities),
        }

    def benchmark_all(self, n_trials: int = 100) -> Dict[str, Any]:
        """Run benchmarks across all dimensions and bit levels."""
        results = {}
        for dim in self.dimensions:
            results[f"dim_{dim}"] = {}
            for bits in self.bits_levels:
                result = self.benchmark_single_vector(dim, bits, n_trials)
                results[f"dim_{dim}"][f"{bits}bit"] = result
        return results

    def print_summary(self, results: Dict[str, Any] = None):
        """Print a formatted summary table."""
        if results is None:
            results = self.benchmark_all()

        print(f"{'Dimension':<12} {'Bits':<6} {'Original':<10} {'Compressed':<12} {'Ratio':<8} {'Avg Sim':<10} {'Min Sim':<10}")
        print("-" * 70)

        for dim_key, bits_results in results.items():
            for bits_key, result in bits_results.items():
                print(
                    f"{result['dimension']:<12} "
                    f"{result['bits']:<6} "
                    f"{result['original_bytes']:<10} "
                    f"{result['avg_compressed_bytes']:<12.0f} "
                    f"{result['compression_ratio']:<8.2f}x "
                    f"{result['avg_similarity']:<10.4f} "
                    f"{result['min_similarity']:<10.4f}"
                )
