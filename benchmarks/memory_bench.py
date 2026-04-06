"""Memory operation benchmarks for TurboMemory."""

import time
import os
import shutil
import tempfile
import statistics
from typing import Dict, Any, List
from turbomemory import TurboMemory


class MemoryBenchmark:
    """Benchmark memory operations: add, query, consolidate."""

    def __init__(self, root: str = None, model_name: str = "all-MiniLM-L6-v2"):
        self.root = root or tempfile.mkdtemp(prefix="turbomemory_bench_")
        self.model_name = model_name
        self.tm = None

    def setup(self):
        """Initialize TurboMemory."""
        self.tm = TurboMemory(root=self.root, model_name=self.model_name)

    def teardown(self):
        """Clean up."""
        if self.tm:
            self.tm.close()
        if os.path.exists(self.root):
            shutil.rmtree(self.root, ignore_errors=True)

    def benchmark_add_memory(self, n: int = 100, bits: int = 6) -> Dict[str, float]:
        """Benchmark adding n memory chunks."""
        self.setup()
        try:
            times = []
            for i in range(n):
                topic = f"topic_{i % 10}"
                text = f"Memory chunk {i}: This is a test memory about topic {topic} with some specific details and numbers like 42 and 3.14."
                
                start = time.perf_counter()
                self.tm.add_memory(topic, text, bits=bits)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            return {
                "operation": "add_memory",
                "count": n,
                "total_time": sum(times),
                "avg_time": statistics.mean(times),
                "p50": statistics.median(times),
                "p95": sorted(times)[int(n * 0.95)],
                "p99": sorted(times)[int(n * 0.99)],
                "ops_per_sec": n / sum(times),
            }
        finally:
            self.teardown()

    def benchmark_query(self, n_add: int = 100, n_query: int = 20, k: int = 5) -> Dict[str, float]:
        """Benchmark query performance."""
        self.setup()
        try:
            # Add memories first
            for i in range(n_add):
                topic = f"topic_{i % 10}"
                text = f"Memory chunk {i}: This is a test memory about topic {topic} with specific details."
                self.tm.add_memory(topic, text)

            # Benchmark queries
            times = []
            for i in range(n_query):
                query = f"Tell me about topic_{i % 10}"
                
                start = time.perf_counter()
                self.tm.query(query, k=k)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            return {
                "operation": "query",
                "memories": n_add,
                "queries": n_query,
                "k": k,
                "total_time": sum(times),
                "avg_time": statistics.mean(times),
                "p50": statistics.median(times),
                "p95": sorted(times)[int(n_query * 0.95)] if n_query > 1 else times[0],
                "p99": sorted(times)[int(n_query * 0.99)] if n_query > 1 else times[0],
                "queries_per_sec": n_query / sum(times),
            }
        finally:
            self.teardown()

    def benchmark_consolidation(self, n: int = 200) -> Dict[str, float]:
        """Benchmark consolidation performance."""
        self.setup()
        try:
            # Add memories with some duplicates
            for i in range(n):
                topic = f"topic_{i % 5}"
                text = f"Memory {i}: Content about {topic}"
                self.tm.add_memory(topic, text)

            # Benchmark consolidation
            start = time.perf_counter()
            from consolidator import run_once
            import argparse
            args = argparse.Namespace(
                threshold=0.93, min_entropy=0.10, staleness_prune=0.90,
                max_chunks=300, merge_threshold=0.85, make_absolute=True,
            )
            run_once(self.tm, args)
            elapsed = time.perf_counter() - start

            stats = self.tm.stats()

            return {
                "operation": "consolidation",
                "initial_chunks": n,
                "final_chunks": stats.get("total_chunks", 0),
                "removed": n - stats.get("total_chunks", 0),
                "total_time": elapsed,
            }
        finally:
            self.teardown()

    def run_all(self) -> Dict[str, Any]:
        """Run all benchmarks."""
        results = {
            "add_memory": self.benchmark_add_memory(),
            "query": self.benchmark_query(),
            "consolidation": self.benchmark_consolidation(),
        }
        return results
