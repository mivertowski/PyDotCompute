#!/usr/bin/env python3
"""
Metal Backend Benchmark Suite for PyDotCompute.

Benchmarks Metal (MLX) backend performance against CPU baseline.
Mirrors the existing CUDA benchmarks for comparison.

Usage:
    python benchmarks/metal_benchmark.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np


def check_metal_available() -> bool:
    """Check if Metal/MLX is available."""
    if sys.platform != "darwin":
        return False
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except ImportError:
        return False


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    operation: str
    size: int | tuple[int, ...]
    cpu_time_ms: float
    metal_time_ms: float
    speedup: float


class MetalBenchmark:
    """Benchmark suite for Metal backend."""

    def __init__(self) -> None:
        self.metal_available = check_metal_available()
        self.results: list[BenchmarkResult] = []

        if self.metal_available:
            from pydotcompute.backends.cpu import CPUBackend
            from pydotcompute.backends.metal import MetalBackend

            self.cpu_backend = CPUBackend()
            self.metal_backend = MetalBackend()

    def _warmup(self, func: callable, *args, n_warmup: int = 3) -> None:  # type: ignore[type-arg]
        """Warmup to exclude JIT compilation time."""
        for _ in range(n_warmup):
            func(*args)

    def benchmark_memory_copy(self, sizes: list[int] | None = None) -> list[BenchmarkResult]:
        """Benchmark memory copy operations."""
        if not self.metal_available:
            print("Metal not available, skipping memory benchmarks")
            return []

        if sizes is None:
            sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

        print("\n=== Memory Copy Benchmark ===")
        results = []

        for size in sizes:
            # Create test data
            data = np.random.randn(size).astype(np.float32)

            # Warmup
            _ = self.metal_backend.copy_to_device(data)
            self.metal_backend.synchronize()

            # CPU baseline (just a copy)
            start = time.perf_counter()
            for _ in range(10):
                _ = data.copy()
            cpu_time = (time.perf_counter() - start) / 10 * 1000

            # Metal copy
            start = time.perf_counter()
            for _ in range(10):
                device_arr = self.metal_backend.copy_to_device(data)
                self.metal_backend.synchronize()
            metal_time = (time.perf_counter() - start) / 10 * 1000

            speedup = cpu_time / metal_time if metal_time > 0 else 0

            result = BenchmarkResult(
                operation="memory_copy",
                size=size,
                cpu_time_ms=cpu_time,
                metal_time_ms=metal_time,
                speedup=speedup,
            )
            results.append(result)
            self.results.append(result)

            size_mb = size * 4 / 1024 / 1024
            print(
                f"  Size: {size:>10,} ({size_mb:.1f}MB) | "
                f"CPU: {cpu_time:.3f}ms | Metal: {metal_time:.3f}ms | "
                f"Speedup: {speedup:.2f}x"
            )

        return results

    def benchmark_vector_add(self, sizes: list[int] | None = None) -> list[BenchmarkResult]:
        """Benchmark vector addition."""
        if not self.metal_available:
            print("Metal not available, skipping vector add benchmarks")
            return []

        if sizes is None:
            sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

        print("\n=== Vector Addition Benchmark ===")
        results = []

        import mlx.core as mx

        for size in sizes:
            # Create test data
            a_np = np.random.randn(size).astype(np.float32)
            b_np = np.random.randn(size).astype(np.float32)

            # CPU baseline
            start = time.perf_counter()
            for _ in range(100):
                _ = a_np + b_np
            cpu_time = (time.perf_counter() - start) / 100 * 1000

            # Metal
            a_mx = mx.array(a_np)
            b_mx = mx.array(b_np)

            # Warmup
            c = mx.add(a_mx, b_mx)
            mx.eval(c)

            start = time.perf_counter()
            for _ in range(100):
                c = mx.add(a_mx, b_mx)
                mx.eval(c)
            metal_time = (time.perf_counter() - start) / 100 * 1000

            speedup = cpu_time / metal_time if metal_time > 0 else 0

            result = BenchmarkResult(
                operation="vector_add",
                size=size,
                cpu_time_ms=cpu_time,
                metal_time_ms=metal_time,
                speedup=speedup,
            )
            results.append(result)
            self.results.append(result)

            print(
                f"  Size: {size:>10,} | "
                f"CPU: {cpu_time:.4f}ms | Metal: {metal_time:.4f}ms | "
                f"Speedup: {speedup:.2f}x"
            )

        return results

    def benchmark_matrix_multiply(
        self, sizes: list[int] | None = None
    ) -> list[BenchmarkResult]:
        """Benchmark matrix multiplication."""
        if not self.metal_available:
            print("Metal not available, skipping matmul benchmarks")
            return []

        if sizes is None:
            sizes = [64, 128, 256, 512, 1024, 2048]

        print("\n=== Matrix Multiplication Benchmark ===")
        results = []

        import mlx.core as mx

        for size in sizes:
            # Create test data
            a_np = np.random.randn(size, size).astype(np.float32)
            b_np = np.random.randn(size, size).astype(np.float32)

            n_iterations = max(1, 100 // (size // 64))

            # CPU baseline
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = np.matmul(a_np, b_np)
            cpu_time = (time.perf_counter() - start) / n_iterations * 1000

            # Metal
            a_mx = mx.array(a_np)
            b_mx = mx.array(b_np)

            # Warmup
            c = mx.matmul(a_mx, b_mx)
            mx.eval(c)

            start = time.perf_counter()
            for _ in range(n_iterations):
                c = mx.matmul(a_mx, b_mx)
                mx.eval(c)
            metal_time = (time.perf_counter() - start) / n_iterations * 1000

            speedup = cpu_time / metal_time if metal_time > 0 else 0

            result = BenchmarkResult(
                operation="matmul",
                size=(size, size),
                cpu_time_ms=cpu_time,
                metal_time_ms=metal_time,
                speedup=speedup,
            )
            results.append(result)
            self.results.append(result)

            # Calculate GFLOPS
            flops = 2 * size**3  # matmul is roughly 2*N^3 operations
            cpu_gflops = flops / (cpu_time / 1000) / 1e9
            metal_gflops = flops / (metal_time / 1000) / 1e9

            print(
                f"  Size: {size:>4}x{size:<4} | "
                f"CPU: {cpu_time:.3f}ms ({cpu_gflops:.1f} GFLOPS) | "
                f"Metal: {metal_time:.3f}ms ({metal_gflops:.1f} GFLOPS) | "
                f"Speedup: {speedup:.2f}x"
            )

        return results

    def benchmark_elementwise_ops(self) -> list[BenchmarkResult]:
        """Benchmark various elementwise operations."""
        if not self.metal_available:
            print("Metal not available, skipping elementwise benchmarks")
            return []

        print("\n=== Elementwise Operations Benchmark ===")
        results = []

        import mlx.core as mx

        size = 1_000_000
        operations = {
            "exp": (np.exp, mx.exp),
            "log": (lambda x: np.log(np.abs(x) + 1e-7), lambda x: mx.log(mx.abs(x) + 1e-7)),
            "sqrt": (lambda x: np.sqrt(np.abs(x)), lambda x: mx.sqrt(mx.abs(x))),
            "sin": (np.sin, mx.sin),
            "cos": (np.cos, mx.cos),
        }

        data_np = np.random.randn(size).astype(np.float32) + 2.0
        data_mx = mx.array(data_np)

        for op_name, (np_op, mx_op) in operations.items():
            # CPU baseline
            start = time.perf_counter()
            for _ in range(100):
                _ = np_op(data_np)
            cpu_time = (time.perf_counter() - start) / 100 * 1000

            # Metal
            # Warmup
            r = mx_op(data_mx)
            mx.eval(r)

            start = time.perf_counter()
            for _ in range(100):
                r = mx_op(data_mx)
                mx.eval(r)
            metal_time = (time.perf_counter() - start) / 100 * 1000

            speedup = cpu_time / metal_time if metal_time > 0 else 0

            result = BenchmarkResult(
                operation=f"elementwise_{op_name}",
                size=size,
                cpu_time_ms=cpu_time,
                metal_time_ms=metal_time,
                speedup=speedup,
            )
            results.append(result)
            self.results.append(result)

            print(
                f"  {op_name:>8}: CPU: {cpu_time:.3f}ms | "
                f"Metal: {metal_time:.3f}ms | Speedup: {speedup:.2f}x"
            )

        return results

    def benchmark_reduction_ops(self) -> list[BenchmarkResult]:
        """Benchmark reduction operations (sum, mean, max, min)."""
        if not self.metal_available:
            print("Metal not available, skipping reduction benchmarks")
            return []

        print("\n=== Reduction Operations Benchmark ===")
        results = []

        import mlx.core as mx

        sizes = [10_000, 100_000, 1_000_000, 10_000_000]
        operations = {
            "sum": (np.sum, mx.sum),
            "mean": (np.mean, mx.mean),
            "max": (np.max, mx.max),
            "min": (np.min, mx.min),
        }

        for size in sizes:
            data_np = np.random.randn(size).astype(np.float32)
            data_mx = mx.array(data_np)

            for op_name, (np_op, mx_op) in operations.items():
                # CPU baseline
                start = time.perf_counter()
                for _ in range(100):
                    _ = np_op(data_np)
                cpu_time = (time.perf_counter() - start) / 100 * 1000

                # Metal
                r = mx_op(data_mx)
                mx.eval(r)

                start = time.perf_counter()
                for _ in range(100):
                    r = mx_op(data_mx)
                    mx.eval(r)
                metal_time = (time.perf_counter() - start) / 100 * 1000

                speedup = cpu_time / metal_time if metal_time > 0 else 0

                result = BenchmarkResult(
                    operation=f"reduce_{op_name}",
                    size=size,
                    cpu_time_ms=cpu_time,
                    metal_time_ms=metal_time,
                    speedup=speedup,
                )
                results.append(result)
                self.results.append(result)

            # Print one line per size with all ops
            print(f"  Size: {size:>10,}")

        return results

    def benchmark_backend_api(self) -> list[BenchmarkResult]:
        """Benchmark the MetalBackend API specifically."""
        if not self.metal_available:
            print("Metal not available, skipping backend API benchmarks")
            return []

        print("\n=== Backend API Benchmark ===")
        results = []

        # Test kernel execution through backend
        import mlx.core as mx

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        def compute_kernel(a: mx.array, b: mx.array) -> mx.array:
            """Sample compute kernel."""
            return mx.sqrt(mx.add(mx.square(a), mx.square(b)))

        sizes = [10_000, 100_000, 1_000_000]

        for size in sizes:
            a = backend.copy_to_device(np.random.randn(size).astype(np.float32))
            b = backend.copy_to_device(np.random.randn(size).astype(np.float32))

            # Warmup
            _ = backend.execute_kernel(compute_kernel, (1,), (1,), a, b)

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                result = backend.execute_kernel(compute_kernel, (1,), (1,), a, b)
            total_time = (time.perf_counter() - start) / 100 * 1000

            avg_execution_time = result.execution_time_ms

            print(
                f"  Size: {size:>10,} | "
                f"Total: {total_time:.3f}ms | "
                f"Kernel: {avg_execution_time:.3f}ms"
            )

        return results

    def run_all(self) -> None:
        """Run all benchmarks."""
        print("=" * 70)
        print("METAL BACKEND BENCHMARK SUITE")
        print("=" * 70)
        print(f"Date: {datetime.now()}")
        print(f"Platform: {sys.platform}")
        print(f"Metal Available: {self.metal_available}")

        if not self.metal_available:
            print("\nMetal is not available on this system.")
            print("Benchmarks require macOS with MLX installed.")
            print("Install with: pip install -e '.[metal]'")
            return

        # Get device info
        import mlx.core as mx

        print(f"\nMLX Metal Info:")
        # Use new API if available
        if hasattr(mx, "get_cache_memory"):
            cache_mem = mx.get_cache_memory()
            peak_mem = mx.get_peak_memory()
        else:
            cache_mem = mx.metal.get_cache_memory()
            peak_mem = mx.metal.get_peak_memory()
        print(f"  Cache Memory: {cache_mem / 1024 / 1024:.1f} MB")
        print(f"  Peak Memory: {peak_mem / 1024 / 1024:.1f} MB")

        # Run benchmarks
        self.benchmark_memory_copy()
        self.benchmark_vector_add()
        self.benchmark_matrix_multiply()
        self.benchmark_elementwise_ops()
        self.benchmark_reduction_ops()
        self.benchmark_backend_api()

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        # Find operations where Metal is faster/slower
        faster = [r for r in self.results if r.speedup > 1.0]
        slower = [r for r in self.results if r.speedup <= 1.0]

        if faster:
            best = max(faster, key=lambda r: r.speedup)
            print(
                f"Best speedup: {best.speedup:.2f}x for {best.operation} "
                f"(size={best.size})"
            )

        if slower:
            worst = min(slower, key=lambda r: r.speedup)
            print(
                f"Worst speedup: {worst.speedup:.2f}x for {worst.operation} "
                f"(size={worst.size})"
            )

        avg_speedup = (
            sum(r.speedup for r in self.results) / len(self.results)
            if self.results
            else 0
        )
        print(f"Average speedup: {avg_speedup:.2f}x")

        # Metal is typically best for large operations
        print("\nNotes:")
        print("- Metal excels at large matrix operations (1000x1000+)")
        print("- Small operations may have overhead from dispatch")
        print("- Unified memory eliminates explicit transfer costs")


if __name__ == "__main__":
    benchmark = MetalBenchmark()
    benchmark.run_all()
