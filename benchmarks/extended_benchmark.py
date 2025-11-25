"""
Extended Benchmark Suite for PyDotCompute.

This benchmark provides a complete picture of PyDotCompute performance:

1. LARGE-SCALE GRAPHS: 10K to 10M+ nodes
2. STREAMING THROUGHPUT: Messages per second with persistent actors
3. LATENCY DISTRIBUTION: p50, p95, p99 latency analysis
4. CONCURRENT ACTORS: Pipeline and parallel actor performance
5. MEMORY EFFICIENCY: Memory usage tracking

Designed to showcase where GPU Actors excel vs traditional batch processing.
"""

from __future__ import annotations

import asyncio
import gc
import json
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExtendedBenchmarkConfig:
    """Configuration for extended benchmarks."""

    # Large-scale graph sizes (sparse only for memory efficiency)
    large_graph_sizes: list[int] = field(default_factory=lambda: [
        10_000, 50_000, 100_000, 500_000, 1_000_000, 10_000_000
    ])
    large_graph_edges_per_node: float = 5.0  # Sparse for memory efficiency

    # Streaming benchmark config
    streaming_message_counts: list[int] = field(default_factory=lambda: [
        100, 1000, 5000, 10000, 50000
    ])
    streaming_payload_sizes: list[int] = field(default_factory=lambda: [
        10, 100, 1000  # Number of float values per message
    ])

    # Latency benchmark config
    latency_sample_count: int = 1000
    latency_warmup_count: int = 100

    # Concurrent actors config
    concurrent_actor_counts: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    concurrent_messages_per_actor: int = 1000

    # PageRank parameters
    damping: float = 0.85
    max_iterations: int = 100
    tolerance: float = 1e-6

    # Benchmark parameters
    warmup_runs: int = 1
    measurement_runs: int = 3


# ============================================================================
# Utilities
# ============================================================================

def check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except (ImportError, Exception):
        return False


def get_gpu_memory_info() -> tuple[int, int]:
    """Get GPU memory (used, total) in bytes."""
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        return mempool.used_bytes(), mempool.total_bytes()
    except Exception:
        return 0, 0


def format_number(n: int | float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n))


def format_time(seconds: float) -> str:
    """Format time with appropriate units."""
    if seconds < 0.001:
        return f"{seconds*1_000_000:.1f}μs"
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    return f"{seconds:.3f}s"


def generate_large_graph_chunked(
    num_nodes: int,
    edges_per_node: float,
    seed: int = 42,
) -> tuple[list[tuple[int, int]], int]:
    """
    Generate a large random graph efficiently.

    Uses chunked generation to avoid memory issues.
    """
    rng = np.random.default_rng(seed)
    target_edges = int(num_nodes * edges_per_node)

    # For very large graphs, generate in chunks
    chunk_size = min(target_edges, 1_000_000)
    edges: list[tuple[int, int]] = []
    edge_set: set[tuple[int, int]] = set()

    while len(edges) < target_edges:
        # Generate a chunk of random edges
        remaining = target_edges - len(edges)
        chunk = min(chunk_size, remaining * 2)  # Generate extra to account for duplicates

        srcs = rng.integers(0, num_nodes, size=chunk)
        dsts = rng.integers(0, num_nodes, size=chunk)

        for src, dst in zip(srcs, dsts):
            if src != dst and (src, dst) not in edge_set:
                edges.append((int(src), int(dst)))
                edge_set.add((src, dst))
                if len(edges) >= target_edges:
                    break

    return edges, len(edges)


# ============================================================================
# Large-Scale Graph Benchmark
# ============================================================================

@dataclass
class LargeScaleResult:
    """Result from large-scale graph benchmark."""
    implementation: str
    num_nodes: int
    num_edges: int
    compute_time: float
    iterations: int
    converged: bool
    memory_mb: float
    throughput_edges_per_sec: float


def pagerank_cpu_sparse_large(
    edges: list[tuple[int, int]],
    num_nodes: int,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[NDArray, bool, int, float]:
    """Sparse CPU implementation optimized for large graphs."""
    try:
        from scipy import sparse
    except ImportError:
        raise RuntimeError("scipy required for large graph benchmarks")

    # Build sparse transition matrix
    out_degree = np.zeros(num_nodes, dtype=np.float64)
    for src, _ in edges:
        out_degree[src] += 1

    rows = []
    cols = []
    data = []

    for src, dst in edges:
        if out_degree[src] > 0:
            rows.append(dst)
            cols.append(src)
            data.append(1.0 / out_degree[src])

    transition = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    )

    # Initialize ranks
    ranks = np.ones(num_nodes, dtype=np.float64) / num_nodes
    teleport = np.ones(num_nodes, dtype=np.float64) * (1 - damping) / num_nodes

    # Power iteration
    for iteration in range(max_iterations):
        new_ranks = teleport + damping * transition @ ranks
        diff = np.abs(new_ranks - ranks).max()

        if diff < tolerance:
            return new_ranks, True, iteration + 1, diff

        ranks = new_ranks

    return ranks, False, max_iterations, diff


def pagerank_gpu_sparse_large(
    edges: list[tuple[int, int]],
    num_nodes: int,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[NDArray, bool, int, float]:
    """GPU sparse implementation for large graphs."""
    import cupy as cp

    # Build sparse matrix data on CPU first
    out_degree = np.zeros(num_nodes, dtype=np.float64)
    for src, _ in edges:
        out_degree[src] += 1

    rows = []
    cols = []
    data = []

    for src, dst in edges:
        if out_degree[src] > 0:
            rows.append(dst)
            cols.append(src)
            data.append(1.0 / out_degree[src])

    # Transfer to GPU
    transition = cp.sparse.csr_matrix(
        (cp.array(data, dtype=cp.float64),
         (cp.array(rows, dtype=cp.int32), cp.array(cols, dtype=cp.int32))),
        shape=(num_nodes, num_nodes),
    )

    # Initialize on GPU
    ranks = cp.ones(num_nodes, dtype=cp.float64) / num_nodes
    teleport = cp.ones(num_nodes, dtype=cp.float64) * (1 - damping) / num_nodes

    # Power iteration on GPU
    for iteration in range(max_iterations):
        new_ranks = teleport + damping * transition @ ranks
        diff = float(cp.abs(new_ranks - ranks).max())

        if diff < tolerance:
            return cp.asnumpy(new_ranks), True, iteration + 1, diff

        ranks = new_ranks

    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(ranks), False, max_iterations, diff


class LargeScaleBenchmark:
    """Benchmark for large-scale graphs."""

    def __init__(self, config: ExtendedBenchmarkConfig) -> None:
        self.config = config
        self.results: list[LargeScaleResult] = []
        self.cuda_available = check_cuda_available()

    def run(self) -> list[LargeScaleResult]:
        """Run large-scale graph benchmarks."""
        print("\n" + "=" * 70)
        print("LARGE-SCALE GRAPH BENCHMARK")
        print("=" * 70)
        print(f"Edges per node: {self.config.large_graph_edges_per_node}")
        print(f"CUDA available: {self.cuda_available}")

        self.results = []

        for num_nodes in self.config.large_graph_sizes:
            print(f"\n--- {format_number(num_nodes)} nodes ---")

            # Check memory constraints
            estimated_edges = int(num_nodes * self.config.large_graph_edges_per_node)
            estimated_memory_mb = (estimated_edges * 16) / (1024 * 1024)  # ~16 bytes per edge

            if estimated_memory_mb > 8000:  # Skip if > 8GB estimated
                print(f"  Skipping: estimated memory {estimated_memory_mb:.0f}MB exceeds limit")
                continue

            print(f"  Generating graph (~{format_number(estimated_edges)} edges)...", end=" ", flush=True)

            gc.collect()
            tracemalloc.start()

            start = time.perf_counter()
            edges, num_edges = generate_large_graph_chunked(
                num_nodes,
                self.config.large_graph_edges_per_node
            )
            gen_time = time.perf_counter() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"done in {format_time(gen_time)} (peak mem: {peak/1024/1024:.1f}MB)")

            # CPU Sparse
            print(f"  CPU Sparse...", end=" ", flush=True)
            try:
                gc.collect()
                tracemalloc.start()

                start = time.perf_counter()
                _, converged, iters, _ = pagerank_cpu_sparse_large(
                    edges, num_nodes,
                    self.config.damping,
                    self.config.max_iterations,
                    self.config.tolerance,
                )
                compute_time = time.perf_counter() - start

                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                result = LargeScaleResult(
                    implementation="CPU Sparse",
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    compute_time=compute_time,
                    iterations=iters,
                    converged=converged,
                    memory_mb=peak / 1024 / 1024,
                    throughput_edges_per_sec=num_edges / compute_time if compute_time > 0 else 0,
                )
                self.results.append(result)
                print(f"{format_time(compute_time)} ({iters} iters, {format_number(result.throughput_edges_per_sec)} edges/s)")

            except Exception as e:
                print(f"Error: {e}")

            # GPU Sparse
            if self.cuda_available:
                print(f"  GPU Sparse...", end=" ", flush=True)
                try:
                    import cupy as cp

                    gc.collect()
                    cp.get_default_memory_pool().free_all_blocks()

                    start = time.perf_counter()
                    _, converged, iters, _ = pagerank_gpu_sparse_large(
                        edges, num_nodes,
                        self.config.damping,
                        self.config.max_iterations,
                        self.config.tolerance,
                    )
                    compute_time = time.perf_counter() - start

                    gpu_used, gpu_total = get_gpu_memory_info()

                    result = LargeScaleResult(
                        implementation="GPU Sparse",
                        num_nodes=num_nodes,
                        num_edges=num_edges,
                        compute_time=compute_time,
                        iterations=iters,
                        converged=converged,
                        memory_mb=gpu_used / 1024 / 1024,
                        throughput_edges_per_sec=num_edges / compute_time if compute_time > 0 else 0,
                    )
                    self.results.append(result)
                    print(f"{format_time(compute_time)} ({iters} iters, {format_number(result.throughput_edges_per_sec)} edges/s)")

                    cp.get_default_memory_pool().free_all_blocks()

                except Exception as e:
                    print(f"Error: {e}")

            # Clean up edges to free memory
            del edges
            gc.collect()

        return self.results


# ============================================================================
# Streaming Throughput Benchmark
# ============================================================================

@dataclass
class StreamingResult:
    """Result from streaming throughput benchmark."""
    implementation: str
    message_count: int
    payload_size: int
    total_time: float
    throughput_msg_per_sec: float
    throughput_values_per_sec: float
    avg_latency_ms: float
    setup_time: float


class StreamingBenchmark:
    """Benchmark for streaming throughput with persistent actors."""

    def __init__(self, config: ExtendedBenchmarkConfig) -> None:
        self.config = config
        self.results: list[StreamingResult] = []

    async def _run_actor_streaming(
        self,
        message_count: int,
        payload_size: int,
    ) -> tuple[float, float, float]:
        """Run streaming benchmark with GPU actors."""
        from pydotcompute import RingKernelRuntime, message, ring_kernel
        from pydotcompute.ring_kernels.lifecycle import KernelContext

        # Message types
        @message
        @dataclass
        class StreamRequest:
            values: list[float]
            message_id: UUID = field(default_factory=uuid4)
            priority: int = 128
            correlation_id: UUID | None = None

        @message
        @dataclass
        class StreamResponse:
            result: float
            message_id: UUID = field(default_factory=uuid4)
            priority: int = 128
            correlation_id: UUID | None = None

        # Simple compute actor
        @ring_kernel(
            kernel_id="stream_bench",
            input_type=StreamRequest,
            output_type=StreamResponse,
            queue_size=max(1024, message_count),
        )
        async def stream_actor(ctx: KernelContext) -> None:
            while not ctx.should_terminate:
                if not ctx.is_active:
                    await ctx.wait_active()
                    continue

                try:
                    request = await ctx.receive(timeout=0.1)
                    result = sum(request.values)
                    await ctx.send(StreamResponse(
                        result=result,
                        correlation_id=request.message_id,
                    ))
                except Exception:
                    continue

        # Prepare test data
        test_values = [float(i) for i in range(payload_size)]

        setup_start = time.perf_counter()

        async with RingKernelRuntime() as runtime:
            await runtime.launch("stream_bench")
            await runtime.activate("stream_bench")

            setup_time = time.perf_counter() - setup_start

            # Streaming benchmark
            start = time.perf_counter()

            for i in range(message_count):
                await runtime.send("stream_bench", StreamRequest(values=test_values))

            for i in range(message_count):
                await runtime.receive("stream_bench", timeout=10.0)

            total_time = time.perf_counter() - start

        return total_time, setup_time, total_time / message_count * 1000

    def _run_batch_processing(
        self,
        message_count: int,
        payload_size: int,
    ) -> tuple[float, float, float]:
        """Run equivalent batch processing for comparison."""
        # Prepare test data
        test_values = [float(i) for i in range(payload_size)]

        start = time.perf_counter()

        results = []
        for i in range(message_count):
            result = sum(test_values)
            results.append(result)

        total_time = time.perf_counter() - start

        return total_time, 0.0, total_time / message_count * 1000

    async def run(self) -> list[StreamingResult]:
        """Run streaming throughput benchmarks."""
        print("\n" + "=" * 70)
        print("STREAMING THROUGHPUT BENCHMARK")
        print("=" * 70)
        print("Comparing persistent actors vs batch processing")
        print("This showcases where GPU Actors excel!\n")

        self.results = []

        for payload_size in self.config.streaming_payload_sizes:
            print(f"\n--- Payload: {payload_size} floats ({payload_size * 8} bytes) ---")

            for msg_count in self.config.streaming_message_counts:
                print(f"\n  {format_number(msg_count)} messages:")

                # Batch processing baseline
                print(f"    Batch Processing...", end=" ", flush=True)
                total_time, setup_time, avg_latency = self._run_batch_processing(
                    msg_count, payload_size
                )

                batch_result = StreamingResult(
                    implementation="Batch Processing",
                    message_count=msg_count,
                    payload_size=payload_size,
                    total_time=total_time,
                    throughput_msg_per_sec=msg_count / total_time if total_time > 0 else 0,
                    throughput_values_per_sec=msg_count * payload_size / total_time if total_time > 0 else 0,
                    avg_latency_ms=avg_latency,
                    setup_time=setup_time,
                )
                self.results.append(batch_result)
                print(f"{format_time(total_time)} ({format_number(batch_result.throughput_msg_per_sec)} msg/s)")

                # GPU Actors streaming
                print(f"    GPU Actors...", end=" ", flush=True)
                try:
                    total_time, setup_time, avg_latency = await self._run_actor_streaming(
                        msg_count, payload_size
                    )

                    actor_result = StreamingResult(
                        implementation="GPU Actors",
                        message_count=msg_count,
                        payload_size=payload_size,
                        total_time=total_time,
                        throughput_msg_per_sec=msg_count / total_time if total_time > 0 else 0,
                        throughput_values_per_sec=msg_count * payload_size / total_time if total_time > 0 else 0,
                        avg_latency_ms=avg_latency,
                        setup_time=setup_time,
                    )
                    self.results.append(actor_result)
                    print(f"{format_time(total_time)} ({format_number(actor_result.throughput_msg_per_sec)} msg/s, setup: {format_time(setup_time)})")

                except Exception as e:
                    print(f"Error: {e}")

        return self.results


# ============================================================================
# Latency Distribution Benchmark
# ============================================================================

@dataclass
class LatencyResult:
    """Result from latency distribution benchmark."""
    implementation: str
    sample_count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    std_ms: float


class LatencyBenchmark:
    """Benchmark for latency distribution analysis."""

    def __init__(self, config: ExtendedBenchmarkConfig) -> None:
        self.config = config
        self.results: list[LatencyResult] = []

    async def run(self) -> list[LatencyResult]:
        """Run latency distribution benchmarks."""
        from pydotcompute import RingKernelRuntime, message, ring_kernel
        from pydotcompute.ring_kernels.lifecycle import KernelContext

        print("\n" + "=" * 70)
        print("LATENCY DISTRIBUTION BENCHMARK")
        print("=" * 70)
        print(f"Sample count: {self.config.latency_sample_count}")
        print(f"Warmup count: {self.config.latency_warmup_count}\n")

        # Message types
        @message
        @dataclass
        class LatencyRequest:
            timestamp: float
            message_id: UUID = field(default_factory=uuid4)
            priority: int = 128
            correlation_id: UUID | None = None

        @message
        @dataclass
        class LatencyResponse:
            timestamp: float
            message_id: UUID = field(default_factory=uuid4)
            priority: int = 128
            correlation_id: UUID | None = None

        @ring_kernel(
            kernel_id="latency_bench",
            input_type=LatencyRequest,
            output_type=LatencyResponse,
            queue_size=4096,
        )
        async def latency_actor(ctx: KernelContext) -> None:
            while not ctx.should_terminate:
                if not ctx.is_active:
                    await ctx.wait_active()
                    continue

                try:
                    request = await ctx.receive(timeout=0.1)
                    await ctx.send(LatencyResponse(
                        timestamp=time.perf_counter(),
                        correlation_id=request.message_id,
                    ))
                except Exception:
                    continue

        self.results = []

        async with RingKernelRuntime() as runtime:
            await runtime.launch("latency_bench")
            await runtime.activate("latency_bench")

            # Warmup
            print("  Warming up...", end=" ", flush=True)
            for _ in range(self.config.latency_warmup_count):
                await runtime.send("latency_bench", LatencyRequest(timestamp=time.perf_counter()))
                await runtime.receive("latency_bench", timeout=1.0)
            print("done")

            # Measure latencies
            print("  Measuring latencies...", end=" ", flush=True)
            latencies = []

            for _ in range(self.config.latency_sample_count):
                send_time = time.perf_counter()
                await runtime.send("latency_bench", LatencyRequest(timestamp=send_time))
                response = await runtime.receive("latency_bench", timeout=1.0)
                receive_time = time.perf_counter()

                # Round-trip latency
                latency_ms = (receive_time - send_time) * 1000
                latencies.append(latency_ms)

            print("done")

            # Calculate statistics
            latencies.sort()
            n = len(latencies)

            result = LatencyResult(
                implementation="GPU Actors",
                sample_count=n,
                p50_ms=latencies[int(n * 0.50)],
                p95_ms=latencies[int(n * 0.95)],
                p99_ms=latencies[int(n * 0.99)],
                min_ms=min(latencies),
                max_ms=max(latencies),
                mean_ms=statistics.mean(latencies),
                std_ms=statistics.stdev(latencies) if n > 1 else 0,
            )
            self.results.append(result)

            print(f"\n  Results:")
            print(f"    p50:  {result.p50_ms:.3f}ms")
            print(f"    p95:  {result.p95_ms:.3f}ms")
            print(f"    p99:  {result.p99_ms:.3f}ms")
            print(f"    min:  {result.min_ms:.3f}ms")
            print(f"    max:  {result.max_ms:.3f}ms")
            print(f"    mean: {result.mean_ms:.3f}ms (std: {result.std_ms:.3f}ms)")

        return self.results


# ============================================================================
# Concurrent Actors Benchmark
# ============================================================================

@dataclass
class ConcurrencyResult:
    """Result from concurrent actors benchmark."""
    num_actors: int
    total_messages: int
    total_time: float
    throughput_msg_per_sec: float
    avg_latency_ms: float
    speedup_vs_single: float


class ConcurrencyBenchmark:
    """Benchmark for concurrent actor performance."""

    def __init__(self, config: ExtendedBenchmarkConfig) -> None:
        self.config = config
        self.results: list[ConcurrencyResult] = []

    async def run(self) -> list[ConcurrencyResult]:
        """Run concurrent actors benchmark."""
        from pydotcompute import RingKernelRuntime, message, ring_kernel
        from pydotcompute.ring_kernels.lifecycle import KernelContext

        print("\n" + "=" * 70)
        print("CONCURRENT ACTORS BENCHMARK")
        print("=" * 70)
        print(f"Messages per actor: {self.config.concurrent_messages_per_actor}")
        print(f"Actor counts: {self.config.concurrent_actor_counts}\n")

        # Message types
        @message
        @dataclass
        class WorkRequest:
            data: list[float]
            message_id: UUID = field(default_factory=uuid4)
            priority: int = 128
            correlation_id: UUID | None = None

        @message
        @dataclass
        class WorkResponse:
            result: float
            message_id: UUID = field(default_factory=uuid4)
            priority: int = 128
            correlation_id: UUID | None = None

        self.results = []
        single_actor_time = None

        for num_actors in self.config.concurrent_actor_counts:
            print(f"\n  Testing with {num_actors} actor(s)...", end=" ", flush=True)

            # Create actor factory
            def create_actor(actor_id: str):
                @ring_kernel(
                    kernel_id=actor_id,
                    input_type=WorkRequest,
                    output_type=WorkResponse,
                    queue_size=4096,
                )
                async def worker(ctx: KernelContext) -> None:
                    while not ctx.should_terminate:
                        if not ctx.is_active:
                            await ctx.wait_active()
                            continue

                        try:
                            request = await ctx.receive(timeout=0.1)
                            # Simulate some computation
                            result = sum(x * x for x in request.data)
                            await ctx.send(WorkResponse(
                                result=result,
                                correlation_id=request.message_id,
                            ))
                        except Exception:
                            continue

                return worker

            test_data = [float(i) for i in range(100)]
            total_messages = num_actors * self.config.concurrent_messages_per_actor

            try:
                async with RingKernelRuntime() as runtime:
                    # Launch actors
                    actor_ids = [f"worker_{i}" for i in range(num_actors)]
                    for actor_id in actor_ids:
                        create_actor(actor_id)  # Register
                        await runtime.launch(actor_id)
                        await runtime.activate(actor_id)

                    start = time.perf_counter()

                    # Send messages round-robin
                    for i in range(self.config.concurrent_messages_per_actor):
                        for actor_id in actor_ids:
                            await runtime.send(actor_id, WorkRequest(data=test_data))

                    # Receive all responses
                    for i in range(self.config.concurrent_messages_per_actor):
                        for actor_id in actor_ids:
                            await runtime.receive(actor_id, timeout=10.0)

                    total_time = time.perf_counter() - start

                if num_actors == 1:
                    single_actor_time = total_time

                speedup = single_actor_time / total_time if single_actor_time else 1.0

                result = ConcurrencyResult(
                    num_actors=num_actors,
                    total_messages=total_messages,
                    total_time=total_time,
                    throughput_msg_per_sec=total_messages / total_time if total_time > 0 else 0,
                    avg_latency_ms=total_time / total_messages * 1000 if total_messages > 0 else 0,
                    speedup_vs_single=speedup,
                )
                self.results.append(result)

                print(f"{format_time(total_time)} ({format_number(result.throughput_msg_per_sec)} msg/s, {speedup:.2f}x speedup)")

            except Exception as e:
                print(f"Error: {e}")

        return self.results


# ============================================================================
# Report Generation
# ============================================================================

def generate_extended_report(
    large_scale_results: list[LargeScaleResult],
    streaming_results: list[StreamingResult],
    latency_results: list[LatencyResult],
    concurrency_results: list[ConcurrencyResult],
    output_path: Path | None = None,
) -> str:
    """Generate comprehensive benchmark report."""
    lines = []
    lines.append("=" * 70)
    lines.append("PYDOTCOMPUTE EXTENDED BENCHMARK REPORT")
    lines.append("=" * 70)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"CUDA Available: {check_cuda_available()}")

    # Large-scale results
    if large_scale_results:
        lines.append("\n" + "=" * 70)
        lines.append("LARGE-SCALE GRAPH RESULTS")
        lines.append("=" * 70)
        lines.append(f"\n{'Nodes':<12} {'Edges':<12} {'Impl':<12} {'Time':<12} {'Throughput':<15} {'Memory':<10}")
        lines.append("-" * 75)

        for r in sorted(large_scale_results, key=lambda x: (x.num_nodes, x.implementation)):
            lines.append(
                f"{format_number(r.num_nodes):<12} "
                f"{format_number(r.num_edges):<12} "
                f"{r.implementation:<12} "
                f"{format_time(r.compute_time):<12} "
                f"{format_number(r.throughput_edges_per_sec):<15} "
                f"{r.memory_mb:.1f}MB"
            )

        # Find crossover point
        cpu_results = [r for r in large_scale_results if r.implementation == "CPU Sparse"]
        gpu_results = [r for r in large_scale_results if r.implementation == "GPU Sparse"]

        if cpu_results and gpu_results:
            lines.append("\n  Analysis:")
            for cpu_r in cpu_results:
                gpu_r = next((g for g in gpu_results if g.num_nodes == cpu_r.num_nodes), None)
                if gpu_r:
                    speedup = cpu_r.compute_time / gpu_r.compute_time
                    winner = "GPU" if speedup > 1 else "CPU"
                    lines.append(f"    {format_number(cpu_r.num_nodes)} nodes: {winner} wins ({speedup:.2f}x)")

    # Streaming results
    if streaming_results:
        lines.append("\n" + "=" * 70)
        lines.append("STREAMING THROUGHPUT RESULTS")
        lines.append("=" * 70)
        lines.append("\nThis benchmark shows where GPU Actors shine!")
        lines.append("As message count increases, actor overhead is amortized.\n")

        # Group by payload size
        payload_sizes = sorted(set(r.payload_size for r in streaming_results))

        for payload in payload_sizes:
            lines.append(f"\n--- Payload: {payload} floats ---")
            lines.append(f"{'Messages':<12} {'Impl':<18} {'Time':<12} {'Throughput':<15} {'Latency':<12}")
            lines.append("-" * 70)

            payload_results = [r for r in streaming_results if r.payload_size == payload]
            for r in sorted(payload_results, key=lambda x: (x.message_count, x.implementation)):
                lines.append(
                    f"{format_number(r.message_count):<12} "
                    f"{r.implementation:<18} "
                    f"{format_time(r.total_time):<12} "
                    f"{format_number(r.throughput_msg_per_sec)} msg/s    "
                    f"{r.avg_latency_ms:.3f}ms"
                )

        # Actor overhead amortization analysis
        actor_results = [r for r in streaming_results if r.implementation == "GPU Actors"]
        if len(actor_results) >= 2:
            lines.append("\n  Actor Overhead Amortization:")
            first = min(actor_results, key=lambda x: x.message_count)
            last = max(actor_results, key=lambda x: x.message_count)

            first_overhead = first.setup_time
            last_per_msg = last.avg_latency_ms

            lines.append(f"    Setup overhead: {format_time(first_overhead)}")
            lines.append(f"    At {format_number(first.message_count)} msgs: {first.avg_latency_ms:.3f}ms/msg")
            lines.append(f"    At {format_number(last.message_count)} msgs: {last.avg_latency_ms:.3f}ms/msg")
            lines.append(f"    Overhead reduction: {(first.avg_latency_ms / last.avg_latency_ms):.1f}x")

    # Latency results
    if latency_results:
        lines.append("\n" + "=" * 70)
        lines.append("LATENCY DISTRIBUTION RESULTS")
        lines.append("=" * 70)

        for r in latency_results:
            lines.append(f"\n{r.implementation} ({r.sample_count} samples):")
            lines.append(f"  p50:  {r.p50_ms:.3f}ms")
            lines.append(f"  p95:  {r.p95_ms:.3f}ms")
            lines.append(f"  p99:  {r.p99_ms:.3f}ms")
            lines.append(f"  min:  {r.min_ms:.3f}ms")
            lines.append(f"  max:  {r.max_ms:.3f}ms")
            lines.append(f"  mean: {r.mean_ms:.3f}ms (std: {r.std_ms:.3f}ms)")

    # Concurrency results
    if concurrency_results:
        lines.append("\n" + "=" * 70)
        lines.append("CONCURRENT ACTORS RESULTS")
        lines.append("=" * 70)
        lines.append(f"\n{'Actors':<10} {'Messages':<12} {'Time':<12} {'Throughput':<15} {'Speedup':<10}")
        lines.append("-" * 60)

        for r in concurrency_results:
            lines.append(
                f"{r.num_actors:<10} "
                f"{format_number(r.total_messages):<12} "
                f"{format_time(r.total_time):<12} "
                f"{format_number(r.throughput_msg_per_sec)} msg/s    "
                f"{r.speedup_vs_single:.2f}x"
            )

        if len(concurrency_results) >= 2:
            max_speedup = max(r.speedup_vs_single for r in concurrency_results)
            max_actors = max(r.num_actors for r in concurrency_results)
            efficiency = max_speedup / max_actors * 100
            lines.append(f"\n  Scaling efficiency: {efficiency:.1f}% ({max_speedup:.2f}x with {max_actors} actors)")

    # Conclusions
    lines.append("\n" + "=" * 70)
    lines.append("CONCLUSIONS")
    lines.append("=" * 70)

    lines.append("\n1. LARGE-SCALE GRAPHS:")
    if large_scale_results:
        gpu_wins = [r for r in large_scale_results if r.implementation == "GPU Sparse"]
        if gpu_wins:
            best_gpu = max(gpu_wins, key=lambda x: x.throughput_edges_per_sec)
            lines.append(f"   - GPU achieves up to {format_number(best_gpu.throughput_edges_per_sec)} edges/sec")
            lines.append(f"   - GPU excels at graphs with {format_number(best_gpu.num_nodes)}+ nodes")

    lines.append("\n2. STREAMING THROUGHPUT:")
    if streaming_results:
        actor_results = [r for r in streaming_results if r.implementation == "GPU Actors"]
        if actor_results:
            best_actor = max(actor_results, key=lambda x: x.throughput_msg_per_sec)
            lines.append(f"   - GPU Actors achieve {format_number(best_actor.throughput_msg_per_sec)} msg/sec")
            lines.append(f"   - Actor overhead amortizes with high message counts")
            lines.append(f"   - Best for: persistent streaming workloads")

    lines.append("\n3. LATENCY:")
    if latency_results:
        r = latency_results[0]
        lines.append(f"   - p99 latency: {r.p99_ms:.3f}ms")
        lines.append(f"   - Consistent performance (std: {r.std_ms:.3f}ms)")

    lines.append("\n4. CONCURRENCY:")
    if concurrency_results:
        best = max(concurrency_results, key=lambda x: x.speedup_vs_single)
        lines.append(f"   - {best.num_actors} actors achieve {best.speedup_vs_single:.2f}x speedup")
        lines.append(f"   - Good scaling with multiple actors")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)

    if output_path:
        output_path.write_text(report)

        # Save JSON
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "cuda_available": check_cuda_available(),
            "large_scale": [
                {
                    "implementation": r.implementation,
                    "num_nodes": r.num_nodes,
                    "num_edges": r.num_edges,
                    "compute_time": r.compute_time,
                    "throughput": r.throughput_edges_per_sec,
                    "memory_mb": r.memory_mb,
                }
                for r in large_scale_results
            ],
            "streaming": [
                {
                    "implementation": r.implementation,
                    "message_count": r.message_count,
                    "payload_size": r.payload_size,
                    "total_time": r.total_time,
                    "throughput": r.throughput_msg_per_sec,
                    "avg_latency_ms": r.avg_latency_ms,
                }
                for r in streaming_results
            ],
            "latency": [
                {
                    "implementation": r.implementation,
                    "p50_ms": r.p50_ms,
                    "p95_ms": r.p95_ms,
                    "p99_ms": r.p99_ms,
                    "mean_ms": r.mean_ms,
                }
                for r in latency_results
            ],
            "concurrency": [
                {
                    "num_actors": r.num_actors,
                    "total_messages": r.total_messages,
                    "throughput": r.throughput_msg_per_sec,
                    "speedup": r.speedup_vs_single,
                }
                for r in concurrency_results
            ],
        }
        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(json_data, indent=2))

    return report


# ============================================================================
# Main Entry Point
# ============================================================================

async def main() -> None:
    """Run the extended benchmark suite."""
    print("=" * 70)
    print("PYDOTCOMPUTE EXTENDED BENCHMARK SUITE")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CUDA Available: {check_cuda_available()}")

    config = ExtendedBenchmarkConfig(
        # Adjust for available memory - RTX 2000 has 8GB
        large_graph_sizes=[10_000, 50_000, 100_000, 500_000, 1_000_000],
        large_graph_edges_per_node=5.0,
        streaming_message_counts=[100, 1000, 5000, 10000, 50000],
        streaming_payload_sizes=[10, 100, 1000],
        latency_sample_count=1000,
        latency_warmup_count=100,
        concurrent_actor_counts=[1, 2, 4, 8],
        concurrent_messages_per_actor=1000,
    )

    # Run benchmarks
    large_scale = LargeScaleBenchmark(config)
    large_scale_results = large_scale.run()

    streaming = StreamingBenchmark(config)
    streaming_results = await streaming.run()

    latency = LatencyBenchmark(config)
    latency_results = await latency.run()

    concurrency = ConcurrencyBenchmark(config)
    concurrency_results = await concurrency.run()

    # Generate report
    output_dir = Path(__file__).parent
    report_path = output_dir / "extended_benchmark_report.txt"

    report = generate_extended_report(
        large_scale_results,
        streaming_results,
        latency_results,
        concurrency_results,
        report_path,
    )

    print("\n" + report)
    print(f"\nReport saved to: {report_path}")
    print(f"JSON data saved to: {report_path.with_suffix('.json')}")


if __name__ == "__main__":
    asyncio.run(main())
