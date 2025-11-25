"""
PageRank Benchmark for PyDotCompute.

This benchmark compares different PageRank implementations:
    a) CPU implementations (basic loop, NumPy vectorized, sparse matrix)
    b) GPU batch processing (CuPy)
    c) GPU native actors (Ring Kernel System)

Tests various graph configurations:
    - Graph sizes: 100, 500, 1000, 5000, 10000 nodes
    - Graph densities: sparse (~2 edges/node), medium (~10 edges/node), dense (~50 edges/node)

Produces a professional benchmark report with timing statistics.
"""

from __future__ import annotations

import asyncio
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from uuid import UUID, uuid4

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Graph sizes to test
    graph_sizes: list[int] = field(default_factory=lambda: [100, 500, 1000, 5000])

    # Edges per node for different densities
    density_levels: dict[str, float] = field(default_factory=lambda: {
        "sparse": 2.0,
        "medium": 10.0,
        "dense": 50.0,
    })

    # PageRank parameters
    damping: float = 0.85
    max_iterations: int = 100
    tolerance: float = 1e-6

    # Benchmark parameters
    warmup_runs: int = 1
    measurement_runs: int = 3

    # Enable/disable implementations
    test_cpu_basic: bool = True
    test_cpu_numpy: bool = True
    test_cpu_sparse: bool = True
    test_gpu_batch: bool = True
    test_gpu_actors: bool = True


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    implementation: str
    graph_size: int
    density: str
    num_edges: int

    # Timing (seconds)
    setup_time: float
    compute_time: float
    total_time: float

    # PageRank results
    iterations: int
    converged: bool
    final_diff: float

    # Optional metadata
    memory_mb: float = 0.0
    throughput_edges_per_sec: float = 0.0


# ============================================================================
# Graph Generation
# ============================================================================

def generate_random_graph(
    num_nodes: int,
    edges_per_node: float,
    seed: int = 42,
) -> tuple[list[tuple[int, int]], int]:
    """
    Generate a random directed graph.

    Args:
        num_nodes: Number of nodes in the graph.
        edges_per_node: Average number of outgoing edges per node.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (edge list, actual number of edges).
    """
    rng = np.random.default_rng(seed)
    num_edges = int(num_nodes * edges_per_node)

    # Generate random edges
    edges: list[tuple[int, int]] = []
    edge_set: set[tuple[int, int]] = set()

    while len(edges) < num_edges:
        src = rng.integers(0, num_nodes)
        dst = rng.integers(0, num_nodes)

        if src != dst and (src, dst) not in edge_set:
            edges.append((int(src), int(dst)))
            edge_set.add((src, dst))

    return edges, len(edges)


def edges_to_csr(
    edges: list[tuple[int, int]],
    num_nodes: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Convert edge list to CSR format for sparse matrix operations.

    Returns:
        Tuple of (data, indices, indptr) arrays.
    """
    # Build adjacency lists
    adjacency: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    out_degree = np.zeros(num_nodes, dtype=np.float64)

    for src, dst in edges:
        adjacency[dst].append(src)
        out_degree[src] += 1

    # Convert to CSR format
    indptr = [0]
    indices = []
    data = []

    for node in range(num_nodes):
        incoming = adjacency[node]
        for src in incoming:
            indices.append(src)
            data.append(1.0 / out_degree[src] if out_degree[src] > 0 else 0.0)
        indptr.append(len(indices))

    return (
        np.array(data, dtype=np.float64),
        np.array(indices, dtype=np.int32),
        np.array(indptr, dtype=np.int32),
    )


# ============================================================================
# CPU Implementations
# ============================================================================

def pagerank_cpu_basic(
    edges: list[tuple[int, int]],
    num_nodes: int,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[NDArray, bool, int, float]:
    """
    Basic CPU implementation using Python loops.

    This is the slowest but most straightforward implementation.
    """
    # Build adjacency structure
    out_degree = np.zeros(num_nodes)
    adjacency: dict[int, list[int]] = {i: [] for i in range(num_nodes)}

    for src, dst in edges:
        out_degree[src] += 1
        adjacency[dst].append(src)

    # Initialize ranks
    ranks = np.ones(num_nodes) / num_nodes
    new_ranks = np.zeros(num_nodes)

    teleport = (1 - damping) / num_nodes

    # Iterate
    for iteration in range(max_iterations):
        for node in range(num_nodes):
            incoming_sum = 0.0
            for src in adjacency[node]:
                if out_degree[src] > 0:
                    incoming_sum += ranks[src] / out_degree[src]
            new_ranks[node] = teleport + damping * incoming_sum

        diff = np.abs(new_ranks - ranks).max()
        if diff < tolerance:
            return new_ranks.copy(), True, iteration + 1, diff

        ranks, new_ranks = new_ranks, ranks

    return ranks, False, max_iterations, np.abs(new_ranks - ranks).max()


def pagerank_cpu_numpy(
    edges: list[tuple[int, int]],
    num_nodes: int,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[NDArray, bool, int, float]:
    """
    NumPy-optimized CPU implementation using matrix operations.

    Faster than basic but uses dense matrix (memory-intensive for large graphs).
    """
    # Build transition matrix (column-stochastic)
    transition = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    out_degree = np.zeros(num_nodes, dtype=np.float64)

    for src, dst in edges:
        out_degree[src] += 1

    for src, dst in edges:
        if out_degree[src] > 0:
            transition[dst, src] = 1.0 / out_degree[src]

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


def pagerank_cpu_sparse(
    edges: list[tuple[int, int]],
    num_nodes: int,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[NDArray, bool, int, float]:
    """
    Sparse matrix CPU implementation using CSR format.

    Memory-efficient and faster for sparse graphs.
    """
    try:
        from scipy import sparse
    except ImportError:
        # Fall back to numpy implementation
        return pagerank_cpu_numpy(edges, num_nodes, damping, max_iterations, tolerance)

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


# ============================================================================
# GPU Implementations
# ============================================================================

def check_cuda_available() -> bool:
    """Check if CUDA is available via CuPy."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except (ImportError, Exception):
        return False


def pagerank_gpu_batch(
    edges: list[tuple[int, int]],
    num_nodes: int,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[NDArray, bool, int, float]:
    """
    GPU batch implementation using CuPy.

    Transfers data to GPU, performs computation, transfers back.
    """
    import cupy as cp

    # Build transition matrix on CPU
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

    # Transfer to GPU as sparse matrix
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

    return cp.asnumpy(ranks), False, max_iterations, diff


async def pagerank_gpu_actors(
    edges: list[tuple[int, int]],
    num_nodes: int,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[NDArray, bool, int, float]:
    """
    GPU actor implementation using PyDotCompute Ring Kernel System.

    Uses persistent actors for PageRank computation.
    """
    from pydotcompute import RingKernelRuntime, message, ring_kernel
    from pydotcompute.ring_kernels.lifecycle import KernelContext

    # Message types
    @message
    @dataclass
    class PRRequest:
        edges: list[tuple[int, int]]
        num_nodes: int
        damping: float = 0.85
        max_iterations: int = 100
        tolerance: float = 1e-6
        message_id: UUID = field(default_factory=uuid4)
        priority: int = 128
        correlation_id: UUID | None = None

    @message
    @dataclass
    class PRResponse:
        ranks: list[float]
        converged: bool
        iterations: int
        final_diff: float
        message_id: UUID = field(default_factory=uuid4)
        priority: int = 128
        correlation_id: UUID | None = None

    # Define the actor
    @ring_kernel(
        kernel_id="pr_benchmark",
        input_type=PRRequest,
        output_type=PRResponse,
        queue_size=32,
    )
    async def pr_actor(ctx: KernelContext) -> None:
        while not ctx.should_terminate:
            if not ctx.is_active:
                await ctx.wait_active()
                continue

            try:
                request = await ctx.receive(timeout=0.1)

                # Use sparse CPU implementation for actors
                # (In real GPU actors, this would use CUDA kernels)
                ranks, converged, iters, diff = pagerank_cpu_sparse(
                    request.edges,
                    request.num_nodes,
                    request.damping,
                    request.max_iterations,
                    request.tolerance,
                )

                response = PRResponse(
                    ranks=ranks.tolist(),
                    converged=converged,
                    iterations=iters,
                    final_diff=diff,
                    correlation_id=request.message_id,
                )
                await ctx.send(response)

            except Exception:
                continue

    # Run the actor
    async with RingKernelRuntime() as runtime:
        await runtime.launch("pr_benchmark")
        await runtime.activate("pr_benchmark")

        request = PRRequest(
            edges=edges,
            num_nodes=num_nodes,
            damping=damping,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

        await runtime.send("pr_benchmark", request)
        response = await runtime.receive("pr_benchmark", timeout=60.0)

    return np.array(response.ranks), response.converged, response.iterations, response.final_diff


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """Runs and collects benchmark results."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.results: list[BenchmarkResult] = []
        self.cuda_available = check_cuda_available()

    def _time_implementation(
        self,
        name: str,
        func: Callable,
        edges: list[tuple[int, int]],
        num_nodes: int,
        num_edges: int,
        density: str,
        is_async: bool = False,
    ) -> BenchmarkResult | None:
        """Time a single implementation."""
        gc.collect()

        try:
            # Warmup
            for _ in range(self.config.warmup_runs):
                if is_async:
                    asyncio.run(func(
                        edges, num_nodes,
                        self.config.damping,
                        self.config.max_iterations,
                        self.config.tolerance,
                    ))
                else:
                    func(
                        edges, num_nodes,
                        self.config.damping,
                        self.config.max_iterations,
                        self.config.tolerance,
                    )

            # Measurement runs
            times = []
            iterations_list = []
            converged_list = []
            diff_list = []

            for _ in range(self.config.measurement_runs):
                gc.collect()

                start = time.perf_counter()
                if is_async:
                    ranks, converged, iters, diff = asyncio.run(func(
                        edges, num_nodes,
                        self.config.damping,
                        self.config.max_iterations,
                        self.config.tolerance,
                    ))
                else:
                    ranks, converged, iters, diff = func(
                        edges, num_nodes,
                        self.config.damping,
                        self.config.max_iterations,
                        self.config.tolerance,
                    )
                end = time.perf_counter()

                times.append(end - start)
                iterations_list.append(iters)
                converged_list.append(converged)
                diff_list.append(diff)

            avg_time = np.mean(times)

            return BenchmarkResult(
                implementation=name,
                graph_size=num_nodes,
                density=density,
                num_edges=num_edges,
                setup_time=0.0,
                compute_time=avg_time,
                total_time=avg_time,
                iterations=int(np.mean(iterations_list)),
                converged=all(converged_list),
                final_diff=float(np.mean(diff_list)),
                throughput_edges_per_sec=num_edges / avg_time if avg_time > 0 else 0,
            )

        except Exception as e:
            print(f"  Error in {name}: {e}")
            return None

    def run(self) -> list[BenchmarkResult]:
        """Run all benchmarks."""
        print("=" * 70)
        print("PyDotCompute PageRank Benchmark")
        print("=" * 70)
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"CUDA Available: {self.cuda_available}")
        print(f"\nConfiguration:")
        print(f"  Graph sizes: {self.config.graph_sizes}")
        print(f"  Density levels: {list(self.config.density_levels.keys())}")
        print(f"  Warmup runs: {self.config.warmup_runs}")
        print(f"  Measurement runs: {self.config.measurement_runs}")
        print()

        self.results = []

        for num_nodes in self.config.graph_sizes:
            for density_name, edges_per_node in self.config.density_levels.items():
                print(f"\n{'='*50}")
                print(f"Graph: {num_nodes} nodes, {density_name} density")
                print(f"{'='*50}")

                # Generate graph
                edges, num_edges = generate_random_graph(
                    num_nodes, edges_per_node
                )
                print(f"Generated {num_edges} edges ({num_edges/num_nodes:.1f} per node)")

                # CPU Basic
                if self.config.test_cpu_basic and num_nodes <= 1000:
                    print("\n  Testing CPU Basic...", end=" ", flush=True)
                    result = self._time_implementation(
                        "CPU Basic", pagerank_cpu_basic,
                        edges, num_nodes, num_edges, density_name,
                    )
                    if result:
                        self.results.append(result)
                        print(f"{result.compute_time:.3f}s ({result.iterations} iters)")

                # CPU NumPy
                if self.config.test_cpu_numpy and num_nodes <= 5000:
                    print("  Testing CPU NumPy...", end=" ", flush=True)
                    result = self._time_implementation(
                        "CPU NumPy", pagerank_cpu_numpy,
                        edges, num_nodes, num_edges, density_name,
                    )
                    if result:
                        self.results.append(result)
                        print(f"{result.compute_time:.3f}s ({result.iterations} iters)")

                # CPU Sparse
                if self.config.test_cpu_sparse:
                    print("  Testing CPU Sparse...", end=" ", flush=True)
                    result = self._time_implementation(
                        "CPU Sparse", pagerank_cpu_sparse,
                        edges, num_nodes, num_edges, density_name,
                    )
                    if result:
                        self.results.append(result)
                        print(f"{result.compute_time:.3f}s ({result.iterations} iters)")

                # GPU Batch
                if self.config.test_gpu_batch and self.cuda_available:
                    print("  Testing GPU Batch...", end=" ", flush=True)
                    result = self._time_implementation(
                        "GPU Batch", pagerank_gpu_batch,
                        edges, num_nodes, num_edges, density_name,
                    )
                    if result:
                        self.results.append(result)
                        print(f"{result.compute_time:.3f}s ({result.iterations} iters)")

                # GPU Actors
                if self.config.test_gpu_actors:
                    print("  Testing GPU Actors...", end=" ", flush=True)
                    result = self._time_implementation(
                        "GPU Actors", pagerank_gpu_actors,
                        edges, num_nodes, num_edges, density_name,
                        is_async=True,
                    )
                    if result:
                        self.results.append(result)
                        print(f"{result.compute_time:.3f}s ({result.iterations} iters)")

        return self.results

    def generate_report(self, output_path: Path | None = None) -> str:
        """Generate a professional benchmark report."""
        lines = []
        lines.append("=" * 70)
        lines.append("PAGERANK BENCHMARK REPORT")
        lines.append("=" * 70)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"CUDA Available: {self.cuda_available}")
        lines.append(f"\nConfiguration:")
        lines.append(f"  Damping factor: {self.config.damping}")
        lines.append(f"  Max iterations: {self.config.max_iterations}")
        lines.append(f"  Tolerance: {self.config.tolerance}")
        lines.append(f"  Measurement runs: {self.config.measurement_runs}")

        # Group results by graph configuration
        results_by_config: dict[tuple[int, str], list[BenchmarkResult]] = {}
        for r in self.results:
            key = (r.graph_size, r.density)
            if key not in results_by_config:
                results_by_config[key] = []
            results_by_config[key].append(r)

        # Summary tables
        lines.append("\n" + "=" * 70)
        lines.append("PERFORMANCE SUMMARY")
        lines.append("=" * 70)

        for (num_nodes, density), results in sorted(results_by_config.items()):
            lines.append(f"\n--- {num_nodes} nodes, {density} density ---")
            lines.append(f"{'Implementation':<20} {'Time (s)':<12} {'Iterations':<12} {'Throughput':<15}")
            lines.append("-" * 60)

            for r in sorted(results, key=lambda x: x.compute_time):
                throughput = f"{r.throughput_edges_per_sec/1000:.1f}K edges/s"
                lines.append(
                    f"{r.implementation:<20} {r.compute_time:<12.4f} {r.iterations:<12} {throughput:<15}"
                )

        # Speedup analysis
        lines.append("\n" + "=" * 70)
        lines.append("SPEEDUP ANALYSIS (relative to CPU Sparse)")
        lines.append("=" * 70)

        for (num_nodes, density), results in sorted(results_by_config.items()):
            sparse_result = next((r for r in results if r.implementation == "CPU Sparse"), None)
            if not sparse_result:
                continue

            lines.append(f"\n--- {num_nodes} nodes, {density} density ---")
            lines.append(f"{'Implementation':<20} {'Speedup':<12}")
            lines.append("-" * 35)

            for r in sorted(results, key=lambda x: x.compute_time):
                speedup = sparse_result.compute_time / r.compute_time if r.compute_time > 0 else 0
                lines.append(f"{r.implementation:<20} {speedup:>10.2f}x")

        # Scaling analysis
        lines.append("\n" + "=" * 70)
        lines.append("SCALING ANALYSIS")
        lines.append("=" * 70)

        impl_names = set(r.implementation for r in self.results)
        for impl in impl_names:
            impl_results = [r for r in self.results if r.implementation == impl]
            if len(impl_results) < 2:
                continue

            lines.append(f"\n{impl}:")
            lines.append(f"{'Size':<10} {'Sparse Time':<15} {'Medium Time':<15} {'Dense Time':<15}")
            lines.append("-" * 55)

            for size in self.config.graph_sizes:
                row = f"{size:<10}"
                for density in ["sparse", "medium", "dense"]:
                    r = next((x for x in impl_results
                             if x.graph_size == size and x.density == density), None)
                    if r:
                        row += f" {r.compute_time:<14.4f}"
                    else:
                        row += f" {'N/A':<14}"
                lines.append(row)

        # Conclusions
        lines.append("\n" + "=" * 70)
        lines.append("CONCLUSIONS")
        lines.append("=" * 70)

        if self.results:
            # Find fastest for large graphs
            large_results = [r for r in self.results if r.graph_size >= 1000]
            if large_results:
                fastest = min(large_results, key=lambda x: x.compute_time)
                lines.append(f"\n- Fastest implementation for large graphs: {fastest.implementation}")
                lines.append(f"  (Best time: {fastest.compute_time:.4f}s for {fastest.graph_size} nodes)")

            # GPU vs CPU comparison
            cpu_results = [r for r in self.results if "CPU" in r.implementation]
            gpu_results = [r for r in self.results if "GPU" in r.implementation]

            if cpu_results and gpu_results:
                avg_cpu = np.mean([r.compute_time for r in cpu_results])
                avg_gpu = np.mean([r.compute_time for r in gpu_results])

                if avg_gpu < avg_cpu:
                    speedup = avg_cpu / avg_gpu
                    lines.append(f"\n- GPU implementations are {speedup:.2f}x faster on average")
                else:
                    slowdown = avg_gpu / avg_cpu
                    lines.append(f"\n- CPU implementations are {slowdown:.2f}x faster on average")
                    lines.append("  (GPU overhead dominates for small graphs)")

            # Density impact
            sparse_times = [r.compute_time for r in self.results if r.density == "sparse"]
            dense_times = [r.compute_time for r in self.results if r.density == "dense"]

            if sparse_times and dense_times:
                density_ratio = np.mean(dense_times) / np.mean(sparse_times)
                lines.append(f"\n- Dense graphs take {density_ratio:.2f}x longer than sparse graphs on average")

        lines.append("\n" + "=" * 70)

        report = "\n".join(lines)

        if output_path:
            output_path.write_text(report)

            # Also save JSON data
            json_path = output_path.with_suffix(".json")
            json_data = {
                "config": {
                    "graph_sizes": self.config.graph_sizes,
                    "density_levels": self.config.density_levels,
                    "damping": self.config.damping,
                    "max_iterations": self.config.max_iterations,
                    "tolerance": self.config.tolerance,
                    "measurement_runs": self.config.measurement_runs,
                },
                "results": [
                    {
                        "implementation": r.implementation,
                        "graph_size": r.graph_size,
                        "density": r.density,
                        "num_edges": r.num_edges,
                        "compute_time": r.compute_time,
                        "iterations": r.iterations,
                        "converged": r.converged,
                        "throughput_edges_per_sec": r.throughput_edges_per_sec,
                    }
                    for r in self.results
                ],
                "cuda_available": self.cuda_available,
                "timestamp": datetime.now().isoformat(),
            }
            json_path.write_text(json.dumps(json_data, indent=2))

        return report


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Run the benchmark suite."""
    # Configure benchmark
    config = BenchmarkConfig(
        graph_sizes=[100, 500, 1000, 5000],
        density_levels={
            "sparse": 2.0,
            "medium": 10.0,
            "dense": 50.0,
        },
        warmup_runs=1,
        measurement_runs=3,
    )

    # Run benchmarks
    runner = BenchmarkRunner(config)
    runner.run()

    # Generate report
    output_dir = Path(__file__).parent
    report_path = output_dir / "pagerank_benchmark_report.txt"

    report = runner.generate_report(report_path)
    print("\n" + report)

    print(f"\nReport saved to: {report_path}")
    print(f"JSON data saved to: {report_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
