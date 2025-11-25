"""
PageRank Actor Example for PyDotCompute.

Demonstrates a more complex ring kernel that implements the PageRank
algorithm using the actor model.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np

from pydotcompute import RingKernelRuntime, UnifiedBuffer, message, ring_kernel

if TYPE_CHECKING:
    from pydotcompute.ring_kernels.lifecycle import KernelContext


# Message types for PageRank
@message
@dataclass
class PageRankRequest:
    """Request to compute PageRank."""

    # Edges as list of (from, to) tuples
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
class PageRankResponse:
    """Response with PageRank results."""

    ranks: list[float]
    converged: bool
    iterations: int
    final_diff: float
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None


def compute_pagerank_cpu(
    edges: list[tuple[int, int]],
    num_nodes: int,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, bool, int, float]:
    """
    CPU implementation of PageRank algorithm.

    Args:
        edges: List of (source, target) edges.
        num_nodes: Number of nodes in the graph.
        damping: Damping factor (typically 0.85).
        max_iterations: Maximum iterations.
        tolerance: Convergence tolerance.

    Returns:
        Tuple of (ranks, converged, iterations, final_diff).
    """
    # Build adjacency matrix
    out_degree = np.zeros(num_nodes)
    adjacency = {}

    for src, dst in edges:
        out_degree[src] += 1
        if dst not in adjacency:
            adjacency[dst] = []
        adjacency[dst].append(src)

    # Initialize ranks
    ranks = np.ones(num_nodes) / num_nodes
    new_ranks = np.zeros(num_nodes)

    # Iterate
    converged = False
    final_diff = 0.0

    for iteration in range(max_iterations):
        # Compute new ranks
        for node in range(num_nodes):
            incoming_sum = 0.0
            for src in adjacency.get(node, []):
                if out_degree[src] > 0:
                    incoming_sum += ranks[src] / out_degree[src]

            new_ranks[node] = (1 - damping) / num_nodes + damping * incoming_sum

        # Check convergence
        diff = np.abs(new_ranks - ranks).max()
        if diff < tolerance:
            converged = True
            final_diff = diff
            ranks = new_ranks.copy()
            return ranks, converged, iteration + 1, final_diff

        ranks, new_ranks = new_ranks, ranks
        final_diff = diff

    return ranks, converged, max_iterations, final_diff


@ring_kernel(
    kernel_id="pagerank",
    input_type=PageRankRequest,
    output_type=PageRankResponse,
    queue_size=256,
)
async def pagerank_actor(
    ctx: KernelContext[PageRankRequest, PageRankResponse],
) -> None:
    """
    Persistent PageRank computation actor.

    Receives graph data, computes PageRank, and returns results.
    This is a CPU implementation; GPU version would use CUDA kernels.
    """
    print(f"[{ctx.kernel_id}] PageRank actor started")

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            # Receive request
            request = await ctx.receive(timeout=0.1)

            print(f"[{ctx.kernel_id}] Processing graph with {request.num_nodes} nodes")

            # Compute PageRank
            ranks, converged, iterations, final_diff = compute_pagerank_cpu(
                edges=request.edges,
                num_nodes=request.num_nodes,
                damping=request.damping,
                max_iterations=request.max_iterations,
                tolerance=request.tolerance,
            )

            # Send response
            response = PageRankResponse(
                ranks=ranks.tolist(),
                converged=converged,
                iterations=iterations,
                final_diff=final_diff,
                correlation_id=request.message_id,
            )
            await ctx.send(response)

            print(f"[{ctx.kernel_id}] Completed in {iterations} iterations")

        except Exception:
            continue

    print(f"[{ctx.kernel_id}] PageRank actor terminated")


async def run_pagerank_example() -> None:
    """Run the PageRank example."""
    print("=" * 60)
    print("PyDotCompute PageRank Actor Example")
    print("=" * 60)

    async with RingKernelRuntime() as runtime:
        print("\n1. Launching PageRank kernel...")
        await runtime.launch("pagerank")
        await runtime.activate("pagerank")
        print("   Kernel active")

        # Create a simple test graph
        # Graph: 0 -> 1 -> 2 -> 0 (cycle) with some additional edges
        print("\n2. Creating test graph...")
        edges = [
            (0, 1),
            (1, 2),
            (2, 0),
            (2, 1),  # 2 also points to 1
            (0, 2),  # 0 also points to 2
        ]
        num_nodes = 3

        print(f"   Nodes: {num_nodes}")
        print(f"   Edges: {edges}")

        print("\n3. Sending PageRank request...")
        request = PageRankRequest(
            edges=edges,
            num_nodes=num_nodes,
            damping=0.85,
            max_iterations=100,
            tolerance=1e-8,
        )
        await runtime.send("pagerank", request)

        print("\n4. Receiving results...")
        response = await runtime.receive("pagerank", timeout=5.0)

        print(f"\n   Results:")
        print(f"   Converged: {response.converged}")
        print(f"   Iterations: {response.iterations}")
        print(f"   Final diff: {response.final_diff:.2e}")
        print(f"   Ranks:")
        for i, rank in enumerate(response.ranks):
            print(f"      Node {i}: {rank:.6f}")

        # Test with a larger graph
        print("\n5. Testing with larger graph...")
        num_large = 100
        np.random.seed(42)

        # Generate random edges
        large_edges = []
        for _ in range(num_large * 3):  # ~3 edges per node on average
            src = np.random.randint(0, num_large)
            dst = np.random.randint(0, num_large)
            if src != dst:
                large_edges.append((int(src), int(dst)))

        large_request = PageRankRequest(
            edges=large_edges,
            num_nodes=num_large,
            damping=0.85,
            max_iterations=200,
        )

        await runtime.send("pagerank", large_request)
        large_response = await runtime.receive("pagerank", timeout=10.0)

        print(f"   Large graph ({num_large} nodes, {len(large_edges)} edges):")
        print(f"   Converged: {large_response.converged}")
        print(f"   Iterations: {large_response.iterations}")

        # Find top 5 ranked nodes
        ranks_with_idx = list(enumerate(large_response.ranks))
        ranks_with_idx.sort(key=lambda x: x[1], reverse=True)

        print(f"   Top 5 nodes:")
        for i, (node, rank) in enumerate(ranks_with_idx[:5]):
            print(f"      {i + 1}. Node {node}: {rank:.6f}")

        print("\n6. Getting telemetry...")
        telemetry = runtime.get_telemetry("pagerank")
        if telemetry:
            print(f"   Requests processed: {telemetry.messages_processed}")

    print("\n" + "=" * 60)
    print("PageRank example completed!")
    print("=" * 60)


# GPU-accelerated version (for reference - requires CUDA)
async def run_gpu_pagerank_example() -> None:
    """
    GPU-accelerated PageRank example.

    This shows how you would implement GPU acceleration
    using UnifiedBuffer and CUDA kernels.
    """
    print("GPU PageRank example (requires CUDA)")

    # Check if CUDA is available
    try:
        import cupy as cp

        cuda_available = cp.cuda.runtime.getDeviceCount() > 0
    except ImportError:
        cuda_available = False

    if not cuda_available:
        print("CUDA not available, skipping GPU example")
        return

    # Create buffers using UnifiedBuffer
    num_nodes = 1000

    ranks_buffer = UnifiedBuffer((num_nodes,), dtype=np.float32)
    ranks_buffer.allocate()
    ranks_buffer.host[:] = 1.0 / num_nodes
    ranks_buffer.mark_host_dirty()

    # Sync to device
    await ranks_buffer.ensure_on_device()

    print(f"Allocated unified buffer: {ranks_buffer}")
    print(f"Buffer state: {ranks_buffer.state.name}")

    # GPU computation would happen here using device data
    # ranks_buffer.device contains the CuPy array

    # Sync back to host
    await ranks_buffer.ensure_on_host()

    print(f"Sum of ranks: {ranks_buffer.host.sum():.6f}")


if __name__ == "__main__":
    asyncio.run(run_pagerank_example())
