"""
Streaming Pipeline Example for PyDotCompute.

Demonstrates chaining multiple ring kernels into a data processing pipeline.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydotcompute import RingKernelRuntime, message, ring_kernel

if TYPE_CHECKING:
    from pydotcompute.ring_kernels.lifecycle import KernelContext


# Pipeline messages
@message
@dataclass
class DataPoint:
    """Raw data point input."""

    value: float
    timestamp: float
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None


@message
@dataclass
class NormalizedData:
    """Normalized data after first stage."""

    original: float
    normalized: float
    timestamp: float
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None


@message
@dataclass
class ProcessedResult:
    """Final processed result."""

    original: float
    normalized: float
    score: float
    classification: str
    timestamp: float
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None


# Stage 1: Normalizer
@ring_kernel(
    kernel_id="normalizer",
    input_type=DataPoint,
    output_type=NormalizedData,
    queue_size=500,
)
async def normalizer_actor(
    ctx: KernelContext[DataPoint, NormalizedData],
) -> None:
    """Normalizes incoming data points to 0-1 range."""
    min_val = 0.0
    max_val = 100.0  # Assume data range

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            data = await ctx.receive(timeout=0.1)

            # Normalize value
            normalized = (data.value - min_val) / (max_val - min_val)
            normalized = max(0.0, min(1.0, normalized))  # Clamp

            result = NormalizedData(
                original=data.value,
                normalized=normalized,
                timestamp=data.timestamp,
                correlation_id=data.message_id,
            )
            await ctx.send(result)

        except Exception:
            continue


# Stage 2: Scorer
@ring_kernel(
    kernel_id="scorer",
    input_type=NormalizedData,
    output_type=ProcessedResult,
    queue_size=500,
)
async def scorer_actor(
    ctx: KernelContext[NormalizedData, ProcessedResult],
) -> None:
    """Scores and classifies normalized data."""
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            data = await ctx.receive(timeout=0.1)

            # Compute score (example: exponential scoring)
            score = data.normalized**2

            # Classify based on score
            if score < 0.3:
                classification = "low"
            elif score < 0.7:
                classification = "medium"
            else:
                classification = "high"

            result = ProcessedResult(
                original=data.original,
                normalized=data.normalized,
                score=score,
                classification=classification,
                timestamp=data.timestamp,
                correlation_id=data.message_id,
            )
            await ctx.send(result)

        except Exception:
            continue


class StreamingPipeline:
    """
    Manages a multi-stage streaming pipeline.

    Connects multiple ring kernels into a processing chain.
    """

    def __init__(self, runtime: RingKernelRuntime) -> None:
        """Initialize the pipeline."""
        self._runtime = runtime
        self._stages: list[str] = []

    async def add_stage(self, kernel_id: str) -> None:
        """Add a stage to the pipeline."""
        await self._runtime.activate(kernel_id)
        self._stages.append(kernel_id)

    async def process(self, data: DataPoint) -> ProcessedResult:
        """
        Process data through all pipeline stages.

        Args:
            data: Input data point.

        Returns:
            Final processed result.
        """
        # Send to first stage
        await self._runtime.send(self._stages[0], data)

        # Get result from first stage
        intermediate = await self._runtime.receive(self._stages[0], timeout=1.0)

        # Send to second stage
        await self._runtime.send(self._stages[1], intermediate)

        # Get final result
        result = await self._runtime.receive(self._stages[1], timeout=1.0)

        return result


async def run_pipeline_example() -> None:
    """Run the streaming pipeline example."""
    import time

    print("=" * 60)
    print("PyDotCompute Streaming Pipeline Example")
    print("=" * 60)

    async with RingKernelRuntime() as runtime:
        print("\n1. Launching pipeline stages...")
        await runtime.launch("normalizer")
        await runtime.launch("scorer")
        print("   Stages launched")

        print("\n2. Creating pipeline...")
        pipeline = StreamingPipeline(runtime)
        await pipeline.add_stage("normalizer")
        await pipeline.add_stage("scorer")
        print("   Pipeline ready")

        print("\n3. Processing data points...")
        test_values = [10.0, 25.0, 50.0, 75.0, 90.0, 100.0, 0.0, -10.0, 150.0]

        results = []
        for value in test_values:
            data = DataPoint(value=value, timestamp=time.time())
            result = await pipeline.process(data)
            results.append(result)
            print(
                f"   {value:6.1f} -> normalized={result.normalized:.3f}, "
                f"score={result.score:.3f}, class={result.classification}"
            )

        print("\n4. Summary:")
        classifications = {}
        for r in results:
            classifications[r.classification] = classifications.get(r.classification, 0) + 1

        for cls, count in sorted(classifications.items()):
            print(f"   {cls}: {count}")

        print("\n5. Telemetry:")
        for stage in ["normalizer", "scorer"]:
            telemetry = runtime.get_telemetry(stage)
            if telemetry:
                print(f"   {stage}: {telemetry.messages_processed} messages processed")

    print("\n" + "=" * 60)
    print("Pipeline example completed!")
    print("=" * 60)


# Parallel processing example
async def run_parallel_example() -> None:
    """Demonstrate parallel processing with multiple workers."""
    print("\n" + "=" * 60)
    print("Parallel Processing Example")
    print("=" * 60)

    async with RingKernelRuntime() as runtime:
        # Launch multiple normalizer workers
        num_workers = 3
        workers = []

        print(f"\n1. Launching {num_workers} parallel workers...")
        for i in range(num_workers):
            worker_id = f"worker_{i}"

            # Define worker inline
            @ring_kernel(kernel_id=worker_id, auto_register=False)
            async def worker(ctx: KernelContext) -> None:  # type: ignore
                while not ctx.should_terminate:
                    if not ctx.is_active:
                        await ctx.wait_active()
                        continue
                    try:
                        msg = await ctx.receive(timeout=0.1)
                        # Simulate work
                        await asyncio.sleep(0.01)
                        await ctx.send(msg)
                    except Exception:
                        continue

            await runtime.launch(worker_id, worker)
            await runtime.activate(worker_id)
            workers.append(worker_id)

        print("   Workers launched and active")

        # Send work to all workers in parallel
        print("\n2. Distributing work...")
        tasks = []
        for i, worker_id in enumerate(workers):
            msg = DataPoint(value=float(i * 10), timestamp=0.0)
            tasks.append(runtime.send(worker_id, msg))

        await asyncio.gather(*tasks)

        # Collect results
        print("\n3. Collecting results...")
        results = []
        for worker_id in workers:
            result = await runtime.receive(worker_id, timeout=1.0)
            results.append(result)
            print(f"   {worker_id}: value={result.value}")

    print("\n" + "=" * 60)
    print("Parallel example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_pipeline_example())
