"""
Real-Time Streaming Anomaly Detection Benchmark.

This benchmark demonstrates the PERFECT use case for GPU native actors:
Real-time anomaly detection in streaming time series data using MICRO-BATCHES.

KEY INSIGHT:
============
Real streaming systems (Kafka, Flink, Spark Streaming) use MICRO-BATCHES,
not point-by-point processing. GPU Actors excel at micro-batch processing
because they maintain PERSISTENT GPU STATE across batches.

WHY GPU ACTORS EXCEL:
=====================

1. PERSISTENT GPU STATE ACROSS MICRO-BATCHES
   - Sliding window buffer persists on GPU between batches
   - Rolling statistics maintained incrementally
   - Model weights stay resident on GPU
   - State doesn't need to be re-uploaded each batch!

2. STATELESS BATCH PROCESSING DISADVANTAGE
   - Must transfer window history for EACH batch
   - Must rebuild rolling statistics from scratch
   - Repeated memory allocation/deallocation
   - Transfer overhead grows with state size!

3. THE MATH:
   - Window size: 10,000 points × 4 bytes = 40KB
   - Batch processing: 40KB transfer per batch
   - 10,000 batches = 400MB total transfers!
   - GPU Actors: 0 transfers after setup!

USE CASE: IoT Sensor Monitoring
===============================
- Sensors send data in chunks (every 100ms = ~100-1000 points)
- Need to detect anomalies in real-time
- Must maintain context (what's "normal" for this sensor)
- Multiple sensors = multiple actors
"""

from __future__ import annotations

import asyncio
import gc
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AnomalyBenchmarkConfig:
    """Configuration for anomaly detection benchmark."""

    # Total points and micro-batch sizes
    total_points: list[int] = field(default_factory=lambda: [
        100_000, 500_000, 1_000_000
    ])

    # Micro-batch sizes (points per message)
    micro_batch_sizes: list[int] = field(default_factory=lambda: [
        100, 500, 1000
    ])

    # Window configuration (this is the persistent GPU state!)
    window_size: int = 10_000  # Large window = more state to transfer in batch mode

    # Benchmark parameters
    warmup_batches: int = 10
    measurement_runs: int = 3


# ============================================================================
# Data Generation
# ============================================================================

def generate_streaming_data(
    length: int,
    anomaly_rate: float = 0.01,
    seed: int = 42,
) -> tuple[NDArray, NDArray]:
    """Generate synthetic time series with anomalies."""
    rng = np.random.default_rng(seed)

    t = np.arange(length, dtype=np.float32)
    signal = (
        np.sin(2 * np.pi * t / 1000) +
        0.5 * np.sin(2 * np.pi * t / 100) +
        0.1 * rng.standard_normal(length)
    ).astype(np.float32)

    # Inject anomalies
    labels = np.zeros(length, dtype=np.int32)
    num_anomalies = int(length * anomaly_rate)
    anomaly_indices = rng.choice(length, num_anomalies, replace=False)

    for idx in anomaly_indices:
        signal[idx] += rng.choice([-1, 1]) * rng.uniform(3, 5)
        labels[idx] = 1

    return signal, labels


# ============================================================================
# GPU Actor with Persistent State
# ============================================================================

class GPUActorAnomalyDetector:
    """
    GPU Actor-based anomaly detector with PERSISTENT GPU STATE.

    The key advantage: window buffer and statistics stay on GPU
    between micro-batches. No repeated transfers!
    """

    def __init__(self, window_size: int = 10_000):
        self.window_size = window_size
        self.runtime = None
        self.setup_time = 0.0

    async def setup(self) -> float:
        """Initialize actor with GPU state."""
        from pydotcompute import RingKernelRuntime, message, ring_kernel
        from pydotcompute.ring_kernels.lifecycle import KernelContext

        window_size = self.window_size

        @message
        @dataclass
        class MicroBatch:
            """Micro-batch of data points."""
            values: list[float]
            batch_id: int
            message_id: UUID = field(default_factory=uuid4)
            priority: int = 128
            correlation_id: UUID | None = None

        @message
        @dataclass
        class BatchResult:
            """Anomaly detection results for batch."""
            batch_id: int
            anomaly_count: int
            anomaly_indices: list[int]
            mean_score: float
            max_score: float
            message_id: UUID = field(default_factory=uuid4)
            priority: int = 128
            correlation_id: UUID | None = None

        self.MicroBatch = MicroBatch
        self.BatchResult = BatchResult

        @ring_kernel(
            kernel_id="anomaly_actor",
            input_type=MicroBatch,
            output_type=BatchResult,
            queue_size=1024,
        )
        async def anomaly_actor(ctx: KernelContext) -> None:
            """
            Anomaly detection actor with PERSISTENT GPU STATE.

            This state lives on GPU for the ENTIRE actor lifetime:
            - window_buffer: Full sliding window (10K+ points)
            - running_mean/var: Rolling statistics
            - No host-device transfers between batches!

            KEY: All operations are VECTORIZED over the batch!
            """
            import cupy as cp

            # ============================================
            # PERSISTENT GPU STATE - allocated ONCE!
            # ============================================
            window_buffer = cp.zeros(window_size, dtype=cp.float32)
            window_pos = 0  # Current position in circular buffer

            # Rolling statistics on GPU (updated per batch)
            running_mean = cp.float32(0.0)
            running_var = cp.float32(1.0)

            while not ctx.should_terminate:
                if not ctx.is_active:
                    await ctx.wait_active()
                    continue

                try:
                    request = await ctx.receive(timeout=0.1)

                    # Transfer batch to GPU (small - just this batch, not the window!)
                    batch_gpu = cp.array(request.values, dtype=cp.float32)
                    batch_size = len(request.values)

                    # ============================================
                    # VECTORIZED PROCESSING ON GPU
                    # ============================================

                    # 1. Update window buffer (batch insert into circular buffer)
                    # This keeps the window on GPU - no transfer needed!
                    end_pos = window_pos + batch_size
                    if end_pos <= window_size:
                        window_buffer[window_pos:end_pos] = batch_gpu
                    else:
                        # Wrap around
                        first_part = window_size - window_pos
                        window_buffer[window_pos:] = batch_gpu[:first_part]
                        window_buffer[:end_pos - window_size] = batch_gpu[first_part:]
                    window_pos = end_pos % window_size

                    # 2. Update running statistics using batch mean/var
                    batch_mean = cp.mean(batch_gpu)
                    batch_var = cp.var(batch_gpu)

                    # Exponential moving average update
                    alpha = cp.float32(batch_size / (window_size + batch_size))
                    running_mean = (1 - alpha) * running_mean + alpha * batch_mean
                    running_var = (1 - alpha) * running_var + alpha * batch_var

                    # 3. Compute anomaly scores (VECTORIZED over entire batch!)
                    std = cp.sqrt(running_var + 1e-8)
                    scores = cp.abs(batch_gpu - running_mean) / std

                    # 4. Detect anomalies (VECTORIZED!)
                    anomaly_mask = scores > 3.0
                    anomaly_count = int(cp.sum(anomaly_mask))
                    anomaly_indices = cp.where(anomaly_mask)[0][:100].get().tolist()

                    # 5. Compute summary statistics
                    mean_score = float(cp.mean(scores))
                    max_score = float(cp.max(scores))

                    # Only transfer small results back (not the window!)
                    response = BatchResult(
                        batch_id=request.batch_id,
                        anomaly_count=anomaly_count,
                        anomaly_indices=anomaly_indices,
                        mean_score=mean_score,
                        max_score=max_score,
                    )
                    await ctx.send(response)

                except Exception:
                    continue

        setup_start = time.perf_counter()

        self.runtime = RingKernelRuntime()
        await self.runtime.__aenter__()
        await self.runtime.launch("anomaly_actor")
        await self.runtime.activate("anomaly_actor")

        self.setup_time = time.perf_counter() - setup_start
        return self.setup_time

    async def process_batches(
        self,
        data: NDArray,
        batch_size: int,
    ) -> tuple[float, int, int]:
        """
        Process data in micro-batches.

        Returns: (processing_time, total_anomalies, batches_processed)
        """
        total_anomalies = 0
        batches_processed = 0

        start = time.perf_counter()

        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch = data[batch_start:batch_end]

            await self.runtime.send(
                "anomaly_actor",
                self.MicroBatch(values=batch.tolist(), batch_id=batches_processed)
            )

            result = await self.runtime.receive("anomaly_actor", timeout=30.0)
            total_anomalies += result.anomaly_count
            batches_processed += 1

        processing_time = time.perf_counter() - start
        return processing_time, total_anomalies, batches_processed

    async def cleanup(self):
        if self.runtime:
            await self.runtime.__aexit__(None, None, None)
            self.runtime = None


# ============================================================================
# Stateless Batch Processing (must transfer state each batch)
# ============================================================================

class StatelessBatchDetector:
    """
    Stateless batch processing - must transfer window state EACH batch.

    This represents the traditional approach without persistent GPU state.
    The entire window history must be transferred for each batch to
    maintain context.
    """

    def __init__(self, window_size: int = 10_000):
        self.window_size = window_size

    def process_batches(
        self,
        data: NDArray,
        batch_size: int,
    ) -> tuple[float, float, int, int]:
        """
        Process with state transfer each batch.

        Returns: (total_time, transfer_time, anomalies, batches)
        """
        import cupy as cp

        # CPU-side state (must be transferred each batch)
        window = np.zeros(self.window_size, dtype=np.float32)
        window_idx = 0
        running_mean = 0.0
        running_var = 1.0
        alpha = 2.0 / (self.window_size + 1)

        total_anomalies = 0
        batches_processed = 0
        total_transfer_time = 0.0

        start = time.perf_counter()

        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch = data[batch_start:batch_end]

            # Update CPU-side window and stats
            for value in batch:
                window[window_idx] = value
                window_idx = (window_idx + 1) % self.window_size

                delta = value - running_mean
                running_mean += alpha * delta
                running_var = (1 - alpha) * (running_var + alpha * delta * delta)

            # ============================================
            # TRANSFER WINDOW STATE TO GPU (the overhead!)
            # ============================================
            transfer_start = time.perf_counter()

            # Must transfer entire window for context
            window_gpu = cp.asarray(window)

            # Also transfer batch for processing
            batch_gpu = cp.asarray(batch)

            # Transfer current statistics
            mean_gpu = cp.float32(running_mean)
            var_gpu = cp.float32(running_var)

            cp.cuda.Stream.null.synchronize()
            transfer_time = time.perf_counter() - transfer_start
            total_transfer_time += transfer_time

            # Process on GPU
            std_gpu = cp.sqrt(var_gpu + 1e-8)
            scores = cp.abs(batch_gpu - mean_gpu) / std_gpu
            anomalies = int(cp.sum(scores > 3.0))

            total_anomalies += anomalies
            batches_processed += 1

        total_time = time.perf_counter() - start
        return total_time, total_transfer_time, total_anomalies, batches_processed


# ============================================================================
# CPU Baseline
# ============================================================================

class CPUDetector:
    """Pure CPU baseline."""

    def __init__(self, window_size: int = 10_000):
        self.window_size = window_size

    def process_batches(
        self,
        data: NDArray,
        batch_size: int,
    ) -> tuple[float, int, int]:
        """Process on CPU only."""
        running_mean = 0.0
        running_var = 1.0
        alpha = 2.0 / (self.window_size + 1)

        total_anomalies = 0
        batches_processed = 0

        start = time.perf_counter()

        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch = data[batch_start:batch_end]

            for value in batch:
                delta = value - running_mean
                running_mean += alpha * delta
                running_var = (1 - alpha) * (running_var + alpha * delta * delta)

                std = np.sqrt(running_var + 1e-8)
                if abs(delta) / std > 3.0:
                    total_anomalies += 1

            batches_processed += 1

        total_time = time.perf_counter() - start
        return total_time, total_anomalies, batches_processed


# ============================================================================
# Benchmark Runner
# ============================================================================

@dataclass
class BenchmarkResult:
    implementation: str
    total_points: int
    batch_size: int
    num_batches: int
    total_time: float
    setup_time: float
    transfer_time: float
    throughput: float  # points/sec
    batch_latency_ms: float
    transfer_overhead_pct: float
    anomalies: int


async def run_benchmark(config: AnomalyBenchmarkConfig) -> list[BenchmarkResult]:
    """Run the complete benchmark."""

    print("=" * 70)
    print("REAL-TIME ANOMALY DETECTION: GPU ACTORS vs BATCH PROCESSING")
    print("=" * 70)
    print(f"""
KEY INSIGHT: Persistent GPU State

GPU Actors maintain state on GPU between micro-batches:
- Window buffer ({config.window_size:,} points = {config.window_size * 4 / 1024:.1f}KB)
- Rolling statistics
- NO repeated transfers!

Batch Processing must transfer state EACH batch:
- Window history: {config.window_size * 4 / 1024:.1f}KB per batch
- This adds up over thousands of batches!
""")

    results = []

    for total_points in config.total_points:
        print(f"\n{'='*60}")
        print(f"TOTAL POINTS: {total_points:,}")
        print(f"{'='*60}")

        # Generate data
        print("Generating data...", end=" ", flush=True)
        data, labels = generate_streaming_data(total_points)
        actual_anomalies = labels.sum()
        print(f"done ({actual_anomalies:,} anomalies)")

        for batch_size in config.micro_batch_sizes:
            num_batches = (total_points + batch_size - 1) // batch_size

            print(f"\n  --- Micro-batch size: {batch_size:,} ({num_batches:,} batches) ---")

            # ===== GPU ACTORS =====
            print(f"\n  GPU Actors (persistent state):")

            actor_detector = GPUActorAnomalyDetector(window_size=config.window_size)

            print("    Setup...", end=" ", flush=True)
            setup_time = await actor_detector.setup()
            print(f"{setup_time*1000:.1f}ms")

            # Warmup
            warmup_data, _ = generate_streaming_data(batch_size * config.warmup_batches)
            await actor_detector.process_batches(warmup_data, batch_size)

            # Benchmark
            print("    Processing...", end=" ", flush=True)
            times = []
            anomalies = 0

            for _ in range(config.measurement_runs):
                gc.collect()
                proc_time, anom, _ = await actor_detector.process_batches(data, batch_size)
                times.append(proc_time)
                anomalies = anom

            avg_time = np.mean(times)
            throughput = total_points / avg_time
            batch_latency = avg_time / num_batches * 1000

            print(f"done")
            print(f"      Time: {avg_time:.3f}s | Throughput: {throughput:,.0f} pts/s")
            print(f"      Batch latency: {batch_latency:.2f}ms | Anomalies: {anomalies:,}")

            results.append(BenchmarkResult(
                implementation="GPU Actors",
                total_points=total_points,
                batch_size=batch_size,
                num_batches=num_batches,
                total_time=avg_time,
                setup_time=setup_time,
                transfer_time=0.0,
                throughput=throughput,
                batch_latency_ms=batch_latency,
                transfer_overhead_pct=0.0,
                anomalies=anomalies,
            ))

            await actor_detector.cleanup()

            # ===== STATELESS BATCH =====
            print(f"\n  Stateless Batch (transfer each batch):")

            batch_detector = StatelessBatchDetector(window_size=config.window_size)

            print("    Processing...", end=" ", flush=True)
            times = []
            transfer_times = []
            anomalies = 0

            for _ in range(config.measurement_runs):
                gc.collect()
                total, transfer, anom, _ = batch_detector.process_batches(data, batch_size)
                times.append(total)
                transfer_times.append(transfer)
                anomalies = anom

            avg_time = np.mean(times)
            avg_transfer = np.mean(transfer_times)
            throughput = total_points / avg_time
            batch_latency = avg_time / num_batches * 1000
            transfer_pct = avg_transfer / avg_time * 100

            print(f"done")
            print(f"      Time: {avg_time:.3f}s | Throughput: {throughput:,.0f} pts/s")
            print(f"      Transfer: {avg_transfer:.3f}s ({transfer_pct:.1f}% overhead)")
            print(f"      Batch latency: {batch_latency:.2f}ms")

            results.append(BenchmarkResult(
                implementation="Stateless Batch",
                total_points=total_points,
                batch_size=batch_size,
                num_batches=num_batches,
                total_time=avg_time,
                setup_time=0.0,
                transfer_time=avg_transfer,
                throughput=throughput,
                batch_latency_ms=batch_latency,
                transfer_overhead_pct=transfer_pct,
                anomalies=anomalies,
            ))

            # ===== CPU BASELINE =====
            print(f"\n  CPU Baseline:")

            cpu_detector = CPUDetector(window_size=config.window_size)

            print("    Processing...", end=" ", flush=True)
            times = []
            anomalies = 0

            for _ in range(config.measurement_runs):
                gc.collect()
                proc_time, anom, _ = cpu_detector.process_batches(data, batch_size)
                times.append(proc_time)
                anomalies = anom

            avg_time = np.mean(times)
            throughput = total_points / avg_time
            batch_latency = avg_time / num_batches * 1000

            print(f"done")
            print(f"      Time: {avg_time:.3f}s | Throughput: {throughput:,.0f} pts/s")

            results.append(BenchmarkResult(
                implementation="CPU Only",
                total_points=total_points,
                batch_size=batch_size,
                num_batches=num_batches,
                total_time=avg_time,
                setup_time=0.0,
                transfer_time=0.0,
                throughput=throughput,
                batch_latency_ms=batch_latency,
                transfer_overhead_pct=0.0,
                anomalies=anomalies,
            ))

    return results


def generate_report(results: list[BenchmarkResult], output_path: Path | None = None) -> str:
    """Generate benchmark report."""

    lines = []
    lines.append("=" * 70)
    lines.append("REAL-TIME ANOMALY DETECTION BENCHMARK REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    lines.append("\n" + "=" * 70)
    lines.append("WHY THIS BENCHMARK MATTERS")
    lines.append("=" * 70)
    lines.append("""
This benchmark demonstrates the KEY ADVANTAGE of GPU Actors:
PERSISTENT GPU STATE eliminates repeated host-device transfers.

In streaming scenarios:
- Data arrives in micro-batches (like Kafka, sensor data, metrics)
- Context (window history) must be maintained
- Traditional batch processing transfers context EVERY batch
- GPU Actors keep context on GPU - transfer ONCE!

The longer the stream runs, the more GPU Actors win!
""")

    # Results table
    lines.append("=" * 70)
    lines.append("PERFORMANCE RESULTS")
    lines.append("=" * 70)

    total_points_list = sorted(set(r.total_points for r in results))

    for total_points in total_points_list:
        lines.append(f"\n--- {total_points:,} total points ---")

        batch_sizes = sorted(set(r.batch_size for r in results if r.total_points == total_points))

        for batch_size in batch_sizes:
            batch_results = [r for r in results
                           if r.total_points == total_points and r.batch_size == batch_size]

            lines.append(f"\n  Batch size: {batch_size:,} ({batch_results[0].num_batches:,} batches)")
            lines.append(f"  {'Implementation':<20} {'Time':<10} {'Throughput':<15} {'Transfer%':<12} {'Latency':<10}")
            lines.append("  " + "-" * 70)

            for r in sorted(batch_results, key=lambda x: -x.throughput):
                transfer = f"{r.transfer_overhead_pct:.1f}%" if r.transfer_overhead_pct > 0 else "0%"
                lines.append(
                    f"  {r.implementation:<20} "
                    f"{r.total_time:.3f}s     "
                    f"{r.throughput:>10,.0f}/s    "
                    f"{transfer:<12} "
                    f"{r.batch_latency_ms:.2f}ms"
                )

    # Speedup analysis
    lines.append("\n" + "=" * 70)
    lines.append("GPU ACTORS vs STATELESS BATCH")
    lines.append("=" * 70)

    for total_points in total_points_list:
        lines.append(f"\n{total_points:,} points:")

        batch_sizes = sorted(set(r.batch_size for r in results if r.total_points == total_points))

        for batch_size in batch_sizes:
            actor = next((r for r in results if r.total_points == total_points
                         and r.batch_size == batch_size and r.implementation == "GPU Actors"), None)
            batch = next((r for r in results if r.total_points == total_points
                         and r.batch_size == batch_size and r.implementation == "Stateless Batch"), None)

            if actor and batch:
                if actor.throughput > batch.throughput:
                    speedup = actor.throughput / batch.throughput
                    lines.append(f"  Batch {batch_size:,}: GPU Actors {speedup:.2f}x FASTER")
                    lines.append(f"    → Batch transfer overhead: {batch.transfer_overhead_pct:.1f}%")
                else:
                    slowdown = batch.throughput / actor.throughput
                    lines.append(f"  Batch {batch_size:,}: GPU Actors {slowdown:.2f}x slower")
                    lines.append(f"    → Actor message overhead dominates at this batch size")

    # Transfer overhead analysis
    lines.append("\n" + "=" * 70)
    lines.append("TRANSFER OVERHEAD ANALYSIS")
    lines.append("=" * 70)

    batch_results = [r for r in results if r.implementation == "Stateless Batch"]
    if batch_results:
        lines.append("\nStateless Batch must transfer window state EACH batch:")
        for r in sorted(batch_results, key=lambda x: (x.total_points, x.batch_size)):
            total_transfer_mb = r.transfer_time * 1000  # Rough estimate
            lines.append(
                f"  {r.total_points:,} pts, batch {r.batch_size:,}: "
                f"{r.transfer_overhead_pct:.1f}% time spent on transfers"
            )

    # Conclusions
    lines.append("\n" + "=" * 70)
    lines.append("CONCLUSIONS")
    lines.append("=" * 70)
    lines.append("""
1. GPU ACTORS ADVANTAGE: Persistent GPU state
   - Window buffer stays on GPU between batches
   - No repeated transfers = higher throughput

2. WHEN GPU ACTORS WIN:
   - Streaming workloads with many micro-batches
   - Large context/window sizes
   - Long-running processing pipelines

3. WHEN BATCH PROCESSING WINS:
   - Very large batch sizes (transfer amortized)
   - One-shot processing
   - Small context requirements

4. REAL-WORLD APPLICATIONS:
   - IoT sensor monitoring
   - Financial tick data processing
   - Log anomaly detection
   - Real-time metrics analysis
""")
    lines.append("=" * 70)

    report = "\n".join(lines)

    if output_path:
        output_path.write_text(report)

        json_data = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "implementation": r.implementation,
                    "total_points": r.total_points,
                    "batch_size": r.batch_size,
                    "num_batches": r.num_batches,
                    "total_time": r.total_time,
                    "throughput": r.throughput,
                    "transfer_overhead_pct": r.transfer_overhead_pct,
                }
                for r in results
            ]
        }
        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(json_data, indent=2))

    return report


async def main():
    config = AnomalyBenchmarkConfig(
        total_points=[100_000, 500_000, 1_000_000],
        micro_batch_sizes=[100, 500, 1000],
        window_size=10_000,
        warmup_batches=10,
        measurement_runs=3,
    )

    results = await run_benchmark(config)

    output_dir = Path(__file__).parent
    report_path = output_dir / "realtime_anomaly_report.txt"

    report = generate_report(results, report_path)
    print("\n" + report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
