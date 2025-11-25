# Building Pipelines

Creating multi-stage processing pipelines with ring kernels.

## Overview

Pipelines chain multiple actors together, where each stage processes data and passes it to the next. This enables modular, scalable data processing.

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Input  │ ──► │ Stage 1 │ ──► │ Stage 2 │ ──► │ Stage 3 │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
```

## Simple Pipeline

### Define Stages

```python
from pydotcompute import ring_kernel, message, RingKernelRuntime
from dataclasses import dataclass, field
from uuid import UUID, uuid4

# Stage 1: Preprocess
@message
@dataclass
class RawData:
    values: list[float]
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None

@message
@dataclass
class PreprocessedData:
    normalized: list[float]
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None

@ring_kernel(kernel_id="preprocess")
async def preprocess_stage(ctx):
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            raw = await ctx.receive(timeout=0.1)

            # Normalize values
            min_val = min(raw.values)
            max_val = max(raw.values)
            normalized = [(v - min_val) / (max_val - min_val) for v in raw.values]

            await ctx.send(PreprocessedData(
                normalized=normalized,
                correlation_id=raw.message_id,
            ))
        except:
            continue

# Stage 2: Compute
@message
@dataclass
class ComputeResult:
    result: float
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None

@ring_kernel(kernel_id="compute")
async def compute_stage(ctx):
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            data = await ctx.receive(timeout=0.1)

            # Compute statistics
            result = sum(data.normalized) / len(data.normalized)

            await ctx.send(ComputeResult(
                result=result,
                correlation_id=data.message_id,
            ))
        except:
            continue
```

### Connect Pipeline

```python
async def run_pipeline():
    async with RingKernelRuntime() as runtime:
        # Launch all stages
        await runtime.launch("preprocess")
        await runtime.launch("compute")

        # Activate all stages
        await runtime.activate("preprocess")
        await runtime.activate("compute")

        await asyncio.sleep(0.1)

        # Send to first stage
        await runtime.send("preprocess", RawData(values=[1, 2, 3, 4, 5]))

        # Get from first stage
        preprocessed = await runtime.receive("preprocess", timeout=1.0)

        # Send to second stage
        await runtime.send("compute", preprocessed)

        # Get final result
        result = await runtime.receive("compute", timeout=1.0)

        print(f"Result: {result.result}")
```

## Pipeline Coordinator

For complex pipelines, use a coordinator:

```python
@ring_kernel(kernel_id="pipeline_coordinator")
async def pipeline_coordinator(ctx):
    """Coordinates multi-stage pipeline execution."""

    stages = ["stage1", "stage2", "stage3"]

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            # Receive initial request
            request = await ctx.receive(timeout=0.1)

            # Track through pipeline
            current_data = request
            for stage in stages:
                # Send to stage
                await runtime.send(stage, current_data)

                # Get result
                current_data = await runtime.receive(stage, timeout=5.0)

                if not current_data.success:
                    # Pipeline failed at this stage
                    await ctx.send(PipelineError(
                        failed_stage=stage,
                        error=current_data.error,
                        correlation_id=request.message_id,
                    ))
                    break
            else:
                # All stages completed
                await ctx.send(PipelineResult(
                    result=current_data,
                    correlation_id=request.message_id,
                ))

        except asyncio.TimeoutError:
            continue
```

## Parallel Pipelines

Run multiple pipeline instances in parallel:

```python
async def parallel_pipeline():
    async with RingKernelRuntime() as runtime:
        # Launch multiple workers per stage
        for i in range(3):
            await runtime.launch(f"preprocess_{i}", preprocess_stage)
            await runtime.launch(f"compute_{i}", compute_stage)

        # Activate all
        for i in range(3):
            await runtime.activate(f"preprocess_{i}")
            await runtime.activate(f"compute_{i}")

        await asyncio.sleep(0.1)

        # Round-robin distribution
        worker_idx = 0
        for data in data_stream:
            stage = f"preprocess_{worker_idx}"
            await runtime.send(stage, data)
            worker_idx = (worker_idx + 1) % 3
```

## Fan-Out / Fan-In

### Fan-Out (One to Many)

```python
@ring_kernel(kernel_id="distributor")
async def distributor(ctx):
    """Distributes work to multiple workers."""
    workers = ["worker_0", "worker_1", "worker_2"]
    idx = 0

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            work = await ctx.receive(timeout=0.1)

            # Send to next worker (round-robin)
            await runtime.send(workers[idx], work)
            idx = (idx + 1) % len(workers)

        except:
            continue
```

### Fan-In (Many to One)

```python
@ring_kernel(kernel_id="aggregator")
async def aggregator(ctx):
    """Collects results from multiple workers."""
    batch = []
    batch_size = 10

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            result = await ctx.receive(timeout=0.1)
            batch.append(result)

            if len(batch) >= batch_size:
                # Aggregate and send
                aggregated = aggregate_results(batch)
                await ctx.send(aggregated)
                batch = []

        except asyncio.TimeoutError:
            # Flush partial batch on timeout
            if batch:
                aggregated = aggregate_results(batch)
                await ctx.send(aggregated)
                batch = []
```

## DAG Pipelines

For directed acyclic graph (DAG) pipelines:

```
        ┌───────┐
        │   A   │
        └───┬───┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌───────┐     ┌───────┐
│   B   │     │   C   │
└───┬───┘     └───┬───┘
     │             │
     └──────┬──────┘
            ▼
        ┌───────┐
        │   D   │
        └───────┘
```

```python
@ring_kernel(kernel_id="dag_coordinator")
async def dag_coordinator(ctx):
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            request = await ctx.receive(timeout=0.1)

            # Stage A
            await runtime.send("stage_a", request)
            a_result = await runtime.receive("stage_a", timeout=5.0)

            # Stages B and C in parallel
            await runtime.send("stage_b", a_result)
            await runtime.send("stage_c", a_result)

            # Gather B and C results
            b_result = await runtime.receive("stage_b", timeout=5.0)
            c_result = await runtime.receive("stage_c", timeout=5.0)

            # Stage D with both inputs
            d_input = MergedInput(b=b_result, c=c_result)
            await runtime.send("stage_d", d_input)
            d_result = await runtime.receive("stage_d", timeout=5.0)

            await ctx.send(FinalResult(
                result=d_result,
                correlation_id=request.message_id,
            ))

        except:
            continue
```

## Backpressure Handling

Handle slow stages:

```python
@ring_kernel(
    kernel_id="slow_stage",
    queue_size=100,
    backpressure=BackpressureStrategy.BLOCK,  # Wait when full
)
async def slow_stage(ctx):
    while not ctx.should_terminate:
        try:
            data = await ctx.receive(timeout=0.1)
            result = slow_processing(data)  # Takes time
            await ctx.send(result)
        except:
            continue
```

### Strategies

| Strategy | Use Case |
|----------|----------|
| `BLOCK` | Guarantee no data loss |
| `REJECT` | Immediate feedback to caller |
| `DROP_OLDEST` | Real-time, latest data matters |

## Error Handling in Pipelines

```python
@message
@dataclass
class StageResult:
    data: Any
    success: bool = True
    error: str | None = None
    stage: str = ""

@ring_kernel(kernel_id="resilient_stage")
async def resilient_stage(ctx):
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            input_data = await ctx.receive(timeout=0.1)

            try:
                result = process(input_data)
                await ctx.send(StageResult(
                    data=result,
                    success=True,
                    stage=ctx.kernel_id,
                ))
            except Exception as e:
                await ctx.send(StageResult(
                    data=None,
                    success=False,
                    error=str(e),
                    stage=ctx.kernel_id,
                ))

        except asyncio.TimeoutError:
            continue
```

## Monitoring Pipelines

Track pipeline metrics:

```python
async with RingKernelRuntime(enable_telemetry=True) as runtime:
    # ... run pipeline ...

    # Get per-stage metrics
    for stage in ["stage1", "stage2", "stage3"]:
        telemetry = runtime.get_telemetry(stage)
        print(f"{stage}:")
        print(f"  Throughput: {telemetry.throughput:.1f} msg/s")
        print(f"  Processed: {telemetry.messages_processed}")
```

## Best Practices

1. **Clear Stage Boundaries**: Each stage should do one thing well

2. **Typed Messages**: Use specific types for each stage transition

3. **Error Propagation**: Include stage information in errors

4. **Backpressure Strategy**: Choose based on requirements

5. **Monitoring**: Track each stage's performance

6. **Timeouts**: Prevent infinite waits

## Next Steps

- [GPU Optimization](gpu-optimization.md): Performance tuning
- [Testing](testing.md): Testing pipelines
