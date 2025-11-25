# PyDotCompute

**A GPU-native actor model with persistent kernels and message passing.**

PyDotCompute is a Python port of DotCompute's Ring Kernel System, bringing the power of GPU-native actors to Python developers. It enables you to create persistent GPU kernels that communicate through high-performance message queues.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Get up and running with PyDotCompute in minutes.

    [:octicons-arrow-right-24: Getting Started](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Concepts**

    ---

    Understand the core concepts behind ring kernels and GPU actors.

    [:octicons-arrow-right-24: Learn Concepts](articles/concepts/ring-kernels.md)

-   :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Complete API documentation for all modules.

    [:octicons-arrow-right-24: API Docs](api/index.md)

-   :material-tools:{ .lg .middle } **Practitioner's Guide**

    ---

    Best practices for building production GPU actors.

    [:octicons-arrow-right-24: Guides](articles/guides/building-actors.md)

</div>

## Why PyDotCompute?

Traditional GPU programming involves launching kernels, waiting for completion, and transferring results back to the host. This approach introduces latency and doesn't scale well for streaming workloads.

**PyDotCompute changes this paradigm:**

```python
from pydotcompute import RingKernelRuntime, ring_kernel, message

@message
class DataPoint:
    value: float

@ring_kernel(kernel_id="processor")
async def stream_processor(ctx):
    """Persistent actor processing streaming data."""
    while not ctx.should_terminate:
        data = await ctx.receive()
        result = process(data)
        await ctx.send(result)

async with RingKernelRuntime() as runtime:
    await runtime.launch("processor")
    await runtime.activate("processor")

    # Stream data continuously
    for point in data_stream:
        await runtime.send("processor", point)
        result = await runtime.receive("processor")
```

### Key Benefits

| Feature | Traditional GPU | PyDotCompute |
|---------|-----------------|--------------|
| Kernel Lifetime | Per-invocation | Persistent |
| Communication | Memory copies | Message queues |
| Latency | High (launch overhead) | Low (always running) |
| Programming Model | Imperative | Actor-based |
| State Management | Manual | Automatic |

## Performance Highlights

| Metric | Value |
|--------|-------|
| Message latency (p50) | **21μs** |
| Message latency (p99) | **131μs** |
| GPU graph processing | **1.7M edges/sec** |
| Actor throughput | **76K msg/sec** |
| Cython queue ops | **0.33μs** |

*Benchmarked with uvloop on Linux.*

## Features

- **Ring Kernel System**: Persistent GPU kernels with infinite processing loops
- **High Performance**: uvloop auto-installation for 21μs message latency
- **Performance Tiers**: From uvloop (default) to Cython extensions
- **Message Passing**: Type-safe, high-performance message serialization
- **Unified Memory**: Transparent host-device memory with lazy synchronization
- **Lifecycle Management**: Two-phase launch with graceful shutdown
- **GPU Telemetry**: Real-time monitoring and performance metrics
- **CPU Backend**: Full compatibility for development and testing

## Architecture Overview

```
PyDotCompute Ring Kernel System
├── Ring Kernels          │ Performance Tiers      │ CUDA Backend
│   • RingKernelRuntime   │ • uvloop (21μs)        │ • Numba JIT
│   • FastMessageQueue    │ • ThreadedRingKernel   │ • CuPy arrays
│   • @ring_kernel        │ • CythonRingKernel     │ • Zero-copy DMA
│   • @message            │ • FastSPSCQueue        │ • PTX caching
├─────────────────────────┴────────────────────────┴─────────────────
│ Memory: UnifiedBuffer, MemoryPool, Accelerator
```

## Installation

=== "Basic (CPU only)"

    ```bash
    pip install pydotcompute
    ```

=== "Fast (Recommended)"

    ```bash
    pip install pydotcompute[fast]
    ```

    Includes uvloop for 21μs message latency.

=== "With CUDA support"

    ```bash
    pip install pydotcompute[cuda,fast]
    ```

=== "Development"

    ```bash
    git clone https://github.com/mivertowski/PyDotCompute.git
    cd PyDotCompute
    pip install -e ".[dev]"
    ```

## Quick Example

```python
import asyncio
from pydotcompute import RingKernelRuntime, ring_kernel, message

@message
class ComputeRequest:
    values: list[float]

@message
class ComputeResponse:
    result: float

@ring_kernel(kernel_id="summer", queue_size=1000)
async def sum_actor(ctx):
    while not ctx.should_terminate:
        request = await ctx.receive()
        total = sum(request.values)
        await ctx.send(ComputeResponse(result=total))

async def main():
    async with RingKernelRuntime() as runtime:
        await runtime.launch("summer")
        await runtime.activate("summer")

        await runtime.send("summer", ComputeRequest(values=[1, 2, 3, 4, 5]))
        response = await runtime.receive("summer")

        print(f"Sum: {response.result}")  # Sum: 15.0

asyncio.run(main())
```

## Next Steps

- **[Quick Start](getting-started/quickstart.md)**: Get running in 5 minutes
- **[First Ring Kernel](getting-started/first-kernel.md)**: Build your first GPU actor
- **[Concepts](articles/concepts/ring-kernels.md)**: Deep dive into the architecture
- **[API Reference](api/index.md)**: Complete API documentation
