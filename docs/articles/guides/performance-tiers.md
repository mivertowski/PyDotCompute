# Performance Tiers

PyDotCompute offers three performance tiers to match your use case. Choose the right tier based on your workload characteristics.

## Overview

| Tier | Implementation | Latency (p50) | Use Case |
|------|---------------|---------------|----------|
| **1 (Default)** | uvloop + FastMessageQueue | **21μs** | Async Python code |
| 2 | ThreadedRingKernel | ~100μs | Blocking I/O, C extensions |
| 3 | CythonRingKernel | **0.33μs** queue ops | Multi-process IPC |

## Key Insight

**uvloop (Tier 1) is optimal for pure Python due to GIL.** Threading adds overhead from context switching. Cython queues shine in multi-process scenarios where the GIL isn't shared.

## Tier 1: Async (Default)

The default tier uses uvloop and FastMessageQueue for optimal async performance.

### Characteristics

- **Latency**: 21μs (p50), 131μs (p99)
- **Throughput**: 76K msg/sec
- **Best for**: Pure Python async code, I/O-bound workloads

### Usage

Automatically enabled when you import `pydotcompute`:

```python
from pydotcompute import RingKernelRuntime, ring_kernel, message

@message
class Request:
    value: int

@ring_kernel(kernel_id="processor")
async def processor(ctx):
    while not ctx.should_terminate:
        msg = await ctx.receive(timeout=0.1)
        if msg:
            await ctx.send(Response(result=msg.value * 2))

async with RingKernelRuntime() as runtime:
    # uvloop is auto-installed for 21μs latency
    await runtime.launch("processor")
    await runtime.activate("processor")
```

### When to Use

- Standard async Python applications
- I/O-bound workloads
- Web services and APIs
- Real-time streaming pipelines

## Tier 2: Threaded

For blocking operations or GIL-releasing C extensions.

### Characteristics

- **Latency**: ~100μs per message
- **Best for**: Blocking I/O, NumPy operations, C extensions
- **Trade-off**: Higher latency but supports blocking code

### Usage

```python
from pydotcompute.ring_kernels import ThreadedRingKernel, ThreadedKernelContext

def blocking_kernel(ctx: ThreadedKernelContext):
    """Kernel that can perform blocking operations."""
    while not ctx.should_terminate:
        msg = ctx.receive(timeout=0.1)  # Blocking receive
        if msg:
            # Blocking operations are OK here
            result = expensive_computation(msg)
            ctx.send(result)

# Use as context manager
with ThreadedRingKernel("worker", blocking_kernel) as kernel:
    kernel.send(request)
    response = kernel.receive(timeout=1.0)
```

### Thread Pool

For managing multiple threaded kernels:

```python
from pydotcompute.ring_kernels import ThreadedKernelPool

with ThreadedKernelPool(max_workers=4) as pool:
    # Launch multiple workers
    pool.launch("worker_1", worker_func)
    pool.launch("worker_2", worker_func)

    # Distribute work
    pool.send("worker_1", task1)
    pool.send("worker_2", task2)
```

### When to Use

- Calling blocking libraries (requests, file I/O)
- NumPy/SciPy operations that release the GIL
- Integrating with C extensions
- Mixed async/blocking workloads

## Tier 3: Cython (Maximum Performance)

For multi-process scenarios requiring ultimate performance.

### Characteristics

- **Queue Operations**: 0.33μs (vs 1.8μs for pure Python)
- **Best for**: Multi-process IPC, high-frequency trading
- **Requirement**: Cython extensions must be built

### Installation

```bash
pip install pydotcompute[cython]
python setup_cython.py build_ext --inplace
```

### Usage

```python
from pydotcompute.ring_kernels import CythonRingKernel, is_cython_kernel_available

# Check availability
if is_cython_kernel_available():
    def fast_kernel(ctx):
        while not ctx.should_terminate:
            msg = ctx.receive(timeout=0.001)  # 1ms timeout
            if msg:
                ctx.send(process(msg))

    with CythonRingKernel("fast_worker", fast_kernel) as kernel:
        kernel.send(request)
        response = kernel.receive()
else:
    # Fallback to threaded kernel
    with ThreadedRingKernel("worker", kernel_func) as kernel:
        ...
```

### When to Use

- Multi-process architectures
- High-frequency message passing
- Latency-critical applications
- When GIL contention is a bottleneck

## Performance Comparison

### Queue Operations

| Queue Type | Put+Get (same thread) |
|------------|----------------------|
| FastMessageQueue (Python) | 1.8μs |
| FastSPSCQueue (Cython) | **0.33μs** |

### Full Actor Roundtrip

```
Tier 1 (uvloop + FastMessageQueue):
  p50:  63μs
  p95:  103μs
  p99:  131μs
  mean: 70μs

Tier 2 (ThreadedRingKernel):
  p50:  ~100μs
  p95:  ~150μs
  p99:  ~200μs

Tier 3 (CythonRingKernel - queue only):
  put+get: 0.33μs
```

## Choosing the Right Tier

```
┌─────────────────────────────────────────────────────────────┐
│                     Start Here                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
         ┌───────────────────────────┐
         │ Is your code async Python? │
         └─────────────┬─────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼ Yes                     ▼ No
   ┌──────────────┐        ┌──────────────────────┐
   │   Tier 1     │        │ Do you need blocking │
   │   (uvloop)   │        │   operations?        │
   └──────────────┘        └──────────┬───────────┘
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                           ▼ Yes                 ▼ No
                    ┌──────────────┐      ┌──────────────┐
                    │   Tier 2     │      │   Tier 3     │
                    │  (Threaded)  │      │   (Cython)   │
                    └──────────────┘      └──────────────┘
```

### Decision Guide

| Your Situation | Recommended Tier |
|---------------|------------------|
| Standard async Python | **Tier 1** |
| Calling blocking APIs | **Tier 2** |
| Multi-process architecture | **Tier 3** |
| Maximum queue performance | **Tier 3** |
| Simple setup, good performance | **Tier 1** |
| NumPy/SciPy heavy computation | **Tier 2** |

## Disabling uvloop

If you need to disable uvloop auto-installation:

```bash
PYDOTCOMPUTE_NO_UVLOOP=1 python my_script.py
```

Or in Python:

```python
import os
os.environ["PYDOTCOMPUTE_NO_UVLOOP"] = "1"

from pydotcompute import RingKernelRuntime
```

## Mixing Tiers

You can use multiple tiers in the same application:

```python
import asyncio
from pydotcompute import RingKernelRuntime, ring_kernel
from pydotcompute.ring_kernels import ThreadedRingKernel

# Tier 1: Async orchestrator
@ring_kernel(kernel_id="orchestrator")
async def orchestrator(ctx):
    while not ctx.should_terminate:
        request = await ctx.receive(timeout=0.1)
        if request:
            # Route to workers
            await ctx.send(process(request))

# Tier 2: Blocking worker
def blocking_worker(ctx):
    while not ctx.should_terminate:
        msg = ctx.receive(timeout=0.1)
        if msg:
            result = blocking_api_call(msg)  # OK to block
            ctx.send(result)

async def main():
    # Start threaded worker
    with ThreadedRingKernel("worker", blocking_worker) as worker:
        # Start async runtime
        async with RingKernelRuntime() as runtime:
            await runtime.launch("orchestrator")
            await runtime.activate("orchestrator")

            # Use both tiers together
            ...

asyncio.run(main())
```

## Lessons Learned

1. **uvloop beats threading for Python**: The GIL makes native threading slower than uvloop's libuv-based event loop for message passing.

2. **Queue operations are fast, synchronization is slow**: Raw queue ops are ~1-2μs, but thread context switching adds 50-100μs.

3. **Cython queues need multi-process**: The Cython FastSPSCQueue achieves 0.33μs but only shines in multi-process scenarios where GIL isn't shared.

## Next Steps

- [Building Actors](building-actors.md): Best practices for actor design
- [GPU Optimization](gpu-optimization.md): Getting the most from GPU
- [Testing](testing.md): Testing your actors
