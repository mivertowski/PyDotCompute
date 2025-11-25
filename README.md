# PyDotCompute

[![PyPI version](https://img.shields.io/pypi/v/pydotcompute.svg)](https://pypi.org/project/pydotcompute/)
[![Python versions](https://img.shields.io/pypi/pyversions/pydotcompute.svg)](https://pypi.org/project/pydotcompute/)
[![License](https://img.shields.io/pypi/l/pydotcompute.svg)](https://github.com/mivertowski/PyDotCompute/blob/main/LICENSE)
[![CI](https://github.com/mivertowski/PyDotCompute/actions/workflows/ci.yml/badge.svg)](https://github.com/mivertowski/PyDotCompute/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://mivertowski.github.io/PyDotCompute/)

A Python port of DotCompute's Ring Kernel System - a GPU-native actor model with persistent kernels and message passing.

## Overview

PyDotCompute brings GPU-native actor model capabilities to Python, enabling developers to create persistent GPU kernels that communicate through message queues. This approach is ideal for:

- Real-time GPU compute pipelines
- Streaming data processing on GPU
- Actor-based GPU programming
- High-throughput message-driven architectures

## Performance Highlights

| Metric | Value |
|--------|-------|
| Message latency (p50) | **21μs** |
| Message latency (p99) | **131μs** |
| GPU graph processing | **1.7M edges/sec** |
| Actor throughput | **76K msg/sec** |
| Cython queue ops | **0.33μs** |

*Benchmarked with uvloop on Linux. See [Benchmarks](#benchmarks) for details.*

## Features

- **Ring Kernel System**: Persistent GPU kernels with infinite loops and message queues
- **High Performance**: uvloop auto-installation for 21μs message latency
- **Message Passing**: Type-safe message serialization with msgpack
- **Unified Memory**: Transparent host-device memory management with lazy synchronization
- **Lifecycle Management**: Two-phase launch (launch -> activate) with graceful shutdown
- **Telemetry**: Real-time GPU monitoring and kernel performance metrics
- **Backend Support**: CPU simulation and CUDA acceleration via Numba/CuPy
- **Performance Tiers**: From uvloop (default) to Cython extensions

## Installation

```bash
# Basic installation (CPU only)
pip install pydotcompute

# With CUDA support
pip install pydotcompute[cuda]

# With performance optimizations (uvloop - Linux/macOS)
pip install pydotcompute[fast]

# With Cython extensions (maximum performance)
pip install pydotcompute[cython]
python setup_cython.py build_ext --inplace

# Development installation
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from pydotcompute import RingKernelRuntime, ring_kernel, message

# Define message types
@message
class ComputeRequest:
    values: list[float]

@message
class ComputeResponse:
    result: float

# Define a ring kernel actor
@ring_kernel(
    kernel_id="compute",
    input_type=ComputeRequest,
    output_type=ComputeResponse,
)
async def compute_actor(ctx):
    while not ctx.should_terminate:
        msg = await ctx.receive()
        result = sum(msg.values)
        await ctx.send(ComputeResponse(result=result))

# Use the runtime (automatically uses uvloop for best performance)
async def main():
    async with RingKernelRuntime() as runtime:
        await runtime.launch("compute")
        await runtime.activate("compute")

        await runtime.send("compute", ComputeRequest(values=[1.0, 2.0, 3.0]))
        response = await runtime.receive("compute")

        print(f"Result: {response.result}")  # 6.0

asyncio.run(main())
```

## Performance Tiers

PyDotCompute offers three performance tiers to match your use case:

| Tier | Implementation | Latency (p50) | Use Case |
|------|---------------|---------------|----------|
| **1 (Default)** | uvloop + FastMessageQueue | **21μs** | Async Python code |
| 2 | ThreadedRingKernel | ~100μs | Blocking I/O, C extensions |
| 3 | CythonRingKernel | **0.33μs** queue ops | Multi-process IPC |

### Tier 1: Async (Default)

Automatically enabled when you import `pydotcompute`. Uses uvloop on Linux/macOS.

```python
async with RingKernelRuntime() as runtime:
    # uvloop is auto-installed for 21μs latency
    await runtime.launch("my_kernel")
    await runtime.activate("my_kernel")
```

### Tier 2: Threaded

For blocking operations or GIL-releasing C extensions:

```python
from pydotcompute.ring_kernels import ThreadedRingKernel, ThreadedKernelContext

def blocking_kernel(ctx: ThreadedKernelContext):
    while not ctx.should_terminate:
        msg = ctx.receive(timeout=0.1)
        if msg:
            ctx.send(process(msg))

with ThreadedRingKernel("worker", blocking_kernel) as kernel:
    kernel.send(request)
    response = kernel.receive()
```

### Tier 3: Cython (Maximum Performance)

For multi-process scenarios or Cython extensions:

```python
from pydotcompute.ring_kernels import CythonRingKernel, is_cython_kernel_available

if is_cython_kernel_available():
    # 0.33μs queue operations
    with CythonRingKernel("fast_worker", my_kernel) as kernel:
        kernel.send(request)
```

## Benchmarks

### Message Latency

```
GPU Actors (1000 samples):
  p50:  63μs
  p95:  103μs
  p99:  131μs
  mean: 70μs
```

### Graph Processing (PageRank)

| Graph Size | CPU Sparse | GPU Batch | Speedup |
|------------|------------|-----------|---------|
| 1K nodes   | 6.8ms      | 64ms      | CPU wins |
| 5K nodes (dense) | 256ms | 200ms | **GPU 1.28x** |
| 1M nodes   | 39.6s      | 4.25s     | **GPU 9.3x** |

**Crossover**: GPU wins at 50K+ nodes

### Streaming Throughput

| Scenario | GPU Actors | Advantage |
|----------|-----------|-----------|
| Persistent state | Yes | No repeated GPU transfers |
| Transfer overhead | 0% | vs 16-28% for batch |
| Best for | Long-running pipelines | Context preservation |

## Architecture

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

## Core Components

### UnifiedBuffer

Transparent host-device memory management:

```python
from pydotcompute import UnifiedBuffer
import numpy as np

buffer = UnifiedBuffer((1000,), dtype=np.float32)
buffer.allocate()

# Write on host
buffer.host[:] = np.random.randn(1000)
buffer.mark_host_dirty()

# Access on device (auto-syncs)
await buffer.ensure_on_device()
device_data = buffer.device
```

### Ring Kernels

Persistent actors with message queues:

```python
@ring_kernel(kernel_id="processor", queue_size=4096)
async def processor(ctx):
    while not ctx.should_terminate:
        msg = await ctx.receive(timeout=0.1)
        # Process message
        await ctx.send(response)
```

### Lifecycle Management

```python
async with RingKernelRuntime() as runtime:
    # Phase 1: Launch (allocate resources)
    await runtime.launch("my_kernel")

    # Phase 2: Activate (start processing)
    await runtime.activate("my_kernel")

    # Use the kernel...

    # Deactivate (pause) or Terminate (cleanup)
    await runtime.deactivate("my_kernel")
    await runtime.reactivate("my_kernel")
```

## Project Structure

```
pydotcompute/
├── core/
│   ├── accelerator.py      # GPU device abstraction
│   ├── unified_buffer.py   # Host-device memory
│   ├── memory_pool.py      # Memory pooling
│   └── orchestrator.py     # Compute coordination
├── ring_kernels/
│   ├── runtime.py          # Main runtime (uvloop)
│   ├── message.py          # Message serialization
│   ├── queue.py            # Async message queues
│   ├── fast_queue.py       # O(1) priority queue
│   ├── lifecycle.py        # Kernel lifecycle
│   ├── telemetry.py        # Performance monitoring
│   ├── _loop.py            # uvloop auto-install
│   ├── sync_queue.py       # Threading queues
│   ├── threaded_kernel.py  # Tier 2 kernel
│   ├── cython_kernel.py    # Tier 3 kernel
│   └── _cython/            # Cython extensions
│       └── fast_spsc.pyx   # 0.33μs queue
├── backends/
│   ├── cpu.py              # CPU simulation
│   └── cuda.py             # CUDA via Numba/CuPy
├── compilation/
│   ├── compiler.py         # Kernel compilation
│   └── cache.py            # PTX caching
└── decorators/
    ├── kernel.py           # @kernel decorator
    ├── ring_kernel.py      # @ring_kernel decorator
    └── validators.py       # Runtime validation
```

## Testing

```bash
# Run all tests (398 passing)
pytest

# Run with coverage
pytest --cov=pydotcompute

# Run only unit tests
pytest tests/unit/

# Skip CUDA tests (if no GPU)
pytest -m "not cuda"

# Run benchmarks
python benchmarks/extended_benchmark.py
python benchmarks/pagerank_benchmark.py
python benchmarks/realtime_anomaly_benchmark.py
```

## Requirements

- Python >= 3.11
- numpy >= 1.26.0
- msgpack >= 1.0.0

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| uvloop | 20-40% faster event loop (Linux/macOS) |
| cupy-cuda12x | CUDA array operations |
| numba | GPU kernel JIT compilation |
| pynvml | GPU monitoring |
| cython | Maximum performance queues |

## Disabling uvloop

If you need to disable uvloop auto-installation:

```bash
PYDOTCOMPUTE_NO_UVLOOP=1 python my_script.py
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and `docs/IMPLEMENTATION_PLAN.md` for the project roadmap.

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Related

- [DotCompute](https://github.com/mivertowski/DotCompute) - Original .NET implementation
- [Numba CUDA](https://numba.readthedocs.io/en/stable/cuda/) - Python CUDA JIT
- [CuPy](https://cupy.dev/) - NumPy-compatible GPU arrays
- [uvloop](https://github.com/MagicStack/uvloop) - Fast asyncio event loop
