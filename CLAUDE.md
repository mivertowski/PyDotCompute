# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyDotCompute is a Python port of DotCompute's Ring Kernel System - a GPU-native actor model with persistent kernels and message passing. It enables developers to create persistent GPU kernels that communicate through message queues, ideal for real-time GPU compute pipelines and streaming data processing.

## Development Commands

```bash
# Install for development
pip install -e ".[dev]"

# Install with CUDA support
pip install -e ".[cuda]"

# Install with performance optimizations (uvloop)
pip install -e ".[fast]"

# Install with Cython extensions (maximum performance)
pip install -e ".[cython]"
python setup_cython.py build_ext --inplace

# Run all tests
pytest

# Run tests with coverage
pytest --cov=pydotcompute

# Run only unit tests
pytest tests/unit/

# Skip CUDA tests (if no GPU)
pytest -m "not cuda"

# Run benchmarks
python benchmarks/extended_benchmark.py
python benchmarks/pagerank_benchmark.py
python benchmarks/realtime_anomaly_benchmark.py

# Type checking
mypy pydotcompute

# Linting
ruff check pydotcompute

# Build documentation
mkdocs serve
```

## Architecture

### Core Components

**Ring Kernel System** (`pydotcompute/ring_kernels/`)
- `runtime.py` - `RingKernelRuntime`: Main coordinator managing kernel lifecycle, message routing, and telemetry. Use as async context manager. Auto-installs uvloop for 21μs message latency.
- `lifecycle.py` - `RingKernel`, `KernelContext`, `KernelState`: Two-phase launch (launch -> activate) with graceful shutdown.
- `message.py` - `@message` decorator and `RingKernelMessage` base class for type-safe msgpack serialization.
- `queue.py` - `MessageQueue`: Async message queues with backpressure strategies (block/reject/drop_oldest).
- `fast_queue.py` - `FastMessageQueue`: O(1) priority banding with 4 bands (SYSTEM/HIGH/NORMAL/LOW). Zero-copy mode for in-process messaging.
- `telemetry.py` - Real-time GPU monitoring and kernel performance metrics.

**Performance Tiers** (`pydotcompute/ring_kernels/`)
- `_loop.py` - uvloop auto-installation + eager_task_factory (Python 3.12+)
- `sync_queue.py` - `SyncQueue`, `SPSCQueue`: Threading-based queues for GIL-releasing workloads
- `threaded_kernel.py` - `ThreadedRingKernel`: Dedicated thread execution for blocking I/O
- `cython_kernel.py` - `CythonRingKernel`: Maximum performance using Cython FastSPSCQueue
- `_cython/fast_spsc.pyx` - Lock-free SPSC queue with 0.33μs operations

**Memory Management** (`pydotcompute/core/`)
- `unified_buffer.py` - `UnifiedBuffer`: Transparent host-device memory with lazy synchronization. Tracks dirty state (HOST_DIRTY/DEVICE_DIRTY/SYNCHRONIZED) to minimize transfers.
- `memory_pool.py` - `MemoryPool`: Memory pooling for buffer reuse.
- `accelerator.py` - `Accelerator`: GPU device abstraction (singleton).
- `orchestrator.py` - `ComputeOrchestrator`: Compute coordination.

**Backends** (`pydotcompute/backends/`)
- `base.py` - `Backend` ABC: Interface all backends implement (allocate, free, copy_to_device, execute_kernel, compile_kernel).
- `cpu.py` - CPU simulation backend.
- `cuda.py` - CUDA backend via Numba JIT and CuPy arrays.

**Decorators** (`pydotcompute/decorators/`)
- `ring_kernel.py` - `@ring_kernel`: Decorator for defining persistent GPU actor kernels. Auto-registers with global registry.
- `kernel.py` - `@kernel`, `@gpu_kernel`: Standard kernel decorators.
- `validators.py` - Runtime validation utilities.

### Performance Tiers

| Tier | Implementation | Latency (p50) | Use Case |
|------|---------------|---------------|----------|
| **1 (Default)** | uvloop + FastMessageQueue | **21μs** | Async Python code |
| 2 | ThreadedRingKernel | ~100μs | Blocking I/O, GIL-releasing |
| 3 | CythonRingKernel + FastSPSCQueue | **0.33μs** queue ops | Multi-process, Cython extensions |

**Key Insight**: uvloop (Tier 1) is optimal for pure Python due to GIL. Threading adds overhead. Cython queues shine in multi-process scenarios.

### Key Patterns

**Ring Kernel Definition:**
```python
@ring_kernel(kernel_id="my_actor", input_type=RequestType, output_type=ResponseType)
async def my_actor(ctx: KernelContext):
    while not ctx.should_terminate:
        msg = await ctx.receive()
        await ctx.send(ResponseType(...))
```

**Runtime Usage (auto-installs uvloop):**
```python
async with RingKernelRuntime() as runtime:
    await runtime.launch("my_actor")      # Phase 1: allocate resources
    await runtime.activate("my_actor")    # Phase 2: start processing
    await runtime.send("my_actor", request)
    response = await runtime.receive("my_actor")
```

**Threaded Kernel (for blocking operations):**
```python
def blocking_kernel(ctx: ThreadedKernelContext):
    while not ctx.should_terminate:
        msg = ctx.receive(timeout=0.1)  # Blocking receive
        if msg:
            ctx.send(process(msg))

with ThreadedRingKernel("worker", blocking_kernel) as kernel:
    kernel.send(request)
    response = kernel.receive()
```

**Buffer State Machine:**
UNINITIALIZED -> HOST_ONLY/DEVICE_ONLY/SYNCHRONIZED
HOST_DIRTY (after host write) -> SYNCHRONIZED (after device access)
DEVICE_DIRTY (after device write) -> SYNCHRONIZED (after host access)

## Performance Benchmarks

### Message Latency (Extended Benchmark)
- **p50**: 63μs (full actor roundtrip)
- **p99**: 131μs
- **Isolated queue**: 21μs p50

### Graph Processing (PageRank Benchmark)
- **GPU wins at**: 50K+ nodes (7-9x faster than CPU)
- **Peak throughput**: 1.7M edges/sec (GPU Sparse)
- **Crossover point**: ~1000 nodes dense, 5000 nodes sparse

### Streaming (Real-Time Anomaly Benchmark)
- **GPU Actors advantage**: Persistent GPU state (no repeated transfers)
- **Transfer overhead**: 16-28% of batch processing time
- **Best for**: Long-running pipelines with context

### Queue Operations
| Queue Type | Put+Get (same thread) |
|------------|----------------------|
| FastMessageQueue (Python) | 1.8μs |
| FastSPSCQueue (Cython) | **0.33μs** |

## Lessons Learned

1. **uvloop beats threading for Python**: The GIL makes native threading slower than uvloop's libuv-based event loop for message passing.

2. **Queue operations are fast, synchronization is slow**: Raw queue ops are ~1-2μs, but thread context switching adds 50-100μs.

3. **Cython queues need multi-process**: The Cython FastSPSCQueue achieves 0.33μs but only shines in multi-process scenarios where GIL isn't shared.

4. **GPU wins at scale**: GPU acceleration becomes beneficial at 50K+ nodes for graphs, and for streaming with persistent state.

5. **Zero-copy matters**: Using `serialize=False` for in-process messaging eliminates serialization overhead.

## Testing

- pytest-asyncio is configured with `asyncio_mode = "auto"` - async tests run automatically.
- Fixtures in `tests/conftest.py` provide `runtime`, `accelerator`, `memory_pool`, `unified_buffer`, `message_queue`.
- CUDA tests are automatically skipped if CUDA is not available.
- Use `@pytest.mark.cuda` for CUDA-specific tests.
- Use `@pytest.mark.slow` for slow-running tests.
- All 398 tests passing.

## Requirements

- Python >= 3.11
- Core: numpy, msgpack
- Performance: uvloop (Linux/macOS, auto-installed)
- CUDA (optional): cupy-cuda12x, numba, pynvml
- Cython (optional): cython >= 3.0.0

## Disabling uvloop

Set environment variable before import:
```bash
PYDOTCOMPUTE_NO_UVLOOP=1 python my_script.py
```
