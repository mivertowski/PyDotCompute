# PyDotCompute Implementation Plan

## Overview

This document outlines the implementation and testing strategy for PyDotCompute - a Python port of DotCompute's Ring Kernel System providing GPU-native actor model capabilities.

## Implementation Phases

### Phase 1: Core Infrastructure

**Goal**: Establish foundational abstractions for GPU memory management and compute orchestration.

#### Components

1. **`core/accelerator.py`** - GPU device abstraction
   - `Accelerator` class for device discovery and selection
   - Device properties (compute capability, memory, cores)
   - Multi-GPU support preparation

2. **`core/unified_buffer.py`** - Cross-device memory abstraction
   - `UnifiedBuffer` class with lazy synchronization
   - Buffer states: UNINITIALIZED, HOST_ONLY, DEVICE_ONLY, SYNCHRONIZED, HOST_DIRTY, DEVICE_DIRTY
   - Automatic host/device transfers on access
   - Support for numpy/cupy arrays

3. **`core/memory_pool.py`** - Memory pooling for allocation efficiency
   - `MemoryPool` class wrapping CuPy's built-in pool
   - Statistics tracking (used/free/total bytes)
   - Configurable initial and max sizes

4. **`core/orchestrator.py`** - High-level compute orchestration
   - `ComputeOrchestrator` for managing kernel execution
   - Resource allocation and cleanup

### Phase 2: Ring Kernel System

**Goal**: Implement the persistent GPU kernel actor model with message passing.

#### Components

1. **`ring_kernels/message.py`** - Message infrastructure
   - `RingKernelMessage` base class with UUID, priority, correlation_id
   - `@message` decorator for user message types
   - msgpack-based serialization/deserialization

2. **`ring_kernels/queue.py`** - Async message queues
   - `MessageQueue` class with asyncio.Queue backend
   - Priority queue support
   - Backpressure strategies: block, reject, drop_oldest

3. **`ring_kernels/lifecycle.py`** - Kernel lifecycle management
   - `KernelState` enum: CREATED, LAUNCHED, ACTIVE, DEACTIVATED, TERMINATING, TERMINATED
   - `RingKernel` class with state machine
   - Context managers for automatic cleanup

4. **`ring_kernels/telemetry.py`** - Performance monitoring
   - `RingKernelTelemetry` dataclass
   - `GPUMonitor` class using pynvml
   - Real-time metrics: messages processed, latency, queue depth

5. **`ring_kernels/runtime.py`** - Main runtime coordinator
   - `RingKernelRuntime` class
   - Kernel registration, launch, activate, terminate
   - Message routing between kernels

### Phase 3: Backend Implementations

**Goal**: Provide CPU simulation and CUDA execution backends.

#### Components

1. **`backends/base.py`** - Backend interface
   - `Backend` abstract base class
   - Common interface for all backends

2. **`backends/cpu.py`** - CPU simulation backend
   - For development and testing without GPU
   - Full API compatibility

3. **`backends/cuda.py`** - CUDA backend
   - Numba CUDA JIT integration
   - CuPy array operations
   - Kernel execution management

### Phase 4: Compilation System

**Goal**: Manage kernel compilation and caching.

#### Components

1. **`compilation/compiler.py`** - Kernel compiler
   - Numba JIT compilation wrapper
   - Signature validation

2. **`compilation/cache.py`** - Compilation cache
   - LRU cache for compiled kernels
   - Disk persistence for PTX

### Phase 5: Decorators & Validators

**Goal**: Provide Pythonic API through decorators.

#### Components

1. **`decorators/kernel.py`** - `@kernel` decorator
   - Basic GPU kernel registration
   - Grid/block configuration

2. **`decorators/ring_kernel.py`** - `@ring_kernel` decorator
   - Full ring kernel actor setup
   - Input/output type specification
   - Queue configuration

3. **`decorators/validators.py`** - Runtime validation
   - Type checking for messages
   - Configuration validation

---

## Testing Strategy

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_accelerator.py
│   ├── test_unified_buffer.py
│   ├── test_memory_pool.py
│   ├── test_message.py
│   ├── test_queue.py
│   ├── test_lifecycle.py
│   ├── test_telemetry.py
│   └── test_decorators.py
├── integration/
│   ├── __init__.py
│   ├── test_runtime.py
│   ├── test_end_to_end.py
│   └── test_backends.py
└── performance/
    ├── __init__.py
    └── test_benchmarks.py
```

### Unit Tests

Each module should have comprehensive unit tests covering:

1. **Happy path** - Normal operation
2. **Edge cases** - Boundary conditions
3. **Error handling** - Invalid inputs, exceptions
4. **State transitions** - For stateful components

### Integration Tests

1. **Runtime Tests**
   - Launch and activate kernels
   - Send and receive messages
   - Graceful shutdown

2. **End-to-End Tests**
   - Complete actor workflows
   - Multi-kernel pipelines
   - Error recovery

### Testing Without GPU

The CPU backend enables testing without NVIDIA hardware:
- All unit tests run on CPU
- Integration tests support backend switching
- CI/CD pipeline uses CPU backend

### Test Fixtures (conftest.py)

```python
@pytest.fixture
def runtime():
    """Provide a clean runtime for each test."""

@pytest.fixture
def unified_buffer():
    """Provide a test buffer."""

@pytest.fixture
def message_queue():
    """Provide a test message queue."""
```

---

## API Design

### High-Level Usage

```python
from pydotcompute import RingKernelRuntime, ring_kernel, message
from pydotcompute.core import UnifiedBuffer

@message
class ComputeRequest:
    data: list[float]

@message
class ComputeResponse:
    result: float

@ring_kernel(kernel_id="compute", input_type=ComputeRequest, output_type=ComputeResponse)
async def compute_actor(ctx):
    while not ctx.should_terminate:
        msg = await ctx.receive()
        result = sum(msg.data)
        await ctx.send(ComputeResponse(result=result))

async def main():
    async with RingKernelRuntime() as runtime:
        await runtime.launch("compute")
        await runtime.activate("compute")

        await runtime.send("compute", ComputeRequest(data=[1.0, 2.0, 3.0]))
        response = await runtime.receive("compute")
        print(f"Result: {response.result}")  # 6.0
```

---

## Dependencies

### Required
- Python >= 3.11
- numpy >= 1.26.0
- msgpack >= 1.0.0

### Optional (GPU)
- cupy-cuda12x >= 13.0.0
- numba >= 0.59.0
- pynvml >= 11.5.0

### Development
- pytest >= 8.0.0
- pytest-asyncio >= 0.23.0
- mypy >= 1.8.0
- ruff >= 0.1.0

---

## Milestones

1. **M1: Core Complete** - UnifiedBuffer, MemoryPool working
2. **M2: Messages Working** - Serialization, queues functional
3. **M3: Runtime Working** - Full lifecycle management
4. **M4: CPU Backend** - All features work on CPU
5. **M5: CUDA Backend** - GPU acceleration working
6. **M6: Examples Complete** - Vector add, PageRank demos
7. **M7: Tests Passing** - 90%+ coverage

---

## File Manifest

```
pydotcompute/
├── __init__.py
├── py.typed
├── core/
│   ├── __init__.py
│   ├── accelerator.py
│   ├── unified_buffer.py
│   ├── memory_pool.py
│   └── orchestrator.py
├── ring_kernels/
│   ├── __init__.py
│   ├── runtime.py
│   ├── message.py
│   ├── queue.py
│   ├── lifecycle.py
│   └── telemetry.py
├── backends/
│   ├── __init__.py
│   ├── base.py
│   ├── cpu.py
│   └── cuda.py
├── compilation/
│   ├── __init__.py
│   ├── compiler.py
│   └── cache.py
├── decorators/
│   ├── __init__.py
│   ├── kernel.py
│   ├── ring_kernel.py
│   └── validators.py
└── exceptions.py
```
