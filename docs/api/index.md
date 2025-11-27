# API Reference

Complete API documentation for PyDotCompute modules.

## Package Structure

```
pydotcompute/
├── __init__.py          # Main exports
├── exceptions.py        # Exception hierarchy
├── core/                # Core abstractions
│   ├── accelerator.py   # Device management
│   ├── unified_buffer.py # Host-device memory
│   ├── memory_pool.py   # Memory pooling
│   └── orchestrator.py  # Compute coordination
├── ring_kernels/        # Actor model
│   ├── runtime.py       # Kernel runtime
│   ├── message.py       # Message types
│   ├── queue.py         # Message queues
│   ├── lifecycle.py     # Kernel lifecycle
│   └── telemetry.py     # Performance metrics
├── backends/            # Compute backends
│   ├── base.py          # Backend interface
│   ├── cpu.py           # CPU simulation
│   ├── cuda.py          # CUDA backend
│   └── metal.py         # Metal backend (macOS)
└── decorators/          # API decorators
    ├── kernel.py        # @kernel decorator
    ├── ring_kernel.py   # @ring_kernel decorator
    └── validators.py    # Type validation
```

## Quick Links

### Core Modules

- **[Accelerator](core/accelerator.md)**: Device detection and management
- **[UnifiedBuffer](core/unified-buffer.md)**: Host-device memory abstraction
- **[MemoryPool](core/memory-pool.md)**: Buffer pooling and reuse
- **[ComputeOrchestrator](core/orchestrator.md)**: Computation coordination

### Ring Kernels

- **[RingKernelRuntime](ring-kernels/runtime.md)**: Main runtime coordinator
- **[Message](ring-kernels/message.md)**: Message types and serialization
- **[MessageQueue](ring-kernels/queue.md)**: Priority message queues
- **[Lifecycle](ring-kernels/lifecycle.md)**: Kernel state management
- **[Telemetry](ring-kernels/telemetry.md)**: Performance monitoring

### Backends

- **[Backend Interface](backends/base.md)**: Abstract backend API
- **[CPUBackend](backends/cpu.md)**: CPU simulation backend
- **[CUDABackend](backends/cuda.md)**: NVIDIA GPU backend
- **[MetalBackend](backends/metal.md)**: Apple Silicon GPU backend (macOS)

### Decorators

- **[@kernel](decorators/kernel.md)**: Low-level kernel decorator
- **[@ring_kernel](decorators/ring-kernel.md)**: Actor decorator
- **[Validators](decorators/validators.md)**: Type validation utilities

### Exceptions

- **[Exceptions](exceptions.md)**: Complete exception hierarchy

## Main Exports

The main `pydotcompute` package exports the most commonly used classes:

```python
from pydotcompute import (
    # Runtime
    RingKernelRuntime,

    # Decorators
    kernel,
    ring_kernel,
    message,

    # Core
    UnifiedBuffer,
    Accelerator,
    get_accelerator,

    # Types
    DeviceType,
    KernelState,
    BackpressureStrategy,
)
```

## Type Hints

PyDotCompute is fully typed. Import types for IDE support:

```python
from pydotcompute.ring_kernels.lifecycle import KernelContext
from pydotcompute.ring_kernels.message import RingKernelMessage
from pydotcompute.core.accelerator import DeviceInfo
```
