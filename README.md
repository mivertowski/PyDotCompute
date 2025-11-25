# PyDotCompute

A Python port of DotCompute's Ring Kernel System - a GPU-native actor model with persistent kernels and message passing.

## Overview

PyDotCompute brings GPU-native actor model capabilities to Python, enabling developers to create persistent GPU kernels that communicate through message queues. This approach is ideal for:

- Real-time GPU compute pipelines
- Streaming data processing on GPU
- Actor-based GPU programming
- High-throughput message-driven architectures

## Features

- **Ring Kernel System**: Persistent GPU kernels with infinite loops and message queues
- **Message Passing**: Type-safe message serialization with msgpack
- **Unified Memory**: Transparent host-device memory management with lazy synchronization
- **Lifecycle Management**: Two-phase launch (launch -> activate) with graceful shutdown
- **Telemetry**: Real-time GPU monitoring and kernel performance metrics
- **Backend Support**: CPU simulation and CUDA acceleration via Numba/CuPy

## Installation

```bash
# Basic installation (CPU only)
pip install pydotcompute

# With CUDA support
pip install pydotcompute[cuda]

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

# Use the runtime
async def main():
    async with RingKernelRuntime() as runtime:
        await runtime.launch("compute")
        await runtime.activate("compute")

        await runtime.send("compute", ComputeRequest(values=[1.0, 2.0, 3.0]))
        response = await runtime.receive("compute")

        print(f"Result: {response.result}")  # 6.0

asyncio.run(main())
```

## Architecture

```
PyDotCompute Ring Kernel System
├── Decorators/Metaclasses    │ Runtime              │ CUDA Backend
│   • @kernel decorator       │ • RingKernelRuntime  │ • Numba JIT
│   • @ring_kernel decorator  │ • asyncio queues     │ • CuPy arrays
│   • @message decorator      │ • Lifecycle mgmt     │ • PTX caching
│   • Type hints + mypy       │ • Telemetry          │ • Zero-copy DMA
├─────────────────────────────┴──────────────────────┴─────────────────
│ Abstractions: Accelerator, UnifiedBuffer, ComputeOrchestrator
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

## Examples

See the `examples/` directory for complete examples:

- `vector_add.py` - Basic ring kernel usage
- `pagerank_actor.py` - PageRank computation actor
- `streaming_pipeline.py` - Multi-stage processing pipeline

Run examples:

```bash
python examples/vector_add.py
python examples/pagerank_actor.py
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pydotcompute

# Run only unit tests
pytest tests/unit/

# Skip CUDA tests (if no GPU)
pytest -m "not cuda"
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
│   ├── runtime.py          # Main runtime
│   ├── message.py          # Message serialization
│   ├── queue.py            # Async message queues
│   ├── lifecycle.py        # Kernel lifecycle
│   └── telemetry.py        # Performance monitoring
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

## Requirements

- Python >= 3.11
- numpy >= 1.26.0
- msgpack >= 1.0.0

### Optional (CUDA support)

- cupy-cuda12x >= 13.0.0
- numba >= 0.59.0
- pynvml >= 11.5.0

## Contributing

Contributions are welcome! Please see the implementation plan in `docs/IMPLEMENTATION_PLAN.md` for the project roadmap.

## License

MIT License - see LICENSE file for details.

## Related

- [DotCompute](https://github.com/mivertowski/DotCompute) - Original .NET implementation
- [Numba CUDA](https://numba.readthedocs.io/en/stable/cuda/) - Python CUDA JIT
- [CuPy](https://cupy.dev/) - NumPy-compatible GPU arrays
