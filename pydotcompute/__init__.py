"""
PyDotCompute - Python port of DotCompute's Ring Kernel System.

A GPU-native actor model framework providing persistent kernels with
message passing for high-performance computing workflows.

Core Features:
    - Ring Kernel System: GPU-based actors with async message passing
    - Unified Memory Management: Transparent host-device synchronization
    - Memory Pooling: Efficient GPU memory allocation via CuPy
    - Kernel Compilation: JIT compilation with Numba CUDA support
    - CPU Fallback: Full API compatibility when CUDA is unavailable

Quick Start:
    >>> from pydotcompute import RingKernelRuntime, message, ring_kernel
    >>>
    >>> @message
    ... class Request:
    ...     value: float
    ...
    >>> @ring_kernel(kernel_id="processor")
    ... async def processor(ctx):
    ...     while True:
    ...         msg = await ctx.receive()
    ...         await ctx.send(msg.value * 2)
    ...
    >>> async with RingKernelRuntime() as runtime:
    ...     await runtime.launch("processor")
    ...     await runtime.activate("processor")
    ...     await runtime.send("processor", Request(value=42.0))

For more information, see the documentation at:
https://mivertowski.github.io/PyDotCompute/
"""

from pydotcompute.core.accelerator import Accelerator
from pydotcompute.core.memory_pool import MemoryPool
from pydotcompute.core.orchestrator import ComputeOrchestrator
from pydotcompute.core.unified_buffer import BufferState, UnifiedBuffer
from pydotcompute.decorators.kernel import gpu_kernel, kernel
from pydotcompute.decorators.ring_kernel import ring_kernel
from pydotcompute.ring_kernels.message import RingKernelMessage, message
from pydotcompute.ring_kernels.runtime import RingKernelRuntime

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Core
    "Accelerator",
    "UnifiedBuffer",
    "BufferState",
    "MemoryPool",
    "ComputeOrchestrator",
    # Ring Kernels
    "RingKernelRuntime",
    "RingKernelMessage",
    # Decorators
    "kernel",
    "gpu_kernel",
    "ring_kernel",
    "message",
]
