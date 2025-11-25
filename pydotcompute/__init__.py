"""
PyDotCompute - Python port of DotCompute's Ring Kernel System.

A GPU-native actor model with persistent kernels and message passing.
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
