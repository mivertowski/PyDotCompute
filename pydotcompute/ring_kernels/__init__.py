"""
Ring Kernel System - GPU-native actor model with persistent kernels.

This module automatically installs uvloop (if available) for improved
asyncio performance on Linux/macOS. To disable auto-installation, set
the environment variable PYDOTCOMPUTE_NO_UVLOOP=1 before importing.
"""

import os as _os

# Auto-install uvloop at import time (before any event loop is created)
# This is the recommended pattern for libraries that want to use uvloop
if not _os.environ.get("PYDOTCOMPUTE_NO_UVLOOP"):
    try:
        import uvloop as _uvloop

        _uvloop.install()
    except ImportError:
        pass

from pydotcompute.ring_kernels._loop import (
    create_optimized_loop,
    get_loop_info,
    install_eager_task_factory,
    install_uvloop,
    is_eager_tasks_installed,
    is_uvloop_available,
    is_uvloop_installed,
    optimize_current_loop,
)
from pydotcompute.ring_kernels.array_message import ArrayMessage, array_message
from pydotcompute.ring_kernels.buffer_registry import (
    BufferHandle,
    BufferRegistry,
    get_buffer,
    get_buffer_registry,
    register_buffer,
    release_buffer,
)
from pydotcompute.ring_kernels.cython_kernel import (
    CythonKernelConfig,
    CythonKernelContext,
    CythonKernelState,
    CythonRingKernel,
    is_cython_kernel_available,
)
from pydotcompute.ring_kernels.fast_queue import (
    FastMessageQueue,
    PriorityBand,
    RingBuffer,
)
from pydotcompute.ring_kernels.lifecycle import KernelState, RingKernel
from pydotcompute.ring_kernels.message import RingKernelMessage, message
from pydotcompute.ring_kernels.queue import MessageQueue
from pydotcompute.ring_kernels.runtime import RingKernelRuntime
from pydotcompute.ring_kernels.sync_queue import SPSCQueue, SyncQueue
from pydotcompute.ring_kernels.telemetry import GPUMonitor, RingKernelTelemetry
from pydotcompute.ring_kernels.threaded_kernel import (
    ThreadedKernelConfig,
    ThreadedKernelContext,
    ThreadedKernelPool,
    ThreadedKernelState,
    ThreadedRingKernel,
)

__all__ = [
    # Runtime
    "RingKernelRuntime",
    "RingKernel",
    "KernelState",
    # Messages
    "RingKernelMessage",
    "message",
    "ArrayMessage",
    "array_message",
    # Queues
    "MessageQueue",
    "FastMessageQueue",
    "RingBuffer",
    "PriorityBand",
    # Buffer Registry (zero-copy)
    "BufferRegistry",
    "BufferHandle",
    "get_buffer_registry",
    "register_buffer",
    "get_buffer",
    "release_buffer",
    # Telemetry
    "RingKernelTelemetry",
    "GPUMonitor",
    # Event Loop Optimization
    "install_uvloop",
    "install_eager_task_factory",
    "is_uvloop_available",
    "is_uvloop_installed",
    "is_eager_tasks_installed",
    "get_loop_info",
    "create_optimized_loop",
    "optimize_current_loop",
    # Threaded Kernel (Tier 2 - Maximum Performance)
    "ThreadedRingKernel",
    "ThreadedKernelContext",
    "ThreadedKernelConfig",
    "ThreadedKernelState",
    "ThreadedKernelPool",
    # Sync Queues (for threaded kernels)
    "SyncQueue",
    "SPSCQueue",
    # Cython Kernel (Tier 3 - Ultimate Performance)
    "CythonRingKernel",
    "CythonKernelContext",
    "CythonKernelConfig",
    "CythonKernelState",
    "is_cython_kernel_available",
]
