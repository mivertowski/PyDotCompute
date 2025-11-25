"""
Compute orchestrator for managing kernel execution.

Provides high-level coordination of GPU compute resources,
kernel launches, and resource cleanup.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID, uuid4

from pydotcompute.core.accelerator import Accelerator, get_accelerator
from pydotcompute.core.memory_pool import MemoryPool, get_memory_pool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass
class KernelLaunchConfig:
    """Configuration for a kernel launch."""

    grid_size: tuple[int, ...] = (1,)
    block_size: tuple[int, ...] = (256,)
    shared_memory_bytes: int = 0
    stream: object | None = None  # CUDA stream

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Ensure grid_size and block_size are tuples
        if isinstance(self.grid_size, int):
            self.grid_size = (self.grid_size,)
        if isinstance(self.block_size, int):
            self.block_size = (self.block_size,)


@dataclass
class KernelExecution:
    """Record of a kernel execution."""

    execution_id: UUID = field(default_factory=uuid4)
    kernel_name: str = ""
    config: KernelLaunchConfig = field(default_factory=KernelLaunchConfig)
    start_time: float = 0.0
    end_time: float = 0.0
    success: bool = False
    error: Exception | None = None

    @property
    def duration_ms(self) -> float:
        """Get execution duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000


class ComputeOrchestrator:
    """
    High-level compute orchestrator.

    Manages kernel execution, resource allocation, and cleanup.
    Provides a unified interface for GPU compute operations.

    Example:
        >>> async with ComputeOrchestrator() as orchestrator:
        ...     await orchestrator.launch_kernel(my_kernel, config, args)
    """

    def __init__(self) -> None:
        """Initialize the compute orchestrator."""
        self._accelerator: Accelerator | None = None
        self._memory_pool: MemoryPool | None = None
        self._registered_kernels: dict[str, Callable[..., Any]] = {}
        self._execution_history: list[KernelExecution] = []
        self._streams: list[object] = []
        self._active = False

    async def __aenter__(self) -> ComputeOrchestrator:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.shutdown()

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        if self._active:
            return

        self._accelerator = get_accelerator()
        self._memory_pool = get_memory_pool()
        self._memory_pool.enable()
        self._active = True

    async def shutdown(self) -> None:
        """Shutdown the orchestrator and cleanup resources."""
        if not self._active:
            return

        # Synchronize all streams
        for stream in self._streams:
            try:
                stream.synchronize()  # type: ignore
            except Exception:
                pass

        # Clear streams
        self._streams.clear()

        # Free memory pool
        if self._memory_pool:
            self._memory_pool.free_all_blocks()

        self._active = False

    @property
    def is_active(self) -> bool:
        """Check if the orchestrator is active."""
        return self._active

    @property
    def accelerator(self) -> Accelerator | None:
        """Get the accelerator instance."""
        return self._accelerator

    @property
    def memory_pool(self) -> MemoryPool | None:
        """Get the memory pool instance."""
        return self._memory_pool

    def register_kernel(self, name: str, kernel_func: Callable[..., Any]) -> None:
        """
        Register a kernel function.

        Args:
            name: Unique name for the kernel.
            kernel_func: The kernel function to register.
        """
        self._registered_kernels[name] = kernel_func

    def get_kernel(self, name: str) -> Callable[..., Any] | None:
        """
        Get a registered kernel by name.

        Args:
            name: Name of the kernel.

        Returns:
            Kernel function or None if not found.
        """
        return self._registered_kernels.get(name)

    async def launch_kernel(
        self,
        kernel: Callable[..., Any] | str,
        config: KernelLaunchConfig,
        *args: Any,
        **kwargs: Any,
    ) -> KernelExecution:
        """
        Launch a kernel with the given configuration.

        Args:
            kernel: Kernel function or registered kernel name.
            config: Launch configuration.
            *args: Positional arguments to pass to the kernel.
            **kwargs: Keyword arguments to pass to the kernel.

        Returns:
            KernelExecution record.
        """
        import time

        # Resolve kernel if string
        if isinstance(kernel, str):
            kernel_func = self._registered_kernels.get(kernel)
            if kernel_func is None:
                raise ValueError(f"Kernel '{kernel}' not registered")
            kernel_name = kernel
        else:
            kernel_func = kernel
            kernel_name = getattr(kernel, "__name__", "unknown")

        # Create execution record
        execution = KernelExecution(
            kernel_name=kernel_name,
            config=config,
            start_time=time.perf_counter(),
        )

        try:
            # Check if it's a Numba CUDA kernel
            if hasattr(kernel_func, "__cuda_kernel__") or hasattr(kernel_func, "forall"):
                # Numba CUDA kernel launch
                kernel_func[config.grid_size, config.block_size](*args, **kwargs)
            else:
                # Regular Python function (CPU or CuPy)
                result = kernel_func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    await result

            execution.success = True

        except Exception as e:
            execution.error = e
            execution.success = False
            raise

        finally:
            execution.end_time = time.perf_counter()
            self._execution_history.append(execution)

        return execution

    def create_stream(self) -> object | None:
        """
        Create a new CUDA stream.

        Returns:
            CUDA stream or None if not available.
        """
        if self._accelerator is None or not self._accelerator.cuda_available:
            return None

        try:
            import cupy as cp

            stream = cp.cuda.Stream()
            self._streams.append(stream)
            return stream
        except ImportError:
            return None

    def synchronize(self) -> None:
        """Synchronize all active streams."""
        if self._accelerator:
            self._accelerator.synchronize()

    def get_execution_history(
        self,
        limit: int | None = None,
        kernel_name: str | None = None,
    ) -> list[KernelExecution]:
        """
        Get kernel execution history.

        Args:
            limit: Maximum number of records to return.
            kernel_name: Filter by kernel name.

        Returns:
            List of KernelExecution records.
        """
        history = self._execution_history

        if kernel_name:
            history = [e for e in history if e.kernel_name == kernel_name]

        if limit:
            history = history[-limit:]

        return history

    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self._execution_history.clear()

    @asynccontextmanager
    async def stream_context(self) -> AsyncIterator[object | None]:
        """
        Context manager for using a CUDA stream.

        Yields:
            CUDA stream or None if not available.
        """
        stream = self.create_stream()

        try:
            if stream is not None:
                try:
                    import cupy as cp

                    with cp.cuda.Stream(stream):
                        yield stream
                except ImportError:
                    yield None
            else:
                yield None
        finally:
            if stream is not None:
                try:
                    stream.synchronize()  # type: ignore
                except Exception:
                    pass

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ComputeOrchestrator(active={self._active}, "
            f"kernels={len(self._registered_kernels)}, "
            f"executions={len(self._execution_history)})"
        )
