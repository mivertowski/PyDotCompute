"""
CPU backend for PyDotCompute.

Provides a CPU-based implementation of the backend interface.
Useful for testing and development without GPU.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np

from pydotcompute.backends.base import Backend, BackendType, KernelExecutionResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


T = TypeVar("T", bound=np.generic)


class CPUBackend(Backend):
    """
    CPU backend implementation.

    Executes kernels on the CPU using NumPy operations.
    Provides full API compatibility for testing without GPU.

    Example:
        >>> backend = CPUBackend()
        >>> arr = backend.allocate((1000,), np.float32)
        >>> result = backend.execute_kernel(my_kernel, (1,), (256,), arr)
    """

    def __init__(self) -> None:
        """Initialize the CPU backend."""
        self._compiled_kernels: dict[str, Callable[..., Any]] = {}

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.CPU

    @property
    def is_available(self) -> bool:
        """Check if this backend is available."""
        return True  # CPU is always available

    @property
    def device_count(self) -> int:
        """Get the number of available devices."""
        return 1  # CPU counts as 1 device

    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[T] | type[T],
    ) -> NDArray[T]:
        """
        Allocate a NumPy array.

        Args:
            shape: Shape of the array.
            dtype: Data type.

        Returns:
            Allocated NumPy array.
        """
        return np.empty(shape, dtype=dtype)

    def allocate_zeros(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[T] | type[T],
    ) -> NDArray[T]:
        """
        Allocate a zero-filled NumPy array.

        Args:
            shape: Shape of the array.
            dtype: Data type.

        Returns:
            Zero-filled NumPy array.
        """
        return np.zeros(shape, dtype=dtype)

    def free(self, array: NDArray[T]) -> None:
        """
        Free a NumPy array.

        For CPU backend, this is a no-op as NumPy handles memory.

        Args:
            array: Array to free.
        """
        # NumPy handles garbage collection
        pass

    def copy_to_device(self, host_array: NDArray[T]) -> NDArray[T]:
        """
        Copy to device (no-op for CPU).

        Args:
            host_array: Source array.

        Returns:
            Same array (CPU has no device memory).
        """
        return host_array.copy()

    def copy_to_host(self, device_array: NDArray[T]) -> NDArray[T]:
        """
        Copy to host (no-op for CPU).

        Args:
            device_array: Source array.

        Returns:
            Same array (CPU has no device memory).
        """
        return device_array.copy()

    def synchronize(self) -> None:
        """Synchronize (no-op for CPU)."""
        # CPU operations are synchronous
        pass

    def execute_kernel(
        self,
        kernel: Callable[..., Any],
        grid_size: tuple[int, ...],
        block_size: tuple[int, ...],
        *args: Any,
        **kwargs: Any,
    ) -> KernelExecutionResult:
        """
        Execute a kernel on the CPU.

        Simulates GPU-style kernel execution by calling the function
        for each thread index.

        Args:
            kernel: Kernel function.
            grid_size: Grid dimensions (for compatibility).
            block_size: Block dimensions (for compatibility).
            *args: Kernel arguments.
            **kwargs: Kernel keyword arguments.

        Returns:
            Execution result.
        """
        start_time = time.perf_counter()

        try:
            # For CPU, we just call the function directly
            # GPU-style indexing is simulated
            result = kernel(*args, **kwargs)

            end_time = time.perf_counter()
            return KernelExecutionResult(
                success=True,
                execution_time_ms=(end_time - start_time) * 1000,
            )

        except Exception as e:
            end_time = time.perf_counter()
            return KernelExecutionResult(
                success=False,
                execution_time_ms=(end_time - start_time) * 1000,
                error=e,
            )

    def compile_kernel(
        self,
        func: Callable[..., Any],
        signature: str | None = None,
    ) -> Callable[..., Any]:
        """
        Compile a kernel for CPU execution.

        For CPU backend, this just returns the function as-is,
        optionally with some basic optimizations.

        Args:
            func: Python function to compile.
            signature: Optional type signature (ignored for CPU).

        Returns:
            The same function (or lightly optimized version).
        """
        # Try to use Numba JIT if available for CPU optimization
        try:
            from numba import jit

            return jit(nopython=True, cache=True)(func)
        except ImportError:
            return func
        except Exception:
            # If Numba fails (e.g., unsupported operations), return original
            return func

    def parallel_for(
        self,
        start: int,
        end: int,
        func: Callable[[int], None],
    ) -> None:
        """
        Execute a function for each index in parallel.

        Simulates GPU thread-level parallelism using Python's
        range iteration.

        Args:
            start: Start index.
            end: End index.
            func: Function to call with each index.
        """
        for i in range(start, end):
            func(i)

    def __repr__(self) -> str:
        """String representation."""
        return "CPUBackend(available=True)"


def vector_add_cpu(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    CPU implementation of vector addition.

    Example kernel that can be used for testing.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Sum of the arrays.
    """
    return a + b


def matrix_multiply_cpu(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    CPU implementation of matrix multiplication.

    Example kernel that can be used for testing.

    Args:
        a: First matrix (M x K).
        b: Second matrix (K x N).

    Returns:
        Product matrix (M x N).
    """
    return np.matmul(a, b)
