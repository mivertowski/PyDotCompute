"""
CUDA backend for PyDotCompute.

Provides CUDA-based implementation using Numba and CuPy.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from pydotcompute.backends.base import Backend, BackendType, KernelExecutionResult
from pydotcompute.exceptions import BackendNotAvailableError, CUDAError

if TYPE_CHECKING:
    from numpy.typing import NDArray


T = TypeVar("T", bound=np.generic)


def _check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except ImportError:
        return False
    except Exception:
        return False


class CUDABackend(Backend):
    """
    CUDA backend implementation using Numba and CuPy.

    Provides GPU-accelerated kernel execution and memory management.

    Example:
        >>> backend = CUDABackend()
        >>> if backend.is_available:
        ...     arr = backend.allocate((1000,), np.float32)
        ...     # Use arr with GPU kernels
    """

    def __init__(self, device_id: int = 0) -> None:
        """
        Initialize the CUDA backend.

        Args:
            device_id: CUDA device ID to use.

        Raises:
            BackendNotAvailableError: If CUDA is not available.
        """
        self._device_id = device_id
        self._cuda_available = _check_cuda_available()
        self._cp = None
        self._cuda = None

        if self._cuda_available:
            try:
                import cupy as cp
                from numba import cuda

                self._cp = cp
                self._cuda = cuda

                # Set device
                cp.cuda.Device(device_id).use()

            except ImportError as e:
                self._cuda_available = False
                raise BackendNotAvailableError(
                    "CUDA",
                    f"Required packages not installed: {e}",
                ) from e

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.CUDA

    @property
    def is_available(self) -> bool:
        """Check if this backend is available."""
        return self._cuda_available

    @property
    def device_count(self) -> int:
        """Get the number of available CUDA devices."""
        if not self._cuda_available or self._cp is None:
            return 0

        try:
            return self._cp.cuda.runtime.getDeviceCount()
        except Exception:
            return 0

    @property
    def device_id(self) -> int:
        """Get the current device ID."""
        return self._device_id

    def set_device(self, device_id: int) -> None:
        """
        Set the current CUDA device.

        Args:
            device_id: Device ID to use.
        """
        if not self._cuda_available or self._cp is None:
            return

        self._device_id = device_id
        self._cp.cuda.Device(device_id).use()

    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[T] | type[T],
    ) -> Any:  # Returns cp.ndarray
        """
        Allocate a CuPy array on the GPU.

        Args:
            shape: Shape of the array.
            dtype: Data type.

        Returns:
            CuPy array on GPU.

        Raises:
            CUDAError: If allocation fails.
        """
        if not self._cuda_available or self._cp is None:
            raise CUDAError("CUDA not available")

        try:
            return self._cp.empty(shape, dtype=dtype)
        except Exception as e:
            raise CUDAError(f"Failed to allocate GPU memory: {e}") from e

    def allocate_zeros(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[T] | type[T],
    ) -> Any:  # Returns cp.ndarray
        """
        Allocate a zero-filled CuPy array on the GPU.

        Args:
            shape: Shape of the array.
            dtype: Data type.

        Returns:
            Zero-filled CuPy array.
        """
        if not self._cuda_available or self._cp is None:
            raise CUDAError("CUDA not available")

        return self._cp.zeros(shape, dtype=dtype)

    def free(self, array: Any) -> None:
        """
        Free a CuPy array.

        Args:
            array: CuPy array to free.
        """
        # CuPy handles garbage collection, but we can clear references
        if hasattr(array, "data"):
            del array

    def copy_to_device(self, host_array: NDArray[T]) -> Any:
        """
        Copy a NumPy array to GPU.

        Args:
            host_array: Source NumPy array.

        Returns:
            CuPy array on GPU.
        """
        if not self._cuda_available or self._cp is None:
            raise CUDAError("CUDA not available")

        return self._cp.asarray(host_array)

    def copy_to_host(self, device_array: Any) -> NDArray[T]:
        """
        Copy a CuPy array to host.

        Args:
            device_array: Source CuPy array.

        Returns:
            NumPy array.
        """
        if not self._cuda_available:
            raise CUDAError("CUDA not available")

        if hasattr(device_array, "get"):
            return device_array.get()
        return np.asarray(device_array)

    def synchronize(self) -> None:
        """Synchronize CUDA operations."""
        if not self._cuda_available or self._cp is None:
            return

        self._cp.cuda.Stream.null.synchronize()

    def execute_kernel(
        self,
        kernel: Callable[..., Any],
        grid_size: tuple[int, ...],
        block_size: tuple[int, ...],
        *args: Any,
        **kwargs: Any,
    ) -> KernelExecutionResult:
        """
        Execute a CUDA kernel.

        Args:
            kernel: Numba CUDA kernel.
            grid_size: Grid dimensions.
            block_size: Block dimensions.
            *args: Kernel arguments.
            **kwargs: Kernel keyword arguments.

        Returns:
            Execution result.
        """
        if not self._cuda_available:
            return KernelExecutionResult(
                success=False,
                execution_time_ms=0,
                error=CUDAError("CUDA not available"),
            )

        start_time = time.perf_counter()

        try:
            # Handle different kernel types
            if hasattr(kernel, "__cuda_kernel__") or str(type(kernel)).find("numba") != -1:
                # Numba CUDA kernel
                kernel[grid_size, block_size](*args, **kwargs)
            elif callable(kernel):
                # CuPy RawKernel or regular function
                if hasattr(kernel, "kernel"):
                    # CuPy RawKernel
                    kernel(grid_size, block_size, args)
                else:
                    # Regular function
                    kernel(*args, **kwargs)

            # Synchronize to get accurate timing
            self.synchronize()

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
        Compile a function as a CUDA kernel.

        Args:
            func: Python function to compile.
            signature: Optional Numba signature string.

        Returns:
            Compiled CUDA kernel.
        """
        if not self._cuda_available or self._cuda is None:
            raise CUDAError("CUDA not available for compilation")

        if signature:
            return self._cuda.jit(signature)(func)
        return self._cuda.jit(func)

    def get_memory_info(self) -> dict[str, int]:
        """
        Get GPU memory information.

        Returns:
            Dictionary with free and total memory in bytes.
        """
        if not self._cuda_available or self._cp is None:
            return {"free": 0, "total": 0, "used": 0}

        mem_info = self._cp.cuda.runtime.memGetInfo()
        return {
            "free": mem_info[0],
            "total": mem_info[1],
            "used": mem_info[1] - mem_info[0],
        }

    def create_stream(self) -> Any:
        """
        Create a CUDA stream.

        Returns:
            CuPy CUDA stream.
        """
        if not self._cuda_available or self._cp is None:
            raise CUDAError("CUDA not available")

        return self._cp.cuda.Stream()

    def __repr__(self) -> str:
        """String representation."""
        if self._cuda_available:
            mem = self.get_memory_info()
            return (
                f"CUDABackend(device={self._device_id}, "
                f"devices={self.device_count}, "
                f"memory_free={mem['free'] // 1024**2}MB)"
            )
        return "CUDABackend(available=False)"


# Example CUDA kernels
def get_vector_add_kernel() -> Callable[..., Any] | None:
    """
    Get a CUDA vector addition kernel.

    Returns:
        Compiled CUDA kernel or None if CUDA not available.
    """
    if not _check_cuda_available():
        return None

    try:
        from numba import cuda

        @cuda.jit
        def vector_add_kernel(a: Any, b: Any, out: Any) -> None:
            idx = cuda.grid(1)
            if idx < out.shape[0]:
                out[idx] = a[idx] + b[idx]

        return vector_add_kernel

    except ImportError:
        return None


def get_matrix_multiply_kernel() -> Callable[..., Any] | None:
    """
    Get a CUDA matrix multiplication kernel.

    Returns:
        Compiled CUDA kernel or None if CUDA not available.
    """
    if not _check_cuda_available():
        return None

    try:
        from numba import cuda, float32

        @cuda.jit
        def matrix_multiply_kernel(A: Any, B: Any, C: Any) -> None:
            """Simple matrix multiplication kernel."""
            i, j = cuda.grid(2)
            if i < C.shape[0] and j < C.shape[1]:
                tmp = float32(0.0)
                for k in range(A.shape[1]):
                    tmp += A[i, k] * B[k, j]
                C[i, j] = tmp

        return matrix_multiply_kernel

    except ImportError:
        return None
