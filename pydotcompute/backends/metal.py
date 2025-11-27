"""
Metal backend for PyDotCompute.

Provides Metal-based implementation using Apple's MLX framework
for macOS/Apple Silicon GPU acceleration.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from pydotcompute.backends.base import Backend, BackendType, KernelExecutionResult
from pydotcompute.exceptions import BackendNotAvailableError, MetalError

if TYPE_CHECKING:
    from numpy.typing import NDArray


T = TypeVar("T", bound=np.generic)
logger = logging.getLogger(__name__)


def _check_metal_available() -> bool:
    """Check if Metal/MLX is available."""
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except ImportError:
        return False
    except Exception:
        return False


class MetalBufferState(Enum):
    """State of a Metal buffer."""

    ALLOCATED = auto()
    FREED = auto()
    IN_USE = auto()


@dataclass
class MetalBufferInfo:
    """Tracking information for a Metal buffer."""

    array_id: int
    size: int
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    state: MetalBufferState
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class MetalBackend(Backend):
    """
    Metal backend implementation using Apple's MLX framework.

    Provides GPU-accelerated kernel execution and memory management
    on macOS with Apple Silicon unified memory architecture.

    Thread Safety:
        This class is thread-safe. All Metal operations are synchronized
        through internal locks.

    Example:
        >>> backend = MetalBackend()
        >>> if backend.is_available:
        ...     arr = backend.allocate((1000,), np.float32)
        ...     result = backend.execute_kernel(my_kernel, (1,), (256,), arr)

    Attributes:
        device_id: Metal device ID (currently only 0 is supported).
    """

    _instance_lock = threading.Lock()
    _instances: weakref.WeakSet[MetalBackend] = weakref.WeakSet()

    def __init__(self, device_id: int = 0) -> None:
        """
        Initialize the Metal backend.

        Args:
            device_id: Metal device ID to use (0 for default GPU).

        Raises:
            BackendNotAvailableError: If Metal/MLX is not available.
        """
        self._device_id = device_id
        self._metal_available = _check_metal_available()
        self._mx: Any = None
        self._lock = threading.RLock()
        self._buffer_registry: dict[int, MetalBufferInfo] = {}
        self._compiled_kernels: dict[str, Callable[..., Any]] = {}

        if self._metal_available:
            try:
                import mlx.core as mx

                self._mx = mx
                logger.info("Metal backend initialized with MLX")

            except ImportError as e:
                self._metal_available = False
                raise BackendNotAvailableError(
                    "Metal",
                    f"MLX not installed: {e}",
                ) from e

        # Register for cleanup
        with self._instance_lock:
            self._instances.add(self)

        # Register atexit cleanup
        atexit.register(self._cleanup)

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.METAL

    @property
    def is_available(self) -> bool:
        """Check if this backend is available."""
        return self._metal_available

    @property
    def device_count(self) -> int:
        """Get the number of available Metal devices."""
        # Apple Silicon typically has 1 unified GPU
        return 1 if self._metal_available else 0

    @property
    def device_id(self) -> int:
        """Get the current device ID."""
        return self._device_id

    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[T] | type[T],
    ) -> Any:  # Returns mlx.core.array
        """
        Allocate an MLX array (backed by Metal unified memory).

        Args:
            shape: Shape of the array.
            dtype: Data type.

        Returns:
            MLX array on GPU.

        Raises:
            MetalError: If allocation fails.
        """
        with self._lock:
            if not self._metal_available or self._mx is None:
                raise MetalError("Metal not available")

            try:
                mlx_dtype = self._numpy_to_mlx_dtype(dtype)
                arr = self._mx.zeros(shape, dtype=mlx_dtype)

                # Register buffer for tracking
                array_id = id(arr)
                self._buffer_registry[array_id] = MetalBufferInfo(
                    array_id=array_id,
                    size=int(np.prod(shape)) * np.dtype(dtype).itemsize,
                    shape=shape,
                    dtype=np.dtype(dtype),
                    state=MetalBufferState.ALLOCATED,
                )

                return arr

            except Exception as e:
                raise MetalError(f"Failed to allocate Metal memory: {e}") from e

    def allocate_zeros(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[T] | type[T],
    ) -> Any:  # Returns mlx.core.array
        """
        Allocate a zero-filled MLX array.

        Args:
            shape: Shape of the array.
            dtype: Data type.

        Returns:
            Zero-filled MLX array.
        """
        # MLX zeros already returns a zero-filled array
        return self.allocate(shape, dtype)

    def free(self, array: Any) -> None:
        """
        Free an MLX array.

        Args:
            array: MLX array to free.
        """
        with self._lock:
            array_id = id(array)
            if array_id in self._buffer_registry:
                info = self._buffer_registry.pop(array_id)
                info.state = MetalBufferState.FREED
                logger.debug(f"Freed Metal buffer: {info.size} bytes")

            # MLX handles garbage collection
            del array

    def copy_to_device(self, host_array: NDArray[T]) -> Any:
        """
        Copy a NumPy array to MLX array (Metal GPU).

        Note: On Apple Silicon, this is logically a copy as memory
        is physically unified. MLX handles optimization automatically.

        Args:
            host_array: Source NumPy array.

        Returns:
            MLX array on GPU.
        """
        with self._lock:
            if not self._metal_available or self._mx is None:
                raise MetalError("Metal not available")

            arr = self._mx.array(host_array)

            # Register buffer
            array_id = id(arr)
            self._buffer_registry[array_id] = MetalBufferInfo(
                array_id=array_id,
                size=host_array.nbytes,
                shape=host_array.shape,
                dtype=host_array.dtype,
                state=MetalBufferState.ALLOCATED,
            )

            return arr

    def copy_to_host(self, device_array: Any) -> NDArray[T]:
        """
        Copy an MLX array to NumPy.

        Forces evaluation of lazy operations and returns NumPy array.

        Args:
            device_array: Source MLX array.

        Returns:
            NumPy array.
        """
        if not self._metal_available:
            raise MetalError("Metal not available")

        # MLX arrays can be converted directly to NumPy
        return np.array(device_array)

    def synchronize(self) -> None:
        """Synchronize Metal operations (evaluate lazy computation graph)."""
        if self._metal_available and self._mx is not None:
            # MLX uses lazy evaluation - eval() forces computation
            self._mx.eval()

    def execute_kernel(
        self,
        kernel: Callable[..., Any],
        grid_size: tuple[int, ...],  # noqa: ARG002 - Required by Backend ABC
        block_size: tuple[int, ...],  # noqa: ARG002 - Required by Backend ABC
        *args: Any,
        **kwargs: Any,
    ) -> KernelExecutionResult:
        """
        Execute a kernel on Metal.

        Supports multiple kernel types:
        - MLX operations (functions using mlx.core)
        - Compiled MSL shaders (via Metal kernel compiler)
        - Regular Python functions (will run on CPU)

        Args:
            kernel: Kernel function.
            grid_size: Grid dimensions (may be ignored for MLX ops).
            block_size: Block dimensions (may be ignored for MLX ops).
            *args: Kernel arguments.
            **kwargs: Kernel keyword arguments.

        Returns:
            Execution result with timing information.
        """
        if not self._metal_available:
            return KernelExecutionResult(
                success=False,
                execution_time_ms=0,
                error=MetalError("Metal not available"),
            )

        start_time = time.perf_counter()

        try:
            # Execute kernel function
            result = kernel(*args, **kwargs)

            # Force evaluation for accurate timing
            if result is not None and self._mx is not None:
                if hasattr(result, "__module__") and "mlx" in str(type(result)):
                    self._mx.eval(result)
                elif isinstance(result, (list, tuple)):
                    # Handle multiple outputs
                    mlx_results = [
                        r
                        for r in result
                        if hasattr(r, "__module__") and "mlx" in str(type(r))
                    ]
                    if mlx_results:
                        self._mx.eval(*mlx_results)

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
        Compile a kernel function for Metal execution.

        MLX uses JIT compilation automatically; this method returns
        a wrapped version that ensures Metal execution and handles
        type conversions.

        For custom MSL shaders, pass the MSL source code as func
        (as a string) and provide a kernel name as signature.

        Args:
            func: Python function to compile (or MSL source string).
            signature: Optional kernel name (for MSL) or type hint.

        Returns:
            Compiled kernel callable.
        """
        if not self._metal_available or self._mx is None:
            raise MetalError("Metal not available for compilation")

        # Check cache
        cache_key = f"{id(func)}_{signature}"
        if cache_key in self._compiled_kernels:
            return self._compiled_kernels[cache_key]

        # Handle MSL source string
        if isinstance(func, str):
            compiled = self._compile_msl_kernel(func, signature or "kernel_main")
            self._compiled_kernels[cache_key] = compiled
            return compiled

        # Wrap Python function for MLX execution
        def metal_kernel(*args: Any, **kwargs: Any) -> Any:
            # Convert NumPy arrays to MLX arrays
            mlx_args = [
                self._mx.array(a) if isinstance(a, np.ndarray) else a for a in args
            ]
            mlx_kwargs = {
                k: self._mx.array(v) if isinstance(v, np.ndarray) else v
                for k, v in kwargs.items()
            }
            result = func(*mlx_args, **mlx_kwargs)
            # Ensure result is evaluated
            if result is not None:
                self._mx.eval(result)
            return result

        self._compiled_kernels[cache_key] = metal_kernel
        return metal_kernel

    def _compile_msl_kernel(
        self,
        source: str,  # noqa: ARG002 - Will be used when MSL compilation is implemented
        kernel_name: str,
    ) -> Callable[..., Any]:
        """
        Compile Metal Shading Language source code.

        Args:
            source: MSL shader source code.
            kernel_name: Name of the kernel function.

        Returns:
            Callable that executes the Metal kernel.

        Note:
            MLX doesn't directly support arbitrary MSL compilation.
            For now, this returns a wrapper that raises NotImplementedError.
            Full MSL support would require PyObjC integration.
        """
        # TODO: Implement full MSL compilation using PyObjC
        # For now, log a warning and return a placeholder
        logger.warning(
            f"MSL kernel compilation not yet fully implemented. "
            f"Kernel '{kernel_name}' will use fallback."
        )

        def msl_kernel_placeholder(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                f"MSL kernel '{kernel_name}' compilation not yet implemented. "
                "Use MLX operations instead or contribute MSL support."
            )

        return msl_kernel_placeholder

    def _numpy_to_mlx_dtype(self, dtype: np.dtype[Any] | type[Any]) -> Any:
        """Convert NumPy dtype to MLX dtype."""
        if self._mx is None:
            raise MetalError("Metal not available")

        dtype = np.dtype(dtype)
        mapping = {
            np.float32: self._mx.float32,
            np.float16: self._mx.float16,
            np.int32: self._mx.int32,
            np.int64: self._mx.int64,
            np.int16: self._mx.int16,
            np.int8: self._mx.int8,
            np.uint32: self._mx.uint32,
            np.uint64: self._mx.uint64,
            np.uint16: self._mx.uint16,
            np.uint8: self._mx.uint8,
            np.bool_: self._mx.bool_,
        }

        # Handle float64 -> float32 (MLX prefers float32)
        if dtype.type == np.float64:
            logger.debug("Converting float64 to float32 for Metal (MLX preference)")
            return self._mx.float32

        mlx_dtype = mapping.get(dtype.type)
        if mlx_dtype is None:
            logger.warning(f"Unknown dtype {dtype}, defaulting to float32")
            return self._mx.float32

        return mlx_dtype

    def get_memory_info(self) -> dict[str, int]:
        """
        Get Metal memory information.

        Returns:
            Dictionary with memory statistics.
        """
        if not self._metal_available or self._mx is None:
            return {"allocated": 0, "buffer_count": 0, "cache_memory": 0, "peak_memory": 0}

        try:
            # Use new API if available, fallback to deprecated
            if hasattr(self._mx, "get_cache_memory"):
                cache_memory = self._mx.get_cache_memory()
                peak_memory = self._mx.get_peak_memory()
            else:
                cache_memory = self._mx.metal.get_cache_memory()
                peak_memory = self._mx.metal.get_peak_memory()

            # Sum allocated from registry
            allocated = sum(
                info.size
                for info in self._buffer_registry.values()
                if info.state == MetalBufferState.ALLOCATED
            )

            return {
                "allocated": allocated,
                "buffer_count": len(self._buffer_registry),
                "cache_memory": cache_memory,
                "peak_memory": peak_memory,
            }
        except Exception as e:
            logger.debug(f"Failed to get memory info: {e}")
            return {"allocated": 0, "buffer_count": 0, "cache_memory": 0, "peak_memory": 0}

    def clear_cache(self) -> None:
        """Clear the Metal memory cache."""
        if self._metal_available and self._mx is not None:
            try:
                # Use new API if available, fallback to deprecated
                if hasattr(self._mx, "clear_cache"):
                    self._mx.clear_cache()
                else:
                    self._mx.metal.clear_cache()
                logger.debug("Metal cache cleared")
            except Exception as e:
                logger.debug(f"Failed to clear cache: {e}")

    def _cleanup(self) -> None:
        """Cleanup resources on exit."""
        with self._lock:
            # Clear buffer registry
            self._buffer_registry.clear()
            # Clear kernel cache
            self._compiled_kernels.clear()
            # Clear MLX cache
            self.clear_cache()

    def __repr__(self) -> str:
        """String representation."""
        if self._metal_available:
            mem = self.get_memory_info()
            return (
                f"MetalBackend(device={self._device_id}, "
                f"available=True, "
                f"buffers={mem['buffer_count']}, "
                f"cache={mem['cache_memory'] // 1024}KB)"
            )
        return "MetalBackend(available=False)"

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        import contextlib

        with contextlib.suppress(Exception):
            self._cleanup()


# Utility functions for Metal operations


def get_vector_add_kernel() -> Callable[..., Any] | None:
    """
    Get a Metal vector addition kernel using MLX.

    Returns:
        Kernel function or None if Metal not available.
    """
    if not _check_metal_available():
        return None

    try:
        import mlx.core as mx

        def vector_add_kernel(a: Any, b: Any) -> Any:
            """Vector addition using MLX."""
            return mx.add(a, b)

        return vector_add_kernel

    except ImportError:
        return None


def get_matrix_multiply_kernel() -> Callable[..., Any] | None:
    """
    Get a Metal matrix multiplication kernel using MLX.

    Returns:
        Kernel function or None if Metal not available.
    """
    if not _check_metal_available():
        return None

    try:
        import mlx.core as mx

        def matrix_multiply_kernel(A: Any, B: Any) -> Any:
            """Matrix multiplication using MLX."""
            return mx.matmul(A, B)

        return matrix_multiply_kernel

    except ImportError:
        return None


def get_elementwise_kernel(operation: str) -> Callable[..., Any] | None:
    """
    Get an elementwise operation kernel.

    Args:
        operation: Operation name (add, sub, mul, div, sqrt, exp, log, sin, cos).

    Returns:
        Kernel function or None if Metal not available.
    """
    if not _check_metal_available():
        return None

    try:
        import mlx.core as mx

        ops = {
            "add": mx.add,
            "sub": mx.subtract,
            "mul": mx.multiply,
            "div": mx.divide,
            "sqrt": mx.sqrt,
            "exp": mx.exp,
            "log": mx.log,
            "sin": mx.sin,
            "cos": mx.cos,
            "abs": mx.abs,
            "square": mx.square,
            "negative": mx.negative,
        }

        return ops.get(operation)

    except ImportError:
        return None
