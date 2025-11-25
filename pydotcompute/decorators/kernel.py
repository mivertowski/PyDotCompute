"""
Kernel decorators for PyDotCompute.

Provides @kernel and @gpu_kernel decorators for defining
compute kernels.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    pass

F = TypeVar("F", bound=Callable[..., Any])


def kernel(
    func: F | None = None,
    *,
    name: str | None = None,
    device: str = "auto",
    fastmath: bool = True,
    cache: bool = True,
) -> F | Callable[[F], F]:
    """
    Decorator to define a compute kernel.

    Automatically selects the best backend (CUDA if available, CPU otherwise)
    and compiles the function for efficient execution.

    Args:
        func: The function to decorate.
        name: Optional kernel name (defaults to function name).
        device: Device target: "auto", "cuda", or "cpu".
        fastmath: Enable fast math optimizations.
        cache: Enable kernel caching.

    Returns:
        Decorated kernel function.

    Example:
        >>> @kernel
        ... def my_compute(a, b, out):
        ...     # Kernel implementation
        ...     pass
        ...
        >>> @kernel(device="cuda", fastmath=True)
        ... def gpu_compute(a, b, out):
        ...     pass
    """

    def decorator(fn: F) -> F:
        kernel_name = name or fn.__name__

        # Store metadata
        fn._kernel_meta = {  # type: ignore
            "name": kernel_name,
            "device": device,
            "fastmath": fastmath,
            "cache": cache,
            "compiled": None,
        }

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Lazy compilation on first call
            meta = fn._kernel_meta  # type: ignore

            if meta["compiled"] is None:
                meta["compiled"] = _compile_kernel(fn, meta)

            compiled = meta["compiled"]
            if compiled is not None:
                return compiled(*args, **kwargs)
            return fn(*args, **kwargs)

        # Copy metadata
        wrapper._kernel_meta = fn._kernel_meta  # type: ignore
        wrapper._original_func = fn  # type: ignore

        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


def gpu_kernel(
    func: F | None = None,
    *,
    name: str | None = None,
    signature: str | None = None,
    fastmath: bool = True,
    debug: bool = False,
    cache: bool = True,
) -> F | Callable[[F], F]:
    """
    Decorator to define a GPU kernel using Numba CUDA.

    This decorator requires CUDA support and will raise an error
    if CUDA is not available.

    Args:
        func: The function to decorate.
        name: Optional kernel name.
        signature: Optional Numba type signature.
        fastmath: Enable fast math optimizations.
        debug: Enable debug mode.
        cache: Enable kernel caching.

    Returns:
        Decorated CUDA kernel function.

    Example:
        >>> @gpu_kernel
        ... def vector_add(a, b, out):
        ...     idx = cuda.grid(1)
        ...     if idx < out.shape[0]:
        ...         out[idx] = a[idx] + b[idx]
        ...
        >>> # Launch with grid/block configuration
        >>> vector_add[grid_size, block_size](a, b, out)
    """

    def decorator(fn: F) -> F:
        kernel_name = name or fn.__name__

        # Try to compile with Numba CUDA
        try:
            from numba import cuda

            if signature:
                compiled = cuda.jit(
                    signature,
                    fastmath=fastmath,
                    debug=debug,
                    cache=cache,
                )(fn)
            else:
                compiled = cuda.jit(
                    fastmath=fastmath,
                    debug=debug,
                    cache=cache,
                )(fn)

            # Store metadata
            compiled._kernel_meta = {  # type: ignore
                "name": kernel_name,
                "signature": signature,
                "device": "cuda",
                "fastmath": fastmath,
                "debug": debug,
            }
            compiled._original_func = fn  # type: ignore

            return compiled  # type: ignore

        except ImportError as e:
            raise RuntimeError(f"CUDA not available for @gpu_kernel. Error: {e}") from e

    if func is not None:
        return decorator(func)
    return decorator


def _compile_kernel(func: Callable[..., Any], meta: dict[str, Any]) -> Callable[..., Any] | None:
    """
    Compile a kernel function based on metadata.

    Args:
        func: Original function.
        meta: Kernel metadata.

    Returns:
        Compiled kernel or None.
    """
    device = meta["device"]

    # Auto-detect device
    if device == "auto":
        try:
            from numba import cuda

            device = "cuda" if cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # Compile for target device
    if device == "cuda":
        try:
            from numba import cuda

            return cuda.jit(fastmath=meta["fastmath"], cache=meta["cache"])(func)
        except ImportError:
            pass

    # CPU fallback
    try:
        from numba import jit

        return jit(nopython=True, fastmath=meta["fastmath"], cache=meta["cache"])(func)
    except ImportError:
        pass

    return None


def cuda_source(source: str) -> Callable[[F], F]:
    """
    Decorator to attach CUDA C source code to a function.

    Used with CuPy RawKernel for custom CUDA C implementations.

    Args:
        source: CUDA C source code string.

    Returns:
        Decorator that attaches the source.

    Example:
        >>> @cuda_source('''
        ... extern "C" __global__
        ... void my_kernel(float* a, float* b, float* out) {
        ...     int idx = blockDim.x * blockIdx.x + threadIdx.x;
        ...     out[idx] = a[idx] + b[idx];
        ... }
        ... ''')
        ... def my_kernel(a, b, out):
        ...     pass
    """

    def decorator(func: F) -> F:
        func.__cuda_source__ = source  # type: ignore
        return func

    return decorator


class KernelRegistry:
    """
    Registry for kernel functions.

    Provides centralized registration and lookup of kernels.
    """

    _instance: KernelRegistry | None = None
    _kernels: dict[str, Callable[..., Any]]

    def __new__(cls) -> KernelRegistry:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._kernels = {}
        return cls._instance

    def register(
        self,
        name: str,
        kernel: Callable[..., Any],
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a kernel.

        Args:
            name: Unique kernel name.
            kernel: Kernel function.
            overwrite: Whether to overwrite existing kernels.
        """
        if name in self._kernels and not overwrite:
            raise ValueError(f"Kernel '{name}' is already registered")
        self._kernels[name] = kernel

    def get(self, name: str) -> Callable[..., Any] | None:
        """
        Get a registered kernel.

        Args:
            name: Kernel name.

        Returns:
            Kernel function or None.
        """
        return self._kernels.get(name)

    def list(self) -> list[str]:
        """List all registered kernel names."""
        return list(self._kernels.keys())

    def clear(self) -> None:
        """Clear all registered kernels."""
        self._kernels.clear()


def get_kernel_registry() -> KernelRegistry:
    """Get the global kernel registry."""
    return KernelRegistry()
