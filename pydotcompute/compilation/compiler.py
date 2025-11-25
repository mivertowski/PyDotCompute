"""
Kernel compiler for PyDotCompute.

Handles compilation of Python functions to GPU kernels
using Numba and CuPy.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydotcompute.exceptions import KernelCompilationError

if TYPE_CHECKING:
    pass


@dataclass
class CompilationOptions:
    """Options for kernel compilation."""

    device: str = "cuda"  # cuda, cpu
    fastmath: bool = True
    debug: bool = False
    lineinfo: bool = False
    opt_level: int = 3
    cache: bool = True


@dataclass
class CompiledKernel:
    """A compiled kernel with metadata."""

    kernel: Callable[..., Any]
    name: str
    source_hash: str
    signature: str | None
    options: CompilationOptions
    compile_time_ms: float = 0.0
    ptx: str | None = None


class KernelCompiler:
    """
    Compiler for GPU kernels.

    Supports compilation via Numba CUDA JIT and CuPy RawKernel.

    Example:
        >>> compiler = KernelCompiler()
        >>> compiled = compiler.compile(my_kernel_func)
        >>> # Use compiled.kernel for execution
    """

    def __init__(self, options: CompilationOptions | None = None) -> None:
        """
        Initialize the kernel compiler.

        Args:
            options: Compilation options.
        """
        self._options = options or CompilationOptions()
        self._compiled_cache: dict[str, CompiledKernel] = {}
        self._cuda_available = self._check_cuda()
        self._numba_available = self._check_numba()

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import cupy as cp

            return cp.cuda.runtime.getDeviceCount() > 0
        except ImportError:
            return False
        except Exception:
            return False

    def _check_numba(self) -> bool:
        """Check if Numba CUDA is available."""
        try:
            from numba import cuda

            return cuda.is_available()
        except ImportError:
            return False
        except Exception:
            return False

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA compilation is available."""
        return self._cuda_available

    @property
    def numba_available(self) -> bool:
        """Check if Numba CUDA is available."""
        return self._numba_available

    def compile(
        self,
        func: Callable[..., Any],
        *,
        signature: str | None = None,
        name: str | None = None,
        options: CompilationOptions | None = None,
    ) -> CompiledKernel:
        """
        Compile a Python function as a GPU kernel.

        Args:
            func: Python function to compile.
            signature: Optional Numba signature string.
            name: Optional kernel name.
            options: Compilation options (uses instance options if None).

        Returns:
            CompiledKernel with the compiled kernel.

        Raises:
            KernelCompilationError: If compilation fails.
        """
        import time

        opts = options or self._options
        kernel_name = name or func.__name__

        # Generate source hash for caching
        source_hash = self._get_source_hash(func, signature, opts)

        # Check cache
        if opts.cache and source_hash in self._compiled_cache:
            return self._compiled_cache[source_hash]

        start_time = time.perf_counter()

        try:
            if opts.device == "cuda" and self._numba_available:
                compiled_kernel = self._compile_numba_cuda(func, signature, opts)
            elif opts.device == "cuda" and self._cuda_available:
                compiled_kernel = self._compile_cupy(func, opts)
            else:
                # CPU fallback
                compiled_kernel = self._compile_cpu(func, opts)

            compile_time = (time.perf_counter() - start_time) * 1000

            result = CompiledKernel(
                kernel=compiled_kernel,
                name=kernel_name,
                source_hash=source_hash,
                signature=signature,
                options=opts,
                compile_time_ms=compile_time,
            )

            # Cache result
            if opts.cache:
                self._compiled_cache[source_hash] = result

            return result

        except Exception as e:
            raise KernelCompilationError(kernel_name, e) from e

    def _compile_numba_cuda(
        self,
        func: Callable[..., Any],
        signature: str | None,
        options: CompilationOptions,
    ) -> Callable[..., Any]:
        """Compile using Numba CUDA JIT."""
        from numba import cuda

        if signature:
            return cuda.jit(
                signature,
                fastmath=options.fastmath,
                debug=options.debug,
                lineinfo=options.lineinfo,
            )(func)

        return cuda.jit(
            fastmath=options.fastmath,
            debug=options.debug,
            lineinfo=options.lineinfo,
        )(func)

    def _compile_cupy(
        self,
        func: Callable[..., Any],
        options: CompilationOptions,
    ) -> Callable[..., Any]:
        """Compile using CuPy (for raw CUDA C kernels)."""
        # CuPy RawKernel expects CUDA C source code
        # This is for advanced use cases where the function
        # contains CUDA C as a string
        source = getattr(func, "__cuda_source__", None)
        if source:
            import cupy as cp

            return cp.RawKernel(source, func.__name__)

        # Fall back to Numba if no CUDA source
        if self._numba_available:
            return self._compile_numba_cuda(func, None, options)

        raise KernelCompilationError(
            func.__name__,
            ValueError("CuPy compilation requires __cuda_source__ attribute"),
        )

    def _compile_cpu(
        self,
        func: Callable[..., Any],
        options: CompilationOptions,
    ) -> Callable[..., Any]:
        """Compile for CPU execution using Numba JIT."""
        try:
            from numba import jit

            return jit(
                nopython=True,
                fastmath=options.fastmath,
                cache=options.cache,
            )(func)
        except ImportError:
            # Return original function if Numba not available
            return func

    def _get_source_hash(
        self,
        func: Callable[..., Any],
        signature: str | None,
        options: CompilationOptions,
    ) -> str:
        """Generate a hash for cache key."""
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            source = func.__name__

        hash_input = f"{source}|{signature}|{options.device}|{options.fastmath}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def get_cached_kernel(self, source_hash: str) -> CompiledKernel | None:
        """
        Get a cached compiled kernel.

        Args:
            source_hash: Hash of the kernel source.

        Returns:
            CompiledKernel or None if not cached.
        """
        return self._compiled_cache.get(source_hash)

    def clear_cache(self) -> int:
        """
        Clear the compilation cache.

        Returns:
            Number of entries cleared.
        """
        count = len(self._compiled_cache)
        self._compiled_cache.clear()
        return count

    def get_cache_stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        return {
            "cached_kernels": len(self._compiled_cache),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KernelCompiler(cuda={self._cuda_available}, "
            f"numba={self._numba_available}, "
            f"cached={len(self._compiled_cache)})"
        )


# Global compiler instance
_global_compiler: KernelCompiler | None = None


def get_compiler() -> KernelCompiler:
    """Get the global kernel compiler instance."""
    global _global_compiler
    if _global_compiler is None:
        _global_compiler = KernelCompiler()
    return _global_compiler


def compile_kernel(
    func: Callable[..., Any],
    *,
    signature: str | None = None,
    **options: Any,
) -> CompiledKernel:
    """
    Compile a kernel using the global compiler.

    Args:
        func: Function to compile.
        signature: Optional signature.
        **options: Compilation options.

    Returns:
        CompiledKernel.
    """
    compiler = get_compiler()
    opts = CompilationOptions(**options) if options else None
    return compiler.compile(func, signature=signature, options=opts)
