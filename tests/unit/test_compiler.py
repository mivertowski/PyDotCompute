"""
Unit tests for the kernel compiler module.

Tests the KernelCompiler class, CompilationOptions, CompiledKernel,
and related compilation utilities.
"""

from __future__ import annotations

from typing import Any

import pytest

from pydotcompute.compilation.compiler import (
    CompilationOptions,
    CompiledKernel,
    KernelCompiler,
    compile_kernel,
    get_compiler,
)
from pydotcompute.exceptions import KernelCompilationError


class TestCompilationOptions:
    """Tests for CompilationOptions dataclass."""

    def test_default_options(self) -> None:
        """Test default compilation options."""
        opts = CompilationOptions()
        assert opts.device == "cuda"
        assert opts.fastmath is True
        assert opts.debug is False
        assert opts.lineinfo is False
        assert opts.opt_level == 3
        assert opts.cache is True

    def test_custom_options(self) -> None:
        """Test custom compilation options."""
        opts = CompilationOptions(
            device="cpu",
            fastmath=False,
            debug=True,
            lineinfo=True,
            opt_level=2,
            cache=False,
        )
        assert opts.device == "cpu"
        assert opts.fastmath is False
        assert opts.debug is True
        assert opts.lineinfo is True
        assert opts.opt_level == 2
        assert opts.cache is False


class TestCompiledKernel:
    """Tests for CompiledKernel dataclass."""

    def test_creation(self) -> None:
        """Test creating a compiled kernel."""

        def dummy_kernel() -> None:
            pass

        kernel = CompiledKernel(
            kernel=dummy_kernel,
            name="test_kernel",
            source_hash="abc123",
            signature="void()",
            options=CompilationOptions(),
        )
        assert kernel.kernel is dummy_kernel
        assert kernel.name == "test_kernel"
        assert kernel.source_hash == "abc123"
        assert kernel.signature == "void()"
        assert kernel.compile_time_ms == 0.0
        assert kernel.ptx is None

    def test_with_compile_time(self) -> None:
        """Test compiled kernel with timing info."""

        def dummy() -> None:
            pass

        kernel = CompiledKernel(
            kernel=dummy,
            name="k",
            source_hash="h",
            signature=None,
            options=CompilationOptions(),
            compile_time_ms=15.5,
            ptx=".version 7.0",
        )
        assert kernel.compile_time_ms == 15.5
        assert kernel.ptx == ".version 7.0"


class TestKernelCompiler:
    """Tests for KernelCompiler class."""

    @pytest.fixture
    def compiler(self) -> KernelCompiler:
        """Provide a fresh compiler instance."""
        return KernelCompiler()

    def test_creation(self, compiler: KernelCompiler) -> None:
        """Test compiler creation."""
        assert compiler._options is not None
        assert isinstance(compiler._options, CompilationOptions)
        assert compiler._compiled_cache == {}

    def test_creation_with_options(self) -> None:
        """Test compiler creation with custom options."""
        opts = CompilationOptions(device="cpu", fastmath=False)
        compiler = KernelCompiler(options=opts)
        assert compiler._options.device == "cpu"
        assert compiler._options.fastmath is False

    def test_cuda_available_property(self, compiler: KernelCompiler) -> None:
        """Test cuda_available property."""
        # Just check it returns a boolean
        assert isinstance(compiler.cuda_available, bool)

    def test_numba_available_property(self, compiler: KernelCompiler) -> None:
        """Test numba_available property."""
        assert isinstance(compiler.numba_available, bool)

    def test_compile_cpu_function(self, compiler: KernelCompiler) -> None:
        """Test compiling a simple CPU function."""

        def simple_func(a: float, b: float) -> float:
            return a + b

        opts = CompilationOptions(device="cpu", cache=False)
        result = compiler.compile(simple_func, options=opts)

        assert result.name == "simple_func"
        assert result.options.device == "cpu"
        assert result.compile_time_ms >= 0

    def test_compile_with_custom_name(self, compiler: KernelCompiler) -> None:
        """Test compiling with custom kernel name."""

        def func() -> None:
            pass

        opts = CompilationOptions(device="cpu", cache=False)
        result = compiler.compile(func, name="my_custom_kernel", options=opts)

        assert result.name == "my_custom_kernel"

    def test_compile_caching(self, compiler: KernelCompiler) -> None:
        """Test that compiled kernels are cached."""

        def cacheable_func(x: float) -> float:
            return x * 2

        opts = CompilationOptions(device="cpu", cache=True)

        # First compilation
        result1 = compiler.compile(cacheable_func, options=opts)

        # Second compilation should return cached
        result2 = compiler.compile(cacheable_func, options=opts)

        assert result1 is result2

    def test_compile_no_caching(self, compiler: KernelCompiler) -> None:
        """Test compiling without caching."""

        def func(x: float) -> float:
            return x

        opts = CompilationOptions(device="cpu", cache=False)

        result1 = compiler.compile(func, options=opts)
        result2 = compiler.compile(func, options=opts)

        # With cache=False, should still work (just not use cache)
        assert result1.name == result2.name

    def test_get_source_hash(self, compiler: KernelCompiler) -> None:
        """Test source hash generation."""

        def func1(a: int) -> int:
            return a + 1

        def func2(a: int) -> int:
            return a + 2

        opts = CompilationOptions()

        hash1 = compiler._get_source_hash(func1, None, opts)
        hash2 = compiler._get_source_hash(func2, None, opts)

        assert hash1 != hash2
        assert len(hash1) == 16

    def test_get_source_hash_with_signature(self, compiler: KernelCompiler) -> None:
        """Test hash differs with different signatures."""

        def func(a: Any) -> Any:
            return a

        opts = CompilationOptions()

        hash1 = compiler._get_source_hash(func, "int32(int32)", opts)
        hash2 = compiler._get_source_hash(func, "float32(float32)", opts)

        assert hash1 != hash2

    def test_get_cached_kernel(self, compiler: KernelCompiler) -> None:
        """Test retrieving cached kernel."""

        def func() -> None:
            pass

        opts = CompilationOptions(device="cpu", cache=True)
        result = compiler.compile(func, options=opts)

        cached = compiler.get_cached_kernel(result.source_hash)
        assert cached is result

    def test_get_cached_kernel_miss(self, compiler: KernelCompiler) -> None:
        """Test cache miss."""
        result = compiler.get_cached_kernel("nonexistent_hash")
        assert result is None

    def test_clear_cache(self, compiler: KernelCompiler) -> None:
        """Test clearing compilation cache."""

        def f1() -> None:
            pass

        def f2() -> None:
            pass

        opts = CompilationOptions(device="cpu", cache=True)
        compiler.compile(f1, options=opts)
        compiler.compile(f2, options=opts)

        assert len(compiler._compiled_cache) == 2

        count = compiler.clear_cache()
        assert count == 2
        assert len(compiler._compiled_cache) == 0

    def test_get_cache_stats(self, compiler: KernelCompiler) -> None:
        """Test getting cache statistics."""

        def f() -> None:
            pass

        opts = CompilationOptions(device="cpu", cache=True)
        compiler.compile(f, options=opts)

        stats = compiler.get_cache_stats()
        assert stats["cached_kernels"] == 1

    def test_repr(self, compiler: KernelCompiler) -> None:
        """Test string representation."""
        repr_str = repr(compiler)
        assert "KernelCompiler" in repr_str
        assert "cuda=" in repr_str
        assert "numba=" in repr_str
        assert "cached=" in repr_str


class TestGetCompiler:
    """Tests for global compiler getter."""

    def test_returns_singleton(self) -> None:
        """Test that get_compiler returns singleton."""
        import pydotcompute.compilation.compiler as compiler_module

        compiler_module._global_compiler = None

        c1 = get_compiler()
        c2 = get_compiler()

        assert c1 is c2

    def test_creates_compiler_if_none(self) -> None:
        """Test that compiler is created if not exists."""
        import pydotcompute.compilation.compiler as compiler_module

        compiler_module._global_compiler = None

        compiler = get_compiler()
        assert compiler is not None
        assert isinstance(compiler, KernelCompiler)


class TestCompileKernel:
    """Tests for compile_kernel convenience function."""

    def test_compile_simple_function(self) -> None:
        """Test compile_kernel function."""

        def add(a: float, b: float) -> float:
            return a + b

        result = compile_kernel(add, device="cpu", cache=False)

        assert result.name == "add"
        assert isinstance(result, CompiledKernel)

    def test_compile_with_signature(self) -> None:
        """Test compile_kernel with signature."""

        def mul(a: Any, b: Any) -> Any:
            return a * b

        result = compile_kernel(mul, signature="float64(float64, float64)", device="cpu")

        assert result.signature == "float64(float64, float64)"


class TestCompilerErrorHandling:
    """Tests for compiler error handling."""

    def test_compile_invalid_function(self) -> None:
        """Test compiling function that can't be inspected."""

        # Create a function that can't have source retrieved
        compiled_func = compile("lambda x: x + 1", "<string>", "eval")
        func = eval(compiled_func)

        compiler = KernelCompiler()
        opts = CompilationOptions(device="cpu", cache=False)

        # Should still compile (falls back to name-based hash)
        result = compiler.compile(func, name="lambda_func", options=opts)
        assert result is not None


class TestCompilerCPUFallback:
    """Tests for CPU fallback compilation."""

    def test_cpu_fallback_returns_callable(self) -> None:
        """Test that CPU fallback returns a callable."""

        def func(x: float) -> float:
            return x * 2

        compiler = KernelCompiler()
        opts = CompilationOptions(device="cpu")
        result = compiler.compile(func, options=opts)

        # The compiled kernel should be callable
        assert callable(result.kernel)
