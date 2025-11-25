"""
Unit tests for compute backends.
"""

from __future__ import annotations

import numpy as np
import pytest

from pydotcompute.backends.base import BackendType, KernelExecutionResult
from pydotcompute.backends.cpu import CPUBackend, matrix_multiply_cpu, vector_add_cpu


class TestCPUBackend:
    """Tests for CPUBackend."""

    def test_backend_type(self) -> None:
        """Test backend type."""
        backend = CPUBackend()

        assert backend.backend_type == BackendType.CPU

    def test_is_available(self) -> None:
        """Test CPU backend is always available."""
        backend = CPUBackend()

        assert backend.is_available

    def test_device_count(self) -> None:
        """Test device count is 1."""
        backend = CPUBackend()

        assert backend.device_count == 1

    def test_allocate(self) -> None:
        """Test array allocation."""
        backend = CPUBackend()

        arr = backend.allocate((10, 20), np.float32)

        assert arr.shape == (10, 20)
        assert arr.dtype == np.float32

    def test_allocate_zeros(self) -> None:
        """Test zero-filled allocation."""
        backend = CPUBackend()

        arr = backend.allocate_zeros((10,), np.float64)

        assert arr.shape == (10,)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, np.zeros(10))

    def test_free(self) -> None:
        """Test array freeing (no-op for CPU)."""
        backend = CPUBackend()

        arr = backend.allocate((10,), np.float32)

        # Should not raise
        backend.free(arr)

    def test_copy_to_device(self) -> None:
        """Test copy to device (returns copy for CPU)."""
        backend = CPUBackend()

        source = np.array([1, 2, 3], dtype=np.float32)
        device_arr = backend.copy_to_device(source)

        np.testing.assert_array_equal(device_arr, source)
        # Should be a copy, not the same array
        assert device_arr is not source

    def test_copy_to_host(self) -> None:
        """Test copy to host (returns copy for CPU)."""
        backend = CPUBackend()

        source = np.array([1, 2, 3], dtype=np.float32)
        host_arr = backend.copy_to_host(source)

        np.testing.assert_array_equal(host_arr, source)
        assert host_arr is not source

    def test_synchronize(self) -> None:
        """Test synchronize (no-op for CPU)."""
        backend = CPUBackend()

        # Should not raise
        backend.synchronize()

    def test_execute_kernel(self) -> None:
        """Test kernel execution."""
        backend = CPUBackend()

        def simple_kernel(data: np.ndarray) -> np.ndarray:
            return data * 2

        data = np.array([1, 2, 3], dtype=np.float32)
        result = backend.execute_kernel(simple_kernel, (1,), (256,), data)

        assert isinstance(result, KernelExecutionResult)
        assert result.success
        assert result.execution_time_ms >= 0

    def test_execute_kernel_error(self) -> None:
        """Test kernel execution error handling."""
        backend = CPUBackend()

        def failing_kernel() -> None:
            raise ValueError("Test error")

        result = backend.execute_kernel(failing_kernel, (1,), (1,))

        assert not result.success
        assert result.error is not None

    def test_compile_kernel(self) -> None:
        """Test kernel compilation."""
        backend = CPUBackend()

        def my_kernel(x: float) -> float:
            return x * 2

        compiled = backend.compile_kernel(my_kernel)

        # Should return a callable
        assert callable(compiled)

    def test_parallel_for(self) -> None:
        """Test parallel_for method."""
        backend = CPUBackend()

        results: list[int] = []

        def append_idx(idx: int) -> None:
            results.append(idx)

        backend.parallel_for(0, 5, append_idx)

        assert results == [0, 1, 2, 3, 4]

    def test_repr(self) -> None:
        """Test string representation."""
        backend = CPUBackend()

        repr_str = repr(backend)

        assert "CPUBackend" in repr_str
        assert "available=True" in repr_str


class TestCPUKernels:
    """Tests for CPU kernel implementations."""

    def test_vector_add(self) -> None:
        """Test CPU vector addition."""
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)

        result = vector_add_cpu(a, b)

        np.testing.assert_array_equal(result, [5, 7, 9])

    def test_matrix_multiply(self) -> None:
        """Test CPU matrix multiplication."""
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)

        result = matrix_multiply_cpu(a, b)

        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


class TestKernelExecutionResult:
    """Tests for KernelExecutionResult."""

    def test_successful_result(self) -> None:
        """Test successful execution result."""
        result = KernelExecutionResult(
            success=True,
            execution_time_ms=1.5,
        )

        assert result.success
        assert result.execution_time_ms == 1.5
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test failed execution result."""
        error = ValueError("Test error")
        result = KernelExecutionResult(
            success=False,
            execution_time_ms=0.1,
            error=error,
        )

        assert not result.success
        assert result.error is error


class TestBackendType:
    """Tests for BackendType enum."""

    def test_backend_types(self) -> None:
        """Test backend type enum values exist."""
        # auto() generates integer values
        assert BackendType.CPU is not None
        assert BackendType.CUDA is not None
        assert BackendType.METAL is not None

    def test_backend_type_names(self) -> None:
        """Test backend type names."""
        assert BackendType.CPU.name == "CPU"
        assert BackendType.CUDA.name == "CUDA"
        assert BackendType.METAL.name == "METAL"


class TestCPUBackendEdgeCases:
    """Edge case tests for CPUBackend."""

    def test_allocate_various_dtypes(self) -> None:
        """Test allocation with various data types."""
        backend = CPUBackend()

        for dtype in [np.float32, np.float64, np.int32, np.int64, np.uint8]:
            arr = backend.allocate((10,), dtype)
            assert arr.dtype == dtype

    def test_allocate_multidimensional(self) -> None:
        """Test multidimensional allocation."""
        backend = CPUBackend()

        arr = backend.allocate((2, 3, 4, 5), np.float32)
        assert arr.shape == (2, 3, 4, 5)

    def test_execute_kernel_with_kwargs(self) -> None:
        """Test kernel execution with keyword arguments."""
        backend = CPUBackend()

        def kernel_with_kwargs(*, multiplier: int = 2) -> int:
            return 10 * multiplier

        result = backend.execute_kernel(
            kernel_with_kwargs,
            grid_size=(1,),
            block_size=(1,),
            multiplier=3,
        )

        assert result.success

    def test_kernel_returning_value(self) -> None:
        """Test kernel that returns a value."""
        backend = CPUBackend()

        def compute_kernel(x: np.ndarray) -> np.ndarray:
            return x * 2

        data = np.array([1, 2, 3], dtype=np.float32)
        result = backend.execute_kernel(compute_kernel, (1,), (1,), data)

        assert result.success

    def test_multiple_kernel_executions(self) -> None:
        """Test multiple sequential kernel executions."""
        backend = CPUBackend()

        def add_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return a + b

        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)

        for _ in range(5):
            result = backend.execute_kernel(add_kernel, (1,), (1,), a, b)
            assert result.success
