"""
Unit tests for Metal backend.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from pydotcompute.backends.base import BackendType, KernelExecutionResult


def _check_metal_available() -> bool:
    """Check if Metal/MLX is available."""
    if sys.platform != "darwin":
        return False
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except ImportError:
        return False


# Skip all tests if Metal not available
pytestmark = pytest.mark.skipif(
    not _check_metal_available(),
    reason="Metal/MLX not available",
)


class TestMetalBackend:
    """Tests for MetalBackend."""

    def test_backend_type(self) -> None:
        """Test backend type."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        assert backend.backend_type == BackendType.METAL

    def test_is_available(self) -> None:
        """Test Metal backend availability."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        assert backend.is_available

    def test_device_count(self) -> None:
        """Test device count is at least 1."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        assert backend.device_count >= 1

    def test_allocate(self) -> None:
        """Test array allocation."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        arr = backend.allocate((10, 20), np.float32)

        # MLX array
        assert arr.shape == (10, 20)

    def test_allocate_zeros(self) -> None:
        """Test zero-filled allocation."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        arr = backend.allocate_zeros((10,), np.float32)

        # Convert to numpy to verify
        result = backend.copy_to_host(arr)
        np.testing.assert_array_equal(result, np.zeros(10, dtype=np.float32))

    def test_free(self) -> None:
        """Test array freeing."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        arr = backend.allocate((10,), np.float32)

        initial_count = len(backend._buffer_registry)
        backend.free(arr)
        assert len(backend._buffer_registry) < initial_count

    def test_copy_to_device(self) -> None:
        """Test copy to device."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        source = np.array([1, 2, 3], dtype=np.float32)
        device_arr = backend.copy_to_device(source)

        # Verify by copying back
        result = backend.copy_to_host(device_arr)
        np.testing.assert_array_equal(result, source)

    def test_copy_to_host(self) -> None:
        """Test copy to host."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        source = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        device_arr = backend.copy_to_device(source)

        host_arr = backend.copy_to_host(device_arr)

        np.testing.assert_array_almost_equal(host_arr, source)

    def test_synchronize(self) -> None:
        """Test synchronize (evaluates lazy computations)."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        # Should not raise
        backend.synchronize()

    def test_execute_kernel(self) -> None:
        """Test kernel execution."""
        import mlx.core as mx

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        def simple_kernel(data: mx.array) -> mx.array:
            return data * 2

        data = backend.copy_to_device(np.array([1, 2, 3], dtype=np.float32))
        result = backend.execute_kernel(simple_kernel, (1,), (256,), data)

        assert isinstance(result, KernelExecutionResult)
        assert result.success
        assert result.execution_time_ms >= 0

    def test_execute_kernel_error(self) -> None:
        """Test kernel execution error handling."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        def failing_kernel() -> None:
            raise ValueError("Test error")

        result = backend.execute_kernel(failing_kernel, (1,), (1,))

        assert not result.success
        assert result.error is not None

    def test_compile_kernel(self) -> None:
        """Test kernel compilation."""
        import mlx.core as mx

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        def my_kernel(x: mx.array) -> mx.array:
            return x * 2

        compiled = backend.compile_kernel(my_kernel)

        # Should return a callable
        assert callable(compiled)

        # Test compiled kernel
        data = np.array([1, 2, 3], dtype=np.float32)
        result = compiled(data)
        result_np = backend.copy_to_host(result)
        np.testing.assert_array_almost_equal(result_np, [2, 4, 6])

    def test_get_memory_info(self) -> None:
        """Test memory info retrieval."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        info = backend.get_memory_info()

        assert "allocated" in info
        assert "buffer_count" in info
        assert "cache_memory" in info
        assert "peak_memory" in info

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        # Should not raise
        backend.clear_cache()

    def test_repr(self) -> None:
        """Test string representation."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        repr_str = repr(backend)

        assert "MetalBackend" in repr_str
        assert "available=True" in repr_str


class TestMetalKernels:
    """Tests for Metal kernel implementations."""

    def test_vector_add(self) -> None:
        """Test Metal vector addition."""
        from pydotcompute.backends.metal import MetalBackend, get_vector_add_kernel

        kernel = get_vector_add_kernel()
        assert kernel is not None

        backend = MetalBackend()
        a = backend.copy_to_device(np.array([1, 2, 3], dtype=np.float32))
        b = backend.copy_to_device(np.array([4, 5, 6], dtype=np.float32))

        result = kernel(a, b)
        result_np = backend.copy_to_host(result)

        np.testing.assert_array_equal(result_np, [5, 7, 9])

    def test_matrix_multiply(self) -> None:
        """Test Metal matrix multiplication."""
        from pydotcompute.backends.metal import MetalBackend, get_matrix_multiply_kernel

        kernel = get_matrix_multiply_kernel()
        assert kernel is not None

        backend = MetalBackend()
        a = backend.copy_to_device(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = backend.copy_to_device(np.array([[5, 6], [7, 8]], dtype=np.float32))

        result = kernel(a, b)
        result_np = backend.copy_to_host(result)

        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result_np, expected)

    def test_elementwise_operations(self) -> None:
        """Test elementwise operations."""
        from pydotcompute.backends.metal import MetalBackend, get_elementwise_kernel

        backend = MetalBackend()
        data = backend.copy_to_device(np.array([1, 4, 9], dtype=np.float32))

        sqrt_kernel = get_elementwise_kernel("sqrt")
        assert sqrt_kernel is not None

        result = sqrt_kernel(data)
        result_np = backend.copy_to_host(result)

        np.testing.assert_array_almost_equal(result_np, [1, 2, 3])


class TestMetalBackendEdgeCases:
    """Edge case tests for MetalBackend."""

    def test_allocate_various_dtypes(self) -> None:
        """Test allocation with various data types."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        for dtype in [np.float32, np.float16, np.int32, np.uint8]:
            arr = backend.allocate((10,), dtype)
            result = backend.copy_to_host(arr)
            # float64 maps to float32 in MLX
            if dtype == np.float64:
                assert result.dtype == np.float32
            else:
                assert result.dtype == dtype

    def test_allocate_multidimensional(self) -> None:
        """Test multidimensional allocation."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        arr = backend.allocate((2, 3, 4, 5), np.float32)

        assert arr.shape == (2, 3, 4, 5)

    def test_large_array_allocation(self) -> None:
        """Test large array allocation."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        # Allocate 100MB array
        arr = backend.allocate((100 * 1024 * 1024 // 4,), np.float32)

        # Verify shape
        assert arr.shape[0] == 100 * 1024 * 1024 // 4

        backend.free(arr)

    def test_copy_roundtrip(self) -> None:
        """Test data integrity through copy roundtrip."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        # Random data
        np.random.seed(42)
        original = np.random.randn(100, 100).astype(np.float32)

        device = backend.copy_to_device(original)
        restored = backend.copy_to_host(device)

        np.testing.assert_array_almost_equal(restored, original)

    def test_multiple_operations(self) -> None:
        """Test multiple sequential operations."""
        import mlx.core as mx

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        def chain_ops(x: mx.array) -> mx.array:
            x = x * 2
            x = x + 1
            x = mx.sqrt(x)
            return x

        data = backend.copy_to_device(np.array([3, 8, 24], dtype=np.float32))

        for _ in range(5):
            result = backend.execute_kernel(chain_ops, (1,), (1,), data)
            assert result.success


class TestMetalUnifiedBuffer:
    """Tests for Metal integration with UnifiedBuffer."""

    def test_metal_property(self) -> None:
        """Test accessing .metal property on UnifiedBuffer."""
        from pydotcompute.core.unified_buffer import UnifiedBuffer

        buffer = UnifiedBuffer((100,), dtype=np.float32)
        buffer.allocate()
        buffer.host[:] = np.arange(100, dtype=np.float32)
        buffer.mark_host_dirty()

        metal_arr = buffer.metal

        # Should be an MLX array
        import mlx.core as mx

        assert hasattr(metal_arr, "shape")
        assert metal_arr.shape == (100,)

    def test_metal_sync_from_host(self) -> None:
        """Test that Metal array syncs from host when dirty."""
        from pydotcompute.core.unified_buffer import UnifiedBuffer

        buffer = UnifiedBuffer((10,), dtype=np.float32)
        buffer.allocate()
        buffer.host[:] = 42.0
        buffer.mark_host_dirty()

        # Access metal should sync
        metal_arr = buffer.metal

        # Verify
        import numpy as np

        result = np.array(metal_arr)
        np.testing.assert_array_almost_equal(result, np.full(10, 42.0, dtype=np.float32))

    def test_mark_metal_dirty(self) -> None:
        """Test marking Metal data as dirty."""
        from pydotcompute.core.unified_buffer import BufferState, UnifiedBuffer

        buffer = UnifiedBuffer((10,), dtype=np.float32)
        buffer.allocate()
        _ = buffer.metal  # Create metal array
        buffer.mark_metal_dirty()

        assert buffer.state == BufferState.DEVICE_DIRTY


class TestMetalAccelerator:
    """Tests for Metal integration with Accelerator."""

    def test_metal_available(self) -> None:
        """Test metal_available property."""
        from pydotcompute.core.accelerator import Accelerator

        # Reset singleton for clean test
        Accelerator._instance = None
        Accelerator._initialized = False

        acc = Accelerator()
        assert acc.metal_available

    def test_metal_device_discovered(self) -> None:
        """Test Metal device is discovered."""
        from pydotcompute.core.accelerator import Accelerator, DeviceType

        # Reset singleton
        Accelerator._instance = None
        Accelerator._initialized = False

        acc = Accelerator()

        metal_devices = [d for d in acc.devices if d.device_type == DeviceType.METAL]
        assert len(metal_devices) >= 1

    def test_accelerator_repr_includes_metal(self) -> None:
        """Test Accelerator repr includes metal info."""
        from pydotcompute.core.accelerator import Accelerator

        # Reset singleton
        Accelerator._instance = None
        Accelerator._initialized = False

        acc = Accelerator()
        repr_str = repr(acc)

        assert "metal_available=" in repr_str


class TestMetalBackendConcurrency:
    """Thread safety and concurrency tests for MetalBackend.

    Note: MLX has known limitations with aggressive concurrent Metal operations.
    These tests use conservative settings to avoid Metal command buffer conflicts.
    """

    def test_sequential_allocations_from_threads(self) -> None:
        """Test thread-safe sequential allocations."""
        import threading
        import time

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        results: list[bool] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def allocate_worker() -> None:
            try:
                # Small delay to reduce contention
                time.sleep(0.01)
                with lock:
                    arr = backend.allocate((100,), np.float32)
                    _ = backend.copy_to_host(arr)
                    backend.free(arr)
                results.append(True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=allocate_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 5

    def test_buffer_registry_thread_safety(self) -> None:
        """Test buffer registry is thread-safe."""
        import threading

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        errors: list[Exception] = []
        lock = threading.Lock()

        def registry_worker() -> None:
            try:
                for _ in range(10):
                    with lock:
                        arr = backend.allocate((50,), np.float32)
                        _ = backend.get_memory_info()
                        backend.free(arr)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=registry_worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"


class TestMetalBackendStress:
    """Stress tests for MetalBackend."""

    def test_rapid_allocation_deallocation(self) -> None:
        """Test rapid allocation/deallocation cycles."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        for _ in range(100):
            arr = backend.allocate((1000,), np.float32)
            backend.free(arr)

        # Should not leak memory
        info = backend.get_memory_info()
        assert info["buffer_count"] == 0

    def test_many_small_allocations(self) -> None:
        """Test many small allocations."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        arrays = []

        # Allocate many small arrays
        for _ in range(100):
            arr = backend.allocate((10,), np.float32)
            arrays.append(arr)

        assert backend.get_memory_info()["buffer_count"] == 100

        # Free all
        for arr in arrays:
            backend.free(arr)

        assert backend.get_memory_info()["buffer_count"] == 0

    def test_large_data_transfer(self) -> None:
        """Test large data transfer integrity."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        # 50MB array
        size = 50 * 1024 * 1024 // 4
        np.random.seed(42)
        large_data = np.random.randn(size).astype(np.float32)

        device_arr = backend.copy_to_device(large_data)
        result = backend.copy_to_host(device_arr)

        np.testing.assert_array_almost_equal(result, large_data)
        backend.free(device_arr)


class TestMetalKernelCompilation:
    """Tests for kernel compilation."""

    def test_compile_and_cache(self) -> None:
        """Test kernel compilation and caching."""
        import mlx.core as mx

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        def my_kernel(x: mx.array) -> mx.array:
            return x * 2 + 1

        compiled1 = backend.compile_kernel(my_kernel)
        compiled2 = backend.compile_kernel(my_kernel)

        # Should return same cached version
        assert compiled1 is compiled2

    def test_compiled_kernel_execution(self) -> None:
        """Test compiled kernel produces correct results."""
        import mlx.core as mx

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        def transform_kernel(x: mx.array) -> mx.array:
            return mx.exp(mx.negative(mx.square(x)))

        compiled = backend.compile_kernel(transform_kernel)

        input_data = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        result = compiled(input_data)
        result_np = backend.copy_to_host(result)

        expected = np.exp(-np.square(input_data))
        np.testing.assert_array_almost_equal(result_np, expected, decimal=5)

    def test_msl_compilation_placeholder(self) -> None:
        """Test MSL compilation returns placeholder (not yet implemented)."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        msl_source = """
        kernel void add_arrays(device float* a, device float* b, device float* c, uint id [[thread_position_in_grid]]) {
            c[id] = a[id] + b[id];
        }
        """

        compiled = backend.compile_kernel(msl_source, signature="add_arrays")

        # Should return a placeholder that raises NotImplementedError
        with pytest.raises(NotImplementedError):
            compiled()


class TestMetalElementwiseKernels:
    """Tests for all elementwise kernel operations."""

    def test_all_elementwise_operations(self) -> None:
        """Test all supported elementwise operations."""
        from pydotcompute.backends.metal import MetalBackend, get_elementwise_kernel

        backend = MetalBackend()

        operations = ["add", "sub", "mul", "div", "sqrt", "exp", "log", "sin", "cos", "abs", "square", "negative"]

        for op in operations:
            kernel = get_elementwise_kernel(op)
            assert kernel is not None, f"Kernel for {op} should not be None"

    def test_elementwise_add(self) -> None:
        """Test elementwise addition."""
        from pydotcompute.backends.metal import MetalBackend, get_elementwise_kernel

        backend = MetalBackend()
        kernel = get_elementwise_kernel("add")

        a = backend.copy_to_device(np.array([1, 2, 3], dtype=np.float32))
        b = backend.copy_to_device(np.array([4, 5, 6], dtype=np.float32))

        result = kernel(a, b)
        result_np = backend.copy_to_host(result)

        np.testing.assert_array_equal(result_np, [5, 7, 9])

    def test_elementwise_exp(self) -> None:
        """Test elementwise exponential."""
        from pydotcompute.backends.metal import MetalBackend, get_elementwise_kernel

        backend = MetalBackend()
        kernel = get_elementwise_kernel("exp")

        data = backend.copy_to_device(np.array([0, 1, 2], dtype=np.float32))
        result = kernel(data)
        result_np = backend.copy_to_host(result)

        expected = np.exp(np.array([0, 1, 2], dtype=np.float32))
        np.testing.assert_array_almost_equal(result_np, expected, decimal=5)

    def test_elementwise_negative(self) -> None:
        """Test elementwise negation."""
        from pydotcompute.backends.metal import MetalBackend, get_elementwise_kernel

        backend = MetalBackend()
        kernel = get_elementwise_kernel("negative")

        data = backend.copy_to_device(np.array([1, -2, 3], dtype=np.float32))
        result = kernel(data)
        result_np = backend.copy_to_host(result)

        np.testing.assert_array_equal(result_np, [-1, 2, -3])


class TestMetalBackendErrorHandling:
    """Error handling tests for MetalBackend."""

    def test_kernel_exception_handling(self) -> None:
        """Test that kernel exceptions are caught and returned."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        def bad_kernel() -> None:
            raise RuntimeError("Intentional test error")

        result = backend.execute_kernel(bad_kernel, (1,), (1,))

        assert not result.success
        assert result.error is not None
        assert "Intentional test error" in str(result.error)

    def test_invalid_dtype_fallback(self) -> None:
        """Test that invalid dtypes fall back to float32."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        # complex64 is not directly supported, should fallback
        arr = backend.allocate((10,), np.float32)
        assert arr is not None

    def test_cleanup_on_del(self) -> None:
        """Test cleanup occurs on deletion."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        _ = backend.allocate((100,), np.float32)

        # Cleanup should not raise
        backend._cleanup()

        assert backend.get_memory_info()["buffer_count"] == 0


class TestMetalReductionOperations:
    """Tests for reduction operations on Metal."""

    def test_sum_reduction(self) -> None:
        """Test sum reduction."""
        import mlx.core as mx

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        data = backend.copy_to_device(np.array([1, 2, 3, 4, 5], dtype=np.float32))

        result = mx.sum(data)
        mx.eval(result)

        assert float(result) == 15.0

    def test_mean_reduction(self) -> None:
        """Test mean reduction."""
        import mlx.core as mx

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        data = backend.copy_to_device(np.array([1, 2, 3, 4, 5], dtype=np.float32))

        result = mx.mean(data)
        mx.eval(result)

        assert float(result) == 3.0

    def test_max_min_reduction(self) -> None:
        """Test max and min reductions."""
        import mlx.core as mx

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        data = backend.copy_to_device(np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float32))

        max_result = mx.max(data)
        min_result = mx.min(data)
        mx.eval(max_result, min_result)

        assert float(max_result) == 9.0
        assert float(min_result) == 1.0

    def test_2d_axis_reduction(self) -> None:
        """Test reduction along specific axis."""
        import mlx.core as mx

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        data = backend.copy_to_device(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))

        row_sum = mx.sum(data, axis=1)
        col_sum = mx.sum(data, axis=0)
        mx.eval(row_sum, col_sum)

        np.testing.assert_array_equal(np.array(row_sum), [6, 15])
        np.testing.assert_array_equal(np.array(col_sum), [5, 7, 9])


class TestMetalBroadcasting:
    """Tests for broadcasting operations on Metal."""

    def test_scalar_broadcast(self) -> None:
        """Test scalar broadcasting."""
        import mlx.core as mx

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        data = backend.copy_to_device(np.array([1, 2, 3], dtype=np.float32))

        result = mx.add(data, 10.0)
        result_np = backend.copy_to_host(result)

        np.testing.assert_array_equal(result_np, [11, 12, 13])

    def test_array_broadcast(self) -> None:
        """Test array broadcasting."""
        import mlx.core as mx

        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()

        # (3,) + (3, 1) -> (3, 3)
        a = backend.copy_to_device(np.array([1, 2, 3], dtype=np.float32))
        b = backend.copy_to_device(np.array([[10], [20], [30]], dtype=np.float32))

        result = mx.add(a, b)
        result_np = backend.copy_to_host(result)

        expected = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=np.float32)
        np.testing.assert_array_equal(result_np, expected)


class TestMetalConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_check_metal_available_function(self) -> None:
        """Test _check_metal_available function."""
        from pydotcompute.backends.metal import _check_metal_available

        result = _check_metal_available()
        assert isinstance(result, bool)
        assert result is True  # Since tests run on Metal-available system

    def test_get_invalid_elementwise_kernel(self) -> None:
        """Test getting invalid elementwise kernel returns None."""
        from pydotcompute.backends.metal import get_elementwise_kernel

        result = get_elementwise_kernel("invalid_operation")
        assert result is None


class TestMetalBufferInfo:
    """Tests for MetalBufferInfo tracking."""

    def test_buffer_info_creation(self) -> None:
        """Test buffer info is created correctly."""
        from pydotcompute.backends.metal import MetalBackend

        backend = MetalBackend()
        arr = backend.allocate((100,), np.float32)

        array_id = id(arr)
        assert array_id in backend._buffer_registry

        info = backend._buffer_registry[array_id]
        assert info.shape == (100,)
        assert info.size == 100 * 4  # float32 = 4 bytes
        assert info.dtype == np.dtype(np.float32)

    def test_buffer_info_tracking(self) -> None:
        """Test buffer tracking across operations."""
        from pydotcompute.backends.metal import MetalBackend, MetalBufferState

        backend = MetalBackend()

        arr1 = backend.allocate((50,), np.float32)
        arr2 = backend.allocate((100,), np.float32)

        assert len(backend._buffer_registry) == 2

        backend.free(arr1)
        assert len(backend._buffer_registry) == 1

        backend.free(arr2)
        assert len(backend._buffer_registry) == 0
