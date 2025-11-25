"""
Unit tests for decorators.
"""

from __future__ import annotations

import pytest

from pydotcompute.decorators.kernel import (
    KernelRegistry,
    cuda_source,
    get_kernel_registry,
    gpu_kernel,
    kernel,
)
from pydotcompute.decorators.ring_kernel import (
    RingKernelBuilder,
    get_ring_kernel_meta,
    is_ring_kernel,
    ring_kernel,
)
from pydotcompute.decorators.validators import (
    RuntimeValidator,
    validate_block_size,
    validate_grid_size,
    validate_priority,
    validate_queue_size,
)
from pydotcompute.exceptions import InvalidConfigurationError


class TestKernelDecorator:
    """Tests for @kernel decorator."""

    def test_basic_decoration(self) -> None:
        """Test basic kernel decoration."""

        @kernel
        def my_kernel(a: list, b: list, out: list) -> None:
            for i in range(len(out)):
                out[i] = a[i] + b[i]

        assert hasattr(my_kernel, "_kernel_meta")
        assert my_kernel._kernel_meta["name"] == "my_kernel"

    def test_decoration_with_options(self) -> None:
        """Test decoration with custom options."""

        @kernel(name="custom_name", device="cpu", fastmath=False)
        def my_kernel(a: list, b: list, out: list) -> None:
            pass

        assert my_kernel._kernel_meta["name"] == "custom_name"
        assert my_kernel._kernel_meta["device"] == "cpu"
        assert my_kernel._kernel_meta["fastmath"] is False

    def test_kernel_execution(self) -> None:
        """Test that decorated kernel can be executed."""

        @kernel(device="cpu")
        def add_one(data: list) -> list:
            return [x + 1 for x in data]

        result = add_one([1, 2, 3])

        assert result == [2, 3, 4]


class TestCudaSourceDecorator:
    """Tests for @cuda_source decorator."""

    def test_attaches_source(self) -> None:
        """Test that cuda_source attaches source code."""
        source_code = """
        extern "C" __global__
        void my_kernel(float* a) {
            int idx = threadIdx.x;
            a[idx] = a[idx] * 2;
        }
        """

        @cuda_source(source_code)
        def my_kernel() -> None:
            pass

        assert hasattr(my_kernel, "__cuda_source__")
        assert "my_kernel" in my_kernel.__cuda_source__


class TestKernelRegistry:
    """Tests for KernelRegistry."""

    def test_singleton(self) -> None:
        """Test that registry is a singleton."""
        reg1 = KernelRegistry()
        reg2 = KernelRegistry()

        assert reg1 is reg2

    def test_register_and_get(self) -> None:
        """Test registering and getting kernels."""
        registry = get_kernel_registry()
        registry.clear()

        def my_kernel() -> None:
            pass

        registry.register("test_kernel", my_kernel)

        retrieved = registry.get("test_kernel")
        assert retrieved is my_kernel

    def test_list_kernels(self) -> None:
        """Test listing registered kernels."""
        registry = get_kernel_registry()
        registry.clear()

        registry.register("kernel1", lambda: None)
        registry.register("kernel2", lambda: None)

        names = registry.list()

        assert "kernel1" in names
        assert "kernel2" in names

    def test_duplicate_registration_raises(self) -> None:
        """Test that duplicate registration raises error."""
        registry = get_kernel_registry()
        registry.clear()

        registry.register("dup_kernel", lambda: None)

        with pytest.raises(ValueError):
            registry.register("dup_kernel", lambda: None)

    def test_overwrite_registration(self) -> None:
        """Test overwrite flag allows replacement."""
        registry = get_kernel_registry()
        registry.clear()

        def first() -> None:
            pass

        def second() -> None:
            pass

        registry.register("kernel", first)
        registry.register("kernel", second, overwrite=True)

        assert registry.get("kernel") is second


class TestRingKernelDecorator:
    """Tests for @ring_kernel decorator."""

    def test_basic_decoration(self) -> None:
        """Test basic ring kernel decoration."""

        @ring_kernel(kernel_id="test_actor", auto_register=False)
        async def my_actor(ctx) -> None:  # type: ignore
            pass

        assert is_ring_kernel(my_actor)

        meta = get_ring_kernel_meta(my_actor)
        assert meta is not None
        assert meta["kernel_id"] == "test_actor"

    def test_decoration_with_options(self) -> None:
        """Test decoration with custom options."""

        @ring_kernel(
            kernel_id="custom_actor",
            queue_size=2000,
            grid_size=4,
            block_size=512,
            auto_register=False,
        )
        async def my_actor(ctx) -> None:  # type: ignore
            pass

        meta = get_ring_kernel_meta(my_actor)
        assert meta is not None
        assert meta["config"]["input_queue_size"] == 2000
        assert meta["config"]["output_queue_size"] == 2000
        assert meta["config"]["grid_size"] == 4
        assert meta["config"]["block_size"] == 512


class TestRingKernelBuilder:
    """Tests for RingKernelBuilder."""

    def test_builder_chain(self) -> None:
        """Test builder method chaining."""

        async def handler(ctx) -> None:  # type: ignore
            pass

        builder = (
            RingKernelBuilder("builder_kernel")
            .with_queue_size(1000)
            .with_grid_size(2)
            .with_block_size(128)
            .with_backpressure("reject")
            .with_handler(handler)
        )

        assert builder._kernel_id == "builder_kernel"
        assert builder._input_queue_size == 1000
        assert builder._grid_size == 2
        assert builder._block_size == 128
        assert builder._backpressure == "reject"

    def test_builder_without_handler_raises(self) -> None:
        """Test that build without handler raises error."""
        builder = RingKernelBuilder("test")

        with pytest.raises(ValueError):
            builder.build()


class TestValidators:
    """Tests for validator functions."""

    def test_validate_queue_size(self) -> None:
        """Test queue size validation."""
        assert validate_queue_size(100) == 100
        assert validate_queue_size(1) == 1
        assert validate_queue_size(1000000) == 1000000

        with pytest.raises(InvalidConfigurationError):
            validate_queue_size(0)

        with pytest.raises(InvalidConfigurationError):
            validate_queue_size(-1)

        with pytest.raises(InvalidConfigurationError):
            validate_queue_size(1000001)

    def test_validate_grid_size(self) -> None:
        """Test grid size validation."""
        assert validate_grid_size(1) == (1,)
        assert validate_grid_size((2, 3)) == (2, 3)
        assert validate_grid_size((1, 2, 3)) == (1, 2, 3)

        with pytest.raises(InvalidConfigurationError):
            validate_grid_size(0)

        with pytest.raises(InvalidConfigurationError):
            validate_grid_size((1, 2, 3, 4))  # Too many dimensions

    def test_validate_block_size(self) -> None:
        """Test block size validation."""
        assert validate_block_size(256) == (256,)
        assert validate_block_size((16, 16)) == (16, 16)

        with pytest.raises(InvalidConfigurationError):
            validate_block_size(0)

        with pytest.raises(InvalidConfigurationError):
            validate_block_size(2048)  # Exceeds max threads

    def test_validate_priority(self) -> None:
        """Test priority validation."""
        assert validate_priority(0) == 0
        assert validate_priority(128) == 128
        assert validate_priority(255) == 255

        with pytest.raises(InvalidConfigurationError):
            validate_priority(-1)

        with pytest.raises(InvalidConfigurationError):
            validate_priority(256)


class TestRuntimeValidator:
    """Tests for RuntimeValidator class."""

    def test_strict_mode(self) -> None:
        """Test strict mode raises exceptions."""
        from dataclasses import dataclass

        validator = RuntimeValidator(strict=True)

        class NotADataclass:
            pass

        with pytest.raises(Exception):
            validator.validate_message_type(NotADataclass)

    def test_non_strict_mode(self) -> None:
        """Test non-strict mode collects warnings."""
        validator = RuntimeValidator(strict=False)

        class NotADataclass:
            pass

        result = validator.validate_message_type(NotADataclass)

        assert result is False
        assert len(validator.warnings) > 0

    def test_validate_kernel_config(self) -> None:
        """Test kernel config validation."""
        validator = RuntimeValidator()

        config = {
            "queue_size": 1000,
            "grid_size": 4,
            "block_size": 256,
        }

        validated = validator.validate_kernel_config(config)

        assert validated["queue_size"] == 1000
        assert validated["grid_size"] == (4,)
        assert validated["block_size"] == (256,)


class TestKernelDecoratorEdgeCases:
    """Edge case tests for kernel decorators."""

    def test_kernel_meta_attributes(self) -> None:
        """Test that kernel metadata is properly stored."""

        @kernel(device="cpu", fastmath=False, cache=False)
        def my_kernel(x: int) -> int:
            return x * 2

        assert my_kernel._kernel_meta["device"] == "cpu"
        assert my_kernel._kernel_meta["fastmath"] is False
        assert my_kernel._kernel_meta["cache"] is False
        assert my_kernel._kernel_meta["compiled"] is None

    def test_kernel_preserves_function_name(self) -> None:
        """Test that decorated function preserves name."""

        @kernel
        def named_function() -> None:
            pass

        assert named_function.__name__ == "named_function"

    def test_kernel_with_return_value(self) -> None:
        """Test kernel that returns a value."""

        @kernel(device="cpu")
        def compute(a: int, b: int) -> int:
            return a + b

        result = compute(3, 4)
        assert result == 7

    def test_cuda_source_decorator(self) -> None:
        """Test cuda_source decorator standalone."""
        source = "void kernel() {}"

        @cuda_source(source)
        def my_kernel() -> None:
            pass

        assert hasattr(my_kernel, "__cuda_source__")
        assert my_kernel.__cuda_source__ == source

    def test_registry_get_nonexistent(self) -> None:
        """Test getting non-existent kernel from registry."""
        registry = get_kernel_registry()
        registry.clear()

        result = registry.get("nonexistent_kernel")
        assert result is None

    def test_registry_clear(self) -> None:
        """Test clearing registry."""
        registry = get_kernel_registry()
        registry.register("temp1", lambda: None, overwrite=True)
        registry.register("temp2", lambda: None, overwrite=True)

        registry.clear()

        assert registry.list() == []


class TestRingKernelDecoratorEdgeCases:
    """Edge case tests for ring_kernel decorator."""

    def test_is_ring_kernel_false(self) -> None:
        """Test is_ring_kernel returns False for non-ring kernels."""

        def regular_function() -> None:
            pass

        assert is_ring_kernel(regular_function) is False

    def test_get_ring_kernel_meta_none(self) -> None:
        """Test get_ring_kernel_meta returns None for non-ring kernels."""

        def regular_function() -> None:
            pass

        result = get_ring_kernel_meta(regular_function)
        assert result is None

    def test_builder_separate_queue_sizes(self) -> None:
        """Test builder with separate input/output queue sizes."""

        async def handler(ctx) -> None:  # type: ignore
            pass

        builder = (
            RingKernelBuilder("test")
            .with_input_queue_size(500)
            .with_output_queue_size(1000)
            .with_handler(handler)
        )

        assert builder._input_queue_size == 500
        assert builder._output_queue_size == 1000

    def test_ring_kernel_with_message_types(self) -> None:
        """Test ring kernel with specified message types."""
        from dataclasses import dataclass

        @dataclass
        class InputMsg:
            value: int

        @dataclass
        class OutputMsg:
            result: int

        @ring_kernel(
            kernel_id="typed_kernel",
            auto_register=False,
        )
        async def typed_actor(ctx) -> None:  # type: ignore
            pass

        meta = get_ring_kernel_meta(typed_actor)
        assert meta is not None
        assert meta["kernel_id"] == "typed_kernel"
