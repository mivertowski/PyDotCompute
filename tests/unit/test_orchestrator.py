"""
Unit tests for the compute orchestrator module.

Tests the ComputeOrchestrator class, KernelLaunchConfig, KernelExecution,
and related resource coordination utilities.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np
import pytest

from pydotcompute.core.orchestrator import (
    ComputeOrchestrator,
    KernelExecution,
    KernelLaunchConfig,
)


class TestKernelLaunchConfig:
    """Tests for KernelLaunchConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default launch configuration."""
        config = KernelLaunchConfig()
        assert config.grid_size == (1,)
        assert config.block_size == (256,)
        assert config.shared_memory_bytes == 0
        assert config.stream is None

    def test_custom_config(self) -> None:
        """Test custom launch configuration."""
        config = KernelLaunchConfig(
            grid_size=(64, 64),
            block_size=(16, 16),
            shared_memory_bytes=1024,
        )
        assert config.grid_size == (64, 64)
        assert config.block_size == (16, 16)
        assert config.shared_memory_bytes == 1024

    def test_int_to_tuple_conversion(self) -> None:
        """Test that integers are converted to tuples."""
        config = KernelLaunchConfig(
            grid_size=128,
            block_size=256,
        )
        assert config.grid_size == (128,)
        assert config.block_size == (256,)

    def test_3d_config(self) -> None:
        """Test 3D grid and block configuration."""
        config = KernelLaunchConfig(
            grid_size=(8, 8, 8),
            block_size=(4, 4, 4),
        )
        assert config.grid_size == (8, 8, 8)
        assert config.block_size == (4, 4, 4)


class TestKernelExecution:
    """Tests for KernelExecution dataclass."""

    def test_default_execution(self) -> None:
        """Test default kernel execution record."""
        exec_record = KernelExecution()
        assert isinstance(exec_record.execution_id, UUID)
        assert exec_record.kernel_name == ""
        assert exec_record.success is False
        assert exec_record.error is None

    def test_custom_execution(self) -> None:
        """Test custom kernel execution record."""
        config = KernelLaunchConfig(grid_size=(32,), block_size=(128,))
        exec_record = KernelExecution(
            kernel_name="my_kernel",
            config=config,
            start_time=1000.0,
            end_time=1001.5,
            success=True,
        )
        assert exec_record.kernel_name == "my_kernel"
        assert exec_record.config.grid_size == (32,)
        assert exec_record.success is True

    def test_duration_ms(self) -> None:
        """Test duration calculation in milliseconds."""
        exec_record = KernelExecution(
            start_time=1.0,
            end_time=1.5,
        )
        assert exec_record.duration_ms == 500.0

    def test_execution_with_error(self) -> None:
        """Test execution record with error."""
        error = ValueError("Test error")
        exec_record = KernelExecution(
            kernel_name="failing_kernel",
            success=False,
            error=error,
        )
        assert exec_record.success is False
        assert exec_record.error is error


class TestComputeOrchestrator:
    """Tests for ComputeOrchestrator class."""

    @pytest.fixture
    async def orchestrator(self) -> ComputeOrchestrator:
        """Provide an initialized orchestrator."""
        orch = ComputeOrchestrator()
        await orch.initialize()
        yield orch
        await orch.shutdown()

    def test_creation(self) -> None:
        """Test orchestrator creation."""
        orch = ComputeOrchestrator()
        assert orch._active is False
        assert orch._registered_kernels == {}
        assert orch._execution_history == []

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager."""
        async with ComputeOrchestrator() as orch:
            assert orch.is_active is True

        assert orch.is_active is False

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """Test initialization."""
        orch = ComputeOrchestrator()
        assert orch.is_active is False

        await orch.initialize()
        assert orch.is_active is True
        assert orch._accelerator is not None
        assert orch._memory_pool is not None

        await orch.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self) -> None:
        """Test that initialize is idempotent."""
        orch = ComputeOrchestrator()
        await orch.initialize()
        await orch.initialize()  # Should not raise
        assert orch.is_active is True
        await orch.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown(self, orchestrator: ComputeOrchestrator) -> None:
        """Test shutdown."""
        assert orchestrator.is_active is True
        await orchestrator.shutdown()
        assert orchestrator.is_active is False

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, orchestrator: ComputeOrchestrator) -> None:
        """Test that shutdown is idempotent."""
        await orchestrator.shutdown()
        await orchestrator.shutdown()  # Should not raise
        assert orchestrator.is_active is False

    @pytest.mark.asyncio
    async def test_is_active_property(self, orchestrator: ComputeOrchestrator) -> None:
        """Test is_active property."""
        assert orchestrator.is_active is True

    @pytest.mark.asyncio
    async def test_accelerator_property(self, orchestrator: ComputeOrchestrator) -> None:
        """Test accelerator property."""
        assert orchestrator.accelerator is not None

    @pytest.mark.asyncio
    async def test_memory_pool_property(self, orchestrator: ComputeOrchestrator) -> None:
        """Test memory_pool property."""
        assert orchestrator.memory_pool is not None

    @pytest.mark.asyncio
    async def test_register_kernel(self, orchestrator: ComputeOrchestrator) -> None:
        """Test kernel registration."""

        def my_kernel(a: Any, b: Any) -> Any:
            return a + b

        orchestrator.register_kernel("add", my_kernel)
        assert "add" in orchestrator._registered_kernels
        assert orchestrator._registered_kernels["add"] is my_kernel

    @pytest.mark.asyncio
    async def test_get_kernel(self, orchestrator: ComputeOrchestrator) -> None:
        """Test getting a registered kernel."""

        def kernel() -> None:
            pass

        orchestrator.register_kernel("test", kernel)

        retrieved = orchestrator.get_kernel("test")
        assert retrieved is kernel

    @pytest.mark.asyncio
    async def test_get_kernel_nonexistent(self, orchestrator: ComputeOrchestrator) -> None:
        """Test getting non-existent kernel."""
        result = orchestrator.get_kernel("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_launch_kernel_function(self, orchestrator: ComputeOrchestrator) -> None:
        """Test launching a kernel function directly."""

        def simple_kernel(arr: np.ndarray) -> np.ndarray:
            return arr * 2

        config = KernelLaunchConfig()
        arr = np.array([1, 2, 3])

        execution = await orchestrator.launch_kernel(simple_kernel, config, arr)

        assert execution.success is True
        assert execution.kernel_name == "simple_kernel"
        assert execution.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_launch_registered_kernel(self, orchestrator: ComputeOrchestrator) -> None:
        """Test launching a registered kernel by name."""

        def add_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return a + b

        orchestrator.register_kernel("add", add_kernel)
        config = KernelLaunchConfig()

        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        execution = await orchestrator.launch_kernel("add", config, a, b)

        assert execution.success is True
        assert execution.kernel_name == "add"

    @pytest.mark.asyncio
    async def test_launch_nonexistent_kernel_raises(
        self, orchestrator: ComputeOrchestrator
    ) -> None:
        """Test that launching non-existent kernel raises."""
        config = KernelLaunchConfig()

        with pytest.raises(ValueError, match="not registered"):
            await orchestrator.launch_kernel("nonexistent", config)

    @pytest.mark.asyncio
    async def test_launch_kernel_failure(self, orchestrator: ComputeOrchestrator) -> None:
        """Test kernel launch failure handling."""

        def failing_kernel() -> None:
            raise RuntimeError("Kernel failed")

        config = KernelLaunchConfig()

        with pytest.raises(RuntimeError):
            await orchestrator.launch_kernel(failing_kernel, config)

        # Check execution was recorded with error
        history = orchestrator.get_execution_history()
        assert len(history) == 1
        assert history[0].success is False
        assert history[0].error is not None

    @pytest.mark.asyncio
    async def test_launch_async_kernel(self, orchestrator: ComputeOrchestrator) -> None:
        """Test launching an async kernel function."""

        async def async_kernel(x: int) -> int:
            return x * 2

        config = KernelLaunchConfig()
        execution = await orchestrator.launch_kernel(async_kernel, config, 5)

        assert execution.success is True

    @pytest.mark.asyncio
    async def test_create_stream(self, orchestrator: ComputeOrchestrator) -> None:
        """Test creating a CUDA stream."""
        stream = orchestrator.create_stream()
        # May be None if CUDA not available
        assert stream is None or hasattr(stream, "synchronize")

    @pytest.mark.asyncio
    async def test_synchronize(self, orchestrator: ComputeOrchestrator) -> None:
        """Test synchronization."""
        # Should not raise
        orchestrator.synchronize()

    @pytest.mark.asyncio
    async def test_get_execution_history(self, orchestrator: ComputeOrchestrator) -> None:
        """Test getting execution history."""

        def k1() -> None:
            pass

        def k2() -> None:
            pass

        config = KernelLaunchConfig()
        await orchestrator.launch_kernel(k1, config)
        await orchestrator.launch_kernel(k2, config)

        history = orchestrator.get_execution_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_get_execution_history_with_limit(
        self, orchestrator: ComputeOrchestrator
    ) -> None:
        """Test getting limited execution history."""

        def k() -> None:
            pass

        config = KernelLaunchConfig()
        for _ in range(5):
            await orchestrator.launch_kernel(k, config)

        history = orchestrator.get_execution_history(limit=3)
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_get_execution_history_by_kernel(
        self, orchestrator: ComputeOrchestrator
    ) -> None:
        """Test filtering execution history by kernel name."""

        def kernel_a() -> None:
            pass

        def kernel_b() -> None:
            pass

        config = KernelLaunchConfig()
        await orchestrator.launch_kernel(kernel_a, config)
        await orchestrator.launch_kernel(kernel_b, config)
        await orchestrator.launch_kernel(kernel_a, config)

        history = orchestrator.get_execution_history(kernel_name="kernel_a")
        assert len(history) == 2
        assert all(e.kernel_name == "kernel_a" for e in history)

    @pytest.mark.asyncio
    async def test_clear_execution_history(self, orchestrator: ComputeOrchestrator) -> None:
        """Test clearing execution history."""

        def k() -> None:
            pass

        config = KernelLaunchConfig()
        await orchestrator.launch_kernel(k, config)
        assert len(orchestrator._execution_history) == 1

        orchestrator.clear_execution_history()
        assert len(orchestrator._execution_history) == 0

    @pytest.mark.asyncio
    async def test_stream_context(self, orchestrator: ComputeOrchestrator) -> None:
        """Test stream context manager."""
        async with orchestrator.stream_context() as stream:
            # Stream may be None if CUDA not available
            pass  # Should not raise

    @pytest.mark.asyncio
    async def test_repr(self, orchestrator: ComputeOrchestrator) -> None:
        """Test string representation."""
        repr_str = repr(orchestrator)
        assert "ComputeOrchestrator" in repr_str
        assert "active=" in repr_str
        assert "kernels=" in repr_str
        assert "executions=" in repr_str


class TestOrchestratorEdgeCases:
    """Edge case tests for ComputeOrchestrator."""

    @pytest.mark.asyncio
    async def test_multiple_orchestrators(self) -> None:
        """Test using multiple orchestrators."""
        async with ComputeOrchestrator() as orch1:
            async with ComputeOrchestrator() as orch2:
                assert orch1.is_active
                assert orch2.is_active

    @pytest.mark.asyncio
    async def test_kernel_with_kwargs(self) -> None:
        """Test launching kernel with keyword arguments."""

        def kernel_with_kwargs(*, multiplier: int = 2) -> int:
            return 10 * multiplier

        async with ComputeOrchestrator() as orch:
            config = KernelLaunchConfig()
            execution = await orch.launch_kernel(
                kernel_with_kwargs, config, multiplier=5
            )
            assert execution.success is True
