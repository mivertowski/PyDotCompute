"""
Unit tests for ring kernel lifecycle management.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pydotcompute.exceptions import KernelStateError
from pydotcompute.ring_kernels.lifecycle import (
    KernelContext,
    KernelState,
    RingKernel,
    RingKernelConfig,
    managed_kernel,
)
from pydotcompute.ring_kernels.message import RingKernelMessage


async def simple_actor(ctx: KernelContext[RingKernelMessage, RingKernelMessage]) -> None:
    """Simple actor for testing."""
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            msg = await ctx.receive(timeout=0.1)
            await ctx.send(msg)
        except Exception:
            continue


class TestRingKernelConfig:
    """Tests for RingKernelConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = RingKernelConfig(kernel_id="test")

        assert config.kernel_id == "test"
        assert config.grid_size == 1
        assert config.block_size == 256
        assert config.input_queue_size == 4096
        assert config.output_queue_size == 4096

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RingKernelConfig(
            kernel_id="custom",
            grid_size=4,
            block_size=512,
            input_queue_size=1000,
            output_queue_size=2000,
        )

        assert config.grid_size == 4
        assert config.block_size == 512
        assert config.input_queue_size == 1000
        assert config.output_queue_size == 2000

    def test_validation(self) -> None:
        """Test configuration validation."""
        with pytest.raises(ValueError):
            RingKernelConfig(kernel_id="test", grid_size=0)

        with pytest.raises(ValueError):
            RingKernelConfig(kernel_id="test", block_size=-1)

        with pytest.raises(ValueError):
            RingKernelConfig(kernel_id="test", input_queue_size=0)


class TestKernelState:
    """Tests for KernelState enum."""

    def test_all_states(self) -> None:
        """Test all states are defined."""
        states = [
            KernelState.CREATED,
            KernelState.LAUNCHED,
            KernelState.ACTIVE,
            KernelState.DEACTIVATED,
            KernelState.TERMINATING,
            KernelState.TERMINATED,
        ]

        assert len(states) == 6


class TestRingKernel:
    """Tests for RingKernel class."""

    def test_creation(self) -> None:
        """Test kernel creation."""
        kernel: RingKernel[RingKernelMessage, RingKernelMessage] = RingKernel(
            "test_kernel",
            simple_actor,
        )

        assert kernel.kernel_id == "test_kernel"
        assert kernel.state == KernelState.CREATED

    @pytest.mark.asyncio
    async def test_launch(self) -> None:
        """Test kernel launch."""
        kernel: RingKernel[RingKernelMessage, RingKernelMessage] = RingKernel(
            "test_kernel",
            simple_actor,
        )

        await kernel.launch()

        assert kernel.state == KernelState.LAUNCHED

    @pytest.mark.asyncio
    async def test_launch_from_wrong_state(self) -> None:
        """Test launch from wrong state raises error."""
        kernel: RingKernel[RingKernelMessage, RingKernelMessage] = RingKernel(
            "test_kernel",
            simple_actor,
        )
        await kernel.launch()

        with pytest.raises(KernelStateError):
            await kernel.launch()

    @pytest.mark.asyncio
    async def test_activate(self) -> None:
        """Test kernel activation."""
        kernel: RingKernel[RingKernelMessage, RingKernelMessage] = RingKernel(
            "test_kernel",
            simple_actor,
        )
        await kernel.launch()

        await kernel.activate()

        assert kernel.state == KernelState.ACTIVE

    @pytest.mark.asyncio
    async def test_activate_from_wrong_state(self) -> None:
        """Test activate from wrong state raises error."""
        kernel: RingKernel[RingKernelMessage, RingKernelMessage] = RingKernel(
            "test_kernel",
            simple_actor,
        )

        with pytest.raises(KernelStateError):
            await kernel.activate()

    @pytest.mark.asyncio
    async def test_deactivate(self) -> None:
        """Test kernel deactivation."""
        kernel: RingKernel[RingKernelMessage, RingKernelMessage] = RingKernel(
            "test_kernel",
            simple_actor,
        )
        await kernel.launch()
        await kernel.activate()

        await kernel.deactivate()

        assert kernel.state == KernelState.DEACTIVATED

    @pytest.mark.asyncio
    async def test_reactivate(self) -> None:
        """Test kernel reactivation."""
        kernel: RingKernel[RingKernelMessage, RingKernelMessage] = RingKernel(
            "test_kernel",
            simple_actor,
        )
        await kernel.launch()
        await kernel.activate()
        await kernel.deactivate()

        await kernel.reactivate()

        assert kernel.state == KernelState.ACTIVE

    @pytest.mark.asyncio
    async def test_terminate(self) -> None:
        """Test kernel termination."""
        kernel: RingKernel[RingKernelMessage, RingKernelMessage] = RingKernel(
            "test_kernel",
            simple_actor,
        )
        await kernel.launch()
        await kernel.activate()

        await kernel.terminate(timeout=1.0)

        assert kernel.state == KernelState.TERMINATED

    @pytest.mark.asyncio
    async def test_send_receive(self) -> None:
        """Test sending and receiving messages."""
        kernel: RingKernel[RingKernelMessage, RingKernelMessage] = RingKernel(
            "test_kernel",
            simple_actor,
        )
        await kernel.launch()
        await kernel.activate()

        # Give actor time to start
        await asyncio.sleep(0.1)

        msg = RingKernelMessage(priority=100)
        await kernel.send(msg)

        response = await kernel.receive(timeout=1.0)

        assert response.message_id == msg.message_id

        await kernel.terminate()

    @pytest.mark.asyncio
    async def test_send_from_wrong_state(self) -> None:
        """Test send from wrong state raises error."""
        kernel: RingKernel[RingKernelMessage, RingKernelMessage] = RingKernel(
            "test_kernel",
            simple_actor,
        )

        with pytest.raises(KernelStateError):
            await kernel.send(RingKernelMessage())


class TestManagedKernel:
    """Tests for managed_kernel context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test managed kernel context manager."""
        async with managed_kernel("test_kernel", simple_actor) as kernel:
            assert kernel.state == KernelState.ACTIVE

            msg = RingKernelMessage()
            await kernel.send(msg)

            # Give time for processing
            await asyncio.sleep(0.1)

        # Kernel should be terminated after context
        assert kernel.state == KernelState.TERMINATED

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_error(self) -> None:
        """Test that context manager cleans up on error."""
        kernel_ref = None

        with pytest.raises(ValueError):
            async with managed_kernel("test_kernel", simple_actor) as kernel:
                kernel_ref = kernel
                raise ValueError("Test error")

        assert kernel_ref is not None
        assert kernel_ref.state == KernelState.TERMINATED
