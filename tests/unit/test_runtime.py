"""
Unit tests for RingKernelRuntime.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pydotcompute.exceptions import (
    KernelAlreadyExistsError,
    KernelNotFoundError,
    KernelStateError,
)
from pydotcompute.ring_kernels.lifecycle import KernelContext, KernelState
from pydotcompute.ring_kernels.message import RingKernelMessage
from pydotcompute.ring_kernels.runtime import RingKernelRuntime


async def echo_actor(ctx: KernelContext[RingKernelMessage, RingKernelMessage]) -> None:
    """Echo actor for testing."""
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            msg = await ctx.receive(timeout=0.1)
            await ctx.send(msg)
        except Exception:
            continue


class TestRingKernelRuntime:
    """Tests for RingKernelRuntime."""

    @pytest.mark.asyncio
    async def test_creation(self) -> None:
        """Test runtime creation."""
        runtime = RingKernelRuntime()

        assert not runtime.is_active
        assert runtime.kernel_ids == []

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test runtime as context manager."""
        async with RingKernelRuntime() as runtime:
            assert runtime.is_active

        assert not runtime.is_active

    @pytest.mark.asyncio
    async def test_launch_kernel(self) -> None:
        """Test launching a kernel."""
        async with RingKernelRuntime() as runtime:
            kernel = await runtime.launch("test_kernel", echo_actor)

            assert "test_kernel" in runtime.kernel_ids
            assert kernel.state == KernelState.LAUNCHED

    @pytest.mark.asyncio
    async def test_launch_duplicate_raises(self) -> None:
        """Test that launching duplicate kernel raises error."""
        async with RingKernelRuntime() as runtime:
            await runtime.launch("test_kernel", echo_actor)

            with pytest.raises(KernelAlreadyExistsError):
                await runtime.launch("test_kernel", echo_actor)

    @pytest.mark.asyncio
    async def test_activate_kernel(self) -> None:
        """Test activating a kernel."""
        async with RingKernelRuntime() as runtime:
            await runtime.launch("test_kernel", echo_actor)

            await runtime.activate("test_kernel")

            assert runtime.get_kernel_state("test_kernel") == KernelState.ACTIVE

    @pytest.mark.asyncio
    async def test_activate_nonexistent_raises(self) -> None:
        """Test that activating nonexistent kernel raises error."""
        async with RingKernelRuntime() as runtime:
            with pytest.raises(KernelNotFoundError):
                await runtime.activate("nonexistent")

    @pytest.mark.asyncio
    async def test_deactivate_kernel(self) -> None:
        """Test deactivating a kernel."""
        async with RingKernelRuntime() as runtime:
            await runtime.launch("test_kernel", echo_actor)
            await runtime.activate("test_kernel")

            await runtime.deactivate("test_kernel")

            assert runtime.get_kernel_state("test_kernel") == KernelState.DEACTIVATED

    @pytest.mark.asyncio
    async def test_reactivate_kernel(self) -> None:
        """Test reactivating a kernel."""
        async with RingKernelRuntime() as runtime:
            await runtime.launch("test_kernel", echo_actor)
            await runtime.activate("test_kernel")
            await runtime.deactivate("test_kernel")

            await runtime.reactivate("test_kernel")

            assert runtime.get_kernel_state("test_kernel") == KernelState.ACTIVE

    @pytest.mark.asyncio
    async def test_terminate_kernel(self) -> None:
        """Test terminating a kernel."""
        async with RingKernelRuntime() as runtime:
            await runtime.launch("test_kernel", echo_actor)
            await runtime.activate("test_kernel")

            await runtime.terminate("test_kernel")

            assert "test_kernel" not in runtime.kernel_ids

    @pytest.mark.asyncio
    async def test_send_receive(self) -> None:
        """Test sending and receiving messages."""
        async with RingKernelRuntime() as runtime:
            await runtime.launch("echo", echo_actor)
            await runtime.activate("echo")

            # Give actor time to start
            await asyncio.sleep(0.1)

            msg = RingKernelMessage(priority=100)
            await runtime.send("echo", msg)

            response = await runtime.receive("echo", timeout=1.0)

            assert response.message_id == msg.message_id

    @pytest.mark.asyncio
    async def test_send_and_receive(self) -> None:
        """Test send_and_receive convenience method."""
        async with RingKernelRuntime() as runtime:
            await runtime.launch("echo", echo_actor)
            await runtime.activate("echo")

            await asyncio.sleep(0.1)

            msg = RingKernelMessage(priority=50)
            response = await runtime.send_and_receive("echo", msg, timeout=1.0)

            assert response.message_id == msg.message_id

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_raises(self) -> None:
        """Test that sending to nonexistent kernel raises error."""
        async with RingKernelRuntime() as runtime:
            with pytest.raises(KernelNotFoundError):
                await runtime.send("nonexistent", RingKernelMessage())

    @pytest.mark.asyncio
    async def test_get_kernel_state(self) -> None:
        """Test getting kernel state."""
        async with RingKernelRuntime() as runtime:
            await runtime.launch("test_kernel", echo_actor)

            state = runtime.get_kernel_state("test_kernel")

            assert state == KernelState.LAUNCHED

    @pytest.mark.asyncio
    async def test_get_telemetry(self) -> None:
        """Test getting kernel telemetry."""
        async with RingKernelRuntime(enable_telemetry=True) as runtime:
            await runtime.launch("test_kernel", echo_actor)

            telemetry = runtime.get_telemetry("test_kernel")

            assert telemetry is not None
            assert telemetry.kernel_id == "test_kernel"

    @pytest.mark.asyncio
    async def test_get_all_telemetry(self) -> None:
        """Test getting all kernel telemetry."""
        async with RingKernelRuntime(enable_telemetry=True) as runtime:
            await runtime.launch("kernel1", echo_actor)
            await runtime.launch("kernel2", echo_actor)

            all_telemetry = runtime.get_all_telemetry()

            assert "kernel1" in all_telemetry
            assert "kernel2" in all_telemetry

    @pytest.mark.asyncio
    async def test_get_summary(self) -> None:
        """Test getting telemetry summary."""
        async with RingKernelRuntime(enable_telemetry=True) as runtime:
            await runtime.launch("kernel1", echo_actor)
            await runtime.launch("kernel2", echo_actor)

            summary = runtime.get_summary()

            assert "kernel_count" in summary
            assert summary["kernel_count"] == 2

    @pytest.mark.asyncio
    async def test_kernel_scope(self) -> None:
        """Test kernel_scope context manager."""
        async with RingKernelRuntime() as runtime:
            async with runtime.kernel_scope("scoped", echo_actor) as kernel:
                assert kernel.state == KernelState.ACTIVE

                msg = RingKernelMessage()
                await runtime.send("scoped", msg)

                await asyncio.sleep(0.1)

            # Kernel should be terminated after scope
            assert "scoped" not in runtime.kernel_ids

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """Test runtime shutdown terminates all kernels."""
        runtime = RingKernelRuntime()
        await runtime.start()

        await runtime.launch("kernel1", echo_actor)
        await runtime.launch("kernel2", echo_actor)
        await runtime.activate("kernel1")
        await runtime.activate("kernel2")

        await runtime.shutdown()

        assert not runtime.is_active
        assert runtime.kernel_ids == []


class TestRingKernelRuntimeTelemetryDisabled:
    """Tests for runtime with telemetry disabled."""

    @pytest.mark.asyncio
    async def test_telemetry_disabled(self) -> None:
        """Test that telemetry methods work when disabled."""
        async with RingKernelRuntime(enable_telemetry=False) as runtime:
            await runtime.launch("test_kernel", echo_actor)

            # Should return None when telemetry is disabled
            telemetry = runtime.get_telemetry("test_kernel")
            assert telemetry is None

            all_telemetry = runtime.get_all_telemetry()
            assert all_telemetry == {}

            summary = runtime.get_summary()
            assert "kernel_count" in summary
