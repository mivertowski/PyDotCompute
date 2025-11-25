"""
End-to-end integration tests.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from pydotcompute import RingKernelRuntime, message, ring_kernel
from pydotcompute.ring_kernels.lifecycle import KernelContext


@message
@dataclass
class ComputeRequest:
    """Test compute request."""

    values: list = field(default_factory=list)
    operation: str = "sum"
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None


@message
@dataclass
class ComputeResponse:
    """Test compute response."""

    result: float = 0.0
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None


@ring_kernel(
    kernel_id="compute_service",
    input_type=ComputeRequest,
    output_type=ComputeResponse,
    queue_size=500,
    auto_register=False,
)
async def compute_service(
    ctx: KernelContext[ComputeRequest, ComputeResponse],
) -> None:
    """Test compute service actor."""
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            request = await ctx.receive(timeout=0.1)

            if request.operation == "sum":
                result = sum(request.values)
            elif request.operation == "mean":
                result = sum(request.values) / len(request.values) if request.values else 0
            elif request.operation == "max":
                result = max(request.values) if request.values else 0
            elif request.operation == "min":
                result = min(request.values) if request.values else 0
            else:
                result = 0

            response = ComputeResponse(
                result=float(result),
                correlation_id=request.message_id,
            )
            await ctx.send(response)

        except Exception:
            continue


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_basic_request_response(self) -> None:
        """Test basic request-response pattern."""
        async with RingKernelRuntime() as runtime:
            await runtime.launch("compute_service", compute_service)
            await runtime.activate("compute_service")

            await asyncio.sleep(0.1)

            request = ComputeRequest(values=[1, 2, 3, 4, 5], operation="sum")
            await runtime.send("compute_service", request)

            response = await runtime.receive("compute_service", timeout=1.0)

            assert response.result == 15.0

    @pytest.mark.asyncio
    async def test_multiple_operations(self) -> None:
        """Test multiple operations."""
        async with RingKernelRuntime() as runtime:
            await runtime.launch("compute_service", compute_service)
            await runtime.activate("compute_service")

            await asyncio.sleep(0.1)

            test_cases = [
                (ComputeRequest(values=[1, 2, 3], operation="sum"), 6.0),
                (ComputeRequest(values=[1, 2, 3], operation="mean"), 2.0),
                (ComputeRequest(values=[1, 5, 3], operation="max"), 5.0),
                (ComputeRequest(values=[4, 2, 3], operation="min"), 2.0),
            ]

            for request, expected in test_cases:
                await runtime.send("compute_service", request)
                response = await runtime.receive("compute_service", timeout=1.0)
                assert response.result == expected

    @pytest.mark.asyncio
    async def test_concurrent_requests(self) -> None:
        """Test handling concurrent requests."""
        async with RingKernelRuntime() as runtime:
            await runtime.launch("compute_service", compute_service)
            await runtime.activate("compute_service")

            await asyncio.sleep(0.1)

            # Send multiple requests concurrently
            requests = [
                ComputeRequest(values=[i, i + 1, i + 2], operation="sum")
                for i in range(10)
            ]

            for req in requests:
                await runtime.send("compute_service", req)

            # Receive all responses
            responses = []
            for _ in range(10):
                resp = await runtime.receive("compute_service", timeout=2.0)
                responses.append(resp)

            assert len(responses) == 10

    @pytest.mark.asyncio
    async def test_deactivate_reactivate(self) -> None:
        """Test deactivating and reactivating a kernel."""
        async with RingKernelRuntime() as runtime:
            await runtime.launch("compute_service", compute_service)
            await runtime.activate("compute_service")

            await asyncio.sleep(0.1)

            # Send request while active
            await runtime.send("compute_service", ComputeRequest(values=[1, 2], operation="sum"))
            resp1 = await runtime.receive("compute_service", timeout=1.0)
            assert resp1.result == 3.0

            # Deactivate
            await runtime.deactivate("compute_service")

            # Reactivate
            await runtime.reactivate("compute_service")

            await asyncio.sleep(0.1)

            # Send request after reactivation
            await runtime.send("compute_service", ComputeRequest(values=[3, 4], operation="sum"))
            resp2 = await runtime.receive("compute_service", timeout=1.0)
            assert resp2.result == 7.0

    @pytest.mark.asyncio
    async def test_multiple_kernels(self) -> None:
        """Test running multiple kernels."""

        @ring_kernel(kernel_id="doubler", auto_register=False)
        async def doubler(ctx: KernelContext) -> None:  # type: ignore
            while not ctx.should_terminate:
                if not ctx.is_active:
                    await ctx.wait_active()
                    continue
                try:
                    msg = await ctx.receive(timeout=0.1)
                    msg.result = msg.result * 2
                    await ctx.send(msg)
                except Exception:
                    continue

        async with RingKernelRuntime() as runtime:
            await runtime.launch("compute_service", compute_service)
            await runtime.launch("doubler", doubler)

            await runtime.activate("compute_service")
            await runtime.activate("doubler")

            await asyncio.sleep(0.1)

            # Both kernels should be active
            assert len(runtime.kernel_ids) == 2

    @pytest.mark.asyncio
    async def test_telemetry_tracking(self) -> None:
        """Test that telemetry is tracked."""
        async with RingKernelRuntime(enable_telemetry=True) as runtime:
            await runtime.launch("compute_service", compute_service)
            await runtime.activate("compute_service")

            await asyncio.sleep(0.1)

            # Send some requests
            for i in range(5):
                await runtime.send(
                    "compute_service",
                    ComputeRequest(values=[i], operation="sum"),
                )
                await runtime.receive("compute_service", timeout=1.0)

            telemetry = runtime.get_telemetry("compute_service")
            assert telemetry is not None

            summary = runtime.get_summary()
            assert summary["kernel_count"] == 1


class TestPipeline:
    """Tests for multi-stage pipelines."""

    @pytest.mark.asyncio
    async def test_two_stage_pipeline(self) -> None:
        """Test a two-stage processing pipeline."""

        @message
        @dataclass
        class Stage1Output:
            value: float = 0.0
            message_id: UUID = field(default_factory=uuid4)
            priority: int = 128
            correlation_id: UUID | None = None

        @ring_kernel(kernel_id="stage1", auto_register=False)
        async def stage1(ctx: KernelContext) -> None:  # type: ignore
            while not ctx.should_terminate:
                if not ctx.is_active:
                    await ctx.wait_active()
                    continue
                try:
                    msg = await ctx.receive(timeout=0.1)
                    output = Stage1Output(value=msg.values[0] * 2)
                    await ctx.send(output)
                except Exception:
                    continue

        @ring_kernel(kernel_id="stage2", auto_register=False)
        async def stage2(ctx: KernelContext) -> None:  # type: ignore
            while not ctx.should_terminate:
                if not ctx.is_active:
                    await ctx.wait_active()
                    continue
                try:
                    msg = await ctx.receive(timeout=0.1)
                    output = ComputeResponse(result=msg.value + 10)
                    await ctx.send(output)
                except Exception:
                    continue

        async with RingKernelRuntime() as runtime:
            await runtime.launch("stage1", stage1)
            await runtime.launch("stage2", stage2)
            await runtime.activate("stage1")
            await runtime.activate("stage2")

            await asyncio.sleep(0.1)

            # Send to stage 1
            await runtime.send("stage1", ComputeRequest(values=[5.0]))

            # Get from stage 1
            stage1_output = await runtime.receive("stage1", timeout=1.0)
            assert stage1_output.value == 10.0

            # Send to stage 2
            await runtime.send("stage2", stage1_output)

            # Get final result
            final = await runtime.receive("stage2", timeout=1.0)
            assert final.result == 20.0
