"""
Vector Addition Example for PyDotCompute.

Demonstrates basic ring kernel usage with a simple vector addition actor.
This example works on both CPU and GPU.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydotcompute import RingKernelRuntime, message, ring_kernel

if TYPE_CHECKING:
    from pydotcompute.ring_kernels.lifecycle import KernelContext


# Define message types using @message decorator
@message
@dataclass
class VectorAddRequest:
    """Request to add two values."""

    a: float
    b: float
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None


@message
@dataclass
class VectorAddResponse:
    """Response with the sum."""

    result: float
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None


# Define the ring kernel actor
@ring_kernel(
    kernel_id="vector_add",
    input_type=VectorAddRequest,
    output_type=VectorAddResponse,
    queue_size=1000,
)
async def vector_add_actor(ctx: KernelContext[VectorAddRequest, VectorAddResponse]) -> None:
    """
    Persistent actor that processes vector additions.

    This actor runs in a loop, receiving requests and sending responses
    until terminated.
    """
    print(f"[{ctx.kernel_id}] Actor started")

    while not ctx.should_terminate:
        try:
            # Wait for active state
            if not ctx.is_active:
                await ctx.wait_active()
                continue

            # Receive a request (with timeout to allow checking termination)
            try:
                request = await ctx.receive(timeout=0.1)
            except Exception:
                continue

            # Process the request
            result = request.a + request.b

            # Create and send response
            response = VectorAddResponse(
                result=result,
                correlation_id=request.message_id,
            )
            await ctx.send(response)

        except Exception as e:
            print(f"[{ctx.kernel_id}] Error: {e}")

    print(f"[{ctx.kernel_id}] Actor terminated")


async def run_vector_add_example() -> None:
    """Run the vector addition example."""
    print("=" * 60)
    print("PyDotCompute Vector Addition Example")
    print("=" * 60)

    async with RingKernelRuntime() as runtime:
        print("\n1. Launching kernel...")
        await runtime.launch("vector_add")
        print("   Kernel launched (resources allocated)")

        print("\n2. Activating kernel...")
        await runtime.activate("vector_add")
        print("   Kernel active (processing messages)")

        print("\n3. Sending requests...")
        test_cases = [
            (1.0, 2.0),
            (10.5, 20.5),
            (-5.0, 5.0),
            (0.0, 0.0),
            (100.0, 200.0),
        ]

        for a, b in test_cases:
            request = VectorAddRequest(a=a, b=b)
            await runtime.send("vector_add", request)
            print(f"   Sent: {a} + {b}")

        print("\n4. Receiving responses...")
        for a, b in test_cases:
            try:
                response = await runtime.receive("vector_add", timeout=1.0)
                print(f"   Received: {a} + {b} = {response.result}")
            except Exception as e:
                print(f"   Error receiving: {e}")

        print("\n5. Getting telemetry...")
        telemetry = runtime.get_telemetry("vector_add")
        if telemetry:
            print(f"   Messages processed: {telemetry.messages_processed}")
            print(f"   Throughput: {telemetry.throughput:.2f} msg/s")

        print("\n6. Shutting down...")
        # Runtime context manager handles termination

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


# Alternative: Manual kernel definition without decorator
class ManualVectorAddKernel:
    """
    Example of manually defining a ring kernel without decorators.

    Useful for more complex scenarios or dynamic kernel creation.
    """

    @staticmethod
    async def handler(
        ctx: KernelContext[VectorAddRequest, VectorAddResponse],
    ) -> None:
        """Process vector addition requests."""
        while not ctx.should_terminate:
            if not ctx.is_active:
                await ctx.wait_active()
                continue

            try:
                request = await ctx.receive(timeout=0.1)
                response = VectorAddResponse(
                    result=request.a + request.b,
                    correlation_id=request.message_id,
                )
                await ctx.send(response)
            except Exception:
                continue


async def run_manual_example() -> None:
    """Run example with manually defined kernel."""
    from pydotcompute.ring_kernels.lifecycle import RingKernelConfig

    async with RingKernelRuntime() as runtime:
        # Launch with explicit handler
        config = RingKernelConfig(
            kernel_id="manual_add",
            input_queue_size=500,
            output_queue_size=500,
        )

        await runtime.launch(
            "manual_add",
            ManualVectorAddKernel.handler,
            config=config,
            input_type=VectorAddRequest,
            output_type=VectorAddResponse,
        )
        await runtime.activate("manual_add")

        # Send and receive
        await runtime.send("manual_add", VectorAddRequest(a=42.0, b=58.0))
        response = await runtime.receive("manual_add", timeout=1.0)
        print(f"Manual kernel result: {response.result}")


if __name__ == "__main__":
    asyncio.run(run_vector_add_example())
