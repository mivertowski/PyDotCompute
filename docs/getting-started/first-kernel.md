# Your First Ring Kernel

This tutorial walks you through building a complete ring kernel actor with proper error handling, telemetry, and best practices.

## What We'll Build

A **Calculator Actor** that:

- Accepts arithmetic operation requests
- Processes them asynchronously
- Returns results with proper error handling
- Tracks performance metrics

## Step 1: Define the Messages

First, define the message types for communication:

```python
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Literal

from pydotcompute import message

@message
@dataclass
class CalculationRequest:
    """Request for a calculation."""
    a: float = 0.0
    b: float = 0.0
    operation: Literal["add", "sub", "mul", "div"] = "add"
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None

@message
@dataclass
class CalculationResponse:
    """Response with calculation result."""
    result: float = 0.0
    success: bool = True
    error: str | None = None
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None
```

!!! tip "Message Design Best Practices"
    - Include an `error` field for error responses
    - Use `correlation_id` to match responses to requests
    - Keep messages serializable (basic types, lists, dicts)

## Step 2: Create the Actor

Now implement the actor logic:

```python
from pydotcompute import ring_kernel
from pydotcompute.ring_kernels.lifecycle import KernelContext

@ring_kernel(
    kernel_id="calculator",
    input_type=CalculationRequest,
    output_type=CalculationResponse,
    queue_size=1000,
)
async def calculator_actor(
    ctx: KernelContext[CalculationRequest, CalculationResponse]
) -> None:
    """
    Calculator actor that performs arithmetic operations.

    This actor demonstrates:
    - Proper message handling
    - Error handling and reporting
    - Graceful shutdown
    """
    print(f"[{ctx.kernel_id}] Calculator started")

    while not ctx.should_terminate:
        # Wait for activation
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            # Receive request with timeout
            request = await ctx.receive(timeout=0.1)

            # Process the operation
            try:
                result = perform_calculation(
                    request.a,
                    request.b,
                    request.operation
                )
                response = CalculationResponse(
                    result=result,
                    success=True,
                    correlation_id=request.message_id,
                )

            except ZeroDivisionError:
                response = CalculationResponse(
                    result=0.0,
                    success=False,
                    error="Division by zero",
                    correlation_id=request.message_id,
                )

            except ValueError as e:
                response = CalculationResponse(
                    result=0.0,
                    success=False,
                    error=str(e),
                    correlation_id=request.message_id,
                )

            # Send response
            await ctx.send(response)

        except Exception:
            # Timeout or other error - continue loop
            continue

    print(f"[{ctx.kernel_id}] Calculator stopped")


def perform_calculation(a: float, b: float, op: str) -> float:
    """Perform the actual calculation."""
    match op:
        case "add":
            return a + b
        case "sub":
            return a - b
        case "mul":
            return a * b
        case "div":
            if b == 0:
                raise ZeroDivisionError()
            return a / b
        case _:
            raise ValueError(f"Unknown operation: {op}")
```

## Step 3: Use the Actor

Create a client that uses the calculator:

```python
import asyncio
from pydotcompute import RingKernelRuntime

async def main():
    async with RingKernelRuntime(enable_telemetry=True) as runtime:
        # Launch and activate
        await runtime.launch("calculator")
        await runtime.activate("calculator")

        # Wait for actor to start
        await asyncio.sleep(0.1)

        # Test cases
        test_cases = [
            (10, 5, "add"),   # 15
            (10, 5, "sub"),   # 5
            (10, 5, "mul"),   # 50
            (10, 5, "div"),   # 2
            (10, 0, "div"),   # Error: division by zero
        ]

        print("\n=== Calculator Tests ===\n")

        for a, b, op in test_cases:
            # Send request
            request = CalculationRequest(a=a, b=b, operation=op)
            await runtime.send("calculator", request)

            # Get response
            response = await runtime.receive("calculator", timeout=1.0)

            # Display result
            if response.success:
                print(f"  {a} {op} {b} = {response.result}")
            else:
                print(f"  {a} {op} {b} = ERROR: {response.error}")

        # Show telemetry
        print("\n=== Telemetry ===\n")
        telemetry = runtime.get_telemetry("calculator")
        if telemetry:
            print(f"  Messages processed: {telemetry.messages_processed}")
            print(f"  Throughput: {telemetry.throughput:.2f} msg/s")

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 4: Run and Test

```bash
python calculator.py
```

Expected output:

```
[calculator] Calculator started

=== Calculator Tests ===

  10 add 5 = 15.0
  10 sub 5 = 5.0
  10 mul 5 = 50.0
  10 div 5 = 2.0
  10 div 0 = ERROR: Division by zero

=== Telemetry ===

  Messages processed: 5
  Throughput: 45.23 msg/s

[calculator] Calculator stopped
```

## Understanding the Pattern

### The Actor Loop

Every ring kernel follows this pattern:

```python
while not ctx.should_terminate:
    if not ctx.is_active:
        await ctx.wait_active()
        continue

    try:
        msg = await ctx.receive(timeout=0.1)
        # Process message...
        await ctx.send(response)
    except:
        continue
```

**Key elements:**

1. **Termination check**: `ctx.should_terminate` - graceful shutdown
2. **Active check**: `ctx.is_active` - pause/resume support
3. **Timeout receive**: Prevents blocking forever, allows shutdown
4. **Error handling**: Catch exceptions, don't crash the actor

### Lifecycle States

```
CREATED → LAUNCHED → ACTIVE ↔ DEACTIVATED → TERMINATED
           ↑                        │
           └────────────────────────┘
```

- **CREATED**: Actor defined but not launched
- **LAUNCHED**: Resources allocated, not processing
- **ACTIVE**: Processing messages
- **DEACTIVATED**: Paused, can reactivate
- **TERMINATED**: Cleaned up, cannot restart

## Advanced Features

### Priority Messages

High-priority messages are processed first:

```python
# Low priority (default)
normal_request = CalculationRequest(a=1, b=2, priority=128)

# High priority
urgent_request = CalculationRequest(a=1, b=2, priority=255)
```

### Correlation Tracking

Match responses to requests:

```python
request = CalculationRequest(a=1, b=2)
await runtime.send("calculator", request)

response = await runtime.receive("calculator")
assert response.correlation_id == request.message_id
```

### Deactivation and Reactivation

Pause processing without losing state:

```python
await runtime.deactivate("calculator")
# Actor paused, messages queue up

await runtime.reactivate("calculator")
# Actor resumes, processes queued messages
```

## Complete Example

See the full example at: [examples/vector_add.py](https://github.com/mivertowski/PyDotCompute/blob/main/examples/vector_add.py)

## Next Steps

- **[Ring Kernel Concepts](../articles/concepts/ring-kernels.md)**: Deep dive into architecture
- **[Building Actors Guide](../articles/guides/building-actors.md)**: Best practices
- **[Pipeline Tutorial](../articles/guides/pipelines.md)**: Multi-stage processing
