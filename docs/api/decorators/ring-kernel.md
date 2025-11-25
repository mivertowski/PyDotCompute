# @ring_kernel

Decorator for creating ring kernel actor functions.

## Overview

The `@ring_kernel` decorator transforms an async function into a ring kernel actor. Actors are persistent functions that process messages in an infinite loop, communicating through queues.

```python
from pydotcompute import ring_kernel

@ring_kernel(kernel_id="my_actor")
async def my_actor(ctx):
    while not ctx.should_terminate:
        msg = await ctx.receive()
        result = process(msg)
        await ctx.send(result)
```

## Decorator Signature

```python
def ring_kernel(
    *,
    kernel_id: str,
    input_type: type | None = None,
    output_type: type | None = None,
    queue_size: int = 1000,
    backpressure: BackpressureStrategy = BackpressureStrategy.BLOCK,
    auto_register: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for ring kernel actor functions.

    Args:
        kernel_id: Unique identifier for the kernel
        input_type: Expected input message type
        output_type: Expected output message type
        queue_size: Size of input/output queues
        backpressure: Strategy when queues are full
        auto_register: Auto-register with runtime

    Returns:
        Decorated actor function
    """
```

## Parameters

### kernel_id

```python
kernel_id: str
```

Unique identifier for the kernel. Used for:

- Launching and managing the kernel
- Sending/receiving messages
- Telemetry and logging

### input_type

```python
input_type: type | None = None
```

Expected type of input messages. If provided, enables runtime type checking.

### output_type

```python
output_type: type | None = None
```

Expected type of output messages. If provided, enables runtime type checking.

### queue_size

```python
queue_size: int = 1000
```

Maximum size of input and output queues. Larger queues buffer more messages but use more memory.

### backpressure

```python
backpressure: BackpressureStrategy = BackpressureStrategy.BLOCK
```

How to handle full queues:

- `BLOCK`: Wait for space (default)
- `REJECT`: Raise error immediately
- `DROP_OLDEST`: Drop oldest message

### auto_register

```python
auto_register: bool = True
```

Whether to automatically register with the global runtime. Set to `False` for testing or manual control.

## Actor Function Signature

```python
async def my_actor(ctx: KernelContext[TIn, TOut]) -> None:
    """
    Actor function signature.

    Args:
        ctx: Context for receiving/sending messages and checking state
    """
```

## Usage Examples

### Basic Actor

```python
from pydotcompute import ring_kernel, message
from dataclasses import dataclass

@message
@dataclass
class Request:
    value: int

@message
@dataclass
class Response:
    result: int

@ring_kernel(kernel_id="doubler")
async def doubler(ctx):
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            request = await ctx.receive(timeout=0.1)
            response = Response(result=request.value * 2)
            await ctx.send(response)
        except:
            continue
```

### With Type Safety

```python
@ring_kernel(
    kernel_id="calculator",
    input_type=CalculationRequest,
    output_type=CalculationResponse,
)
async def calculator(ctx: KernelContext[CalculationRequest, CalculationResponse]):
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            req = await ctx.receive(timeout=0.1)
            # req is typed as CalculationRequest
            result = calculate(req.a, req.b, req.operation)
            await ctx.send(CalculationResponse(result=result))
        except:
            continue
```

### Custom Queue Configuration

```python
@ring_kernel(
    kernel_id="high_throughput",
    queue_size=10000,
    backpressure=BackpressureStrategy.DROP_OLDEST,
)
async def high_throughput_actor(ctx):
    """Actor for high-volume streaming data."""
    while not ctx.should_terminate:
        try:
            msg = await ctx.receive(timeout=0.01)
            await ctx.send(process(msg))
        except:
            continue
```

### Without Auto-Registration

```python
@ring_kernel(kernel_id="test_actor", auto_register=False)
async def test_actor(ctx):
    while not ctx.should_terminate:
        try:
            msg = await ctx.receive(timeout=0.1)
            await ctx.send(msg)
        except:
            continue

# Must manually provide function to launch
async with RingKernelRuntime() as runtime:
    await runtime.launch("test_actor", test_actor)
```

### Error Handling

```python
@ring_kernel(kernel_id="resilient")
async def resilient_actor(ctx):
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            request = await ctx.receive(timeout=0.1)

            try:
                result = risky_operation(request)
                await ctx.send(SuccessResponse(data=result))
            except ValueError as e:
                await ctx.send(ErrorResponse(error=str(e)))
            except Exception as e:
                await ctx.send(ErrorResponse(error="Internal error"))

        except asyncio.TimeoutError:
            continue
```

### Stateful Actor

```python
@ring_kernel(kernel_id="counter")
async def counter_actor(ctx):
    """Actor that maintains state across messages."""
    count = 0

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            msg = await ctx.receive(timeout=0.1)

            if msg.action == "increment":
                count += 1
            elif msg.action == "decrement":
                count -= 1
            elif msg.action == "reset":
                count = 0

            await ctx.send(CountResponse(count=count))
        except:
            continue
```

## Actor Loop Pattern

The recommended pattern for all actors:

```python
@ring_kernel(kernel_id="template")
async def template_actor(ctx):
    # Optional: One-time initialization
    print(f"[{ctx.kernel_id}] Starting...")

    while not ctx.should_terminate:
        # Handle deactivation
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            # Receive with timeout for responsive shutdown
            msg = await ctx.receive(timeout=0.1)

            # Process message
            result = process(msg)

            # Send response
            await ctx.send(result)

        except asyncio.TimeoutError:
            # No message - loop back to check termination
            continue
        except Exception as e:
            # Log error but don't crash
            print(f"Error: {e}")
            continue

    # Optional: Cleanup
    print(f"[{ctx.kernel_id}] Terminated")
```

## Best Practices

1. **Always Check Termination**: Use `while not ctx.should_terminate`

2. **Handle Deactivation**: Check `is_active` and use `wait_active()`

3. **Use Timeout**: Allows responsive shutdown

4. **Never Block Forever**: Timeouts prevent actor from hanging

5. **Error Recovery**: Catch exceptions to prevent actor crash

6. **Correlation IDs**: Set `correlation_id` in responses

## Notes

- Actors run in asyncio tasks
- One actor instance per `kernel_id`
- State is private to the actor (no shared state)
- Use messages for all communication
- Actors are single-threaded (no internal concurrency)
