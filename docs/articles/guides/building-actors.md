# Building Actors

Best practices for creating robust ring kernel actors.

## Actor Structure

### Basic Template

Every actor should follow this structure:

```python
from pydotcompute import ring_kernel, message
from pydotcompute.ring_kernels.lifecycle import KernelContext

@ring_kernel(kernel_id="my_actor")
async def my_actor(ctx: KernelContext) -> None:
    """
    Actor description.

    Explain what this actor does and any important behavior.
    """
    # === INITIALIZATION ===
    # One-time setup (load models, initialize state)
    state = initialize_state()

    # === MAIN LOOP ===
    while not ctx.should_terminate:
        # Handle deactivation
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            # Receive with timeout
            request = await ctx.receive(timeout=0.1)

            # Process request
            response = process(request, state)

            # Send response
            await ctx.send(response)

        except asyncio.TimeoutError:
            continue
        except Exception as e:
            # Log but don't crash
            log_error(e)
            continue

    # === CLEANUP ===
    cleanup(state)
```

## Message Design

### Request Messages

```python
@message
@dataclass
class CalculateRequest:
    """Request to perform calculation."""

    # Business fields
    a: float
    b: float
    operation: str = "add"

    # Standard fields (added by @message but shown for clarity)
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None
```

### Response Messages

```python
@message
@dataclass
class CalculateResponse:
    """Response with calculation result."""

    # Result field
    result: float

    # Success/error fields
    success: bool = True
    error: str | None = None
    error_code: str | None = None

    # Standard fields
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None
```

### Design Guidelines

1. **Include Error Information**: Always have `success` and `error` fields
2. **Use Error Codes**: Machine-readable error classification
3. **Minimal Fields**: Only include necessary data
4. **Default Values**: Provide sensible defaults
5. **Type Hints**: Enable IDE support and validation

## Error Handling

### Structured Error Responses

```python
@ring_kernel(kernel_id="calculator")
async def calculator(ctx: KernelContext) -> None:
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            request = await ctx.receive(timeout=0.1)

            # Business logic with error handling
            try:
                result = calculate(request.a, request.b, request.operation)
                await ctx.send(CalculateResponse(
                    result=result,
                    success=True,
                    correlation_id=request.message_id,
                ))

            except ZeroDivisionError:
                await ctx.send(CalculateResponse(
                    result=0.0,
                    success=False,
                    error="Division by zero",
                    error_code="DIVIDE_BY_ZERO",
                    correlation_id=request.message_id,
                ))

            except ValueError as e:
                await ctx.send(CalculateResponse(
                    result=0.0,
                    success=False,
                    error=str(e),
                    error_code="INVALID_INPUT",
                    correlation_id=request.message_id,
                ))

        except asyncio.TimeoutError:
            continue
        except Exception as e:
            # Unexpected error - log but continue
            logging.exception("Unexpected error in calculator")
            continue
```

### Error Categories

| Type | Handling | Example |
|------|----------|---------|
| Business errors | Return error response | Invalid input |
| Infrastructure errors | Log and continue | Network timeout |
| Fatal errors | Let crash (supervisor will restart) | Out of memory |

## Stateful Actors

### Maintaining State

```python
@ring_kernel(kernel_id="counter")
async def counter(ctx: KernelContext) -> None:
    """Counter that maintains state across messages."""

    # Private state
    count = 0
    history: list[int] = []

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            request = await ctx.receive(timeout=0.1)

            if request.action == "increment":
                count += 1
            elif request.action == "decrement":
                count -= 1
            elif request.action == "reset":
                count = 0

            history.append(count)

            await ctx.send(CounterResponse(
                count=count,
                correlation_id=request.message_id,
            ))

        except asyncio.TimeoutError:
            continue
```

### State Persistence

For durable state, persist periodically:

```python
@ring_kernel(kernel_id="persistent_actor")
async def persistent_actor(ctx: KernelContext) -> None:
    # Load persisted state
    state = load_state_from_disk("actor_state.json")
    messages_since_save = 0

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            request = await ctx.receive(timeout=0.1)

            # Update state
            update_state(state, request)
            messages_since_save += 1

            # Periodic persistence
            if messages_since_save >= 100:
                save_state_to_disk("actor_state.json", state)
                messages_since_save = 0

            await ctx.send(Response(...))

        except asyncio.TimeoutError:
            continue

    # Save on shutdown
    save_state_to_disk("actor_state.json", state)
```

## Resource Management

### Loading Heavy Resources

```python
@ring_kernel(kernel_id="ml_inference")
async def ml_inference(ctx: KernelContext) -> None:
    """ML inference actor with model loading."""

    # Load model once at startup
    print(f"[{ctx.kernel_id}] Loading model...")
    model = load_large_model("model.pt")
    print(f"[{ctx.kernel_id}] Model loaded!")

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            request = await ctx.receive(timeout=0.1)

            # Use pre-loaded model
            prediction = model.predict(request.data)

            await ctx.send(InferenceResponse(
                prediction=prediction,
                correlation_id=request.message_id,
            ))

        except asyncio.TimeoutError:
            continue

    # Cleanup
    print(f"[{ctx.kernel_id}] Unloading model...")
    del model
```

### Buffer Management

```python
from pydotcompute.core.memory_pool import get_memory_pool

@ring_kernel(kernel_id="buffer_processor")
async def buffer_processor(ctx: KernelContext) -> None:
    pool = get_memory_pool()

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            request = await ctx.receive(timeout=0.1)

            # Acquire buffer from pool
            buffer = pool.acquire(request.shape, dtype=np.float32)
            try:
                # Process using buffer
                buffer.copy_from(request.data)
                result = process_on_gpu(buffer.device)

                await ctx.send(ProcessResponse(
                    result=result.tolist(),
                    correlation_id=request.message_id,
                ))
            finally:
                # Always release buffer
                pool.release(buffer)

        except asyncio.TimeoutError:
            continue
```

## Testing Actors

### Unit Testing

```python
import pytest
from pydotcompute import RingKernelRuntime

@pytest.mark.asyncio
async def test_calculator_add():
    async with RingKernelRuntime() as runtime:
        await runtime.launch("calculator", calculator)
        await runtime.activate("calculator")

        await asyncio.sleep(0.1)

        # Send request
        request = CalculateRequest(a=10, b=5, operation="add")
        await runtime.send("calculator", request)

        # Verify response
        response = await runtime.receive("calculator", timeout=1.0)
        assert response.success
        assert response.result == 15.0
        assert response.correlation_id == request.message_id

@pytest.mark.asyncio
async def test_calculator_divide_by_zero():
    async with RingKernelRuntime() as runtime:
        await runtime.launch("calculator", calculator)
        await runtime.activate("calculator")

        await asyncio.sleep(0.1)

        request = CalculateRequest(a=10, b=0, operation="div")
        await runtime.send("calculator", request)

        response = await runtime.receive("calculator", timeout=1.0)
        assert not response.success
        assert response.error_code == "DIVIDE_BY_ZERO"
```

## Performance Tips

1. **Use Appropriate Timeouts**: Short for responsive, longer for efficiency

2. **Batch Processing**: Process multiple messages per iteration

3. **Async Operations**: Don't block the event loop

4. **Memory Pooling**: Reuse buffers for reduced allocation

5. **Profile**: Measure before optimizing

## Anti-Patterns

### Don't Block the Event Loop

```python
# BAD: Blocking operation
while not ctx.should_terminate:
    request = await ctx.receive(timeout=0.1)
    time.sleep(1)  # BLOCKS EVENT LOOP!
    await ctx.send(response)

# GOOD: Use async sleep
while not ctx.should_terminate:
    request = await ctx.receive(timeout=0.1)
    await asyncio.sleep(1)  # Non-blocking
    await ctx.send(response)
```

### Don't Share State

```python
# BAD: Shared mutable state
shared_counter = {"count": 0}

@ring_kernel(kernel_id="worker")
async def worker(ctx):
    while not ctx.should_terminate:
        shared_counter["count"] += 1  # RACE CONDITION!

# GOOD: Private state
@ring_kernel(kernel_id="worker")
async def worker(ctx):
    count = 0  # Private to this actor
    while not ctx.should_terminate:
        count += 1
```

### Don't Ignore Errors

```python
# BAD: Swallowing errors
except Exception:
    pass

# GOOD: Log and continue
except Exception as e:
    logging.exception("Error processing request")
    continue
```

## Next Steps

- [Pipelines](pipelines.md): Multi-stage processing
- [GPU Optimization](gpu-optimization.md): Performance tuning
- [Testing](testing.md): Comprehensive testing
