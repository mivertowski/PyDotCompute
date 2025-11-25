# Lifecycle

Kernel state management and lifecycle transitions.

## Overview

Ring kernels follow a well-defined lifecycle with explicit state transitions. This enables proper resource management, graceful shutdown, and pause/resume functionality.

```
CREATED → LAUNCHED → ACTIVE ↔ DEACTIVATED → TERMINATING → TERMINATED
```

## Enums

### KernelState

```python
class KernelState(Enum):
    """Possible states for a ring kernel."""

    CREATED = "created"          # Defined but not launched
    LAUNCHED = "launched"        # Resources allocated, not processing
    ACTIVE = "active"            # Processing messages
    DEACTIVATED = "deactivated"  # Paused, can reactivate
    TERMINATING = "terminating"  # Shutdown in progress
    TERMINATED = "terminated"    # Fully stopped
```

## Classes

### RingKernel

```python
@dataclass
class RingKernel:
    """Represents a ring kernel instance."""

    kernel_id: str
    func: Callable
    input_queue: MessageQueue
    output_queue: MessageQueue
    state: KernelState = KernelState.CREATED
    task: asyncio.Task | None = None
```

### KernelContext

```python
class KernelContext(Generic[TIn, TOut]):
    """
    Context provided to ring kernel functions.

    This is the primary interface actors use to:
    - Receive input messages
    - Send output messages
    - Check lifecycle state
    - Wait for state changes
    """
```

## KernelContext Properties

### kernel_id

```python
@property
def kernel_id(self) -> str:
    """The kernel's unique identifier."""
```

### should_terminate

```python
@property
def should_terminate(self) -> bool:
    """Whether termination has been requested."""
```

### is_active

```python
@property
def is_active(self) -> bool:
    """Whether the kernel is in ACTIVE state."""
```

### state

```python
@property
def state(self) -> KernelState:
    """Current kernel state."""
```

## KernelContext Methods

### receive

```python
async def receive(self, *, timeout: float | None = None) -> TIn:
    """
    Receive a message from the input queue.

    Args:
        timeout: Maximum wait time

    Returns:
        The received message

    Raises:
        asyncio.TimeoutError: If timeout exceeded
    """
```

### send

```python
async def send(self, message: TOut) -> None:
    """
    Send a message to the output queue.

    Args:
        message: Message to send
    """
```

### wait_active

```python
async def wait_active(self) -> None:
    """Wait until the kernel is activated."""
```

### request_termination

```python
def request_termination(self) -> None:
    """Request graceful termination."""
```

## State Transitions

### Valid Transitions

| From | To | Method |
|------|----|--------|
| CREATED | LAUNCHED | `runtime.launch()` |
| LAUNCHED | ACTIVE | `runtime.activate()` |
| ACTIVE | DEACTIVATED | `runtime.deactivate()` |
| DEACTIVATED | ACTIVE | `runtime.reactivate()` |
| ACTIVE | TERMINATING | `runtime.terminate()` |
| DEACTIVATED | TERMINATING | `runtime.terminate()` |
| LAUNCHED | TERMINATING | `runtime.terminate()` |
| TERMINATING | TERMINATED | (automatic) |

### State Diagram

```
                    ┌──────────────────┐
                    │     CREATED      │
                    └────────┬─────────┘
                             │ launch()
                    ┌────────▼─────────┐
                    │     LAUNCHED     │
                    └────────┬─────────┘
                             │ activate()
                    ┌────────▼─────────┐
         ┌─────────►│      ACTIVE      │◄─────────┐
         │          └────────┬─────────┘          │
         │                   │ deactivate()       │
         │          ┌────────▼─────────┐          │
         │          │   DEACTIVATED    │──────────┘
         │          └────────┬─────────┘ reactivate()
         │                   │
         │                   │ terminate()
         │          ┌────────▼─────────┐
         └──────────│   TERMINATING    │
         terminate()└────────┬─────────┘
                             │ (automatic)
                    ┌────────▼─────────┐
                    │    TERMINATED    │
                    └──────────────────┘
```

## Usage Examples

### Standard Actor Loop

```python
@ring_kernel(kernel_id="worker")
async def worker(ctx: KernelContext):
    print(f"[{ctx.kernel_id}] Started")

    while not ctx.should_terminate:
        # Check if we should pause
        if not ctx.is_active:
            print(f"[{ctx.kernel_id}] Waiting for activation...")
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
            # No message - check termination and continue
            continue

    print(f"[{ctx.kernel_id}] Terminated")
```

### Checking State in Application

```python
async with RingKernelRuntime() as runtime:
    await runtime.launch("worker")

    state = runtime.get_state("worker")
    assert state == KernelState.LAUNCHED

    await runtime.activate("worker")

    state = runtime.get_state("worker")
    assert state == KernelState.ACTIVE
```

### Graceful Shutdown

```python
async with RingKernelRuntime() as runtime:
    await runtime.launch("worker")
    await runtime.activate("worker")

    # ... do work ...

    # Graceful termination with timeout
    await runtime.terminate("worker", timeout=5.0)

    state = runtime.get_state("worker")
    assert state == KernelState.TERMINATED
```

### Pause and Resume

```python
async with RingKernelRuntime() as runtime:
    await runtime.launch("worker")
    await runtime.activate("worker")

    # Process some messages
    for msg in batch1:
        await runtime.send("worker", msg)

    # Pause processing
    await runtime.deactivate("worker")
    assert runtime.get_state("worker") == KernelState.DEACTIVATED

    # Messages queue but don't process
    for msg in batch2:
        await runtime.send("worker", msg)

    # Resume
    await runtime.reactivate("worker")
    assert runtime.get_state("worker") == KernelState.ACTIVE

    # Now batch2 messages are processed
```

## Context Manager

### managed_kernel

```python
@asynccontextmanager
async def managed_kernel(
    runtime: RingKernelRuntime,
    kernel_id: str,
    kernel_func: Callable | None = None,
) -> AsyncIterator[str]:
    """
    Context manager for automatic kernel lifecycle.

    Launches, activates, and terminates a kernel automatically.
    """
```

Usage:

```python
async with RingKernelRuntime() as runtime:
    async with managed_kernel(runtime, "worker", worker_func) as kernel_id:
        # Kernel is launched and active
        await runtime.send(kernel_id, message)
        response = await runtime.receive(kernel_id)
    # Kernel automatically terminated
```

## Best Practices

1. **Always Check Termination**: Use `while not ctx.should_terminate`

2. **Use Timeout in Receive**: Allows responsive shutdown

3. **Handle Deactivation**: Check `is_active` and use `wait_active()`

4. **Graceful Shutdown**: Allow time for in-flight messages

5. **Use Context Manager**: For automatic cleanup

## Notes

- Invalid state transitions raise `KernelStateError`
- Terminated kernels cannot be restarted
- The runtime context manager terminates all kernels on exit
- State is tracked per-kernel, not globally
