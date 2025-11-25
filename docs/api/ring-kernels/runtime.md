# RingKernelRuntime

The main coordinator for managing ring kernel actors.

## Overview

`RingKernelRuntime` is the central class for creating, managing, and communicating with ring kernel actors. It handles lifecycle management, message routing, and telemetry collection.

```python
from pydotcompute import RingKernelRuntime

async with RingKernelRuntime() as runtime:
    await runtime.launch("my_actor")
    await runtime.activate("my_actor")

    await runtime.send("my_actor", request)
    response = await runtime.receive("my_actor")
```

## Class Definition

```python
class RingKernelRuntime:
    """Runtime coordinator for ring kernel actors."""

    def __init__(
        self,
        *,
        enable_telemetry: bool = False,
        default_queue_size: int = 1000,
        backend: str = "auto",
    ) -> None:
        """
        Create a runtime instance.

        Args:
            enable_telemetry: Enable performance monitoring
            default_queue_size: Default message queue size
            backend: Backend to use ("auto", "cpu", "cuda")
        """
```

## Lifecycle Methods

### launch

```python
async def launch(
    self,
    kernel_id: str,
    kernel_func: Callable | None = None,
    *,
    queue_size: int | None = None,
) -> None:
    """
    Launch a kernel (Phase 1: allocate resources).

    Args:
        kernel_id: Unique identifier for the kernel
        kernel_func: Kernel function (uses registered if None)
        queue_size: Override default queue size

    Raises:
        KernelNotFoundError: If kernel_id not registered and no func provided
        KernelStateError: If kernel already launched
    """
```

### activate

```python
async def activate(self, kernel_id: str) -> None:
    """
    Activate a kernel (Phase 2: start processing).

    Args:
        kernel_id: Kernel to activate

    Raises:
        KernelNotFoundError: If kernel doesn't exist
        KernelStateError: If kernel not in LAUNCHED state
    """
```

### deactivate

```python
async def deactivate(self, kernel_id: str) -> None:
    """
    Deactivate a kernel (pause processing).

    Messages continue to queue but won't be processed.

    Args:
        kernel_id: Kernel to deactivate

    Raises:
        KernelNotFoundError: If kernel doesn't exist
        KernelStateError: If kernel not active
    """
```

### reactivate

```python
async def reactivate(self, kernel_id: str) -> None:
    """
    Reactivate a deactivated kernel.

    Args:
        kernel_id: Kernel to reactivate

    Raises:
        KernelNotFoundError: If kernel doesn't exist
        KernelStateError: If kernel not deactivated
    """
```

### terminate

```python
async def terminate(self, kernel_id: str, *, timeout: float = 5.0) -> None:
    """
    Terminate a kernel (graceful shutdown).

    Args:
        kernel_id: Kernel to terminate
        timeout: Maximum wait time for graceful shutdown

    Raises:
        KernelNotFoundError: If kernel doesn't exist
    """
```

### terminate_all

```python
async def terminate_all(self, *, timeout: float = 10.0) -> None:
    """
    Terminate all kernels.

    Args:
        timeout: Maximum total wait time
    """
```

## Message Methods

### send

```python
async def send(
    self,
    kernel_id: str,
    message: Any,
    *,
    timeout: float | None = None,
) -> None:
    """
    Send a message to a kernel's input queue.

    Args:
        kernel_id: Target kernel
        message: Message to send
        timeout: Maximum wait time if queue is full

    Raises:
        KernelNotFoundError: If kernel doesn't exist
        QueueFullError: If queue full and timeout exceeded
    """
```

### receive

```python
async def receive(
    self,
    kernel_id: str,
    *,
    timeout: float | None = None,
) -> Any:
    """
    Receive a message from a kernel's output queue.

    Args:
        kernel_id: Source kernel
        timeout: Maximum wait time

    Returns:
        The received message

    Raises:
        KernelNotFoundError: If kernel doesn't exist
        asyncio.TimeoutError: If timeout exceeded
    """
```

### send_batch

```python
async def send_batch(
    self,
    kernel_id: str,
    messages: list[Any],
    *,
    timeout: float | None = None,
) -> int:
    """
    Send multiple messages to a kernel.

    Args:
        kernel_id: Target kernel
        messages: Messages to send
        timeout: Timeout per message

    Returns:
        Number of messages successfully sent
    """
```

## Query Methods

### get_state

```python
def get_state(self, kernel_id: str) -> KernelState:
    """
    Get current state of a kernel.

    Args:
        kernel_id: Kernel to query

    Returns:
        Current KernelState

    Raises:
        KernelNotFoundError: If kernel doesn't exist
    """
```

### kernel_ids

```python
@property
def kernel_ids(self) -> list[str]:
    """List of all registered kernel IDs."""
```

### active_kernels

```python
@property
def active_kernels(self) -> list[str]:
    """List of currently active kernel IDs."""
```

## Telemetry Methods

### get_telemetry

```python
def get_telemetry(self, kernel_id: str) -> KernelTelemetry | None:
    """
    Get telemetry for a kernel.

    Args:
        kernel_id: Kernel to query

    Returns:
        KernelTelemetry or None if telemetry disabled
    """
```

### get_summary

```python
def get_summary(self) -> dict[str, Any]:
    """
    Get summary of all kernels.

    Returns:
        Dictionary with runtime statistics
    """
```

## Context Manager

```python
async def __aenter__(self) -> RingKernelRuntime:
    """Enter async context."""

async def __aexit__(self, *args) -> None:
    """Exit async context, terminating all kernels."""
```

## Usage Examples

### Complete Lifecycle

```python
from pydotcompute import RingKernelRuntime, ring_kernel, message

@message
class Request:
    value: int

@message
class Response:
    result: int

@ring_kernel(kernel_id="doubler")
async def doubler(ctx):
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue
        try:
            msg = await ctx.receive(timeout=0.1)
            await ctx.send(Response(result=msg.value * 2))
        except:
            continue

async def main():
    async with RingKernelRuntime(enable_telemetry=True) as runtime:
        # Phase 1: Launch (allocate resources)
        await runtime.launch("doubler")

        # Phase 2: Activate (start processing)
        await runtime.activate("doubler")

        # Wait for startup
        await asyncio.sleep(0.1)

        # Send request
        await runtime.send("doubler", Request(value=21))

        # Receive response
        response = await runtime.receive("doubler", timeout=1.0)
        print(f"Result: {response.result}")  # 42

        # Check telemetry
        telemetry = runtime.get_telemetry("doubler")
        print(f"Messages processed: {telemetry.messages_processed}")

    # Runtime automatically terminates all kernels on exit
```

### Multiple Kernels

```python
async with RingKernelRuntime() as runtime:
    # Launch multiple kernels
    await runtime.launch("processor1", processor_func)
    await runtime.launch("processor2", processor_func)
    await runtime.launch("aggregator", aggregator_func)

    # Activate all
    for kernel_id in runtime.kernel_ids:
        await runtime.activate(kernel_id)

    # Use pipeline...
```

### Pause and Resume

```python
async with RingKernelRuntime() as runtime:
    await runtime.launch("worker")
    await runtime.activate("worker")

    # Process some work...
    for req in batch1:
        await runtime.send("worker", req)

    # Pause for maintenance
    await runtime.deactivate("worker")

    # Messages queue up but aren't processed
    for req in batch2:
        await runtime.send("worker", req)

    # Resume processing
    await runtime.reactivate("worker")

    # Queued messages now processed
```

## Notes

- Always use as async context manager for proper cleanup
- Kernels must be launched before activation
- The runtime is the single point of contact for actors
- Telemetry adds minimal overhead when enabled
