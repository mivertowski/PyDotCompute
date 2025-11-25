# Lifecycle Management

Understanding kernel states and transitions.

## Overview

Ring kernels follow a structured lifecycle that enables proper resource management, graceful shutdown, and operational control.

## States

### CREATED

The kernel is defined but not yet launched.

```python
@ring_kernel(kernel_id="worker")
async def worker(ctx):
    ...

# Kernel is CREATED but not in runtime
```

### LAUNCHED

Resources are allocated, but processing hasn't started.

```python
await runtime.launch("worker")
# State: LAUNCHED
# - Input/output queues created
# - asyncio task not yet running
```

### ACTIVE

The kernel is running and processing messages.

```python
await runtime.activate("worker")
# State: ACTIVE
# - asyncio task running
# - ctx.is_active == True
# - Processing messages
```

### DEACTIVATED

Processing is paused, but resources remain allocated.

```python
await runtime.deactivate("worker")
# State: DEACTIVATED
# - asyncio task paused
# - ctx.is_active == False
# - Messages still queue
```

### TERMINATING

Shutdown has been requested, waiting for graceful exit.

```python
await runtime.terminate("worker")
# State: TERMINATING
# - ctx.should_terminate == True
# - Waiting for actor loop to exit
```

### TERMINATED

The kernel has stopped and resources are released.

```python
# After terminate completes
# State: TERMINATED
# - asyncio task done
# - Queues cleared
# - Cannot restart
```

## State Diagram

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
         │ terminate()       │ terminate()
         │          ┌────────▼─────────┐
         └──────────┤   TERMINATING    │◄─── (from any running state)
                    └────────┬─────────┘
                             │ (graceful exit)
                    ┌────────▼─────────┐
                    │    TERMINATED    │
                    └──────────────────┘
```

## Two-Phase Launch

The launch → activate pattern provides control over startup:

### Phase 1: Launch

```python
# Allocate resources for all kernels
await runtime.launch("kernel_a")
await runtime.launch("kernel_b")
await runtime.launch("kernel_c")

# At this point:
# - All queues created
# - All tasks created (but waiting)
# - Memory allocated
# - Kernels compiled (if needed)
```

### Phase 2: Activate

```python
# Start processing together
await runtime.activate("kernel_a")
await runtime.activate("kernel_b")
await runtime.activate("kernel_c")

# Now all kernels are processing
```

### Benefits

1. **Coordinated Startup**: Start multiple kernels simultaneously
2. **Validation**: Verify all kernels can launch before activating any
3. **Warm-up**: Pre-allocate resources, warm caches
4. **Dependency Ordering**: Launch dependencies before dependents

## Actor Loop Pattern

The standard actor loop handles all lifecycle states:

```python
@ring_kernel(kernel_id="worker")
async def worker(ctx):
    # Optional: One-time initialization
    print(f"[{ctx.kernel_id}] Initializing...")

    # Main processing loop
    while not ctx.should_terminate:
        # Handle deactivation
        if not ctx.is_active:
            print(f"[{ctx.kernel_id}] Paused, waiting...")
            await ctx.wait_active()
            print(f"[{ctx.kernel_id}] Resumed!")
            continue

        try:
            # Process messages with timeout
            msg = await ctx.receive(timeout=0.1)
            result = process(msg)
            await ctx.send(result)

        except asyncio.TimeoutError:
            # No message - loop back to check termination
            continue

    # Optional: Cleanup
    print(f"[{ctx.kernel_id}] Cleaning up...")
```

### Key Points

1. **`while not ctx.should_terminate`**: Check for shutdown request

2. **`if not ctx.is_active`**: Handle pause/resume

3. **`await ctx.wait_active()`**: Block until reactivated

4. **`timeout=0.1`**: Allows responsive shutdown

5. **`except asyncio.TimeoutError`**: Timeouts are normal, not errors

## Pause and Resume

Deactivation pauses processing without losing state:

```python
async with RingKernelRuntime() as runtime:
    await runtime.launch("worker")
    await runtime.activate("worker")

    # Process some messages
    for msg in batch_1:
        await runtime.send("worker", msg)
    for _ in range(len(batch_1)):
        await runtime.receive("worker")

    # Pause for maintenance
    await runtime.deactivate("worker")

    # Messages queue up but aren't processed
    for msg in batch_2:
        await runtime.send("worker", msg)

    # Resume processing
    await runtime.reactivate("worker")

    # Queued messages now processed
    for _ in range(len(batch_2)):
        await runtime.receive("worker")
```

### Use Cases for Deactivation

- **Maintenance**: Apply configuration changes
- **Load Balancing**: Temporarily reduce processing
- **Debugging**: Pause to inspect state
- **Batching**: Accumulate messages, then process

## Graceful Shutdown

Termination allows in-flight work to complete:

```python
# Request termination with timeout
await runtime.terminate("worker", timeout=5.0)

# What happens:
# 1. ctx.should_terminate becomes True
# 2. Actor loop exits on next check
# 3. Runtime waits up to 5 seconds
# 4. If timeout, force stops
# 5. State becomes TERMINATED
```

### Terminating All Kernels

```python
# Terminate all at once
await runtime.terminate_all(timeout=10.0)
```

### Using Context Manager

The recommended approach for automatic cleanup:

```python
async with RingKernelRuntime() as runtime:
    await runtime.launch("worker")
    await runtime.activate("worker")

    # Use kernel...

# Automatically terminates all on exit
```

## Error Handling

### Invalid State Transitions

```python
from pydotcompute.exceptions import KernelStateError

try:
    await runtime.activate("worker")  # Not launched!
except KernelStateError as e:
    print(f"State: {e.current_state}, need: {e.expected_states}")
```

### Valid Transitions

| From | Action | To | Valid |
|------|--------|-----|-------|
| CREATED | launch() | LAUNCHED | ✓ |
| CREATED | activate() | - | ✗ |
| LAUNCHED | activate() | ACTIVE | ✓ |
| LAUNCHED | deactivate() | - | ✗ |
| ACTIVE | deactivate() | DEACTIVATED | ✓ |
| ACTIVE | launch() | - | ✗ |
| DEACTIVATED | reactivate() | ACTIVE | ✓ |
| DEACTIVATED | activate() | - | ✗ |
| TERMINATED | any | - | ✗ |

## Monitoring State

```python
# Check current state
state = runtime.get_state("worker")
print(f"State: {state.name}")

# List all kernels
for kernel_id in runtime.kernel_ids:
    state = runtime.get_state(kernel_id)
    print(f"{kernel_id}: {state.name}")

# List only active
for kernel_id in runtime.active_kernels:
    print(f"Active: {kernel_id}")
```

## Best Practices

1. **Always Use Context Manager**: Ensures cleanup

2. **Use Two-Phase Launch**: Better control and validation

3. **Implement Graceful Shutdown**: Check `should_terminate`

4. **Handle Deactivation**: Support pause/resume

5. **Set Appropriate Timeouts**: For responsive shutdown

6. **Log State Changes**: For debugging

## Next Steps

- [Building Actors Guide](../guides/building-actors.md): Best practices
- [Testing Guide](../guides/testing.md): Testing lifecycle
