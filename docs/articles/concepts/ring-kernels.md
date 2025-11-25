# Ring Kernels

Understanding the core abstraction in PyDotCompute.

## What is a Ring Kernel?

A **ring kernel** is a persistent compute unit that runs continuously, processing messages in an infinite loop. Unlike traditional GPU kernels that launch, execute, and terminate, ring kernels stay alive and wait for work.

```python
@ring_kernel(kernel_id="processor")
async def processor(ctx):
    while not ctx.should_terminate:  # Infinite loop
        msg = await ctx.receive()     # Wait for work
        result = process(msg)         # Do work
        await ctx.send(result)        # Return result
```

The name "ring" comes from the circular nature of the processing loop and the ring buffer queues used for communication.

## Traditional vs Ring Kernel

### Traditional GPU Kernels

```
Host                         Device
─────────────────────────────────────────
Prepare data
Copy to GPU        ───────►
Launch kernel      ───────►  Execute
Wait for completion ◄───────
Copy from GPU      ◄───────
Process result

(Repeat for next batch)
```

**Problems:**

- Launch overhead on every invocation
- Memory transfer latency
- No persistent state
- Synchronous execution model

### Ring Kernels

```
Host                         Device
─────────────────────────────────────────
Launch once        ───────►  Start loop
                              │
Send message       ───────►   Wait ◄─┐
                              │      │
Receive result     ◄───────   Process │
                              │      │
Send message       ───────►   Wait ◄─┤
                              │      │
Receive result     ◄───────   Process │
                              │      │
                   ...        └──────┘

Terminate         ───────►   Exit loop
```

**Benefits:**

- One-time launch overhead
- Continuous processing
- Persistent state
- Asynchronous message passing

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     RingKernelRuntime                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │   Kernel A   │   │   Kernel B   │   │   Kernel C   │       │
│  ├─────────────┤   ├─────────────┤   ├─────────────┤        │
│  │ Input Queue  │   │ Input Queue  │   │ Input Queue  │       │
│  │ Output Queue │   │ Output Queue │   │ Output Queue │       │
│  │ State Machine│   │ State Machine│   │ State Machine│       │
│  │ Telemetry    │   │ Telemetry    │   │ Telemetry    │       │
│  └─────────────┘   └─────────────┘   └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### The Context Object

Every ring kernel receives a `KernelContext` that provides:

```python
ctx.kernel_id          # Unique identifier
ctx.should_terminate   # Termination flag
ctx.is_active          # Active state flag
ctx.receive()          # Get input message
ctx.send()             # Put output message
ctx.wait_active()      # Wait for activation
```

## Lifecycle

Ring kernels follow a well-defined lifecycle:

```
                    ┌──────────────────┐
                    │     CREATED      │  Definition only
                    └────────┬─────────┘
                             │ launch()
                    ┌────────▼─────────┐
                    │     LAUNCHED     │  Resources allocated
                    └────────┬─────────┘
                             │ activate()
                    ┌────────▼─────────┐
         ┌─────────►│      ACTIVE      │◄─────────┐  Processing
         │          └────────┬─────────┘          │
         │                   │ deactivate()       │
         │          ┌────────▼─────────┐          │
         │          │   DEACTIVATED    │──────────┘  Paused
         │          └────────┬─────────┘ reactivate()
         │                   │
         │                   │ terminate()
         │          ┌────────▼─────────┐
         └──────────│   TERMINATING    │  Shutting down
         terminate()└────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    TERMINATED    │  Stopped
                    └──────────────────┘
```

### Two-Phase Launch

The two-phase launch (launch → activate) enables:

1. **Resource Pre-allocation**: Allocate memory, compile kernels
2. **Coordinated Startup**: Start multiple kernels together
3. **Warm-up**: Pre-warm caches before processing
4. **Testing**: Verify setup before production

```python
# Phase 1: Allocate resources
await runtime.launch("kernel_a")
await runtime.launch("kernel_b")
await runtime.launch("kernel_c")

# Phase 2: Start all together
await runtime.activate("kernel_a")
await runtime.activate("kernel_b")
await runtime.activate("kernel_c")
```

## Message Flow

Messages flow through priority queues:

```
Producer                    Ring Kernel                    Consumer
   │                            │                              │
   │  ┌──────────────────┐      │      ┌──────────────────┐   │
   ├──►│   Input Queue    │──────┼──────►│  Output Queue    │───►
   │  │  (Priority Heap)  │      │      │  (Priority Heap)  │   │
   │  └──────────────────┘      │      └──────────────────┘   │
   │                            │                              │
   │                     ┌──────▼──────┐                       │
   │                     │   Process   │                       │
   │                     └─────────────┘                       │
```

### Priority Processing

Higher priority messages are processed first:

```python
# Priority 255 (highest) processed first
await runtime.send("worker", UrgentRequest(priority=255))

# Priority 128 (normal) processed second
await runtime.send("worker", NormalRequest(priority=128))

# Priority 64 (low) processed last
await runtime.send("worker", BackgroundRequest(priority=64))
```

## Backpressure

When queues fill up, backpressure strategies apply:

| Strategy | Behavior |
|----------|----------|
| `BLOCK` | Wait for space (default) |
| `REJECT` | Raise error immediately |
| `DROP_OLDEST` | Drop oldest message |

```python
@ring_kernel(
    kernel_id="high_volume",
    queue_size=10000,
    backpressure=BackpressureStrategy.DROP_OLDEST,
)
async def high_volume_processor(ctx):
    ...
```

## Use Cases

### Stream Processing

```python
@ring_kernel(kernel_id="stream_processor")
async def stream_processor(ctx):
    while not ctx.should_terminate:
        data_point = await ctx.receive()
        result = transform(data_point)
        await ctx.send(result)
```

### Service Pattern

```python
@ring_kernel(kernel_id="inference_service")
async def inference_service(ctx):
    model = load_model()  # One-time initialization

    while not ctx.should_terminate:
        request = await ctx.receive()
        prediction = model.predict(request.data)
        await ctx.send(InferenceResponse(prediction=prediction))
```

### Pipeline Stage

```python
@ring_kernel(kernel_id="stage_1")
async def stage_1(ctx):
    while not ctx.should_terminate:
        raw = await ctx.receive()
        preprocessed = preprocess(raw)
        await ctx.send(preprocessed)

@ring_kernel(kernel_id="stage_2")
async def stage_2(ctx):
    while not ctx.should_terminate:
        preprocessed = await ctx.receive()
        result = compute(preprocessed)
        await ctx.send(result)
```

## Best Practices

1. **Use Timeouts**: Always use `timeout` in `receive()` for responsive shutdown

2. **Check Termination**: Always loop on `while not ctx.should_terminate`

3. **Handle Deactivation**: Check `is_active` and use `wait_active()`

4. **Error Recovery**: Catch exceptions to prevent actor crash

5. **Correlation IDs**: Link responses to requests

6. **Keep State Private**: Don't share state between kernels

## Next Steps

- [Message Passing](message-passing.md): Deep dive into messaging
- [Memory Management](memory-management.md): Buffer handling
- [Lifecycle](lifecycle.md): State transitions
