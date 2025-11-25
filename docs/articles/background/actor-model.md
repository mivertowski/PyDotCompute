# The Actor Model

Understanding the theoretical foundation of ring kernels.

## What is the Actor Model?

The **Actor Model** is a mathematical model for concurrent computation introduced by Carl Hewitt in 1973. It treats "actors" as the fundamental unit of computation.

### Core Principles

1. **Everything is an Actor**: Actors are the basic building blocks
2. **Actors are Isolated**: No shared state between actors
3. **Communication via Messages**: Actors interact only through messages
4. **Async Processing**: Messages are processed asynchronously

## Actor Properties

Each actor has:

```
┌─────────────────────────────────────┐
│              ACTOR                  │
├─────────────────────────────────────┤
│  Mailbox (Queue)                    │ ◄── Messages arrive here
├─────────────────────────────────────┤
│  Behavior (Logic)                   │     Process messages
├─────────────────────────────────────┤
│  State (Private)                    │     Internal state
└─────────────────────────────────────┘
```

When an actor receives a message, it can:

1. **Send messages** to other actors
2. **Create new actors**
3. **Change its behavior** for the next message

## Why Actors for GPU Computing?

### Traditional GPU Model

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Launch    │ ───► │   Execute   │ ───► │  Complete   │
│   Kernel    │      │             │      │   Return    │
└─────────────┘      └─────────────┘      └─────────────┘
     │                                           │
     └───────────── Repeat for each call ────────┘
```

Problems:

- Launch overhead every call
- No persistent state
- Synchronous blocking

### Actor GPU Model

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU ACTOR (Persistent)                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │     ┌─────────────┐                                   │  │
│  │     │   Receive   │ ◄─── Messages from host           │  │
│  │     └──────┬──────┘                                   │  │
│  │            │                                          │  │
│  │     ┌──────▼──────┐                                   │  │
│  │     │   Process   │      Uses persistent state        │  │
│  │     └──────┬──────┘                                   │  │
│  │            │                                          │  │
│  │     ┌──────▼──────┐                                   │  │
│  │     │    Send     │ ───► Results to host              │  │
│  │     └──────┬──────┘                                   │  │
│  │            │                                          │  │
│  │            └──────────────────────────────────────┐   │  │
│  │                        Loop                       │   │  │
│  └───────────────────────────────────────────────────┘   │  │
└─────────────────────────────────────────────────────────────┘
```

Benefits:

- One-time launch overhead
- Persistent state (models, caches)
- Asynchronous message processing
- Natural fit for streaming

## Actor Model in PyDotCompute

### Ring Kernel as Actor

```python
@ring_kernel(kernel_id="processor")
async def processor(ctx):
    # State: Private to this actor
    model = load_model()
    cache = {}

    # Behavior: Message processing loop
    while not ctx.should_terminate:
        # Mailbox: Receive messages
        msg = await ctx.receive()

        # Process and respond
        result = model.predict(msg.data)
        await ctx.send(Response(result=result))
```

### Message Passing

```python
# Producer sends message (fire-and-forget)
await runtime.send("processor", Request(data=x))

# Consumer receives response (async)
response = await runtime.receive("processor")
```

### Isolation

```python
# Each actor has private state
@ring_kernel(kernel_id="counter_a")
async def counter_a(ctx):
    count = 0  # Private to counter_a
    ...

@ring_kernel(kernel_id="counter_b")
async def counter_b(ctx):
    count = 0  # Private to counter_b, independent of counter_a
    ...
```

## Comparison with Other Models

### Threads

| Aspect | Threads | Actors |
|--------|---------|--------|
| Communication | Shared memory | Messages |
| Synchronization | Locks, mutexes | Message ordering |
| State | Shared | Private |
| Deadlocks | Possible | Avoided by design |

### CSP (Go channels)

| Aspect | CSP | Actors |
|--------|-----|--------|
| Identity | Anonymous | Named |
| Channels | Shared | Private mailbox |
| Blocking | Synchronous | Asynchronous |

### Traditional GPU

| Aspect | Traditional | Ring Kernel |
|--------|-------------|-------------|
| Lifetime | Per-call | Persistent |
| State | None | Persistent |
| Communication | Memory copy | Messages |
| Latency | High (launch) | Low (running) |

## Benefits of Actor Model

### 1. Concurrency Safety

No shared mutable state means no race conditions:

```python
# No locks needed!
@ring_kernel(kernel_id="safe_counter")
async def safe_counter(ctx):
    count = 0  # Only this actor touches this

    while not ctx.should_terminate:
        msg = await ctx.receive()
        if msg.action == "increment":
            count += 1  # No race condition possible
        await ctx.send(CountResponse(count=count))
```

### 2. Scalability

Add more actors for more parallelism:

```python
# Scale horizontally
for i in range(num_workers):
    await runtime.launch(f"worker_{i}", worker_fn)
    await runtime.activate(f"worker_{i}")
```

### 3. Fault Isolation

Actor crashes don't affect others:

```python
# worker_a crashes, but worker_b continues
@ring_kernel(kernel_id="worker_a")
async def worker_a(ctx):
    raise Exception("Crash!")  # Only affects worker_a

@ring_kernel(kernel_id="worker_b")
async def worker_b(ctx):
    # Still running fine
    ...
```

### 4. Location Transparency

Actors can run anywhere:

```python
# Same code works locally or distributed
await runtime.send("worker", message)  # Local
await runtime.send("remote_worker", message)  # Could be remote
```

## Design Patterns

### Request-Response

```
Client ──Request──► Actor
Client ◄─Response── Actor
```

### Pipeline

```
Actor A ──► Actor B ──► Actor C
```

### Fan-Out / Fan-In

```
           ┌──► Worker 1 ──┐
Distributor├──► Worker 2 ──┼──► Aggregator
           └──► Worker 3 ──┘
```

### Supervision

```
Supervisor
    │
    ├── Worker 1 (restarts on crash)
    ├── Worker 2
    └── Worker 3
```

## Further Reading

- [Hewitt, C. "Actor Model of Computation" (2010)](https://arxiv.org/abs/1008.1459)
- [Agha, G. "Actors: A Model of Concurrent Computation" (1986)](https://dl.acm.org/doi/book/10.5555/7929)
- [Armstrong, J. "Programming Erlang" (2007)](https://pragprog.com/titles/jaerlang2/programming-erlang-2nd-edition/)

## Next Steps

- [GPU Computing Background](gpu-computing.md): GPU architecture
- [Ring Kernels Concept](../concepts/ring-kernels.md): Implementation details
