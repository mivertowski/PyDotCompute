# Quick Start

Get PyDotCompute running in 5 minutes with this quick start guide.

## Step 1: Install PyDotCompute

```bash
pip install pydotcompute
```

## Step 2: Create Your First Actor

Create a file called `hello_actor.py`:

```python
import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from pydotcompute import RingKernelRuntime, ring_kernel, message

# Step 2a: Define message types
@message
@dataclass
class GreetRequest:
    name: str = ""
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None

@message
@dataclass
class GreetResponse:
    greeting: str = ""
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None

# Step 2b: Define the actor
@ring_kernel(kernel_id="greeter")
async def greeter_actor(ctx):
    """Actor that generates personalized greetings."""
    print("[greeter] Actor started!")

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            # Wait for a request
            request = await ctx.receive(timeout=0.1)

            # Process the request
            greeting = f"Hello, {request.name}! Welcome to PyDotCompute."

            # Send the response
            response = GreetResponse(
                greeting=greeting,
                correlation_id=request.message_id,
            )
            await ctx.send(response)

        except Exception:
            # Timeout - check for termination and continue
            continue

    print("[greeter] Actor terminated.")

# Step 2c: Use the actor
async def main():
    print("=== PyDotCompute Quick Start ===\n")

    async with RingKernelRuntime() as runtime:
        # Launch the actor
        print("1. Launching actor...")
        await runtime.launch("greeter")

        # Activate the actor
        print("2. Activating actor...")
        await runtime.activate("greeter")

        # Give the actor time to start
        await asyncio.sleep(0.1)

        # Send a request
        print("3. Sending request...")
        await runtime.send("greeter", GreetRequest(name="Developer"))

        # Receive the response
        print("4. Receiving response...")
        response = await runtime.receive("greeter", timeout=1.0)

        print(f"\n>>> {response.greeting}\n")

    print("=== Done! ===")

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 3: Run It

```bash
python hello_actor.py
```

Expected output:

```
=== PyDotCompute Quick Start ===

1. Launching actor...
2. Activating actor...
[greeter] Actor started!
3. Sending request...
4. Receiving response...

>>> Hello, Developer! Welcome to PyDotCompute.

[greeter] Actor terminated.
=== Done! ===
```

## Understanding the Code

### Message Types

Messages are the data that flows between your application and actors:

```python
@message
@dataclass
class GreetRequest:
    name: str = ""
    # These fields are added automatically by @message:
    # message_id: UUID
    # priority: int
    # correlation_id: UUID | None
```

The `@message` decorator:

- Adds serialization support (msgpack)
- Adds unique message IDs
- Enables priority-based processing
- Supports request-response correlation

### Ring Kernel Actor

The actor is a persistent function that processes messages:

```python
@ring_kernel(kernel_id="greeter")
async def greeter_actor(ctx):
    while not ctx.should_terminate:
        request = await ctx.receive()
        # Process...
        await ctx.send(response)
```

Key points:

- Actors run in an infinite loop
- They check `ctx.should_terminate` for graceful shutdown
- `ctx.receive()` gets messages from the input queue
- `ctx.send()` puts messages in the output queue

### Runtime Usage

The runtime manages actor lifecycles:

```python
async with RingKernelRuntime() as runtime:
    await runtime.launch("greeter")    # Phase 1: Allocate resources
    await runtime.activate("greeter")  # Phase 2: Start processing
    # Use the actor...
    # Automatic termination on context exit
```

## Next Steps

Now that you've run your first actor:

1. **[Build a Complete Actor](first-kernel.md)**: Learn about message queues, priority, and error handling
2. **[Understand Ring Kernels](../articles/concepts/ring-kernels.md)**: Deep dive into the architecture
3. **[Explore Examples](https://github.com/mivertowski/PyDotCompute/tree/main/examples)**: See more complex use cases

## Quick Reference

| Concept | Description |
|---------|-------------|
| `@message` | Decorator for message types |
| `@ring_kernel` | Decorator for actor functions |
| `RingKernelRuntime` | Manages actor lifecycles |
| `ctx.receive()` | Get message from input queue |
| `ctx.send()` | Put message in output queue |
| `ctx.should_terminate` | Check if shutdown requested |
