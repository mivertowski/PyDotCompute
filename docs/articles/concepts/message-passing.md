# Message Passing

How ring kernels communicate through typed messages.

## Overview

Ring kernels communicate exclusively through **message passing**. Messages are immutable data objects that flow through priority queues between the application and actors.

```python
@message
@dataclass
class WorkRequest:
    data: list[float]
    operation: str = "sum"

# Send message to actor
await runtime.send("worker", WorkRequest(data=[1, 2, 3]))

# Receive response from actor
response = await runtime.receive("worker")
```

## Why Message Passing?

### Thread Safety

Messages are copied between queues, eliminating shared state:

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Producer   │  copy   │    Queue    │  copy   │   Actor     │
│             │ ──────► │             │ ──────► │             │
│  (owns msg) │         │ (owns copy) │         │ (owns copy) │
└─────────────┘         └─────────────┘         └─────────────┘
```

### Decoupling

Producers and consumers are decoupled:

- Actors don't know who sends messages
- Producers don't know actor internals
- Either side can be replaced independently

### Asynchrony

Message queues enable async processing:

- Producers don't block waiting for actors
- Actors process at their own pace
- Backpressure handled by queue strategies

## Message Structure

### Required Fields

Every message has these fields (added by `@message`):

```python
@message
@dataclass
class MyMessage:
    # Your fields
    data: str

    # Added automatically:
    message_id: UUID      # Unique identifier
    priority: int = 128   # Processing priority (0-255)
    correlation_id: UUID | None = None  # Links responses to requests
```

### Message ID

Every message has a unique ID for:

- Tracking and logging
- Deduplication
- Correlation

```python
msg = WorkRequest(data=[1, 2, 3])
print(msg.message_id)  # UUID('a1b2c3d4-...')
```

### Priority

Priority determines processing order (higher = first):

| Range | Level | Use Case |
|-------|-------|----------|
| 0-63 | Low | Background tasks |
| 64-127 | Below Normal | Batch processing |
| 128 | Normal | Default |
| 129-191 | Above Normal | Important requests |
| 192-255 | High | Urgent/real-time |

```python
# High priority request
urgent = WorkRequest(data=[1, 2, 3], priority=255)

# Low priority background task
background = WorkRequest(data=[1, 2, 3], priority=32)
```

### Correlation ID

Link responses to their requests:

```python
# In actor
request = await ctx.receive()
response = WorkResponse(
    result=compute(request),
    correlation_id=request.message_id,  # Link to request
)
await ctx.send(response)

# In client
request = WorkRequest(data=[1, 2, 3])
await runtime.send("worker", request)

response = await runtime.receive("worker")
assert response.correlation_id == request.message_id
```

## Serialization

Messages are serialized using msgpack for efficient transfer:

```python
msg = WorkRequest(data=[1.0, 2.0, 3.0])

# Serialize (internal)
data = msg.serialize()  # bytes

# Deserialize (internal)
restored = WorkRequest.deserialize(data)
```

### Supported Types

| Type | Serialized As |
|------|---------------|
| `int` | Integer |
| `float` | Float |
| `str` | String |
| `bool` | Boolean |
| `bytes` | Binary |
| `None` | Nil |
| `list` | Array |
| `tuple` | Array |
| `dict` | Map |
| `UUID` | Binary (16 bytes) |
| `datetime` | ISO string |
| Nested `@message` | Map |

### Large Data

For large data, use `UnifiedBuffer` instead of message fields:

```python
# Don't do this for large arrays
@message
@dataclass
class BadRequest:
    huge_array: list[float]  # Slow to serialize!

# Do this instead
@message
@dataclass
class GoodRequest:
    buffer_id: str  # Reference to UnifiedBuffer
    size: int
```

## Request-Response Pattern

The most common pattern:

```python
@message
@dataclass
class CalculateRequest:
    expression: str
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None

@message
@dataclass
class CalculateResponse:
    result: float
    success: bool = True
    error: str | None = None
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None

# Usage
request = CalculateRequest(expression="2 + 2")
await runtime.send("calculator", request)

response = await runtime.receive("calculator", timeout=5.0)
if response.success:
    print(f"Result: {response.result}")
else:
    print(f"Error: {response.error}")
```

## Fire-and-Forget Pattern

When you don't need a response:

```python
@message
@dataclass
class LogEvent:
    level: str
    message: str
    timestamp: float = field(default_factory=time.time)

# Send without waiting for response
await runtime.send("logger", LogEvent(level="INFO", message="Started"))
# Continue immediately
```

## Pub-Sub Pattern

One producer, multiple consumers:

```python
# Publisher sends to topic
await runtime.send("topic_processor", Event(data="update"))

# Topic processor broadcasts to subscribers
@ring_kernel(kernel_id="topic_processor")
async def topic_processor(ctx):
    subscribers = ["sub1", "sub2", "sub3"]

    while not ctx.should_terminate:
        event = await ctx.receive(timeout=0.1)
        for sub in subscribers:
            await forward_to(sub, event)
```

## Pipeline Pattern

Chain of actors:

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Input  │ ──► │ Stage 1 │ ──► │ Stage 2 │ ──► │ Stage 3 │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
```

```python
# Send to first stage
await runtime.send("stage1", RawData(values=[1, 2, 3]))

# Connect stages in actor
@ring_kernel(kernel_id="stage1")
async def stage1(ctx):
    while not ctx.should_terminate:
        raw = await ctx.receive(timeout=0.1)
        processed = preprocess(raw)
        # Forward to next stage
        await runtime.send("stage2", processed)
        await ctx.send(Ack())  # Acknowledge to caller
```

## Error Handling

Include error information in responses:

```python
@message
@dataclass
class Response:
    success: bool = True
    error: str | None = None
    error_code: int | None = None
    result: Any = None

# In actor
try:
    result = risky_operation(request)
    await ctx.send(Response(success=True, result=result))
except ValidationError as e:
    await ctx.send(Response(success=False, error=str(e), error_code=400))
except Exception as e:
    await ctx.send(Response(success=False, error="Internal error", error_code=500))
```

## Best Practices

1. **Keep Messages Small**: Large data should use buffers

2. **Include Error Fields**: Always handle failure cases

3. **Use Correlation IDs**: Essential for request-response

4. **Set Appropriate Priority**: Don't abuse high priority

5. **Design for Immutability**: Messages shouldn't be modified

6. **Version Messages**: Consider backwards compatibility

## Next Steps

- [Memory Management](memory-management.md): Handling large data
- [Lifecycle](lifecycle.md): State transitions
