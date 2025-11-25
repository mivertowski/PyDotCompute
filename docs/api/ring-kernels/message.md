# Message

Message types and serialization for ring kernel communication.

## Overview

Messages are the primary means of communication between your application and ring kernel actors. The `@message` decorator adds serialization, unique IDs, priority, and correlation support.

```python
from pydotcompute import message
from dataclasses import dataclass

@message
@dataclass
class MyRequest:
    data: list[float]
    operation: str = "process"
```

## Decorator

### @message

```python
def message(cls: type[T]) -> type[T]:
    """
    Decorator that enhances a dataclass for ring kernel messaging.

    Adds:
    - message_id: UUID (auto-generated)
    - priority: int (default 128)
    - correlation_id: UUID | None (for request-response matching)
    - Serialization via msgpack
    - Deserialization support

    Args:
        cls: A dataclass to enhance

    Returns:
        Enhanced message class
    """
```

## Base Class

### RingKernelMessage

```python
@dataclass
class RingKernelMessage:
    """Base class for all ring kernel messages."""

    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None
```

## Methods Added by @message

### serialize

```python
def serialize(self) -> bytes:
    """
    Serialize message to bytes using msgpack.

    Returns:
        Serialized message bytes
    """
```

### deserialize (classmethod)

```python
@classmethod
def deserialize(cls, data: bytes) -> Self:
    """
    Deserialize message from bytes.

    Args:
        data: Serialized message bytes

    Returns:
        Deserialized message instance
    """
```

## Fields

### message_id

```python
message_id: UUID = field(default_factory=uuid4)
```

Unique identifier for this message. Auto-generated using UUID4.

### priority

```python
priority: int = 128
```

Message priority (0-255). Higher values = higher priority.

- 0-63: Low priority
- 64-127: Below normal
- 128: Normal (default)
- 129-191: Above normal
- 192-255: High priority

### correlation_id

```python
correlation_id: UUID | None = None
```

Optional ID linking responses to requests. Set this to `request.message_id` in responses.

## Usage Examples

### Basic Message Definition

```python
from pydotcompute import message
from dataclasses import dataclass, field
from uuid import UUID, uuid4

@message
@dataclass
class CalculationRequest:
    a: float = 0.0
    b: float = 0.0
    operation: str = "add"
    # These are added automatically but can be explicit:
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None

@message
@dataclass
class CalculationResponse:
    result: float = 0.0
    success: bool = True
    error: str | None = None
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None
```

### Request-Response Correlation

```python
# Send request
request = CalculationRequest(a=10, b=5, operation="add")
await runtime.send("calculator", request)

# Receive response
response = await runtime.receive("calculator", timeout=1.0)

# Verify correlation
assert response.correlation_id == request.message_id
```

### Priority Messages

```python
# Normal priority (default)
normal_msg = CalculationRequest(a=1, b=2)

# High priority
urgent_msg = CalculationRequest(a=1, b=2, priority=255)

# Low priority background task
background_msg = CalculationRequest(a=1, b=2, priority=32)
```

### Serialization

```python
# Serialize
msg = CalculationRequest(a=1.5, b=2.5, operation="mul")
data = msg.serialize()

# Deserialize
restored = CalculationRequest.deserialize(data)
assert restored.a == 1.5
assert restored.operation == "mul"
```

### In Actor Context

```python
@ring_kernel(kernel_id="calculator")
async def calculator(ctx):
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            request = await ctx.receive(timeout=0.1)

            # Process...
            result = calculate(request.a, request.b, request.operation)

            # Correlate response to request
            response = CalculationResponse(
                result=result,
                correlation_id=request.message_id,  # Link to request
            )

            await ctx.send(response)
        except:
            continue
```

## Supported Types

The `@message` decorator supports these field types:

| Type | Notes |
|------|-------|
| `int`, `float`, `bool`, `str` | Basic types |
| `bytes` | Binary data |
| `list`, `tuple` | Collections |
| `dict` | Maps (string keys recommended) |
| `UUID` | Serialized as bytes |
| `None` | Null values |
| `datetime` | Serialized as ISO string |
| Nested `@message` types | Recursive serialization |

## Best Practices

1. **Always Use Dataclasses**: The `@message` decorator requires `@dataclass`

2. **Include Error Fields**: For responses, include `success` and `error` fields

3. **Use Correlation IDs**: Always set `correlation_id` in responses

4. **Keep Messages Small**: Large data should use `UnifiedBuffer`, not messages

5. **Use Type Hints**: Enable IDE support and validation

6. **Default Values**: Provide defaults for optional fields

## Notes

- Messages are immutable after creation
- Serialization uses msgpack for efficiency
- Large messages may impact queue performance
- UUIDs are serialized as 16-byte binary
