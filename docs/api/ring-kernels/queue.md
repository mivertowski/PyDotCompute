# MessageQueue

Priority-based async message queues for ring kernels.

## Overview

`MessageQueue` provides async-safe message queuing with priority ordering. Messages with higher priority values are dequeued first.

```python
from pydotcompute.ring_kernels.queue import MessageQueue

queue = MessageQueue(max_size=1000)

# Add message
await queue.put(message)

# Get message (highest priority first)
msg = await queue.get(timeout=1.0)
```

## Enums

### BackpressureStrategy

```python
class BackpressureStrategy(Enum):
    """How to handle queue full conditions."""

    BLOCK = "block"        # Wait for space (default)
    REJECT = "reject"      # Raise QueueFullError immediately
    DROP_OLDEST = "drop"   # Drop oldest message to make room
```

## Classes

### MessageQueue

```python
class MessageQueue(Generic[T]):
    """Priority-based async message queue."""

    def __init__(
        self,
        max_size: int = 1000,
        backpressure: BackpressureStrategy = BackpressureStrategy.BLOCK,
    ) -> None:
        """
        Create a message queue.

        Args:
            max_size: Maximum number of messages
            backpressure: Strategy when queue is full
        """
```

## Methods

### put

```python
async def put(
    self,
    message: T,
    *,
    timeout: float | None = None,
) -> None:
    """
    Add a message to the queue.

    Args:
        message: Message to add
        timeout: Maximum wait time (for BLOCK strategy)

    Raises:
        QueueFullError: If queue full and strategy is REJECT or timeout exceeded
    """
```

### get

```python
async def get(
    self,
    *,
    timeout: float | None = None,
) -> T:
    """
    Get the highest priority message.

    Args:
        timeout: Maximum wait time

    Returns:
        The highest priority message

    Raises:
        asyncio.TimeoutError: If timeout exceeded and no message available
    """
```

### get_nowait

```python
def get_nowait(self) -> T | None:
    """
    Get message without waiting.

    Returns:
        Message or None if queue empty
    """
```

### peek

```python
def peek(self) -> T | None:
    """
    View highest priority message without removing it.

    Returns:
        Message or None if queue empty
    """
```

### clear

```python
def clear(self) -> int:
    """
    Remove all messages from the queue.

    Returns:
        Number of messages cleared
    """
```

## Properties

### size

```python
@property
def size(self) -> int:
    """Current number of messages in queue."""
```

### max_size

```python
@property
def max_size(self) -> int:
    """Maximum queue capacity."""
```

### is_empty

```python
@property
def is_empty(self) -> bool:
    """Whether the queue is empty."""
```

### is_full

```python
@property
def is_full(self) -> bool:
    """Whether the queue is at capacity."""
```

## Usage Examples

### Basic Queue Operations

```python
from pydotcompute.ring_kernels.queue import MessageQueue

async def example():
    queue = MessageQueue(max_size=100)

    # Add messages
    await queue.put(Message(data="first", priority=128))
    await queue.put(Message(data="urgent", priority=255))
    await queue.put(Message(data="background", priority=64))

    # Get messages (priority order)
    msg1 = await queue.get()  # urgent (255)
    msg2 = await queue.get()  # first (128)
    msg3 = await queue.get()  # background (64)
```

### Backpressure Strategies

```python
from pydotcompute.ring_kernels.queue import MessageQueue, BackpressureStrategy

# Block until space available (default)
blocking_queue = MessageQueue(
    max_size=10,
    backpressure=BackpressureStrategy.BLOCK,
)

# Reject immediately when full
rejecting_queue = MessageQueue(
    max_size=10,
    backpressure=BackpressureStrategy.REJECT,
)

# Drop oldest to make room
dropping_queue = MessageQueue(
    max_size=10,
    backpressure=BackpressureStrategy.DROP_OLDEST,
)
```

### Handling Full Queue

```python
from pydotcompute.exceptions import QueueFullError

queue = MessageQueue(max_size=10, backpressure=BackpressureStrategy.REJECT)

# Fill the queue
for i in range(10):
    await queue.put(Message(data=i))

# This will raise QueueFullError
try:
    await queue.put(Message(data="overflow"))
except QueueFullError:
    print("Queue is full!")
```

### Timeout Handling

```python
import asyncio

queue = MessageQueue(max_size=100)

# Try to get with timeout
try:
    msg = await queue.get(timeout=1.0)
except asyncio.TimeoutError:
    print("No message available within timeout")

# Put with timeout (for BLOCK strategy)
try:
    await queue.put(message, timeout=0.5)
except QueueFullError:
    print("Could not add message within timeout")
```

### Non-blocking Operations

```python
queue = MessageQueue(max_size=100)

# Non-blocking get
msg = queue.get_nowait()
if msg is None:
    print("Queue is empty")

# Peek without removing
msg = queue.peek()
if msg is not None:
    print(f"Next message: {msg.data}")
```

### Queue Monitoring

```python
queue = MessageQueue(max_size=1000)

# Add some messages
for i in range(500):
    await queue.put(Message(data=i))

print(f"Queue size: {queue.size}")
print(f"Queue capacity: {queue.max_size}")
print(f"Is empty: {queue.is_empty}")
print(f"Is full: {queue.is_full}")
print(f"Utilization: {queue.size / queue.max_size:.1%}")
```

## Priority Ordering

Messages are ordered by priority (higher = first) and then by insertion order (FIFO within same priority):

```python
queue = MessageQueue(max_size=100)

await queue.put(Message(data="A", priority=100))  # 1st in
await queue.put(Message(data="B", priority=200))  # 2nd in
await queue.put(Message(data="C", priority=100))  # 3rd in
await queue.put(Message(data="D", priority=150))  # 4th in

# Dequeue order: B (200), D (150), A (100), C (100)
# A comes before C because they have same priority and A was inserted first
```

## Thread Safety

- All async methods are safe for concurrent use
- Use `asyncio.Lock` if you need atomic sequences of operations
- The queue uses internal locking for consistency

## Notes

- Priority range is 0-255 (higher = more urgent)
- Messages must have a `priority` attribute (added by `@message`)
- Queue size is bounded to prevent memory exhaustion
- Dropped messages (DROP_OLDEST) are lost permanently
