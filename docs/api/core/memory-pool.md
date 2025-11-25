# MemoryPool

Buffer pooling and reuse for reduced allocation overhead.

## Overview

`MemoryPool` manages a cache of `UnifiedBuffer` instances, reducing allocation overhead for frequently created/destroyed buffers of the same size.

```python
from pydotcompute.core.memory_pool import get_memory_pool

pool = get_memory_pool()

# Acquire a buffer from the pool
buffer = pool.acquire((1024,), dtype=np.float32)

# Use the buffer...
buffer.host[:] = data

# Release back to pool for reuse
pool.release(buffer)
```

## Classes

### MemoryPool

```python
class MemoryPool:
    """Pool of reusable UnifiedBuffer instances."""

    def __init__(
        self,
        max_pool_size: int = 100,
        max_buffer_size: int = 1024 * 1024 * 1024,  # 1 GB
    ) -> None:
        """
        Create a memory pool.

        Args:
            max_pool_size: Maximum number of buffers to cache
            max_buffer_size: Maximum size of individual buffer to pool
        """
```

## Functions

### get_memory_pool

```python
def get_memory_pool() -> MemoryPool:
    """Get the global memory pool instance (singleton)."""
```

## Methods

### acquire

```python
def acquire(
    self,
    shape: tuple[int, ...],
    dtype: np.dtype = np.float32,
    *,
    pinned: bool = False,
) -> UnifiedBuffer:
    """
    Acquire a buffer from the pool.

    Returns a pooled buffer if available, otherwise creates a new one.

    Args:
        shape: Buffer dimensions
        dtype: NumPy data type
        pinned: Use pinned memory

    Returns:
        A UnifiedBuffer ready for use
    """
```

### release

```python
def release(self, buffer: UnifiedBuffer) -> None:
    """
    Release a buffer back to the pool.

    The buffer may be reused by future acquire() calls.

    Args:
        buffer: Buffer to release
    """
```

### clear

```python
def clear(self) -> None:
    """Clear all pooled buffers, freeing memory."""
```

### stats

```python
def stats(self) -> dict[str, Any]:
    """
    Get pool statistics.

    Returns:
        Dictionary with:
        - pool_size: Current number of pooled buffers
        - total_acquired: Total buffers acquired
        - total_released: Total buffers released
        - cache_hits: Times a pooled buffer was reused
        - cache_misses: Times a new buffer was created
    """
```

## Properties

### pool_size

```python
@property
def pool_size(self) -> int:
    """Current number of buffers in the pool."""
```

### total_bytes

```python
@property
def total_bytes(self) -> int:
    """Total bytes of pooled buffers."""
```

## Usage Examples

### Basic Pooling

```python
from pydotcompute.core.memory_pool import get_memory_pool
import numpy as np

pool = get_memory_pool()

# First acquisition - creates new buffer
buf1 = pool.acquire((1000,), dtype=np.float32)
buf1.host[:] = np.random.randn(1000)

# Release back to pool
pool.release(buf1)

# Second acquisition - reuses pooled buffer
buf2 = pool.acquire((1000,), dtype=np.float32)
# buf2 is the same buffer as buf1!
```

### Context Manager Pattern

```python
from contextlib import contextmanager

@contextmanager
def pooled_buffer(shape, dtype=np.float32):
    """Context manager for automatic buffer release."""
    pool = get_memory_pool()
    buffer = pool.acquire(shape, dtype)
    try:
        yield buffer
    finally:
        pool.release(buffer)

# Usage
with pooled_buffer((1024, 1024)) as buf:
    buf.host[:] = compute_data()
    result = process(buf.device)
# Buffer automatically released
```

### Monitoring Pool Usage

```python
pool = get_memory_pool()

# Perform operations...
for _ in range(100):
    buf = pool.acquire((1000,), dtype=np.float32)
    # ... use buffer ...
    pool.release(buf)

# Check statistics
stats = pool.stats()
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Hit rate: {stats['cache_hits'] / stats['total_acquired']:.1%}")
```

### Clearing the Pool

```python
pool = get_memory_pool()

# After a batch of work, clear to free memory
pool.clear()

print(f"Pool size after clear: {pool.pool_size}")  # 0
```

## Performance Considerations

1. **Size Matching**: Buffers are reused only when shape and dtype match exactly

2. **Pool Limits**: Buffers larger than `max_buffer_size` are not pooled

3. **Memory Pressure**: Call `clear()` when memory is tight

4. **Thread Safety**: The pool is thread-safe for concurrent access

## Notes

- The global pool is a singleton - all code shares the same pool
- Buffers are not cleared when released - data from previous use may be present
- Pool statistics are useful for tuning `max_pool_size`
