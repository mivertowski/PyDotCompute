# UnifiedBuffer

Host-device memory abstraction with lazy synchronization.

## Overview

`UnifiedBuffer` provides a unified view of memory that can exist on both host (CPU) and device (GPU). It tracks which copy is current and automatically synchronizes when needed.

```python
from pydotcompute import UnifiedBuffer
import numpy as np

# Create a buffer
buffer = UnifiedBuffer(shape=(1024,), dtype=np.float32)

# Work on host
buffer.host[:] = np.random.randn(1024)

# Access on device (auto-syncs)
device_data = buffer.device
```

## Classes

### BufferState

```python
class BufferState(Enum):
    """Tracks which copy of data is authoritative."""
    UNINITIALIZED = "uninitialized"  # No data yet
    HOST_ONLY = "host_only"          # Only host has data
    DEVICE_ONLY = "device_only"      # Only device has data
    SYNCHRONIZED = "synchronized"     # Both copies match
    HOST_DIRTY = "host_dirty"        # Host modified, device stale
    DEVICE_DIRTY = "device_dirty"    # Device modified, host stale
```

### UnifiedBuffer

```python
class UnifiedBuffer(Generic[T]):
    """Unified host-device memory buffer with lazy synchronization."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype = np.float32,
        *,
        pinned: bool = False,
    ) -> None:
        """
        Create a unified buffer.

        Args:
            shape: Buffer dimensions
            dtype: NumPy data type
            pinned: Use pinned (page-locked) host memory for faster transfers
        """
```

## Properties

### host

```python
@property
def host(self) -> np.ndarray:
    """
    Get host (CPU) view of data.

    Automatically syncs from device if device is dirty.
    Marks buffer as HOST_DIRTY after modification.
    """
```

### device

```python
@property
def device(self) -> Any:
    """
    Get device (GPU) view of data.

    Automatically syncs from host if host is dirty.
    Returns CuPy array if CUDA available, else NumPy array.
    """
```

### shape

```python
@property
def shape(self) -> tuple[int, ...]:
    """Buffer dimensions."""
```

### dtype

```python
@property
def dtype(self) -> np.dtype:
    """Data type."""
```

### nbytes

```python
@property
def nbytes(self) -> int:
    """Total size in bytes."""
```

### state

```python
@property
def state(self) -> BufferState:
    """Current synchronization state."""
```

## Methods

### sync_to_device

```python
def sync_to_device(self) -> None:
    """Explicitly sync host data to device."""
```

### sync_to_host

```python
def sync_to_host(self) -> None:
    """Explicitly sync device data to host."""
```

### mark_host_dirty

```python
def mark_host_dirty(self) -> None:
    """Mark host data as modified (device copy is stale)."""
```

### mark_device_dirty

```python
def mark_device_dirty(self) -> None:
    """Mark device data as modified (host copy is stale)."""
```

### fill

```python
def fill(self, value: T) -> None:
    """Fill buffer with a value on the current device."""
```

### copy_from

```python
def copy_from(self, data: np.ndarray) -> None:
    """Copy data from a NumPy array to the buffer."""
```

### to_numpy

```python
def to_numpy(self) -> np.ndarray:
    """Get a copy of the data as a NumPy array."""
```

## Usage Examples

### Basic Usage

```python
from pydotcompute import UnifiedBuffer
import numpy as np

# Create buffer
buf = UnifiedBuffer((1000,), dtype=np.float32)

# Initialize on host
buf.host[:] = np.arange(1000, dtype=np.float32)

# Use on device (automatically syncs)
device_array = buf.device
# ... GPU computation ...

# Read back (automatically syncs if device modified)
result = buf.to_numpy()
```

### Pinned Memory for Faster Transfers

```python
# Use pinned memory for frequent host-device transfers
buf = UnifiedBuffer((1024, 1024), dtype=np.float32, pinned=True)
```

### Explicit Synchronization

```python
buf = UnifiedBuffer((1000,), dtype=np.float32)

# Modify on host
buf.host[:] = np.random.randn(1000)

# Explicitly sync before kernel launch
buf.sync_to_device()

# After GPU kernel modifies device data
buf.mark_device_dirty()

# Explicitly sync back
buf.sync_to_host()
```

### State Tracking

```python
buf = UnifiedBuffer((100,), dtype=np.float32)

print(buf.state)  # UNINITIALIZED

buf.host[:] = 1.0
print(buf.state)  # HOST_DIRTY

buf.sync_to_device()
print(buf.state)  # SYNCHRONIZED
```

## Performance Tips

1. **Minimize Transfers**: Access `host` or `device` properties sparingly - each access may trigger a sync

2. **Batch Operations**: Do all host work, then sync once, then do all device work

3. **Use Pinned Memory**: For buffers that transfer frequently, use `pinned=True`

4. **Explicit Sync**: For performance-critical code, use explicit `sync_to_device()` / `sync_to_host()` instead of relying on lazy sync

5. **Check State**: Use `state` property to understand when syncs will occur

## Notes

- On systems without CUDA, `device` returns the same NumPy array as `host`
- Pinned memory requires CUDA and may be limited by system resources
- Large buffers should use explicit sync to avoid unexpected latency
