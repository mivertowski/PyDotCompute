# Memory Management

Managing host and device memory with UnifiedBuffer.

## Overview

PyDotCompute provides `UnifiedBuffer` for seamless host-device memory management. It tracks which copy is current and automatically synchronizes when needed.

## The Memory Challenge

GPU programming traditionally requires explicit memory management:

```python
# Traditional approach (manual)
host_data = np.array([1, 2, 3], dtype=np.float32)
device_data = cuda.to_device(host_data)  # Copy to GPU
kernel(device_data)                       # GPU computation
host_result = device_data.copy_to_host() # Copy back
```

Problems:

- Easy to forget synchronization
- Redundant copies
- Manual state tracking
- Error-prone

## UnifiedBuffer Solution

`UnifiedBuffer` automates memory management:

```python
from pydotcompute import UnifiedBuffer

# Create unified buffer
buf = UnifiedBuffer((1000,), dtype=np.float32)

# Work on host
buf.host[:] = np.random.randn(1000)

# Use on device (auto-syncs)
device_array = buf.device  # Automatically copies to GPU

# Read back (auto-syncs if device modified)
result = buf.to_numpy()
```

## Buffer States

The buffer tracks its state:

```
┌─────────────────┐
│  UNINITIALIZED  │  No data allocated
└────────┬────────┘
         │ first access
┌────────▼────────┐
│    HOST_ONLY    │  Data on host only
└────────┬────────┘
         │ .device access
┌────────▼────────┐
│  SYNCHRONIZED   │  Both copies match
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌────────┐
│HOST    │  │DEVICE  │
│DIRTY   │  │DIRTY   │
└────────┘  └────────┘
```

### State Transitions

| From | Action | To |
|------|--------|-----|
| UNINITIALIZED | `.host` access | HOST_ONLY |
| HOST_ONLY | `.device` access | SYNCHRONIZED |
| SYNCHRONIZED | host modified | HOST_DIRTY |
| SYNCHRONIZED | device modified | DEVICE_DIRTY |
| HOST_DIRTY | `.device` access | SYNCHRONIZED |
| DEVICE_DIRTY | `.host` access | SYNCHRONIZED |

## Lazy Synchronization

Synchronization happens only when needed:

```python
buf = UnifiedBuffer((1000,), dtype=np.float32)

# Write on host
buf.host[:] = data  # State: HOST_DIRTY

# No sync yet - still HOST_DIRTY
print(buf.state)  # HOST_DIRTY

# Access device - triggers sync
device_data = buf.device  # Sync: host → device
print(buf.state)  # SYNCHRONIZED
```

## Explicit Synchronization

For performance-critical code, use explicit sync:

```python
buf = UnifiedBuffer((1000,), dtype=np.float32)

# Prepare data
buf.host[:] = data

# Explicitly sync before kernel launch
buf.sync_to_device()

# Run GPU kernel (modifies device data)
my_kernel(buf.device)

# Mark device as modified
buf.mark_device_dirty()

# Explicitly sync before reading
buf.sync_to_host()

# Now safe to read
result = buf.host[:]
```

## Pinned Memory

For faster transfers, use pinned (page-locked) memory:

```python
# Pinned memory for frequent transfers
buf = UnifiedBuffer((1000,), dtype=np.float32, pinned=True)

# Transfers are faster due to DMA
buf.host[:] = data
device_view = buf.device  # Faster copy
```

**When to use pinned memory:**

- Streaming workloads with frequent transfers
- Real-time processing
- Large batch operations

**When to avoid:**

- Limited system memory
- Many small buffers (overhead)
- Infrequent transfers

## Memory Pooling

Reduce allocation overhead with pooling:

```python
from pydotcompute.core.memory_pool import get_memory_pool

pool = get_memory_pool()

# Acquire from pool (fast if cached)
buf = pool.acquire((1000,), dtype=np.float32)

# Use buffer...
buf.host[:] = data
process(buf.device)

# Release back to pool (not deallocated)
pool.release(buf)

# Next acquire may reuse the buffer
buf2 = pool.acquire((1000,), dtype=np.float32)  # Same buffer!
```

## Large Data Handling

For very large data, avoid message serialization:

```python
@message
@dataclass
class ProcessRequest:
    # Don't include large arrays in messages!
    buffer_id: str  # Reference to shared buffer
    offset: int
    size: int

# Shared buffer registry
buffers: dict[str, UnifiedBuffer] = {}

def create_work_buffer(data: np.ndarray) -> str:
    buf_id = str(uuid4())
    buf = UnifiedBuffer(data.shape, data.dtype)
    buf.copy_from(data)
    buffers[buf_id] = buf
    return buf_id

# Send just the reference
buf_id = create_work_buffer(large_array)
await runtime.send("processor", ProcessRequest(
    buffer_id=buf_id,
    offset=0,
    size=len(large_array),
))
```

## Memory Patterns

### Read-Modify-Write

```python
buf = UnifiedBuffer((1000,), dtype=np.float32)

# Read on host
buf.host[:] = input_data  # HOST_DIRTY

# Modify on device
gpu_kernel(buf.device)    # Syncs, then SYNCHRONIZED
buf.mark_device_dirty()   # DEVICE_DIRTY

# Read on host
result = buf.to_numpy()   # Syncs back
```

### Double Buffering

```python
# Two buffers for overlap
buf_a = UnifiedBuffer((1000,), dtype=np.float32)
buf_b = UnifiedBuffer((1000,), dtype=np.float32)

while data_available():
    # Process buf_a on GPU while filling buf_b on CPU
    buf_a.sync_to_device()
    gpu_task = launch_async(kernel, buf_a.device)

    # Meanwhile, fill buf_b on host
    buf_b.host[:] = get_next_batch()

    # Wait for GPU
    await gpu_task

    # Swap buffers
    buf_a, buf_b = buf_b, buf_a
```

### Batch Processing

```python
pool = get_memory_pool()
results = []

for batch in batches:
    buf = pool.acquire(batch.shape, batch.dtype)
    try:
        buf.copy_from(batch)
        process_on_gpu(buf.device)
        results.append(buf.to_numpy().copy())
    finally:
        pool.release(buf)
```

## Memory Best Practices

1. **Minimize Transfers**: Keep data on GPU as long as possible

2. **Use Pooling**: Reduce allocation overhead

3. **Explicit Sync for Timing**: Use explicit sync for benchmarks

4. **Pinned Memory for Streaming**: Enable for high-throughput

5. **Batch Operations**: Process multiple items per transfer

6. **Check State**: Debug with `buf.state`

7. **Don't Serialize Large Data**: Use buffer references

## GPU Memory Monitoring

```python
from pydotcompute import get_accelerator

acc = get_accelerator()

# Before allocation
free_before, total = acc.get_memory_info()

# Allocate
buf = UnifiedBuffer((10_000_000,), dtype=np.float32)
_ = buf.device  # Force device allocation

# After allocation
free_after, _ = acc.get_memory_info()

print(f"Allocated: {(free_before - free_after) / 1e6:.1f} MB")
```

## Next Steps

- [Lifecycle](lifecycle.md): Kernel state management
- [GPU Optimization Guide](../guides/gpu-optimization.md): Performance tips
