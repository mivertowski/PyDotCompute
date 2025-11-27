# GPU Optimization

Performance tuning for GPU-accelerated ring kernels.

## Overview

This guide covers techniques to maximize GPU performance with PyDotCompute across CUDA (NVIDIA) and Metal (Apple Silicon) backends.

## Memory Optimization

### Minimize Host-Device Transfers

Transfers are expensive. Keep data on the GPU:

```python
# BAD: Transfer every iteration
for batch in batches:
    gpu_data = cp.asarray(batch)  # Transfer
    result = kernel(gpu_data)
    cpu_result = cp.asnumpy(result)  # Transfer back

# GOOD: Batch transfers
all_data = cp.asarray(np.concatenate(batches))  # One transfer
results = process_all(all_data)  # All on GPU
final = cp.asnumpy(results)  # One transfer back
```

### Use UnifiedBuffer Efficiently

```python
from pydotcompute import UnifiedBuffer

# Minimize state transitions
buf = UnifiedBuffer((10000,), dtype=np.float32)

# Do ALL host work first
buf.host[:5000] = data_a
buf.host[5000:] = data_b

# Then sync once
buf.sync_to_device()

# Do ALL GPU work
result = gpu_kernel(buf.device)

# Sync back once
buf.sync_to_host()
```

### Use Pinned Memory (CUDA)

For streaming workloads on NVIDIA GPUs:

```python
# Enable pinned memory for fast transfers
buf = UnifiedBuffer((10000,), dtype=np.float32, pinned=True)

# DMA transfers are faster with pinned memory
```

!!! note "Metal Advantage"
    On Apple Silicon with Metal, pinned memory is not needed. The unified memory architecture means CPU and GPU share the same physical memory, eliminating transfer overhead entirely.

### Memory Pooling

Reduce allocation overhead:

```python
from pydotcompute.core.memory_pool import get_memory_pool

pool = get_memory_pool()

for batch in batches:
    buf = pool.acquire((batch_size,), dtype=np.float32)
    try:
        process(buf)
    finally:
        pool.release(buf)  # Returns to pool, not freed
```

## Kernel Optimization

### Block Size Selection

Choose appropriate thread block sizes:

```python
from pydotcompute import kernel

# Good block sizes: 128, 256, 512
@kernel(block=(256,))
def my_kernel(data, out):
    i = cuda.grid(1)
    if i < len(data):
        out[i] = process(data[i])
```

**Guidelines:**

| Data Size | Block Size |
|-----------|------------|
| < 1K | 64-128 |
| 1K - 100K | 256 |
| > 100K | 256-512 |

### Memory Coalescing

Access memory sequentially:

```python
# BAD: Strided access
@cuda.jit
def bad_kernel(data, out):
    i = cuda.grid(1)
    # Threads access non-contiguous memory
    out[i] = data[i * stride]

# GOOD: Coalesced access
@cuda.jit
def good_kernel(data, out):
    i = cuda.grid(1)
    # Threads access contiguous memory
    out[i] = data[i]
```

### Shared Memory

Use shared memory for data reuse:

```python
@cuda.jit
def reduce_kernel(data, out):
    # Allocate shared memory
    shared = cuda.shared.array(256, dtype=float32)

    i = cuda.grid(1)
    tid = cuda.threadIdx.x

    # Load to shared memory (coalesced)
    if i < len(data):
        shared[tid] = data[i]
    else:
        shared[tid] = 0

    cuda.syncthreads()

    # Reduction in shared memory (fast)
    s = 128
    while s > 0:
        if tid < s:
            shared[tid] += shared[tid + s]
        cuda.syncthreads()
        s //= 2

    if tid == 0:
        out[cuda.blockIdx.x] = shared[0]
```

### Avoid Warp Divergence

Keep threads in a warp on the same path:

```python
# BAD: Warp divergence
@cuda.jit
def bad_kernel(data, out):
    i = cuda.grid(1)
    if i % 2 == 0:  # Half the warp goes one way
        out[i] = process_a(data[i])
    else:  # Other half goes another
        out[i] = process_b(data[i])

# GOOD: No divergence
@cuda.jit
def good_kernel(data, out):
    i = cuda.grid(1)
    # All threads do same work
    out[i] = process(data[i])
```

## Actor Optimization

### Batch Processing

Process multiple messages per iteration:

```python
@ring_kernel(kernel_id="batch_processor", queue_size=10000)
async def batch_processor(ctx):
    batch_size = 100

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            # Collect batch
            batch = []
            for _ in range(batch_size):
                try:
                    msg = await ctx.receive(timeout=0.01)
                    batch.append(msg)
                except asyncio.TimeoutError:
                    break

            if batch:
                # Process batch on GPU
                results = process_batch_gpu(batch)

                # Send responses
                for msg, result in zip(batch, results):
                    await ctx.send(Response(
                        result=result,
                        correlation_id=msg.message_id,
                    ))

        except:
            continue
```

### Queue Size Tuning

Balance memory vs. latency:

```python
# High throughput, more memory
@ring_kernel(kernel_id="high_throughput", queue_size=10000)
async def high_throughput(ctx):
    ...

# Low latency, less buffering
@ring_kernel(kernel_id="low_latency", queue_size=100)
async def low_latency(ctx):
    ...
```

### Priority Processing

Handle urgent work first:

```python
# High-priority messages processed first
urgent = Request(data=x, priority=255)
normal = Request(data=y, priority=128)

await runtime.send("worker", urgent)  # Processed first
await runtime.send("worker", normal)  # Processed second
```

## Async Streams

Overlap compute and transfer:

```python
import cupy as cp

@ring_kernel(kernel_id="stream_processor")
async def stream_processor(ctx):
    # Create CUDA streams
    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()

    current_buf = None
    next_buf = None

    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue

        try:
            # Receive next work
            msg = await ctx.receive(timeout=0.1)

            # Process current while loading next
            if current_buf is not None:
                with stream1:
                    result = kernel(current_buf)

            # Load next data
            with stream2:
                next_buf = cp.asarray(msg.data)

            # Synchronize
            stream1.synchronize()
            stream2.synchronize()

            if current_buf is not None:
                await ctx.send(Response(result=result))

            current_buf = next_buf

        except asyncio.TimeoutError:
            continue
```

## Profiling

### GPU Profiling

```python
import cupy as cp

# Profile GPU operations
with cp.cuda.profile():
    result = my_kernel(data)

# Or use NVIDIA Nsight
# nsys profile python my_script.py
```

### Actor Telemetry

```python
async with RingKernelRuntime(enable_telemetry=True) as runtime:
    await runtime.launch("worker")
    await runtime.activate("worker")

    # Run workload...

    telemetry = runtime.get_telemetry("worker")
    print(f"Throughput: {telemetry.throughput:.1f} msg/s")
    print(f"Messages: {telemetry.messages_processed}")
```

### Timing

```python
import time
import cupy as cp

# Accurate GPU timing
cp.cuda.Device().synchronize()  # Wait for GPU
start = time.perf_counter()

result = my_kernel(data)

cp.cuda.Device().synchronize()  # Wait for GPU
elapsed = time.perf_counter() - start

print(f"Kernel time: {elapsed*1000:.2f} ms")
```

## Common Bottlenecks

### Problem: Low GPU Utilization

**Symptoms:** GPU usage < 50%

**Solutions:**

1. Increase batch size
2. Use larger queue sizes
3. Reduce host-device transfers
4. Use async streams

### Problem: Memory Transfer Bound

**Symptoms:** High transfer time, low compute time

**Solutions:**

1. Keep data on GPU longer
2. Use pinned memory
3. Batch transfers
4. Overlap compute and transfer

### Problem: Kernel Launch Overhead

**Symptoms:** Many small kernels, high overhead

**Solutions:**

1. Fuse kernels
2. Increase work per kernel
3. Use persistent kernels (ring kernels!)

## Best Practices Summary

| Area | Recommendation |
|------|----------------|
| Transfers | Minimize, batch, use pinned memory (CUDA) |
| Memory | Pool buffers, use UnifiedBuffer |
| Kernels | Coalesced access, avoid divergence (CUDA) |
| Actors | Batch processing, tune queue size |
| Profiling | Measure before optimizing |
| Metal | Leverage unified memory, use MLX operations |

## Next Steps

- [Testing](testing.md): Performance testing
- [Building Actors](building-actors.md): Actor design
