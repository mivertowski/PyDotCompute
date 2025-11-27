# GPU Computing Fundamentals

Understanding GPU architecture and why persistent kernels matter.

## GPU Architecture

PyDotCompute supports multiple GPU backends:

- **CUDA**: NVIDIA GPUs (Windows, Linux)
- **Metal**: Apple Silicon GPUs via MLX (macOS)
- **CPU**: Fallback simulation for development/testing

### CPU vs GPU

```
CPU (Few powerful cores)          GPU (Many simple cores)
┌─────────────────────┐          ┌─────────────────────────────────┐
│  ┌─────┐  ┌─────┐   │          │ ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐ │
│  │Core │  │Core │   │          │ └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘ │
│  │  1  │  │  2  │   │          │ ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐ │
│  └─────┘  └─────┘   │          │ └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘ │
│  ┌─────┐  ┌─────┐   │          │ ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐ │
│  │Core │  │Core │   │          │ └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘ │
│  │  3  │  │  4  │   │          │         ... 1000s of cores       │
│  └─────┘  └─────┘   │          │ ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐ │
│                     │          │ └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘ │
└─────────────────────┘          └─────────────────────────────────┘
    Complex tasks                      Parallel tasks
    Low latency                        High throughput
```

### NVIDIA GPU Hierarchy

```
GPU
└── Streaming Multiprocessors (SMs)
    └── Blocks (Thread Blocks)
        └── Warps (32 threads)
            └── Threads
```

| Level | Typical Count | Characteristics |
|-------|---------------|-----------------|
| SMs | 80-100+ | Independent processors |
| Blocks | 1000s | Scheduled to SMs |
| Warps | 32 threads | Execute in lockstep |
| Threads | 100,000s | Lightweight |

## Traditional GPU Programming

### The Typical Flow

```
1. Allocate host memory
2. Initialize data on host
3. Allocate device memory
4. Copy data to device        ← Transfer latency
5. Launch kernel              ← Launch overhead
6. Wait for completion        ← Synchronization
7. Copy results to host       ← Transfer latency
8. Free device memory
```

### Example (CUDA/CuPy)

```python
import cupy as cp
import numpy as np

# Host data
host_data = np.random.randn(1000000).astype(np.float32)

# Copy to device
device_data = cp.asarray(host_data)  # ~0.5ms for 4MB

# Kernel launch
result = cp.square(device_data)       # ~0.01ms

# Copy back
host_result = cp.asnumpy(result)      # ~0.5ms
```

**Observation**: Transfer time dominates computation time!

### The Problem with Small Kernels

```
Traditional approach:
For each batch:
    copy_to_device()     ─── 500μs
    launch_kernel()      ─── 10μs  (actual work)
    copy_from_device()   ─── 500μs

Total: 1010μs per batch
Efficiency: 10/1010 = 1%
```

## Persistent Kernels

### The Innovation

Keep the kernel running and feed it data:

```
Persistent kernel approach:
launch_kernel() once ─── 10μs (one-time)

For each batch:
    send_to_queue()      ─── 1μs
    kernel_processes()   ─── 10μs
    receive_from_queue() ─── 1μs

Total: 12μs per batch (after launch)
Efficiency: 10/12 = 83%
```

### How Ring Kernels Work

```
┌─────────────────────────────────────────────────────────────┐
│                         GPU                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    Ring Kernel                        │  │
│  │                                                       │  │
│  │   ┌─────────┐    ┌───────────┐    ┌─────────┐        │  │
│  │   │  Input  │───►│  Process  │───►│ Output  │        │  │
│  │   │  Queue  │    │  (Loop)   │    │  Queue  │        │  │
│  │   └─────────┘    └───────────┘    └─────────┘        │  │
│  │        ▲                               │              │  │
│  │        │         ┌─────────┐          │              │  │
│  │        │         │  State  │          │              │  │
│  │        │         └─────────┘          │              │  │
│  └────────│───────────────────────────────│──────────────┘  │
│           │                               │                 │
│           │    ┌────────────────────┐     │                 │
│           │    │   Unified Memory   │     │                 │
│           │    │   (Host-Device)    │     │                 │
│           │    └────────────────────┘     │                 │
│           │               ▲               ▼                 │
└───────────│───────────────│───────────────│─────────────────┘
            │               │               │
      ┌─────┴───────────────┴───────────────┴─────┐
      │                   HOST                     │
      │     send()                    receive()    │
      └────────────────────────────────────────────┘
```

## Memory Hierarchy

### GPU Memory Types

```
┌─────────────────────────────────────────────────────────┐
│                       Global Memory                      │
│                    (Large, ~24-80GB, Slow)               │
├─────────────────────────────────────────────────────────┤
│   ┌─────────────────┐     ┌─────────────────┐          │
│   │  Shared Memory  │     │  Shared Memory  │          │
│   │  (Fast, ~48KB)  │     │  (Fast, ~48KB)  │          │
│   │   Per Block     │     │   Per Block     │          │
│   └─────────────────┘     └─────────────────┘          │
│   ┌──┐ ┌──┐ ┌──┐ ┌──┐     ┌──┐ ┌──┐ ┌──┐ ┌──┐          │
│   │R │ │R │ │R │ │R │     │R │ │R │ │R │ │R │          │
│   │e │ │e │ │e │ │e │     │e │ │e │ │e │ │e │          │
│   │g │ │g │ │g │ │g │     │g │ │g │ │g │ │g │          │
│   └──┘ └──┘ └──┘ └──┘     └──┘ └──┘ └──┘ └──┘          │
│    Thread Registers (Fastest, Per-Thread)               │
└─────────────────────────────────────────────────────────┘
```

| Memory Type | Size | Bandwidth | Latency | Scope |
|-------------|------|-----------|---------|-------|
| Registers | ~256KB | ~20TB/s | 1 cycle | Thread |
| Shared | ~48KB | ~10TB/s | ~20 cycles | Block |
| L1 Cache | ~128KB | ~2TB/s | ~30 cycles | SM |
| L2 Cache | ~6MB | ~1TB/s | ~200 cycles | GPU |
| Global | 24-80GB | ~900GB/s | ~400 cycles | All |
| Host | GB-TB | ~25GB/s | ~10K cycles | CPU |

### Unified Memory

PyDotCompute's `UnifiedBuffer` abstracts memory across backends:

=== "CUDA"

    ```python
    from pydotcompute import UnifiedBuffer

    # Single buffer, accessible from both host and device
    buf = UnifiedBuffer((1000,), dtype=np.float32)

    # Host access
    buf.host[:] = data      # Automatic page migration

    # Device access
    result = kernel(buf.device)  # Data migrates to GPU

    # Host access again
    output = buf.host[:]    # Data migrates back
    ```

=== "Metal (macOS)"

    ```python
    from pydotcompute import UnifiedBuffer

    # On Apple Silicon, memory is truly unified
    buf = UnifiedBuffer((1000,), dtype=np.float32)

    # Host access
    buf.host[:] = data

    # Metal access (no physical transfer needed!)
    metal_array = buf.metal  # Returns MLX array

    # CPU and GPU share the same physical memory
    output = buf.host[:]    # Virtually free
    ```

!!! tip "Apple Silicon Advantage"
    Apple Silicon's unified memory architecture means CPU and GPU share the same physical memory. This eliminates the traditional host-device transfer bottleneck, making Metal particularly efficient for streaming workloads.

## Kernel Launch Overhead

### What Happens at Launch

1. **Driver Setup**: ~5-10μs
2. **Command Buffer**: ~2-5μs
3. **Kernel Dispatch**: ~1-2μs
4. **First Thread Start**: ~5-10μs

**Total**: ~15-30μs per launch

### Why Persistent Kernels Help

```
100 small computations:

Traditional:
  100 × (15μs launch + 10μs compute) = 2500μs

Persistent:
  1 × 15μs launch + 100 × 10μs compute = 1015μs

Speedup: 2.5x
```

For streaming workloads, the difference is even larger.

## Streaming Multiprocessors (SMs)

### SM Structure

```
┌─────────────────────────────────────────────────┐
│                      SM                          │
├─────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────┐    │
│  │         Warp Schedulers (4)             │    │
│  └─────────────────────────────────────────┘    │
│                                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │   CUDA    │  │   CUDA    │  │   CUDA    │   │
│  │   Cores   │  │   Cores   │  │   Cores   │   │
│  │   (32)    │  │   (32)    │  │   (32)    │   │
│  └───────────┘  └───────────┘  └───────────┘   │
│                                                  │
│  ┌───────────────────┐  ┌───────────────────┐  │
│  │  Tensor Cores (4) │  │  Register File    │  │
│  └───────────────────┘  │  (256KB)          │  │
│                          └───────────────────┘  │
│  ┌───────────────────────────────────────────┐ │
│  │          Shared Memory (96KB)             │ │
│  └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### Occupancy

**Occupancy** = Active Warps / Maximum Warps per SM

Higher occupancy hides latency:

```python
# Higher occupancy (more concurrent warps)
@kernel(block=(256,))  # 256 threads = 8 warps
def high_occupancy_kernel(...):
    ...

# Lower occupancy (fewer warps, more resources each)
@kernel(block=(64,))   # 64 threads = 2 warps
def low_occupancy_kernel(...):
    # More registers/shared memory per thread
    ...
```

## PyDotCompute's Approach

PyDotCompute addresses these GPU challenges:

| Challenge | Traditional | PyDotCompute |
|-----------|-------------|--------------|
| Launch overhead | Every call | Once |
| Memory transfer | Every call | Minimized |
| State management | Manual | Automatic |
| Synchronization | Explicit | Message-based |
| Memory tracking | Manual | UnifiedBuffer |
| Backend portability | Vendor-specific | Multi-backend (CUDA, Metal, CPU) |

## Next Steps

- [DotCompute Comparison](dotcompute-comparison.md): Origin story
- [Ring Kernels](../concepts/ring-kernels.md): Implementation
