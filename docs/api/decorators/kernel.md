# @kernel

Low-level decorator for GPU kernel functions.

## Overview

The `@kernel` decorator marks a function for GPU compilation. It provides fine-grained control over kernel configuration, suitable for performance-critical code.

```python
from pydotcompute import kernel

@kernel(grid=(256,), block=(128,))
def vector_add(a, b, out):
    i = cuda.grid(1)
    if i < len(out):
        out[i] = a[i] + b[i]
```

## Decorator Signature

```python
def kernel(
    *,
    grid: tuple[int, ...] | None = None,
    block: tuple[int, ...] | None = None,
    shared_memory: int = 0,
    signature: tuple[type, ...] | None = None,
    backend: str = "auto",
    cache: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for GPU kernel functions.

    Args:
        grid: Grid dimensions (blocks per grid)
        block: Block dimensions (threads per block)
        shared_memory: Bytes of shared memory per block
        signature: Explicit type signature
        backend: Backend to use ("auto", "cpu", "cuda")
        cache: Whether to cache compiled kernel

    Returns:
        Decorated kernel function
    """
```

## Parameters

### grid

```python
grid: tuple[int, ...] | None = None
```

Grid dimensions specifying the number of thread blocks.

- 1D: `(blocks_x,)`
- 2D: `(blocks_x, blocks_y)`
- 3D: `(blocks_x, blocks_y, blocks_z)`

### block

```python
block: tuple[int, ...] | None = None
```

Block dimensions specifying threads per block.

- 1D: `(threads_x,)`
- 2D: `(threads_x, threads_y)`
- 3D: `(threads_x, threads_y, threads_z)`

### shared_memory

```python
shared_memory: int = 0
```

Bytes of dynamic shared memory to allocate per block.

### signature

```python
signature: tuple[type, ...] | None = None
```

Explicit type signature for kernel arguments. If not provided, inferred from first call.

### backend

```python
backend: str = "auto"
```

Target backend:

- `"auto"`: Use CUDA if available, else CPU
- `"cuda"`: Force CUDA (error if unavailable)
- `"cpu"`: Force CPU backend

### cache

```python
cache: bool = True
```

Whether to cache the compiled kernel to disk.

## Usage Examples

### Basic Kernel

```python
from pydotcompute import kernel
from numba import cuda

@kernel
def add_one(arr, out):
    i = cuda.grid(1)
    if i < len(arr):
        out[i] = arr[i] + 1

# Call with arrays
add_one[grid, block](input_arr, output_arr)
```

### Fixed Configuration

```python
@kernel(grid=(100,), block=(256,))
def process(data, result):
    i = cuda.grid(1)
    if i < len(data):
        result[i] = data[i] ** 2
```

### Dynamic Grid Calculation

```python
@kernel(block=(256,))
def flexible_kernel(data, out):
    i = cuda.grid(1)
    if i < len(data):
        out[i] = data[i] * 2

# Grid calculated automatically based on data size
n = len(data)
grid = (n + 255) // 256
flexible_kernel[grid, (256,)](data, out)
```

### 2D Kernel

```python
@kernel(block=(16, 16))
def matrix_add(a, b, c):
    i, j = cuda.grid(2)
    if i < c.shape[0] and j < c.shape[1]:
        c[i, j] = a[i, j] + b[i, j]

# Calculate 2D grid
grid_x = (height + 15) // 16
grid_y = (width + 15) // 16
matrix_add[(grid_x, grid_y), (16, 16)](a, b, c)
```

### With Shared Memory

```python
@kernel(block=(256,), shared_memory=256 * 4)  # 256 floats
def reduce_sum(data, out):
    shared = cuda.shared.array(256, dtype=float32)

    i = cuda.grid(1)
    tid = cuda.threadIdx.x

    # Load to shared memory
    if i < len(data):
        shared[tid] = data[i]
    else:
        shared[tid] = 0

    cuda.syncthreads()

    # Reduction in shared memory
    s = 128
    while s > 0:
        if tid < s:
            shared[tid] += shared[tid + s]
        cuda.syncthreads()
        s //= 2

    if tid == 0:
        out[cuda.blockIdx.x] = shared[0]
```

### Explicit Signature

```python
from numba import float32, int32

@kernel(signature=(float32[:], float32[:], int32))
def scale(arr, out, factor):
    i = cuda.grid(1)
    if i < len(arr):
        out[i] = arr[i] * factor
```

### CPU Fallback

```python
@kernel(backend="cpu")
def cpu_kernel(a, b, out):
    for i in range(len(out)):
        out[i] = a[i] + b[i]
```

## Thread Indexing

### 1D Grid

```python
i = cuda.grid(1)  # Global thread index
```

### 2D Grid

```python
i, j = cuda.grid(2)  # Global (row, col) index
```

### 3D Grid

```python
i, j, k = cuda.grid(3)  # Global (x, y, z) index
```

### Manual Calculation

```python
# 1D
i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

# 2D
i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
```

## Best Practices

1. **Bounds Checking**: Always check array bounds in kernels

2. **Block Size**: Use powers of 2 (128, 256, 512)

3. **Warp Size**: CUDA warps are 32 threads

4. **Memory Coalescing**: Sequential threads access sequential memory

5. **Occupancy**: Balance blocks/threads with shared memory

## Notes

- `@kernel` is for compute kernels, not actor functions
- Use `@ring_kernel` for actor-based processing
- Kernels should be pure functions (no side effects)
- First call may be slow due to JIT compilation
