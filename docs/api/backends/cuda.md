# CUDABackend

NVIDIA GPU backend using Numba and CuPy.

## Overview

`CUDABackend` provides GPU-accelerated computation using NVIDIA CUDA. It leverages Numba for kernel compilation and CuPy for array operations.

```python
from pydotcompute.backends.cuda import CUDABackend

if CUDABackend.check_available():
    backend = CUDABackend()
```

## Requirements

- NVIDIA GPU with compute capability 3.5+
- CUDA Toolkit 11.x or 12.x
- Python packages: `cupy`, `numba`

Install with:

```bash
pip install pydotcompute[cuda]
```

## Class Definition

```python
class CUDABackend(Backend):
    """NVIDIA CUDA compute backend."""

    def __init__(
        self,
        *,
        device_id: int = 0,
        enable_caching: bool = True,
    ) -> None:
        """
        Create a CUDA backend.

        Args:
            device_id: GPU device to use
            enable_caching: Cache compiled kernels

        Raises:
            RuntimeError: If CUDA is not available
        """
```

## Class Methods

### check_available

```python
@classmethod
def check_available(cls) -> bool:
    """
    Check if CUDA backend can be used.

    Returns:
        True if CUDA is available
    """
```

## Properties

### name

```python
@property
def name(self) -> str:
    """Returns 'cuda'."""
```

### is_available

```python
@property
def is_available(self) -> bool:
    """Whether CUDA is available."""
```

### device_id

```python
@property
def device_id(self) -> int:
    """Current GPU device ID."""
```

### device_name

```python
@property
def device_name(self) -> str:
    """Name of the current GPU."""
```

### compute_capability

```python
@property
def compute_capability(self) -> tuple[int, int]:
    """GPU compute capability (major, minor)."""
```

## Methods

### compile_kernel

```python
def compile_kernel(
    self,
    func: Callable,
    signature: tuple[type, ...],
) -> Callable:
    """
    Compile a CUDA kernel using Numba.

    Args:
        func: Python function with CUDA-compatible code
        signature: Argument types

    Returns:
        Compiled CUDA kernel
    """
```

### allocate

```python
def allocate(
    self,
    shape: tuple[int, ...],
    dtype: np.dtype = np.float32,
) -> Any:
    """
    Allocate GPU memory.

    Returns:
        CuPy array
    """
```

### to_device

```python
def to_device(self, data: np.ndarray) -> Any:
    """
    Copy data from CPU to GPU.

    Args:
        data: NumPy array

    Returns:
        CuPy array on GPU
    """
```

### to_host

```python
def to_host(self, data: Any) -> np.ndarray:
    """
    Copy data from GPU to CPU.

    Args:
        data: CuPy array

    Returns:
        NumPy array
    """
```

### synchronize

```python
def synchronize(self) -> None:
    """Wait for all GPU operations to complete."""
```

### get_memory_info

```python
def get_memory_info(self) -> tuple[int, int]:
    """
    Get GPU memory info.

    Returns:
        (free_bytes, total_bytes)
    """
```

### launch_kernel

```python
def launch_kernel(
    self,
    kernel: Callable,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    *args: Any,
) -> None:
    """
    Launch CUDA kernel with explicit configuration.

    Args:
        kernel: Compiled CUDA kernel
        grid: Grid dimensions (blocks per grid)
        block: Block dimensions (threads per block)
        *args: Kernel arguments
    """
```

## Usage Examples

### Basic GPU Computation

```python
from pydotcompute.backends.cuda import CUDABackend
import numpy as np

# Create backend
backend = CUDABackend(device_id=0)

# Allocate GPU arrays
a_gpu = backend.to_device(np.random.randn(1000).astype(np.float32))
b_gpu = backend.to_device(np.random.randn(1000).astype(np.float32))
c_gpu = backend.allocate((1000,), dtype=np.float32)

# Use CuPy for computation
import cupy as cp
c_gpu = a_gpu + b_gpu

# Copy result back
result = backend.to_host(c_gpu)
```

### CUDA Kernel with Numba

```python
from numba import cuda

@cuda.jit
def vector_add(a, b, c):
    i = cuda.grid(1)
    if i < len(c):
        c[i] = a[i] + b[i]

# Compile and launch
backend = CUDABackend()

a_gpu = backend.to_device(a)
b_gpu = backend.to_device(b)
c_gpu = backend.allocate(a.shape)

threads_per_block = 256
blocks = (len(a) + threads_per_block - 1) // threads_per_block

backend.launch_kernel(
    vector_add,
    grid=(blocks,),
    block=(threads_per_block,),
    a_gpu, b_gpu, c_gpu
)

backend.synchronize()
result = backend.to_host(c_gpu)
```

### Memory Management

```python
backend = CUDABackend()

# Check available memory
free, total = backend.get_memory_info()
print(f"GPU Memory: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

# Allocate based on available memory
max_elements = free // 4  # float32 = 4 bytes
arr = backend.allocate((min(max_elements, 1_000_000),), dtype=np.float32)
```

### Multi-GPU

```python
# Use specific GPU
backend_0 = CUDABackend(device_id=0)
backend_1 = CUDABackend(device_id=1)

print(f"GPU 0: {backend_0.device_name}")
print(f"GPU 1: {backend_1.device_name}")

# Allocate on different GPUs
data_gpu0 = backend_0.allocate((1000,))
data_gpu1 = backend_1.allocate((1000,))
```

### Kernel Caching

```python
# Enable caching (default)
backend = CUDABackend(enable_caching=True)

# First call compiles
kernel = backend.compile_kernel(my_func, (np.float32,))

# Subsequent calls use cached version
kernel = backend.compile_kernel(my_func, (np.float32,))  # Fast
```

## Performance Tips

1. **Minimize Transfers**: Keep data on GPU as long as possible

2. **Use Async Operations**: Overlap compute and transfer

3. **Coalesced Access**: Access memory sequentially for best bandwidth

4. **Appropriate Block Size**: 128-256 threads per block is typical

5. **Kernel Caching**: Enable to avoid recompilation

## Compute Capability Requirements

| Feature | Min CC |
|---------|--------|
| Basic operations | 3.5 |
| Shared memory atomics | 6.0 |
| Tensor cores | 7.0 |
| Advanced features | 8.0+ |

## Notes

- Falls back to CPUBackend if CUDA unavailable
- Kernel caching uses disk storage
- Memory is automatically freed when arrays go out of scope
- Use `synchronize()` before timing GPU code
