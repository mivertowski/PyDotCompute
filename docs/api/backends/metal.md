# MetalBackend

Apple Silicon GPU backend using MLX for macOS.

## Overview

`MetalBackend` provides GPU-accelerated computation on macOS using Apple's Metal framework via the MLX library. It leverages Apple Silicon's unified memory architecture for efficient data handling.

```python
from pydotcompute.backends.metal import MetalBackend

backend = MetalBackend()
if backend.is_available:
    # GPU acceleration on Apple Silicon
    data = backend.copy_to_device(np.array([1, 2, 3], dtype=np.float32))
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Python package: `mlx>=0.4.0`

Install with:

```bash
pip install pydotcompute[metal]
```

## Class Definition

```python
class MetalBackend(Backend):
    """Metal compute backend using Apple MLX framework."""

    def __init__(self, device_id: int = 0) -> None:
        """
        Create a Metal backend.

        Args:
            device_id: Metal device ID (0 for default GPU)

        Raises:
            BackendNotAvailableError: If Metal/MLX is not available
        """
```

## Properties

### backend_type

```python
@property
def backend_type(self) -> BackendType:
    """Returns BackendType.METAL."""
```

### is_available

```python
@property
def is_available(self) -> bool:
    """Whether Metal is available on this system."""
```

### device_count

```python
@property
def device_count(self) -> int:
    """Number of Metal devices (typically 1 on Apple Silicon)."""
```

### device_id

```python
@property
def device_id(self) -> int:
    """Current device ID."""
```

## Methods

### allocate

```python
def allocate(
    self,
    shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> Any:
    """
    Allocate an MLX array on Metal GPU.

    Args:
        shape: Array dimensions
        dtype: NumPy data type

    Returns:
        MLX array (backed by Metal unified memory)

    Raises:
        MetalError: If allocation fails
    """
```

### allocate_zeros

```python
def allocate_zeros(
    self,
    shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> Any:
    """
    Allocate a zero-filled MLX array.

    Args:
        shape: Array dimensions
        dtype: Data type

    Returns:
        Zero-filled MLX array
    """
```

### free

```python
def free(self, array: Any) -> None:
    """
    Free an MLX array from the buffer registry.

    Args:
        array: MLX array to free

    Note:
        MLX handles actual memory deallocation via garbage collection.
    """
```

### copy_to_device

```python
def copy_to_device(self, host_array: np.ndarray) -> Any:
    """
    Copy a NumPy array to MLX array (Metal GPU).

    Note:
        On Apple Silicon, this is logically a copy as memory
        is physically unified. MLX handles optimization automatically.

    Args:
        host_array: Source NumPy array

    Returns:
        MLX array on GPU
    """
```

### copy_to_host

```python
def copy_to_host(self, device_array: Any) -> np.ndarray:
    """
    Copy an MLX array to NumPy.

    Forces evaluation of lazy operations and returns NumPy array.

    Args:
        device_array: Source MLX array

    Returns:
        NumPy array
    """
```

### synchronize

```python
def synchronize(self) -> None:
    """
    Synchronize Metal operations.

    Evaluates the lazy computation graph, ensuring all
    pending MLX operations are completed.
    """
```

### execute_kernel

```python
def execute_kernel(
    self,
    kernel: Callable[..., Any],
    grid_size: tuple[int, ...],
    block_size: tuple[int, ...],
    *args: Any,
    **kwargs: Any,
) -> KernelExecutionResult:
    """
    Execute a kernel on Metal.

    Supports multiple kernel types:
    - MLX operations (functions using mlx.core)
    - Regular Python functions

    Args:
        kernel: Kernel function
        grid_size: Grid dimensions (ignored for MLX ops)
        block_size: Block dimensions (ignored for MLX ops)
        *args: Kernel arguments
        **kwargs: Kernel keyword arguments

    Returns:
        KernelExecutionResult with timing information
    """
```

### compile_kernel

```python
def compile_kernel(
    self,
    func: Callable[..., Any],
    signature: str | None = None,
) -> Callable[..., Any]:
    """
    Compile a kernel function for Metal execution.

    MLX uses JIT compilation automatically; this method returns
    a wrapped version that ensures Metal execution and handles
    type conversions.

    Args:
        func: Python function to compile
        signature: Optional type hint or kernel name

    Returns:
        Compiled kernel callable that accepts NumPy arrays
    """
```

### get_memory_info

```python
def get_memory_info(self) -> dict[str, int]:
    """
    Get Metal memory information.

    Returns:
        Dictionary with:
        - allocated: Total bytes tracked in buffer registry
        - buffer_count: Number of registered buffers
        - cache_memory: MLX cache memory usage
        - peak_memory: MLX peak memory usage
    """
```

### clear_cache

```python
def clear_cache(self) -> None:
    """Clear the Metal memory cache."""
```

## Utility Functions

### get_vector_add_kernel

```python
def get_vector_add_kernel() -> Callable[..., Any] | None:
    """
    Get a Metal vector addition kernel.

    Returns:
        Kernel function or None if Metal not available

    Example:
        kernel = get_vector_add_kernel()
        result = kernel(a, b)  # a + b
    """
```

### get_matrix_multiply_kernel

```python
def get_matrix_multiply_kernel() -> Callable[..., Any] | None:
    """
    Get a Metal matrix multiplication kernel.

    Returns:
        Kernel function or None if Metal not available

    Example:
        kernel = get_matrix_multiply_kernel()
        result = kernel(A, B)  # A @ B
    """
```

### get_elementwise_kernel

```python
def get_elementwise_kernel(operation: str) -> Callable[..., Any] | None:
    """
    Get an elementwise operation kernel.

    Args:
        operation: One of: add, sub, mul, div, sqrt, exp, log,
                   sin, cos, abs, square, negative

    Returns:
        Kernel function or None if operation not found

    Example:
        sqrt_kernel = get_elementwise_kernel("sqrt")
        result = sqrt_kernel(data)
    """
```

## Usage Examples

### Basic GPU Computation

```python
from pydotcompute.backends.metal import MetalBackend
import numpy as np

# Create backend
backend = MetalBackend()

# Copy data to Metal GPU
a = backend.copy_to_device(np.array([1, 2, 3], dtype=np.float32))
b = backend.copy_to_device(np.array([4, 5, 6], dtype=np.float32))

# Use MLX for computation
import mlx.core as mx
c = mx.add(a, b)
mx.eval(c)  # Force evaluation

# Copy result back
result = backend.copy_to_host(c)
print(result)  # [5. 7. 9.]
```

### Using Pre-built Kernels

```python
from pydotcompute.backends.metal import (
    MetalBackend,
    get_vector_add_kernel,
    get_matrix_multiply_kernel,
    get_elementwise_kernel,
)
import numpy as np

backend = MetalBackend()

# Vector addition
add_kernel = get_vector_add_kernel()
a = backend.copy_to_device(np.array([1, 2, 3], dtype=np.float32))
b = backend.copy_to_device(np.array([4, 5, 6], dtype=np.float32))
result = add_kernel(a, b)

# Matrix multiplication
matmul_kernel = get_matrix_multiply_kernel()
A = backend.copy_to_device(np.random.randn(100, 100).astype(np.float32))
B = backend.copy_to_device(np.random.randn(100, 100).astype(np.float32))
C = matmul_kernel(A, B)

# Elementwise operations
sqrt_kernel = get_elementwise_kernel("sqrt")
data = backend.copy_to_device(np.array([1, 4, 9], dtype=np.float32))
roots = sqrt_kernel(data)  # [1, 2, 3]
```

### Compiling Custom Kernels

```python
import mlx.core as mx
from pydotcompute.backends.metal import MetalBackend
import numpy as np

backend = MetalBackend()

# Define a custom kernel using MLX operations
def gaussian_kernel(x: mx.array) -> mx.array:
    return mx.exp(mx.negative(mx.square(x)))

# Compile the kernel
compiled = backend.compile_kernel(gaussian_kernel)

# Use with NumPy arrays (auto-converted)
input_data = np.linspace(-3, 3, 100).astype(np.float32)
output = compiled(input_data)
result = backend.copy_to_host(output)
```

### Kernel Execution with Timing

```python
from pydotcompute.backends.metal import MetalBackend
import mlx.core as mx
import numpy as np

backend = MetalBackend()

def compute_kernel(a, b):
    return mx.sqrt(mx.add(mx.square(a), mx.square(b)))

a = backend.copy_to_device(np.random.randn(10000).astype(np.float32))
b = backend.copy_to_device(np.random.randn(10000).astype(np.float32))

result = backend.execute_kernel(compute_kernel, (1,), (1,), a, b)

if result.success:
    print(f"Execution time: {result.execution_time_ms:.3f}ms")
else:
    print(f"Error: {result.error}")
```

### Memory Management

```python
backend = MetalBackend()

# Check memory usage
info = backend.get_memory_info()
print(f"Buffers: {info['buffer_count']}")
print(f"Cache: {info['cache_memory'] / 1024:.1f} KB")
print(f"Peak: {info['peak_memory'] / 1024:.1f} KB")

# Allocate arrays
arrays = [backend.allocate((1000,), np.float32) for _ in range(10)]

# Check updated count
print(f"Buffers after allocation: {backend.get_memory_info()['buffer_count']}")

# Free arrays
for arr in arrays:
    backend.free(arr)

# Clear cache
backend.clear_cache()
```

## Integration with UnifiedBuffer

```python
from pydotcompute.core.unified_buffer import UnifiedBuffer
import numpy as np

# Create a UnifiedBuffer
buffer = UnifiedBuffer((1000,), dtype=np.float32)
buffer.allocate()

# Write data on host
buffer.host[:] = np.random.randn(1000).astype(np.float32)
buffer.mark_host_dirty()

# Access on Metal GPU (auto-syncs from host)
metal_array = buffer.metal  # Returns MLX array

# Perform computation
import mlx.core as mx
result = mx.sum(metal_array)
mx.eval(result)
print(f"Sum: {float(result)}")

# Mark Metal as dirty if modified
buffer.mark_metal_dirty()
```

## Performance Tips

1. **Batch Operations**: MLX uses lazy evaluation - batch multiple operations before calling `mx.eval()`

2. **Unified Memory**: Apple Silicon's unified memory means data doesn't need explicit transfer - leverage this for large datasets

3. **Avoid Small Operations**: Metal has dispatch overhead - use for larger arrays (1000+ elements)

4. **Use Pre-built Kernels**: `get_vector_add_kernel()` etc. are optimized

5. **Clear Cache**: Call `clear_cache()` after processing large batches to free memory

## Supported Data Types

| NumPy dtype | MLX dtype | Notes |
|-------------|-----------|-------|
| float32 | float32 | Preferred type |
| float16 | float16 | Good for ML workloads |
| float64 | float32 | Auto-converted (MLX prefers float32) |
| int32 | int32 | |
| int64 | int64 | |
| int16 | int16 | |
| int8 | int8 | |
| uint32 | uint32 | |
| uint64 | uint64 | |
| uint16 | uint16 | |
| uint8 | uint8 | |
| bool | bool_ | |

## Notes

- Metal backend requires macOS with Apple Silicon
- MLX uses lazy evaluation - call `mx.eval()` to force computation
- Unified memory architecture eliminates explicit host-device transfers
- Thread safety is ensured through internal locks
- Buffer registry tracks allocations to prevent leaks
