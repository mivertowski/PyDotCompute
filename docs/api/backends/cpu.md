# CPUBackend

CPU simulation backend for development and testing.

## Overview

`CPUBackend` provides a CPU-based implementation of the backend interface. It enables full PyDotCompute functionality without requiring GPU hardware, making it ideal for development, testing, and CI/CD environments.

```python
from pydotcompute.backends.cpu import CPUBackend

backend = CPUBackend()
```

## Class Definition

```python
class CPUBackend(Backend):
    """CPU-based compute backend."""

    def __init__(
        self,
        *,
        num_threads: int | None = None,
        use_numba: bool = True,
    ) -> None:
        """
        Create a CPU backend.

        Args:
            num_threads: Thread count (None = auto)
            use_numba: Use Numba JIT if available
        """
```

## Properties

### name

```python
@property
def name(self) -> str:
    """Returns 'cpu'."""
```

### is_available

```python
@property
def is_available(self) -> bool:
    """Always returns True."""
```

### num_threads

```python
@property
def num_threads(self) -> int:
    """Number of threads used for parallel operations."""
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
    Compile a kernel for CPU execution.

    If Numba is available and use_numba=True, uses @njit.
    Otherwise returns the function unchanged.
    """
```

### allocate

```python
def allocate(
    self,
    shape: tuple[int, ...],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Allocate a NumPy array.

    Returns:
        NumPy array (zeros)
    """
```

### to_device

```python
def to_device(self, data: np.ndarray) -> np.ndarray:
    """
    "Transfer" to device (no-op for CPU).

    Returns:
        The same array (CPU has no device transfer)
    """
```

### to_host

```python
def to_host(self, data: np.ndarray) -> np.ndarray:
    """
    "Transfer" from device (no-op for CPU).

    Returns:
        The same array
    """
```

### synchronize

```python
def synchronize(self) -> None:
    """No-op for CPU (all operations are synchronous)."""
```

### get_memory_info

```python
def get_memory_info(self) -> tuple[int, int]:
    """
    Get system RAM info.

    Returns:
        (available_bytes, total_bytes)
    """
```

## Usage Examples

### Basic Usage

```python
from pydotcompute.backends.cpu import CPUBackend

backend = CPUBackend()

# Allocate array
arr = backend.allocate((1000, 1000), dtype=np.float32)

# Compile kernel
@backend.compile_kernel
def square(x, out):
    for i in range(len(x)):
        out[i] = x[i] ** 2

# Use compiled kernel
output = backend.allocate((1000,))
square(input_data, output)
```

### With Numba Acceleration

```python
# Enable Numba JIT (default if available)
backend = CPUBackend(use_numba=True)

# This will be JIT-compiled
def my_kernel(a, b, out):
    for i in range(len(out)):
        out[i] = a[i] + b[i]

compiled = backend.compile_kernel(my_kernel, (np.float32, np.float32, np.float32))
```

### Thread Control

```python
# Use specific thread count
backend = CPUBackend(num_threads=4)

# Use all available cores
backend = CPUBackend(num_threads=None)  # Default

print(f"Using {backend.num_threads} threads")
```

### Memory Info

```python
backend = CPUBackend()

free, total = backend.get_memory_info()
print(f"RAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
```

## Performance Considerations

1. **Numba JIT**: Enable for compute-intensive kernels (~10-100x speedup)

2. **Thread Count**: Match to physical cores for best performance

3. **Memory Locality**: CPU cache benefits from sequential access

4. **No Transfer Overhead**: CPU has no host-device transfer cost

## When to Use CPU Backend

| Scenario | Recommendation |
|----------|----------------|
| Development | CPU backend |
| Unit tests | CPU backend |
| CI/CD | CPU backend |
| Small datasets | CPU backend |
| Large datasets + GPU | CUDA backend |
| Production + GPU | CUDA backend |

## Notes

- CPUBackend is the default when CUDA is unavailable
- All async operations are actually synchronous
- Memory allocations are standard NumPy arrays
- Compatible with all PyDotCompute features
