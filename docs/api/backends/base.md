# Backend Interface

Abstract base class defining the compute backend interface.

## Overview

The `Backend` abstract base class defines the interface that all compute backends must implement. This enables PyDotCompute to work seamlessly across CPU and GPU environments.

```python
from pydotcompute.backends.base import Backend

class MyCustomBackend(Backend):
    def compile_kernel(self, func, signature):
        # Custom compilation logic
        ...
```

## Abstract Base Class

### Backend

```python
from abc import ABC, abstractmethod

class Backend(ABC):
    """Abstract base class for compute backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend is available on the current system."""

    @abstractmethod
    def compile_kernel(
        self,
        func: Callable,
        signature: tuple[type, ...],
    ) -> Callable:
        """
        Compile a kernel function for this backend.

        Args:
            func: Python function to compile
            signature: Argument types

        Returns:
            Compiled kernel callable
        """

    @abstractmethod
    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> Any:
        """
        Allocate memory on this backend.

        Args:
            shape: Array dimensions
            dtype: Data type

        Returns:
            Backend-specific array
        """

    @abstractmethod
    def to_device(self, data: np.ndarray) -> Any:
        """
        Transfer data to this backend.

        Args:
            data: NumPy array

        Returns:
            Backend-specific array
        """

    @abstractmethod
    def to_host(self, data: Any) -> np.ndarray:
        """
        Transfer data from this backend to host.

        Args:
            data: Backend-specific array

        Returns:
            NumPy array
        """

    @abstractmethod
    def synchronize(self) -> None:
        """Wait for all pending operations to complete."""
```

## Optional Methods

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
    Launch a kernel with explicit grid/block configuration.

    Default implementation just calls the kernel.
    GPU backends override for proper launch.
    """
```

### get_memory_info

```python
def get_memory_info(self) -> tuple[int, int]:
    """
    Get (free, total) memory in bytes.

    Returns:
        Tuple of (free_bytes, total_bytes)
    """
```

### supports_feature

```python
def supports_feature(self, feature: str) -> bool:
    """
    Check if backend supports a specific feature.

    Args:
        feature: Feature name (e.g., "async", "streams", "shared_memory")

    Returns:
        Whether the feature is supported
    """
```

## Built-in Backends

| Backend | Module | Description |
|---------|--------|-------------|
| `CPUBackend` | `pydotcompute.backends.cpu` | CPU simulation for development |
| `CUDABackend` | `pydotcompute.backends.cuda` | NVIDIA GPU via Numba/CuPy |

## Usage Examples

### Checking Backend Availability

```python
from pydotcompute.backends.cpu import CPUBackend
from pydotcompute.backends.cuda import CUDABackend

cpu = CPUBackend()
print(f"CPU available: {cpu.is_available}")  # Always True

cuda = CUDABackend()
print(f"CUDA available: {cuda.is_available}")  # True if GPU present
```

### Backend Selection

```python
from pydotcompute.backends import get_backend

# Auto-select best available
backend = get_backend("auto")

# Force specific backend
cpu_backend = get_backend("cpu")
cuda_backend = get_backend("cuda")  # Raises if unavailable
```

### Implementing Custom Backend

```python
from pydotcompute.backends.base import Backend
import numpy as np

class OpenCLBackend(Backend):
    """Example custom OpenCL backend."""

    @property
    def name(self) -> str:
        return "opencl"

    @property
    def is_available(self) -> bool:
        try:
            import pyopencl
            return True
        except ImportError:
            return False

    def compile_kernel(self, func, signature):
        # OpenCL kernel compilation
        ...

    def allocate(self, shape, dtype):
        # OpenCL buffer allocation
        ...

    def to_device(self, data):
        # Copy to OpenCL buffer
        ...

    def to_host(self, data):
        # Copy from OpenCL buffer
        ...

    def synchronize(self):
        # Wait for OpenCL queue
        ...
```

## Notes

- CPUBackend is always available as fallback
- CUDABackend requires NVIDIA GPU and CUDA toolkit
- Custom backends can be registered with the runtime
- Backend selection happens at runtime launch
