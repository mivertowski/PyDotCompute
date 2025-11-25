# ComputeOrchestrator

Coordination of compute operations across devices.

## Overview

`ComputeOrchestrator` provides high-level coordination for executing compute operations across CPU and GPU devices. It manages work distribution and result collection.

```python
from pydotcompute.core.orchestrator import ComputeOrchestrator

orchestrator = ComputeOrchestrator()

# Submit work
result = await orchestrator.execute(
    kernel_fn,
    inputs=[data1, data2],
    output_shape=(1000,),
)
```

## Classes

### ComputeOrchestrator

```python
class ComputeOrchestrator:
    """Coordinates compute operations across devices."""

    def __init__(
        self,
        preferred_device: DeviceType | None = None,
        enable_profiling: bool = False,
    ) -> None:
        """
        Create an orchestrator.

        Args:
            preferred_device: Preferred device type (auto-selects if None)
            enable_profiling: Enable operation profiling
        """
```

## Methods

### execute

```python
async def execute(
    self,
    kernel: Callable,
    inputs: list[UnifiedBuffer],
    output_shape: tuple[int, ...],
    output_dtype: np.dtype = np.float32,
    **kwargs: Any,
) -> UnifiedBuffer:
    """
    Execute a kernel with inputs and return output.

    Args:
        kernel: Kernel function to execute
        inputs: Input buffers
        output_shape: Shape of output buffer
        output_dtype: Data type of output
        **kwargs: Additional kernel arguments

    Returns:
        Output buffer with results
    """
```

### execute_batch

```python
async def execute_batch(
    self,
    kernel: Callable,
    batch_inputs: list[list[UnifiedBuffer]],
    output_shape: tuple[int, ...],
    output_dtype: np.dtype = np.float32,
) -> list[UnifiedBuffer]:
    """
    Execute a kernel on multiple input batches.

    Args:
        kernel: Kernel function
        batch_inputs: List of input lists
        output_shape: Shape per output
        output_dtype: Data type of outputs

    Returns:
        List of output buffers
    """
```

### map

```python
async def map(
    self,
    kernel: Callable,
    data: UnifiedBuffer,
    chunk_size: int = 1024,
) -> UnifiedBuffer:
    """
    Apply kernel to data in chunks.

    Args:
        kernel: Element-wise kernel function
        data: Input buffer
        chunk_size: Size of each chunk

    Returns:
        Output buffer with mapped results
    """
```

### reduce

```python
async def reduce(
    self,
    kernel: Callable,
    data: UnifiedBuffer,
    initial: float = 0.0,
) -> float:
    """
    Reduce data using kernel.

    Args:
        kernel: Reduction kernel (a, b) -> c
        data: Input buffer
        initial: Initial value

    Returns:
        Reduced scalar result
    """
```

## Properties

### device

```python
@property
def device(self) -> DeviceInfo:
    """Currently selected compute device."""
```

### profiling_enabled

```python
@property
def profiling_enabled(self) -> bool:
    """Whether profiling is enabled."""
```

## Usage Examples

### Basic Execution

```python
from pydotcompute.core.orchestrator import ComputeOrchestrator
from pydotcompute import UnifiedBuffer
import numpy as np

async def example():
    orchestrator = ComputeOrchestrator()

    # Create input buffers
    a = UnifiedBuffer((1000,), dtype=np.float32)
    b = UnifiedBuffer((1000,), dtype=np.float32)
    a.host[:] = np.random.randn(1000)
    b.host[:] = np.random.randn(1000)

    # Define kernel
    def add_kernel(x, y, out):
        for i in range(len(out)):
            out[i] = x[i] + y[i]

    # Execute
    result = await orchestrator.execute(
        add_kernel,
        inputs=[a, b],
        output_shape=(1000,),
    )

    return result.to_numpy()
```

### Batch Processing

```python
async def batch_example():
    orchestrator = ComputeOrchestrator()

    # Create batches of inputs
    batches = []
    for _ in range(10):
        buf = UnifiedBuffer((100,), dtype=np.float32)
        buf.host[:] = np.random.randn(100)
        batches.append([buf])

    # Process all batches
    results = await orchestrator.execute_batch(
        square_kernel,
        batches,
        output_shape=(100,),
    )

    return [r.to_numpy() for r in results]
```

### With Profiling

```python
orchestrator = ComputeOrchestrator(enable_profiling=True)

result = await orchestrator.execute(kernel, inputs, (1000,))

# Access profiling data
profile = orchestrator.get_profile()
print(f"Execution time: {profile['execution_time_ms']:.2f} ms")
print(f"Memory transferred: {profile['bytes_transferred']} bytes")
```

## Notes

- The orchestrator automatically selects GPU if available
- Use `preferred_device` to force CPU or GPU execution
- Profiling adds overhead - disable for production
- Batch execution may be more efficient than individual calls
