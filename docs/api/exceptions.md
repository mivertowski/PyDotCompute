# Exceptions

Complete exception hierarchy for PyDotCompute error handling.

## Overview

PyDotCompute uses a structured exception hierarchy to provide clear, actionable error messages. All exceptions inherit from `PyDotComputeError`.

```python
from pydotcompute.exceptions import (
    PyDotComputeError,
    KernelNotFoundError,
    KernelStateError,
    QueueFullError,
)

try:
    await runtime.send("unknown", message)
except KernelNotFoundError as e:
    print(f"Kernel not found: {e.kernel_id}")
```

## Exception Hierarchy

```
PyDotComputeError
├── KernelError
│   ├── KernelNotFoundError
│   ├── KernelStateError
│   ├── KernelAlreadyExistsError
│   └── KernelCompilationError
├── QueueError
│   ├── QueueFullError
│   └── QueueEmptyError
├── MessageError
│   ├── MessageSerializationError
│   └── MessageValidationError
├── BackendError
│   ├── BackendNotAvailableError
│   ├── BackendExecutionError
│   └── MetalError
│       └── MSLCompilationError
└── MemoryError
    ├── AllocationError
    └── OutOfMemoryError
```

## Base Exception

### PyDotComputeError

```python
class PyDotComputeError(Exception):
    """Base exception for all PyDotCompute errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
```

## Kernel Exceptions

### KernelError

```python
class KernelError(PyDotComputeError):
    """Base exception for kernel-related errors."""

    def __init__(self, kernel_id: str, message: str) -> None:
        self.kernel_id = kernel_id
        super().__init__(f"[{kernel_id}] {message}")
```

### KernelNotFoundError

```python
class KernelNotFoundError(KernelError):
    """Raised when a kernel ID is not found."""

    def __init__(self, kernel_id: str) -> None:
        super().__init__(kernel_id, f"Kernel not found: '{kernel_id}'")
```

**When raised:**

- `runtime.send()` with unknown kernel_id
- `runtime.receive()` with unknown kernel_id
- `runtime.activate()` with unknown kernel_id
- `runtime.get_state()` with unknown kernel_id

### KernelStateError

```python
class KernelStateError(KernelError):
    """Raised when kernel is in wrong state for operation."""

    def __init__(
        self,
        kernel_id: str,
        current_state: KernelState,
        expected_states: list[KernelState],
    ) -> None:
        self.current_state = current_state
        self.expected_states = expected_states
        expected = ", ".join(s.name for s in expected_states)
        super().__init__(
            kernel_id,
            f"Invalid state: {current_state.name}, expected: {expected}"
        )
```

**When raised:**

- `activate()` on non-LAUNCHED kernel
- `deactivate()` on non-ACTIVE kernel
- `reactivate()` on non-DEACTIVATED kernel

### KernelAlreadyExistsError

```python
class KernelAlreadyExistsError(KernelError):
    """Raised when launching a kernel that already exists."""

    def __init__(self, kernel_id: str) -> None:
        super().__init__(kernel_id, "Kernel already exists")
```

### KernelCompilationError

```python
class KernelCompilationError(KernelError):
    """Raised when kernel compilation fails."""

    def __init__(self, kernel_id: str, reason: str) -> None:
        self.reason = reason
        super().__init__(kernel_id, f"Compilation failed: {reason}")
```

## Queue Exceptions

### QueueError

```python
class QueueError(PyDotComputeError):
    """Base exception for queue-related errors."""
    pass
```

### QueueFullError

```python
class QueueFullError(QueueError):
    """Raised when queue is full and cannot accept messages."""

    def __init__(self, queue_size: int) -> None:
        self.queue_size = queue_size
        super().__init__(f"Queue is full (size: {queue_size})")
```

**When raised:**

- `queue.put()` with REJECT strategy and queue full
- `queue.put()` with BLOCK strategy and timeout exceeded
- `runtime.send()` when input queue full

### QueueEmptyError

```python
class QueueEmptyError(QueueError):
    """Raised when queue is empty on non-blocking get."""

    def __init__(self) -> None:
        super().__init__("Queue is empty")
```

## Message Exceptions

### MessageError

```python
class MessageError(PyDotComputeError):
    """Base exception for message-related errors."""
    pass
```

### MessageSerializationError

```python
class MessageSerializationError(MessageError):
    """Raised when message serialization fails."""

    def __init__(self, message_type: str, reason: str) -> None:
        self.message_type = message_type
        self.reason = reason
        super().__init__(f"Cannot serialize {message_type}: {reason}")
```

### MessageValidationError

```python
class MessageValidationError(MessageError):
    """Raised when message validation fails."""

    def __init__(self, message_type: str, field: str, reason: str) -> None:
        self.message_type = message_type
        self.field = field
        self.reason = reason
        super().__init__(f"{message_type}.{field}: {reason}")
```

## Backend Exceptions

### BackendError

```python
class BackendError(PyDotComputeError):
    """Base exception for backend-related errors."""

    def __init__(self, backend: str, message: str) -> None:
        self.backend = backend
        super().__init__(f"[{backend}] {message}")
```

### BackendNotAvailableError

```python
class BackendNotAvailableError(BackendError):
    """Raised when requested backend is not available."""

    def __init__(self, backend: str) -> None:
        super().__init__(backend, "Backend not available")
```

### BackendExecutionError

```python
class BackendExecutionError(BackendError):
    """Raised when backend execution fails."""

    def __init__(self, backend: str, reason: str) -> None:
        self.reason = reason
        super().__init__(backend, f"Execution failed: {reason}")
```

### MetalError

```python
class MetalError(BackendError):
    """Raised for Metal-specific errors on macOS."""

    def __init__(self, message: str) -> None:
        super().__init__("Metal", message)
```

**When raised:**

- Metal/MLX not available on the system
- Metal memory allocation fails
- Metal operations fail

### MSLCompilationError

```python
class MSLCompilationError(MetalError):
    """Raised when Metal Shading Language compilation fails."""

    def __init__(self, shader_name: str, error_message: str) -> None:
        self.shader_name = shader_name
        self.error_message = error_message
        super().__init__(
            f"Failed to compile MSL shader '{shader_name}': {error_message}"
        )
```

**When raised:**

- Custom MSL shader compilation fails
- Invalid Metal shader syntax

## Memory Exceptions

### MemoryError

```python
class MemoryError(PyDotComputeError):
    """Base exception for memory-related errors."""
    pass
```

### AllocationError

```python
class AllocationError(MemoryError):
    """Raised when memory allocation fails."""

    def __init__(self, size: int, reason: str) -> None:
        self.size = size
        self.reason = reason
        super().__init__(f"Cannot allocate {size} bytes: {reason}")
```

### OutOfMemoryError

```python
class OutOfMemoryError(MemoryError):
    """Raised when system/device runs out of memory."""

    def __init__(self, requested: int, available: int) -> None:
        self.requested = requested
        self.available = available
        super().__init__(
            f"Out of memory: requested {requested} bytes, "
            f"available {available} bytes"
        )
```

## Usage Examples

### Handling Kernel Errors

```python
from pydotcompute.exceptions import KernelNotFoundError, KernelStateError

try:
    await runtime.activate("my_kernel")
except KernelNotFoundError as e:
    print(f"Unknown kernel: {e.kernel_id}")
    # Maybe launch it first?
    await runtime.launch(e.kernel_id, my_kernel_func)
    await runtime.activate(e.kernel_id)
except KernelStateError as e:
    print(f"Wrong state: {e.current_state}, need: {e.expected_states}")
```

### Handling Queue Errors

```python
from pydotcompute.exceptions import QueueFullError
import asyncio

try:
    await runtime.send("worker", message, timeout=1.0)
except QueueFullError:
    print("Queue full, retrying with backoff...")
    await asyncio.sleep(0.1)
    await runtime.send("worker", message, timeout=5.0)
```

### Generic Error Handling

```python
from pydotcompute.exceptions import PyDotComputeError

try:
    # PyDotCompute operations...
    pass
except PyDotComputeError as e:
    # Catch all PyDotCompute errors
    print(f"Error: {e.message}")
    raise
```

## Best Practices

1. **Catch Specific Exceptions**: Handle specific error types for precise recovery

2. **Check Exception Attributes**: Use attributes like `kernel_id` for context

3. **Log Before Re-raising**: Log errors with full context

4. **Graceful Degradation**: Fall back to alternatives when possible

5. **Don't Catch Base Exception**: Avoid catching `PyDotComputeError` unless necessary

## Notes

- All exceptions include helpful error messages
- Exception attributes provide programmatic access to details
- Use `str(exception)` for logging
- Exceptions are picklable for multiprocessing
