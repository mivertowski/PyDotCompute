"""
PyDotCompute exception hierarchy.

This module defines the complete exception hierarchy for PyDotCompute,
providing specific exception types for different error categories:

- KernelError: Kernel lifecycle and management issues
- MessageError: Message serialization and routing problems
- QueueError: Message queue capacity and timeout issues
- BufferError: Memory buffer allocation and synchronization
- BackendError: Compute backend availability and execution
- CompilationError: Kernel compilation failures
- ValidationError: Configuration and type validation errors

All exceptions inherit from PyDotComputeError for easy catching.
"""

from __future__ import annotations


class PyDotComputeError(Exception):
    """Base exception for all PyDotCompute errors."""

    pass


class KernelError(PyDotComputeError):
    """Base exception for kernel-related errors."""

    pass


class KernelNotFoundError(KernelError):
    """Raised when a requested kernel does not exist."""

    def __init__(self, kernel_id: str, available: list[str] | None = None) -> None:
        self.kernel_id = kernel_id
        self.available = available or []
        msg = f"Kernel '{kernel_id}' not found."
        if self.available:
            msg += f" Available kernels: {self.available}"
        msg += f"\nHint: Did you forget to call runtime.launch('{kernel_id}')?"
        super().__init__(msg)


class KernelStateError(KernelError):
    """Raised when a kernel operation is invalid for the current state."""

    def __init__(self, kernel_id: str, current_state: str, operation: str) -> None:
        self.kernel_id = kernel_id
        self.current_state = current_state
        self.operation = operation
        super().__init__(f"Cannot {operation} kernel '{kernel_id}' from state '{current_state}'")


class KernelAlreadyExistsError(KernelError):
    """Raised when attempting to register a kernel that already exists."""

    def __init__(self, kernel_id: str) -> None:
        self.kernel_id = kernel_id
        super().__init__(f"Kernel '{kernel_id}' is already registered")


class MessageError(PyDotComputeError):
    """Base exception for message-related errors."""

    pass


class MessageSerializationError(MessageError):
    """Raised when message serialization fails."""

    def __init__(self, message_type: type, cause: Exception) -> None:
        self.message_type = message_type
        self.cause = cause
        super().__init__(f"Failed to serialize message of type '{message_type.__name__}': {cause}")


class MessageDeserializationError(MessageError):
    """Raised when message deserialization fails."""

    def __init__(self, message_type: type, cause: Exception) -> None:
        self.message_type = message_type
        self.cause = cause
        super().__init__(
            f"Failed to deserialize message to type '{message_type.__name__}': {cause}"
        )


class QueueError(PyDotComputeError):
    """Base exception for queue-related errors."""

    pass


class QueueFullError(QueueError):
    """Raised when a queue is full and cannot accept more messages."""

    def __init__(self, kernel_id: str, queue_size: int) -> None:
        self.kernel_id = kernel_id
        self.queue_size = queue_size
        super().__init__(f"Input queue for kernel '{kernel_id}' is full (size: {queue_size})")


class QueueTimeoutError(QueueError):
    """Raised when a queue operation times out."""

    def __init__(self, kernel_id: str, timeout: float, operation: str) -> None:
        self.kernel_id = kernel_id
        self.timeout = timeout
        self.operation = operation
        super().__init__(
            f"Timeout ({timeout}s) waiting to {operation} message for kernel '{kernel_id}'"
        )


class BufferError(PyDotComputeError):
    """Base exception for buffer-related errors."""

    pass


class BufferNotAllocatedError(BufferError):
    """Raised when accessing a buffer that hasn't been allocated."""

    def __init__(self) -> None:
        super().__init__("Buffer has not been allocated. Call allocate() first.")


class BufferSyncError(BufferError):
    """Raised when buffer synchronization fails."""

    def __init__(self, direction: str, cause: Exception) -> None:
        self.direction = direction
        self.cause = cause
        super().__init__(f"Failed to sync buffer {direction}: {cause}")


class BackendError(PyDotComputeError):
    """Base exception for backend-related errors."""

    pass


class BackendNotAvailableError(BackendError):
    """Raised when a requested backend is not available."""

    def __init__(self, backend_name: str, reason: str) -> None:
        self.backend_name = backend_name
        self.reason = reason
        super().__init__(f"Backend '{backend_name}' is not available: {reason}")


class CUDAError(BackendError):
    """Raised for CUDA-specific errors."""

    pass


class MetalError(BackendError):
    """Raised for Metal-specific errors."""

    pass


class MSLCompilationError(MetalError):
    """Raised when Metal Shading Language compilation fails."""

    def __init__(self, shader_name: str, error_message: str) -> None:
        self.shader_name = shader_name
        self.error_message = error_message
        super().__init__(f"Failed to compile MSL shader '{shader_name}': {error_message}")


class CompilationError(PyDotComputeError):
    """Base exception for compilation-related errors."""

    pass


class KernelCompilationError(CompilationError):
    """Raised when kernel compilation fails."""

    def __init__(self, kernel_name: str, cause: Exception) -> None:
        self.kernel_name = kernel_name
        self.cause = cause
        super().__init__(f"Failed to compile kernel '{kernel_name}': {cause}")


class ValidationError(PyDotComputeError):
    """Base exception for validation-related errors."""

    pass


class InvalidConfigurationError(ValidationError):
    """Raised when configuration is invalid."""

    def __init__(self, parameter: str, value: object, reason: str) -> None:
        self.parameter = parameter
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid configuration: {parameter}={value!r} - {reason}")


class TypeValidationError(ValidationError):
    """Raised when type validation fails."""

    def __init__(self, expected: type, actual: type, context: str) -> None:
        self.expected = expected
        self.actual = actual
        self.context = context
        super().__init__(
            f"Type mismatch in {context}: expected {expected.__name__}, got {actual.__name__}"
        )
