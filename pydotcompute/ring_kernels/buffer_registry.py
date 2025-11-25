"""
Buffer Registry for Zero-Copy Array Transfer.

Provides a registry for large arrays to avoid serialization overhead.
Arrays are stored by reference and only their IDs are passed in messages.

Performance impact:
- Without registry: 1MB array → ~1ms serialization + copy
- With registry: 1MB array → ~1μs reference lookup
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID, uuid4

import numpy as np

if TYPE_CHECKING:
    pass

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore
    HAS_CUPY = False

ArrayType = TypeVar("ArrayType", bound=np.ndarray)


@dataclass
class BufferHandle:
    """Handle to a registered buffer."""

    buffer_id: UUID
    shape: tuple[int, ...]
    dtype: str
    device: str  # 'cpu' or 'cuda:N'
    size_bytes: int
    ref_count: int = 1

    def __repr__(self) -> str:
        return (
            f"BufferHandle(id={self.buffer_id.hex[:8]}, "
            f"shape={self.shape}, dtype={self.dtype}, device={self.device})"
        )


class BufferRegistry:
    """
    Thread-safe registry for zero-copy array transfer.

    Registers numpy/cupy arrays and returns lightweight handles.
    Arrays are stored by reference - no copying occurs during registration.

    Example:
        >>> registry = BufferRegistry()
        >>> arr = np.random.randn(1000000).astype(np.float32)
        >>> handle = registry.register(arr)
        >>> # Pass handle.buffer_id in message instead of array
        >>> retrieved = registry.get(handle.buffer_id)
        >>> assert retrieved is arr  # Same object, no copy
    """

    _instance: BufferRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> BufferRegistry:
        """Singleton pattern for global registry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry."""
        if getattr(self, "_initialized", False):
            return

        self._buffers: dict[UUID, np.ndarray | Any] = {}
        self._handles: dict[UUID, BufferHandle] = {}
        self._registry_lock = threading.RLock()
        self._stats = {
            "registered": 0,
            "retrieved": 0,
            "released": 0,
            "bytes_registered": 0,
        }
        self._initialized = True

    def register(self, array: np.ndarray | Any) -> BufferHandle:
        """
        Register an array and return a handle.

        Args:
            array: numpy or cupy array to register.

        Returns:
            BufferHandle with metadata for the array.
        """
        buffer_id = uuid4()

        # Determine device
        device = f"cuda:{array.device.id}" if HAS_CUPY and isinstance(array, cp.ndarray) else "cpu"

        # Calculate size
        size_bytes = array.nbytes

        handle = BufferHandle(
            buffer_id=buffer_id,
            shape=array.shape,
            dtype=str(array.dtype),
            device=device,
            size_bytes=size_bytes,
        )

        with self._registry_lock:
            self._buffers[buffer_id] = array
            self._handles[buffer_id] = handle
            self._stats["registered"] += 1
            self._stats["bytes_registered"] += size_bytes

        return handle

    def get(self, buffer_id: UUID) -> np.ndarray | Any | None:
        """
        Retrieve an array by its buffer ID.

        Args:
            buffer_id: UUID of the registered buffer.

        Returns:
            The registered array or None if not found.
        """
        with self._registry_lock:
            array = self._buffers.get(buffer_id)
            if array is not None:
                self._stats["retrieved"] += 1
            return array

    def get_handle(self, buffer_id: UUID) -> BufferHandle | None:
        """
        Get the handle for a buffer.

        Args:
            buffer_id: UUID of the registered buffer.

        Returns:
            BufferHandle or None if not found.
        """
        with self._registry_lock:
            return self._handles.get(buffer_id)

    def release(self, buffer_id: UUID) -> bool:
        """
        Release a buffer from the registry.

        Args:
            buffer_id: UUID of the buffer to release.

        Returns:
            True if buffer was released, False if not found.
        """
        with self._registry_lock:
            if buffer_id in self._buffers:
                handle = self._handles.pop(buffer_id, None)
                del self._buffers[buffer_id]
                self._stats["released"] += 1
                if handle:
                    self._stats["bytes_registered"] -= handle.size_bytes
                return True
            return False

    def increment_ref(self, buffer_id: UUID) -> bool:
        """Increment reference count for a buffer."""
        with self._registry_lock:
            handle = self._handles.get(buffer_id)
            if handle:
                handle.ref_count += 1
                return True
            return False

    def decrement_ref(self, buffer_id: UUID) -> int:
        """
        Decrement reference count and release if zero.

        Returns:
            New reference count, or -1 if buffer not found.
        """
        with self._registry_lock:
            handle = self._handles.get(buffer_id)
            if handle:
                handle.ref_count -= 1
                if handle.ref_count <= 0:
                    self.release(buffer_id)
                    return 0
                return handle.ref_count
            return -1

    def clear(self) -> int:
        """
        Clear all buffers from the registry.

        Returns:
            Number of buffers cleared.
        """
        with self._registry_lock:
            count = len(self._buffers)
            self._buffers.clear()
            self._handles.clear()
            self._stats["bytes_registered"] = 0
            return count

    def get_stats(self) -> dict[str, int]:
        """Get registry statistics."""
        with self._registry_lock:
            return {
                **self._stats,
                "active_buffers": len(self._buffers),
            }

    def __len__(self) -> int:
        """Get number of registered buffers."""
        return len(self._buffers)

    def __contains__(self, buffer_id: UUID) -> bool:
        """Check if buffer is registered."""
        return buffer_id in self._buffers


# Global registry instance
_global_registry: BufferRegistry | None = None


def get_buffer_registry() -> BufferRegistry:
    """Get the global buffer registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = BufferRegistry()
    return _global_registry


def register_buffer(array: np.ndarray | Any) -> BufferHandle:
    """
    Register an array in the global registry.

    Convenience function for quick buffer registration.

    Args:
        array: numpy or cupy array to register.

    Returns:
        BufferHandle for the registered array.
    """
    return get_buffer_registry().register(array)


def get_buffer(buffer_id: UUID) -> np.ndarray | Any | None:
    """
    Get an array from the global registry.

    Convenience function for quick buffer retrieval.

    Args:
        buffer_id: UUID of the registered buffer.

    Returns:
        The registered array or None.
    """
    return get_buffer_registry().get(buffer_id)


def release_buffer(buffer_id: UUID) -> bool:
    """
    Release a buffer from the global registry.

    Args:
        buffer_id: UUID of the buffer to release.

    Returns:
        True if released, False if not found.
    """
    return get_buffer_registry().release(buffer_id)
