"""
Backend base classes and interfaces.

Defines the abstract interface that all backends must implement.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


T = TypeVar("T", bound=np.generic)


class BackendType(Enum):
    """Type of compute backend."""

    CPU = auto()
    CUDA = auto()
    METAL = auto()  # Future support


@dataclass
class KernelExecutionResult:
    """Result of a kernel execution."""

    success: bool
    execution_time_ms: float
    error: Exception | None = None


class Backend(ABC):
    """
    Abstract base class for compute backends.

    All backends must implement this interface to provide
    a consistent API for kernel execution and memory management.
    """

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        ...

    @property
    @abstractmethod
    def device_count(self) -> int:
        """Get the number of available devices."""
        ...

    @abstractmethod
    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[T],
    ) -> NDArray[T]:
        """
        Allocate an array on this backend.

        Args:
            shape: Shape of the array.
            dtype: Data type.

        Returns:
            Allocated array.
        """
        ...

    @abstractmethod
    def free(self, array: NDArray[T]) -> None:
        """
        Free an array allocated by this backend.

        Args:
            array: Array to free.
        """
        ...

    @abstractmethod
    def copy_to_device(self, host_array: NDArray[T]) -> NDArray[T]:
        """
        Copy a host array to this backend's device.

        Args:
            host_array: Source host array.

        Returns:
            Device array.
        """
        ...

    @abstractmethod
    def copy_to_host(self, device_array: NDArray[T]) -> NDArray[T]:
        """
        Copy a device array to host.

        Args:
            device_array: Source device array.

        Returns:
            Host array.
        """
        ...

    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize all pending operations."""
        ...

    @abstractmethod
    def execute_kernel(
        self,
        kernel: Callable[..., Any],
        grid_size: tuple[int, ...],
        block_size: tuple[int, ...],
        *args: Any,
        **kwargs: Any,
    ) -> KernelExecutionResult:
        """
        Execute a kernel on this backend.

        Args:
            kernel: Kernel function.
            grid_size: Grid dimensions.
            block_size: Block dimensions.
            *args: Kernel arguments.
            **kwargs: Kernel keyword arguments.

        Returns:
            Execution result.
        """
        ...

    @abstractmethod
    def compile_kernel(
        self,
        func: Callable[..., Any],
        signature: str | None = None,
    ) -> Callable[..., Any]:
        """
        Compile a kernel function for this backend.

        Args:
            func: Python function to compile.
            signature: Optional type signature.

        Returns:
            Compiled kernel.
        """
        ...


class BackendArray(Generic[T]):
    """
    Array wrapper that tracks which backend owns it.

    Provides a unified interface for arrays on different backends.
    """

    def __init__(
        self,
        data: NDArray[T],
        backend: Backend,
        *,
        owns_data: bool = True,
    ) -> None:
        """
        Initialize a backend array.

        Args:
            data: The underlying array.
            backend: Backend that owns this array.
            owns_data: Whether this wrapper owns the data.
        """
        self._data = data
        self._backend = backend
        self._owns_data = owns_data

    @property
    def data(self) -> NDArray[T]:
        """Get the underlying array."""
        return self._data

    @property
    def backend(self) -> Backend:
        """Get the backend."""
        return self._backend

    @property
    def shape(self) -> tuple[int, ...]:
        """Get array shape."""
        return self._data.shape

    @property
    def dtype(self) -> np.dtype[T]:
        """Get array dtype."""
        return self._data.dtype

    @property
    def nbytes(self) -> int:
        """Get total size in bytes."""
        return self._data.nbytes

    def to_host(self) -> NDArray[T]:
        """Copy to host memory."""
        return self._backend.copy_to_host(self._data)

    def synchronize(self) -> None:
        """Synchronize this array's backend."""
        self._backend.synchronize()

    def __del__(self) -> None:
        """Free memory if owned."""
        if self._owns_data:
            with contextlib.suppress(Exception):
                self._backend.free(self._data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BackendArray(shape={self.shape}, dtype={self.dtype}, "
            f"backend={self._backend.backend_type.name})"
        )
