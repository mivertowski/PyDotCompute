"""
Unified memory buffer with lazy synchronization.

Provides transparent host-device memory management with automatic
synchronization when needed.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from pydotcompute.exceptions import BufferNotAllocatedError, BufferSyncError, MetalError

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    try:
        import cupy as cp
    except ImportError:
        cp = None  # type: ignore


T = TypeVar("T", bound=np.generic)


class BufferState(Enum):
    """State of the unified buffer."""

    UNINITIALIZED = auto()
    HOST_ONLY = auto()
    DEVICE_ONLY = auto()
    SYNCHRONIZED = auto()
    HOST_DIRTY = auto()
    DEVICE_DIRTY = auto()


class UnifiedBuffer(Generic[T]):
    """
    Unified memory buffer with lazy synchronization.

    Provides transparent host-device memory management. Data is only
    transferred when needed, based on dirty state tracking.

    Example:
        >>> buffer = UnifiedBuffer((1000,), dtype=np.float32)
        >>> buffer.allocate()
        >>> buffer.host[:] = np.random.randn(1000)
        >>> buffer.mark_host_dirty()
        >>> # Data will be synced to device on first device access
        >>> device_data = buffer.device  # Triggers sync
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        dtype: DTypeLike = np.float32,
        *,
        pinned: bool = False,
    ) -> None:
        """
        Initialize a unified buffer.

        Args:
            shape: Shape of the buffer.
            dtype: Data type of the buffer elements.
            pinned: Whether to use pinned (page-locked) host memory.
        """
        if isinstance(shape, int):
            shape = (shape,)

        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._pinned = pinned
        self._state = BufferState.UNINITIALIZED

        self._host_data: NDArray[T] | None = None
        self._device_data: object | None = None  # cp.ndarray when available

        self._cuda_available = False
        try:
            import cupy as cp

            self._cuda_available = cp.cuda.runtime.getDeviceCount() > 0
        except ImportError:
            pass

        # Metal support
        self._metal_available = False
        self._metal_data: object | None = None  # mlx.core.array when available
        try:
            import mlx.core as mx

            self._metal_available = mx.metal.is_available()
        except ImportError:
            pass

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the buffer shape."""
        return self._shape

    @property
    def dtype(self) -> np.dtype[T]:
        """Get the buffer dtype."""
        return self._dtype  # type: ignore[return-value]

    @property
    def size(self) -> int:
        """Get the total number of elements."""
        result = 1
        for dim in self._shape:
            result *= dim
        return result

    @property
    def nbytes(self) -> int:
        """Get the total size in bytes."""
        return self.size * self._dtype.itemsize

    @property
    def state(self) -> BufferState:
        """Get the current buffer state."""
        return self._state

    @property
    def is_allocated(self) -> bool:
        """Check if the buffer is allocated."""
        return self._state != BufferState.UNINITIALIZED

    def allocate(self, *, host: bool = True, device: bool = True) -> UnifiedBuffer[T]:
        """
        Allocate the buffer memory.

        Args:
            host: Whether to allocate host memory.
            device: Whether to allocate device memory (if CUDA available).

        Returns:
            Self for method chaining.
        """
        if host:
            if self._pinned and self._cuda_available:
                try:
                    import cupy as cp

                    pinned_mem = cp.cuda.alloc_pinned_memory(self.nbytes)
                    self._host_data = np.frombuffer(pinned_mem, dtype=self._dtype).reshape(
                        self._shape
                    )
                except Exception:
                    # Fall back to regular allocation
                    self._host_data = np.empty(self._shape, dtype=self._dtype)
            else:
                self._host_data = np.empty(self._shape, dtype=self._dtype)

        if device and self._cuda_available:
            try:
                import cupy as cp

                self._device_data = cp.empty(self._shape, dtype=self._dtype)
            except Exception:
                pass

        # Set initial state
        if host and device and self._device_data is not None:
            self._state = BufferState.SYNCHRONIZED
        elif host:
            self._state = BufferState.HOST_ONLY
        elif device and self._device_data is not None:
            self._state = BufferState.DEVICE_ONLY
        else:
            self._state = BufferState.HOST_ONLY
            self._host_data = np.empty(self._shape, dtype=self._dtype)

        return self

    def free(self) -> None:
        """Free the buffer memory."""
        self._host_data = None
        self._device_data = None
        self._state = BufferState.UNINITIALIZED

    @property
    def host(self) -> NDArray[T]:
        """
        Get the host array, synchronizing from device if needed.

        Returns:
            NumPy array with host data.

        Raises:
            BufferNotAllocatedError: If buffer is not allocated.
        """
        if self._state == BufferState.UNINITIALIZED:
            raise BufferNotAllocatedError()

        # Sync from device if needed
        if self._state == BufferState.DEVICE_DIRTY:
            self._sync_device_to_host()

        # Allocate host if only on device
        if self._host_data is None:
            self._host_data = np.empty(self._shape, dtype=self._dtype)
            if self._state == BufferState.DEVICE_ONLY:
                self._sync_device_to_host()

        return self._host_data

    @property
    def device(self) -> object:
        """
        Get the device array, synchronizing from host if needed.

        Returns:
            CuPy array with device data (or NumPy array if CUDA not available).

        Raises:
            BufferNotAllocatedError: If buffer is not allocated.
        """
        if self._state == BufferState.UNINITIALIZED:
            raise BufferNotAllocatedError()

        # If CUDA not available, return host array
        if not self._cuda_available:
            return self.host

        # Sync from host if needed
        if self._state == BufferState.HOST_DIRTY:
            self._sync_host_to_device()

        # Allocate device if only on host
        if self._device_data is None:
            try:
                import cupy as cp

                self._device_data = cp.empty(self._shape, dtype=self._dtype)
                if self._state == BufferState.HOST_ONLY:
                    self._sync_host_to_device()
            except Exception:
                return self.host

        return self._device_data

    @property
    def metal(self) -> object:
        """
        Get the Metal (MLX) array, synchronizing from host if needed.

        Returns:
            MLX array with Metal data.

        Raises:
            BufferNotAllocatedError: If buffer is not allocated.
            MetalError: If Metal is not available.
        """
        if self._state == BufferState.UNINITIALIZED:
            raise BufferNotAllocatedError()

        if not self._metal_available:
            raise MetalError("Metal not available")

        # Sync from host if needed
        if self._state == BufferState.HOST_DIRTY:
            self._sync_host_to_metal()

        # Create Metal array if not exists
        if self._metal_data is None:
            try:
                import mlx.core as mx

                if self._host_data is not None:
                    self._metal_data = mx.array(self._host_data)
                else:
                    mlx_dtype = self._numpy_to_mlx_dtype()
                    self._metal_data = mx.zeros(self._shape, dtype=mlx_dtype)
                self._state = BufferState.SYNCHRONIZED
            except Exception as e:
                raise MetalError(f"Failed to create Metal array: {e}") from e

        return self._metal_data

    def _numpy_to_mlx_dtype(self) -> object:
        """Convert NumPy dtype to MLX dtype."""
        import mlx.core as mx

        mapping = {
            np.float32: mx.float32,
            np.float16: mx.float16,
            np.int32: mx.int32,
            np.int64: mx.int64,
            np.int16: mx.int16,
            np.int8: mx.int8,
            np.uint32: mx.uint32,
            np.uint64: mx.uint64,
            np.uint16: mx.uint16,
            np.uint8: mx.uint8,
            np.bool_: mx.bool_,
        }

        # Handle float64 -> float32 (MLX prefers float32)
        if self._dtype.type == np.float64:
            return mx.float32

        return mapping.get(self._dtype.type, mx.float32)

    def _sync_host_to_metal(self) -> None:
        """Synchronize data from host to Metal."""
        if self._host_data is None or not self._metal_available:
            return

        try:
            import mlx.core as mx

            self._metal_data = mx.array(self._host_data)
            self._state = BufferState.SYNCHRONIZED

        except Exception as e:
            raise BufferSyncError("host->metal", e) from e

    def _sync_metal_to_host(self) -> None:
        """Synchronize data from Metal to host."""
        if self._metal_data is None:
            return

        try:
            import numpy as np

            if self._host_data is None:
                self._host_data = np.empty(self._shape, dtype=self._dtype)

            self._host_data[:] = np.array(self._metal_data)
            self._state = BufferState.SYNCHRONIZED

        except Exception as e:
            raise BufferSyncError("metal->host", e) from e

    def mark_host_dirty(self) -> None:
        """Mark the host data as modified (needs sync to device)."""
        if self._state == BufferState.UNINITIALIZED:
            raise BufferNotAllocatedError()
        self._state = BufferState.HOST_DIRTY

    def mark_device_dirty(self) -> None:
        """Mark the device data as modified (needs sync to host)."""
        if self._state == BufferState.UNINITIALIZED:
            raise BufferNotAllocatedError()
        self._state = BufferState.DEVICE_DIRTY

    def mark_metal_dirty(self) -> None:
        """Mark the Metal data as modified (needs sync to host)."""
        if self._state == BufferState.UNINITIALIZED:
            raise BufferNotAllocatedError()
        self._state = BufferState.DEVICE_DIRTY

    def _sync_host_to_device(self) -> None:
        """Synchronize data from host to device."""
        if self._host_data is None or not self._cuda_available:
            return

        try:
            import cupy as cp

            if self._device_data is None:
                self._device_data = cp.empty(self._shape, dtype=self._dtype)

            self._device_data.set(self._host_data)  # type: ignore
            self._state = BufferState.SYNCHRONIZED

        except Exception as e:
            raise BufferSyncError("host->device", e) from e

    def _sync_device_to_host(self) -> None:
        """Synchronize data from device to host."""
        if self._device_data is None:
            return

        try:
            if self._host_data is None:
                self._host_data = np.empty(self._shape, dtype=self._dtype)

            self._host_data[:] = self._device_data.get()  # type: ignore
            self._state = BufferState.SYNCHRONIZED

        except Exception as e:
            raise BufferSyncError("device->host", e) from e

    def synchronize(self) -> None:
        """Force synchronization in both directions based on state."""
        if self._state == BufferState.HOST_DIRTY:
            self._sync_host_to_device()
        elif self._state == BufferState.DEVICE_DIRTY:
            self._sync_device_to_host()

    async def ensure_on_device(self) -> None:
        """Async method to ensure data is on device."""
        _ = self.device
        if self._state == BufferState.HOST_DIRTY:
            self._sync_host_to_device()

    async def ensure_on_host(self) -> None:
        """Async method to ensure data is on host."""
        _ = self.host
        if self._state == BufferState.DEVICE_DIRTY:
            self._sync_device_to_host()

    def copy_from(self, data: NDArray[T]) -> None:
        """
        Copy data from a numpy array to the buffer.

        Args:
            data: Source numpy array.
        """
        if self._state == BufferState.UNINITIALIZED:
            self.allocate()

        if self._host_data is None:
            self._host_data = np.empty(self._shape, dtype=self._dtype)

        np.copyto(self._host_data, data)
        self.mark_host_dirty()

    def copy_to(self, data: NDArray[T]) -> None:
        """
        Copy buffer data to a numpy array.

        Args:
            data: Destination numpy array.
        """
        np.copyto(data, self.host)

    def zeros(self) -> UnifiedBuffer[T]:
        """Fill the buffer with zeros."""
        if self._state == BufferState.UNINITIALIZED:
            self.allocate()

        if self._host_data is not None:
            self._host_data.fill(0)
            self.mark_host_dirty()

        return self

    def fill(self, value: T) -> UnifiedBuffer[T]:
        """Fill the buffer with a value."""
        if self._state == BufferState.UNINITIALIZED:
            self.allocate()

        if self._host_data is not None:
            self._host_data.fill(value)
            self.mark_host_dirty()

        return self

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UnifiedBuffer(shape={self._shape}, dtype={self._dtype}, "
            f"state={self._state.name}, cuda={self._cuda_available}, "
            f"metal={self._metal_available})"
        )

    def __len__(self) -> int:
        """Get the length of the first dimension."""
        return self._shape[0]

    def __getitem__(self, key: object) -> NDArray[T]:
        """Get item from host array."""
        return self.host[key]  # type: ignore

    def __setitem__(self, key: object, value: object) -> None:
        """Set item in host array and mark dirty."""
        self.host[key] = value  # type: ignore
        self.mark_host_dirty()
