"""
Unit tests for UnifiedBuffer.
"""

from __future__ import annotations

import numpy as np
import pytest

from pydotcompute.core.unified_buffer import BufferState, UnifiedBuffer
from pydotcompute.exceptions import BufferNotAllocatedError


class TestUnifiedBuffer:
    """Tests for UnifiedBuffer class."""

    def test_creation(self) -> None:
        """Test buffer creation with various shapes."""
        # 1D buffer
        buf1d = UnifiedBuffer((100,), dtype=np.float32)
        assert buf1d.shape == (100,)
        assert buf1d.dtype == np.float32
        assert buf1d.state == BufferState.UNINITIALIZED

        # 2D buffer
        buf2d = UnifiedBuffer((10, 20), dtype=np.float64)
        assert buf2d.shape == (10, 20)
        assert buf2d.dtype == np.float64

        # Integer shape
        buf_int = UnifiedBuffer(50, dtype=np.int32)
        assert buf_int.shape == (50,)

    def test_allocation(self) -> None:
        """Test buffer allocation."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((100,), dtype=np.float32)

        assert not buf.is_allocated

        buf.allocate()

        assert buf.is_allocated
        assert buf.state in (BufferState.HOST_ONLY, BufferState.SYNCHRONIZED)
        assert buf.host is not None
        assert buf.host.shape == (100,)

    def test_allocation_host_only(self) -> None:
        """Test allocation with host only."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((100,), dtype=np.float32)
        buf.allocate(host=True, device=False)

        assert buf.is_allocated
        assert buf.state == BufferState.HOST_ONLY

    def test_free(self) -> None:
        """Test buffer deallocation."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((100,), dtype=np.float32)
        buf.allocate()

        assert buf.is_allocated

        buf.free()

        assert not buf.is_allocated
        assert buf.state == BufferState.UNINITIALIZED

    def test_host_access_without_allocation_raises(self) -> None:
        """Test that accessing host data before allocation raises."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((100,), dtype=np.float32)

        with pytest.raises(BufferNotAllocatedError):
            _ = buf.host

    def test_device_access_without_allocation_raises(self) -> None:
        """Test that accessing device data before allocation raises."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((100,), dtype=np.float32)

        with pytest.raises(BufferNotAllocatedError):
            _ = buf.device

    def test_host_data_modification(self) -> None:
        """Test modifying host data."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((10,), dtype=np.float32)
        buf.allocate()

        buf.host[:] = np.arange(10, dtype=np.float32)

        np.testing.assert_array_equal(buf.host, np.arange(10, dtype=np.float32))

    def test_dirty_state_tracking(self) -> None:
        """Test dirty state tracking."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((10,), dtype=np.float32)
        buf.allocate()

        buf.mark_host_dirty()
        assert buf.state == BufferState.HOST_DIRTY

        buf.mark_device_dirty()
        assert buf.state == BufferState.DEVICE_DIRTY

    def test_copy_from(self) -> None:
        """Test copying data from numpy array."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((10,), dtype=np.float32)
        buf.allocate()

        source = np.arange(10, dtype=np.float32)
        buf.copy_from(source)

        np.testing.assert_array_equal(buf.host, source)
        assert buf.state == BufferState.HOST_DIRTY

    def test_copy_to(self) -> None:
        """Test copying data to numpy array."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((10,), dtype=np.float32)
        buf.allocate()
        buf.host[:] = np.arange(10, dtype=np.float32)

        dest = np.zeros(10, dtype=np.float32)
        buf.copy_to(dest)

        np.testing.assert_array_equal(dest, np.arange(10, dtype=np.float32))

    def test_zeros(self) -> None:
        """Test filling buffer with zeros."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((10,), dtype=np.float32)
        buf.allocate()
        buf.host[:] = 1.0

        buf.zeros()

        np.testing.assert_array_equal(buf.host, np.zeros(10, dtype=np.float32))

    def test_fill(self) -> None:
        """Test filling buffer with a value."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((10,), dtype=np.float32)
        buf.allocate()

        buf.fill(np.float32(42.0))

        np.testing.assert_array_equal(buf.host, np.full(10, 42.0, dtype=np.float32))

    def test_size_and_nbytes(self) -> None:
        """Test size calculations."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((10, 20), dtype=np.float32)

        assert buf.size == 200
        assert buf.nbytes == 200 * 4  # float32 = 4 bytes

    def test_len(self) -> None:
        """Test __len__ method."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((10, 20), dtype=np.float32)

        assert len(buf) == 10

    def test_getitem_setitem(self) -> None:
        """Test indexing operations."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((10,), dtype=np.float32)
        buf.allocate()

        buf[0] = 42.0
        buf[5] = 100.0

        assert buf[0] == 42.0
        assert buf[5] == 100.0
        assert buf.state == BufferState.HOST_DIRTY

    def test_repr(self) -> None:
        """Test string representation."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((10,), dtype=np.float32)

        repr_str = repr(buf)

        assert "UnifiedBuffer" in repr_str
        assert "10" in repr_str
        assert "float32" in repr_str


class TestUnifiedBufferAsync:
    """Async tests for UnifiedBuffer."""

    @pytest.mark.asyncio
    async def test_ensure_on_host(self) -> None:
        """Test async ensure_on_host method."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((10,), dtype=np.float32)
        buf.allocate()
        buf.host[:] = np.arange(10, dtype=np.float32)

        await buf.ensure_on_host()

        np.testing.assert_array_equal(buf.host, np.arange(10, dtype=np.float32))

    @pytest.mark.asyncio
    async def test_ensure_on_device(self) -> None:
        """Test async ensure_on_device method."""
        buf: UnifiedBuffer[np.float32] = UnifiedBuffer((10,), dtype=np.float32)
        buf.allocate()
        buf.host[:] = np.arange(10, dtype=np.float32)
        buf.mark_host_dirty()

        await buf.ensure_on_device()

        # Device data should be synced (or host if CUDA not available)
        device_data = buf.device
        assert device_data is not None
