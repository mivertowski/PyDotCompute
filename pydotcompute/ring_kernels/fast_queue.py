"""
High-Performance Message Queue with Ring Buffer and Priority Banding.

Optimized for minimal latency with:
- Pre-allocated ring buffer (no allocation on hot path)
- Priority banding with O(1) selection (vs O(log n) heap)
- Lock-free reads when possible
- Batch operations for throughput

Target: <10μs enqueue/dequeue latency.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Generic, TypeVar

from pydotcompute.exceptions import QueueFullError, QueueTimeoutError
from pydotcompute.ring_kernels.message import RingKernelMessage

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=RingKernelMessage)


class PriorityBand(IntEnum):
    """Priority bands for O(1) priority selection."""

    SYSTEM = 0  # Priority 224-255: System/control messages
    HIGH = 1  # Priority 160-223: High priority
    NORMAL = 2  # Priority 64-159: Normal priority
    LOW = 3  # Priority 0-63: Low priority/bulk


def get_priority_band(priority: int) -> PriorityBand:
    """Map priority value (0-255) to band."""
    if priority >= 224:
        return PriorityBand.SYSTEM
    elif priority >= 160:
        return PriorityBand.HIGH
    elif priority >= 64:
        return PriorityBand.NORMAL
    else:
        return PriorityBand.LOW


@dataclass
class QueueStats:
    """Statistics for the fast queue."""

    total_enqueued: int = 0
    total_dequeued: int = 0
    total_dropped: int = 0
    current_size: int = 0
    peak_size: int = 0
    band_counts: dict[str, int] | None = None


class FastMessageQueue(Generic[T]):
    """
    High-performance message queue with priority banding.

    Uses 4 priority bands instead of a heap for O(1) priority selection.
    Each band is a simple deque for FIFO ordering within the band.

    Performance characteristics:
    - Enqueue: O(1) - append to appropriate band
    - Dequeue: O(1) - pop from highest non-empty band
    - No heap operations, no sorting

    Example:
        >>> queue: FastMessageQueue[MyMessage] = FastMessageQueue(maxsize=10000)
        >>> await queue.put(MyMessage(priority=200))  # Goes to HIGH band
        >>> msg = await queue.get()  # Returns from highest priority band
    """

    __slots__ = (
        "_maxsize",
        "_bands",
        "_size",
        "_condition",
        "_stats",
        "_kernel_id",
        "_serialize",
    )

    def __init__(
        self,
        maxsize: int = 4096,
        *,
        kernel_id: str = "",
        serialize: bool = False,
    ) -> None:
        """
        Initialize the fast queue.

        Args:
            maxsize: Maximum total messages across all bands.
            kernel_id: ID of the kernel this queue belongs to.
            serialize: If True, serialize messages before storing (for IPC).
                       If False (default), store direct object references
                       for zero-copy in-process message passing.
        """
        self._maxsize = maxsize
        self._kernel_id = kernel_id
        self._serialize = serialize

        # 4 priority bands - each is a deque for O(1) append/pop
        self._bands: tuple[deque[T], deque[T], deque[T], deque[T]] = (
            deque(),  # SYSTEM
            deque(),  # HIGH
            deque(),  # NORMAL
            deque(),  # LOW
        )

        self._size = 0
        self._condition = asyncio.Condition()

        # Statistics
        self._stats = QueueStats()

    @property
    def maxsize(self) -> int:
        """Get the maximum queue size."""
        return self._maxsize

    @property
    def qsize(self) -> int:
        """Get the current queue size."""
        return self._size

    @property
    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self._size == 0

    @property
    def full(self) -> bool:
        """Check if the queue is full."""
        return self._size >= self._maxsize

    async def put(
        self,
        message: T,
        *,
        timeout: float | None = None,
    ) -> bool:
        """
        Put a message in the queue.

        O(1) operation - appends to the appropriate priority band.

        Args:
            message: Message to enqueue.
            timeout: Maximum time to wait if full (None for no timeout).

        Returns:
            True if message was enqueued.

        Raises:
            QueueFullError: If queue is full and timeout expires.
        """
        async with self._condition:
            # Wait for space if full
            if self.full:
                if timeout == 0:
                    raise QueueFullError(self._kernel_id, self._maxsize)

                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(lambda: not self.full),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    raise QueueFullError(self._kernel_id, self._maxsize) from None

            # Get priority band - O(1)
            priority = getattr(message, "priority", 128)
            band = get_priority_band(priority)

            # Conditionally serialize (for IPC) or store direct reference (zero-copy)
            if self._serialize and hasattr(message, "serialize"):
                stored = message.serialize()
            else:
                stored = message

            # Append to band - O(1)
            self._bands[band].append(stored)
            self._size += 1

            # Update stats
            self._stats.total_enqueued += 1
            if self._size > self._stats.peak_size:
                self._stats.peak_size = self._size

            # Notify waiters
            self._condition.notify()
            return True

    async def get(
        self,
        *,
        timeout: float | None = None,
    ) -> T:
        """
        Get a message from the queue.

        O(1) operation - pops from highest non-empty priority band.

        Args:
            timeout: Maximum time to wait if empty (None for no timeout).

        Returns:
            Message from the queue.

        Raises:
            QueueTimeoutError: If timeout expires.
        """
        async with self._condition:
            # Wait for message if empty
            if self.empty:
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(lambda: not self.empty),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    raise QueueTimeoutError(
                        self._kernel_id, timeout or 0, "receive"
                    ) from None

            # Find highest priority non-empty band - O(1) with 4 bands
            for band in self._bands:
                if band:
                    message = band.popleft()
                    self._size -= 1
                    self._stats.total_dequeued += 1

                    # Notify waiters (space available)
                    self._condition.notify()
                    return message

            # Should never reach here if size > 0
            raise RuntimeError("Queue inconsistency: size > 0 but all bands empty")

    def put_nowait(self, message: T) -> bool:
        """
        Put a message without waiting.

        Args:
            message: Message to enqueue.

        Returns:
            True if successful, False if queue is full.
        """
        if self.full:
            return False

        priority = getattr(message, "priority", 128)
        band = get_priority_band(priority)

        # Conditionally serialize (for IPC) or store direct reference (zero-copy)
        if self._serialize and hasattr(message, "serialize"):
            stored = message.serialize()
        else:
            stored = message

        self._bands[band].append(stored)
        self._size += 1
        self._stats.total_enqueued += 1

        if self._size > self._stats.peak_size:
            self._stats.peak_size = self._size

        return True

    def get_nowait(self) -> T | None:
        """
        Get a message without waiting.

        Returns:
            Message or None if queue is empty.
        """
        if self.empty:
            return None

        for band in self._bands:
            if band:
                message = band.popleft()
                self._size -= 1
                self._stats.total_dequeued += 1
                return message

        return None

    def put_batch(self, messages: list[T]) -> int:
        """
        Put multiple messages at once.

        More efficient than individual puts for bulk operations.

        Args:
            messages: List of messages to enqueue.

        Returns:
            Number of messages successfully enqueued.
        """
        count = 0
        space = self._maxsize - self._size

        for message in messages[:space]:
            priority = getattr(message, "priority", 128)
            band = get_priority_band(priority)

            # Conditionally serialize (for IPC) or store direct reference (zero-copy)
            if self._serialize and hasattr(message, "serialize"):
                stored = message.serialize()
            else:
                stored = message

            self._bands[band].append(stored)
            self._size += 1
            count += 1

        self._stats.total_enqueued += count
        if self._size > self._stats.peak_size:
            self._stats.peak_size = self._size

        return count

    def get_batch(self, max_count: int) -> list[T]:
        """
        Get multiple messages at once.

        More efficient than individual gets for bulk operations.

        Args:
            max_count: Maximum number of messages to retrieve.

        Returns:
            List of messages (may be fewer than max_count).
        """
        messages: list[T] = []

        for band in self._bands:
            while band and len(messages) < max_count:
                messages.append(band.popleft())
                self._size -= 1

            if len(messages) >= max_count:
                break

        self._stats.total_dequeued += len(messages)
        return messages

    def clear(self) -> int:
        """
        Clear all messages from the queue.

        Returns:
            Number of messages cleared.
        """
        count = self._size
        for band in self._bands:
            band.clear()
        self._size = 0
        return count

    def get_stats(self) -> dict[str, int | dict[str, int]]:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue statistics including per-band counts.
        """
        return {
            "current_size": self._size,
            "max_size": self._maxsize,
            "peak_size": self._stats.peak_size,
            "total_enqueued": self._stats.total_enqueued,
            "total_dequeued": self._stats.total_dequeued,
            "total_dropped": self._stats.total_dropped,
            "band_counts": {
                "system": len(self._bands[PriorityBand.SYSTEM]),
                "high": len(self._bands[PriorityBand.HIGH]),
                "normal": len(self._bands[PriorityBand.NORMAL]),
                "low": len(self._bands[PriorityBand.LOW]),
            },
        }

    def __len__(self) -> int:
        """Get queue size."""
        return self._size

    def __repr__(self) -> str:
        """String representation."""
        return f"FastMessageQueue(size={self._size}/{self._maxsize})"


class RingBuffer(Generic[T]):
    """
    Fixed-size ring buffer for pre-allocated message slots.

    Provides O(1) enqueue/dequeue with no allocation on hot path.
    Use when message rate is very high and consistent.

    Note: This is a simpler queue without priority support.
    For priority support, use FastMessageQueue.
    """

    __slots__ = ("_buffer", "_capacity", "_head", "_tail", "_size", "_condition")

    def __init__(self, capacity: int = 4096) -> None:
        """
        Initialize the ring buffer.

        Args:
            capacity: Fixed buffer size (must be power of 2 for optimal performance).
        """
        # Round up to power of 2 for fast modulo
        self._capacity = 1 << (capacity - 1).bit_length()
        self._buffer: list[T | None] = [None] * self._capacity
        self._head = 0  # Next read position
        self._tail = 0  # Next write position
        self._size = 0
        self._condition = asyncio.Condition()

    @property
    def capacity(self) -> int:
        """Get buffer capacity."""
        return self._capacity

    @property
    def size(self) -> int:
        """Get current size."""
        return self._size

    @property
    def empty(self) -> bool:
        """Check if empty."""
        return self._size == 0

    @property
    def full(self) -> bool:
        """Check if full."""
        return self._size >= self._capacity

    async def put(self, item: T, *, timeout: float | None = None) -> bool:
        """Put an item in the buffer."""
        async with self._condition:
            if self.full:
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(lambda: not self.full),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    return False

            # Write to tail position
            self._buffer[self._tail] = item
            self._tail = (self._tail + 1) & (self._capacity - 1)  # Fast modulo
            self._size += 1

            self._condition.notify()
            return True

    async def get(self, *, timeout: float | None = None) -> T | None:
        """Get an item from the buffer."""
        async with self._condition:
            if self.empty:
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(lambda: not self.empty),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    return None

            # Read from head position
            item = self._buffer[self._head]
            self._buffer[self._head] = None  # Clear slot
            self._head = (self._head + 1) & (self._capacity - 1)  # Fast modulo
            self._size -= 1

            self._condition.notify()
            return item

    def put_nowait(self, item: T) -> bool:
        """Put without waiting."""
        if self.full:
            return False

        self._buffer[self._tail] = item
        self._tail = (self._tail + 1) & (self._capacity - 1)
        self._size += 1
        return True

    def get_nowait(self) -> T | None:
        """Get without waiting."""
        if self.empty:
            return None

        item = self._buffer[self._head]
        self._buffer[self._head] = None
        self._head = (self._head + 1) & (self._capacity - 1)
        self._size -= 1
        return item

    def clear(self) -> int:
        """Clear the buffer."""
        count = self._size
        for i in range(self._capacity):
            self._buffer[i] = None
        self._head = 0
        self._tail = 0
        self._size = 0
        return count

    def __len__(self) -> int:
        """Get size."""
        return self._size

    def __repr__(self) -> str:
        """String representation."""
        return f"RingBuffer(size={self._size}/{self._capacity})"
