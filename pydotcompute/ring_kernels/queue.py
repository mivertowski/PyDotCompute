"""
Async message queues for ring kernels.

Provides priority-aware message queuing with backpressure support.
"""

from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Generic, TypeVar

from pydotcompute.exceptions import QueueFullError, QueueTimeoutError
from pydotcompute.ring_kernels.message import RingKernelMessage

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=RingKernelMessage)


class BackpressureStrategy(Enum):
    """Strategy for handling queue overflow."""

    BLOCK = auto()  # Block until space is available
    REJECT = auto()  # Reject new messages when full
    DROP_OLDEST = auto()  # Drop oldest messages to make room


@dataclass(order=True)
class PriorityItem(Generic[T]):
    """Wrapper for priority queue items."""

    priority: int
    sequence: int  # For FIFO ordering within same priority
    message: T = field(compare=False)


class MessageQueue(Generic[T]):
    """
    Async message queue with priority support.

    Provides a high-performance message queue for ring kernel communication.
    Supports priority ordering and configurable backpressure strategies.

    Example:
        >>> queue: MessageQueue[MyMessage] = MessageQueue(maxsize=1000)
        >>> await queue.put(MyMessage(value=42))
        >>> msg = await queue.get(timeout=1.0)
    """

    def __init__(
        self,
        maxsize: int = 4096,
        *,
        backpressure: BackpressureStrategy = BackpressureStrategy.BLOCK,
        use_priority: bool = True,
        kernel_id: str = "",
    ) -> None:
        """
        Initialize the message queue.

        Args:
            maxsize: Maximum number of messages in the queue.
            backpressure: Strategy for handling overflow.
            use_priority: Whether to use priority ordering.
            kernel_id: ID of the kernel this queue belongs to.
        """
        self._maxsize = maxsize
        self._backpressure = backpressure
        self._use_priority = use_priority
        self._kernel_id = kernel_id

        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self._priority_heap: list[PriorityItem[T]] = []
        self._sequence = 0
        self._lock = asyncio.Lock()

        # Statistics
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._total_dropped = 0

    @property
    def maxsize(self) -> int:
        """Get the maximum queue size."""
        return self._maxsize

    @property
    def qsize(self) -> int:
        """Get the current queue size."""
        if self._use_priority:
            return len(self._priority_heap)
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self.qsize == 0

    @property
    def full(self) -> bool:
        """Check if the queue is full."""
        return self.qsize >= self._maxsize

    async def put(
        self,
        message: T,
        *,
        timeout: float | None = None,
    ) -> bool:
        """
        Put a message in the queue.

        Args:
            message: Message to enqueue.
            timeout: Maximum time to wait (None for no timeout).

        Returns:
            True if message was enqueued, False if rejected.

        Raises:
            QueueFullError: If queue is full and strategy is REJECT.
            QueueTimeoutError: If timeout expires.
        """
        if self._use_priority:
            return await self._put_priority(message, timeout)

        # Handle backpressure
        if self.full:
            if self._backpressure == BackpressureStrategy.REJECT:
                raise QueueFullError(self._kernel_id, self._maxsize)
            elif self._backpressure == BackpressureStrategy.DROP_OLDEST:
                try:
                    self._queue.get_nowait()
                    self._total_dropped += 1
                except asyncio.QueueEmpty:
                    pass

        try:
            if timeout is not None:
                await asyncio.wait_for(
                    self._queue.put(message),
                    timeout=timeout,
                )
            else:
                await self._queue.put(message)

            self._total_enqueued += 1
            return True

        except asyncio.TimeoutError:
            raise QueueTimeoutError(self._kernel_id, timeout or 0, "send") from None

    async def _put_priority(
        self,
        message: T,
        timeout: float | None,
    ) -> bool:
        """Put a message in the priority queue."""
        async with self._lock:
            if self.full:
                if self._backpressure == BackpressureStrategy.REJECT:
                    raise QueueFullError(self._kernel_id, self._maxsize)
                elif self._backpressure == BackpressureStrategy.DROP_OLDEST:
                    if self._priority_heap:
                        heapq.heappop(self._priority_heap)
                        self._total_dropped += 1

            # Use negative priority for max-heap behavior (higher priority first)
            priority = -(getattr(message, "priority", 128))
            item = PriorityItem(
                priority=priority,
                sequence=self._sequence,
                message=message,
            )
            self._sequence += 1

            heapq.heappush(self._priority_heap, item)
            self._total_enqueued += 1
            return True

    async def get(
        self,
        *,
        timeout: float | None = None,
    ) -> T:
        """
        Get a message from the queue.

        Args:
            timeout: Maximum time to wait (None for no timeout).

        Returns:
            Message from the queue.

        Raises:
            QueueTimeoutError: If timeout expires.
        """
        if self._use_priority:
            return await self._get_priority(timeout)

        try:
            if timeout is not None:
                message = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout,
                )
            else:
                message = await self._queue.get()

            self._total_dequeued += 1
            return message

        except asyncio.TimeoutError:
            raise QueueTimeoutError(self._kernel_id, timeout or 0, "receive") from None

    async def _get_priority(self, timeout: float | None) -> T:
        """Get a message from the priority queue."""
        start_time = asyncio.get_event_loop().time()

        while True:
            async with self._lock:
                if self._priority_heap:
                    item = heapq.heappop(self._priority_heap)
                    self._total_dequeued += 1
                    return item.message

            # Check timeout
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    raise QueueTimeoutError(self._kernel_id, timeout, "receive")

            # Brief sleep before retry
            await asyncio.sleep(0.001)

    def get_nowait(self) -> T | None:
        """
        Get a message without waiting.

        Returns:
            Message or None if queue is empty.
        """
        if self._use_priority:
            if self._priority_heap:
                item = heapq.heappop(self._priority_heap)
                self._total_dequeued += 1
                return item.message
            return None

        try:
            message = self._queue.get_nowait()
            self._total_dequeued += 1
            return message
        except asyncio.QueueEmpty:
            return None

    def put_nowait(self, message: T) -> bool:
        """
        Put a message without waiting.

        Args:
            message: Message to enqueue.

        Returns:
            True if successful, False if queue is full.
        """
        if self.full:
            if self._backpressure == BackpressureStrategy.REJECT:
                return False
            elif self._backpressure == BackpressureStrategy.DROP_OLDEST:
                self.get_nowait()
                self._total_dropped += 1

        if self._use_priority:
            priority = -(getattr(message, "priority", 128))
            item = PriorityItem(
                priority=priority,
                sequence=self._sequence,
                message=message,
            )
            self._sequence += 1
            heapq.heappush(self._priority_heap, item)
        else:
            try:
                self._queue.put_nowait(message)
            except asyncio.QueueFull:
                return False

        self._total_enqueued += 1
        return True

    def clear(self) -> int:
        """
        Clear all messages from the queue.

        Returns:
            Number of messages cleared.
        """
        if self._use_priority:
            count = len(self._priority_heap)
            self._priority_heap.clear()
            return count

        count = 0
        while True:
            try:
                self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        return count

    def get_stats(self) -> dict[str, int]:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue statistics.
        """
        return {
            "current_size": self.qsize,
            "max_size": self._maxsize,
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_dropped": self._total_dropped,
        }

    def __len__(self) -> int:
        """Get queue size."""
        return self.qsize

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MessageQueue(size={self.qsize}/{self._maxsize}, "
            f"priority={self._use_priority}, "
            f"backpressure={self._backpressure.name})"
        )
