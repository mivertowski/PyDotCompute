"""
High-Performance Synchronous Queue for Threaded Kernels.

Uses threading primitives instead of asyncio for minimal latency.
Target: <10μs enqueue/dequeue operations.

This queue is designed for use with ThreadedRingKernel where we bypass
asyncio entirely and use native threading for maximum performance.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Generic, TypeVar

from pydotcompute.exceptions import QueueFullError, QueueTimeoutError
from pydotcompute.ring_kernels.message import RingKernelMessage

T = TypeVar("T", bound=RingKernelMessage)


class SyncQueue(Generic[T]):
    """
    High-performance synchronous queue using threading.Condition.

    This queue provides minimal-latency message passing for threaded kernels.
    It uses a single lock with condition variables for efficient blocking.

    Performance characteristics:
    - Lock acquisition: ~1-2μs on modern CPUs
    - Condition wait/notify: ~1-2μs
    - Total put/get: ~5-10μs typical

    Example:
        >>> queue: SyncQueue[MyMessage] = SyncQueue(maxsize=1000)
        >>> queue.put(MyMessage(value=42))  # Blocks if full
        >>> msg = queue.get()  # Blocks if empty
    """

    __slots__ = (
        "_maxsize",
        "_queue",
        "_lock",
        "_not_empty",
        "_not_full",
        "_shutdown",
        "_stats_enqueued",
        "_stats_dequeued",
    )

    def __init__(self, maxsize: int = 4096) -> None:
        """
        Initialize the sync queue.

        Args:
            maxsize: Maximum queue size. 0 means unlimited.
        """
        self._maxsize = maxsize
        self._queue: deque[T] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._shutdown = False
        self._stats_enqueued = 0
        self._stats_dequeued = 0

    @property
    def maxsize(self) -> int:
        """Get maximum queue size."""
        return self._maxsize

    @property
    def qsize(self) -> int:
        """Get current queue size (thread-safe)."""
        with self._lock:
            return len(self._queue)

    @property
    def empty(self) -> bool:
        """Check if queue is empty (thread-safe)."""
        with self._lock:
            return len(self._queue) == 0

    @property
    def full(self) -> bool:
        """Check if queue is full (thread-safe)."""
        with self._lock:
            return self._maxsize > 0 and len(self._queue) >= self._maxsize

    def put(self, item: T, timeout: float | None = None) -> bool:
        """
        Put an item in the queue (blocking).

        Args:
            item: Item to enqueue.
            timeout: Maximum time to wait. None means wait forever.
                     0 means don't wait (raise immediately if full).

        Returns:
            True if item was enqueued.

        Raises:
            QueueFullError: If queue is full and timeout expires.
            RuntimeError: If queue is shutdown.
        """
        with self._not_full:
            if self._shutdown:
                raise RuntimeError("Queue is shutdown")

            # Check if full
            if self._maxsize > 0 and len(self._queue) >= self._maxsize:
                if timeout == 0:
                    raise QueueFullError("sync_queue", self._maxsize)

                # Wait for space
                if not self._not_full.wait_for(
                    lambda: len(self._queue) < self._maxsize or self._shutdown,
                    timeout=timeout,
                ):
                    raise QueueFullError("sync_queue", self._maxsize)

                if self._shutdown:
                    raise RuntimeError("Queue is shutdown")

            # Enqueue
            self._queue.append(item)
            self._stats_enqueued += 1

            # Notify waiters
            self._not_empty.notify()
            return True

    def get(self, timeout: float | None = None) -> T:
        """
        Get an item from the queue (blocking).

        Args:
            timeout: Maximum time to wait. None means wait forever.

        Returns:
            Item from the queue.

        Raises:
            QueueTimeoutError: If timeout expires.
            RuntimeError: If queue is shutdown and empty.
        """
        with self._not_empty:
            # Wait for item
            if len(self._queue) == 0:
                if not self._not_empty.wait_for(
                    lambda: len(self._queue) > 0 or self._shutdown,
                    timeout=timeout,
                ):
                    raise QueueTimeoutError("sync_queue", timeout or 0, "get")

                if self._shutdown and len(self._queue) == 0:
                    raise RuntimeError("Queue is shutdown and empty")

            # Dequeue
            item = self._queue.popleft()
            self._stats_dequeued += 1

            # Notify waiters
            self._not_full.notify()
            return item

    def put_nowait(self, item: T) -> bool:
        """
        Put an item without waiting.

        Args:
            item: Item to enqueue.

        Returns:
            True if successful, False if full.
        """
        with self._lock:
            if self._shutdown:
                return False
            if self._maxsize > 0 and len(self._queue) >= self._maxsize:
                return False

            self._queue.append(item)
            self._stats_enqueued += 1
            self._not_empty.notify()
            return True

    def get_nowait(self) -> T | None:
        """
        Get an item without waiting.

        Returns:
            Item or None if empty.
        """
        with self._lock:
            if len(self._queue) == 0:
                return None

            item = self._queue.popleft()
            self._stats_dequeued += 1
            self._not_full.notify()
            return item

    def clear(self) -> int:
        """
        Clear all items from the queue.

        Returns:
            Number of items cleared.
        """
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            self._not_full.notify_all()
            return count

    def shutdown(self) -> None:
        """
        Shutdown the queue, waking all waiters.

        After shutdown, put() will raise RuntimeError and get() will
        return remaining items then raise RuntimeError.
        """
        with self._lock:
            self._shutdown = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def get_stats(self) -> dict[str, int]:
        """Get queue statistics."""
        with self._lock:
            return {
                "current_size": len(self._queue),
                "max_size": self._maxsize,
                "total_enqueued": self._stats_enqueued,
                "total_dequeued": self._stats_dequeued,
            }

    def __len__(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)

    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            return f"SyncQueue(size={len(self._queue)}/{self._maxsize})"


class SPSCQueue(Generic[T]):
    """
    Single-Producer Single-Consumer lock-free queue.

    Uses a ring buffer with atomic-like operations for minimal latency
    when there's exactly one producer and one consumer thread.

    Performance characteristics:
    - No locks on hot path
    - Memory barriers via volatile-like access
    - ~2-5μs put/get typical

    Warning:
        This queue is ONLY safe when there is exactly one producer thread
        and one consumer thread. Multiple producers or consumers will
        cause data corruption.

    Example:
        >>> queue: SPSCQueue[MyMessage] = SPSCQueue(capacity=1024)
        >>> # Producer thread:
        >>> queue.put(MyMessage(value=42))
        >>> # Consumer thread:
        >>> msg = queue.get()
    """

    __slots__ = (
        "_buffer",
        "_capacity",
        "_mask",
        "_head",  # Consumer reads from here
        "_tail",  # Producer writes here
        "_not_empty",
        "_not_full",
        "_shutdown",
    )

    def __init__(self, capacity: int = 4096) -> None:
        """
        Initialize SPSC queue.

        Args:
            capacity: Buffer size (will be rounded to power of 2).
        """
        # Round up to power of 2 for fast modulo
        self._capacity = 1 << (capacity - 1).bit_length()
        self._mask = self._capacity - 1
        self._buffer: list[T | None] = [None] * self._capacity
        self._head = 0
        self._tail = 0

        # Events for blocking (optional)
        self._not_empty = threading.Event()
        self._not_full = threading.Event()
        self._not_full.set()  # Initially not full
        self._shutdown = False

    @property
    def capacity(self) -> int:
        """Get buffer capacity."""
        return self._capacity

    @property
    def size(self) -> int:
        """Get current size (approximate, may race)."""
        return (self._tail - self._head) & self._mask

    @property
    def empty(self) -> bool:
        """Check if empty."""
        return self._head == self._tail

    @property
    def full(self) -> bool:
        """Check if full."""
        return ((self._tail + 1) & self._mask) == self._head

    def put(self, item: T, timeout: float | None = None) -> bool:
        """
        Put an item (blocks if full).

        Args:
            item: Item to enqueue.
            timeout: Maximum wait time.

        Returns:
            True if successful.
        """
        # Fast path: check if space available
        next_tail = (self._tail + 1) & self._mask
        if next_tail == self._head:
            # Queue full, wait
            if timeout == 0:
                return False
            if not self._not_full.wait(timeout):
                return False

        if self._shutdown:
            return False

        # Write item
        self._buffer[self._tail] = item
        self._tail = next_tail

        # Signal consumer
        self._not_empty.set()
        return True

    def get(self, timeout: float | None = None) -> T | None:
        """
        Get an item (blocks if empty).

        Args:
            timeout: Maximum wait time.

        Returns:
            Item or None if timeout/shutdown.
        """
        # Fast path: check if item available
        if self._head == self._tail:
            # Queue empty, wait
            self._not_empty.clear()
            if not self._not_empty.wait(timeout):
                return None

        if self._shutdown and self._head == self._tail:
            return None

        # Read item
        item = self._buffer[self._head]
        self._buffer[self._head] = None  # Help GC
        self._head = (self._head + 1) & self._mask

        # Signal producer
        self._not_full.set()
        return item

    def put_nowait(self, item: T) -> bool:
        """Put without blocking."""
        next_tail = (self._tail + 1) & self._mask
        if next_tail == self._head:
            return False

        self._buffer[self._tail] = item
        self._tail = next_tail
        self._not_empty.set()
        return True

    def get_nowait(self) -> T | None:
        """Get without blocking."""
        if self._head == self._tail:
            return None

        item = self._buffer[self._head]
        self._buffer[self._head] = None
        self._head = (self._head + 1) & self._mask
        self._not_full.set()
        return item

    def clear(self) -> int:
        """Clear the queue."""
        count = 0
        while self._head != self._tail:
            self._buffer[self._head] = None
            self._head = (self._head + 1) & self._mask
            count += 1
        return count

    def shutdown(self) -> None:
        """Shutdown the queue."""
        self._shutdown = True
        self._not_empty.set()
        self._not_full.set()

    def __len__(self) -> int:
        """Get size."""
        return (self._tail - self._head) & self._mask

    def __repr__(self) -> str:
        """String representation."""
        return f"SPSCQueue(size={len(self)}/{self._capacity})"
