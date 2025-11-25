"""
Unit tests for message queues.
"""

from __future__ import annotations

import asyncio

import pytest

from pydotcompute.exceptions import QueueFullError, QueueTimeoutError
from pydotcompute.ring_kernels.message import RingKernelMessage
from pydotcompute.ring_kernels.queue import BackpressureStrategy, MessageQueue


class TestMessageQueue:
    """Tests for MessageQueue class."""

    def test_creation(self) -> None:
        """Test queue creation."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(maxsize=100)

        assert queue.maxsize == 100
        assert queue.qsize == 0
        assert queue.empty
        assert not queue.full

    def test_creation_with_options(self) -> None:
        """Test queue creation with options."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(
            maxsize=50,
            backpressure=BackpressureStrategy.REJECT,
            use_priority=False,
            kernel_id="test_kernel",
        )

        assert queue.maxsize == 50
        assert queue._backpressure == BackpressureStrategy.REJECT
        assert not queue._use_priority

    @pytest.mark.asyncio
    async def test_put_and_get(self) -> None:
        """Test basic put and get operations."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(maxsize=100)
        msg = RingKernelMessage(priority=100)

        await queue.put(msg)

        assert queue.qsize == 1

        received = await queue.get()

        assert received.message_id == msg.message_id
        assert queue.qsize == 0

    @pytest.mark.asyncio
    async def test_fifo_ordering(self) -> None:
        """Test FIFO ordering without priority."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(
            maxsize=100,
            use_priority=False,
        )

        msgs = [RingKernelMessage(priority=i) for i in range(5)]
        for msg in msgs:
            await queue.put(msg)

        for expected in msgs:
            received = await queue.get()
            assert received.message_id == expected.message_id

    @pytest.mark.asyncio
    async def test_priority_ordering(self) -> None:
        """Test priority ordering."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(
            maxsize=100,
            use_priority=True,
        )

        # Add messages with different priorities
        low = RingKernelMessage(priority=50)
        medium = RingKernelMessage(priority=100)
        high = RingKernelMessage(priority=200)

        await queue.put(low)
        await queue.put(high)
        await queue.put(medium)

        # Should receive in priority order (high first)
        first = await queue.get()
        second = await queue.get()
        third = await queue.get()

        assert first.priority == 200
        assert second.priority == 100
        assert third.priority == 50

    @pytest.mark.asyncio
    async def test_backpressure_reject(self) -> None:
        """Test reject backpressure strategy."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(
            maxsize=2,
            backpressure=BackpressureStrategy.REJECT,
        )

        await queue.put(RingKernelMessage())
        await queue.put(RingKernelMessage())

        with pytest.raises(QueueFullError):
            await queue.put(RingKernelMessage())

    @pytest.mark.asyncio
    async def test_backpressure_drop_oldest(self) -> None:
        """Test drop_oldest backpressure strategy."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(
            maxsize=2,
            backpressure=BackpressureStrategy.DROP_OLDEST,
        )

        msg1 = RingKernelMessage(priority=1)
        msg2 = RingKernelMessage(priority=2)
        msg3 = RingKernelMessage(priority=3)

        await queue.put(msg1)
        await queue.put(msg2)
        await queue.put(msg3)  # Should drop msg1

        assert queue.qsize == 2

        # Get remaining messages
        received = await queue.get()
        # The oldest (lowest priority in heap) should have been dropped

    @pytest.mark.asyncio
    async def test_timeout_on_get(self) -> None:
        """Test timeout when getting from empty queue."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(maxsize=100)

        with pytest.raises(QueueTimeoutError):
            await queue.get(timeout=0.1)

    @pytest.mark.asyncio
    async def test_get_nowait(self) -> None:
        """Test get_nowait method."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(maxsize=100)

        # Empty queue returns None
        result = queue.get_nowait()
        assert result is None

        # With message returns message
        msg = RingKernelMessage()
        await queue.put(msg)

        result = queue.get_nowait()
        assert result is not None
        assert result.message_id == msg.message_id

    def test_put_nowait(self) -> None:
        """Test put_nowait method."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(maxsize=2)

        assert queue.put_nowait(RingKernelMessage())
        assert queue.put_nowait(RingKernelMessage())

        # Full queue
        queue._backpressure = BackpressureStrategy.REJECT
        assert not queue.put_nowait(RingKernelMessage())

    def test_clear(self) -> None:
        """Test clear method."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(
            maxsize=100,
            use_priority=True,
        )

        for _ in range(10):
            queue.put_nowait(RingKernelMessage())

        assert queue.qsize == 10

        cleared = queue.clear()

        assert cleared == 10
        assert queue.qsize == 0
        assert queue.empty

    def test_stats(self) -> None:
        """Test get_stats method."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(maxsize=100)

        for _ in range(5):
            queue.put_nowait(RingKernelMessage())

        for _ in range(3):
            queue.get_nowait()

        stats = queue.get_stats()

        assert stats["current_size"] == 2
        assert stats["max_size"] == 100
        assert stats["total_enqueued"] == 5
        assert stats["total_dequeued"] == 3

    def test_len(self) -> None:
        """Test __len__ method."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(maxsize=100)

        for i in range(5):
            queue.put_nowait(RingKernelMessage())
            assert len(queue) == i + 1

    def test_repr(self) -> None:
        """Test __repr__ method."""
        queue: MessageQueue[RingKernelMessage] = MessageQueue(
            maxsize=100,
            use_priority=True,
        )

        repr_str = repr(queue)

        assert "MessageQueue" in repr_str
        assert "100" in repr_str
