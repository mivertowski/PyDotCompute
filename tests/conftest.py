"""
Pytest configuration and shared fixtures.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncGenerator, Generator

import numpy as np
import pytest

from pydotcompute.core.accelerator import Accelerator
from pydotcompute.core.memory_pool import MemoryPool
from pydotcompute.core.unified_buffer import UnifiedBuffer
from pydotcompute.ring_kernels.message import RingKernelMessage, message
from pydotcompute.ring_kernels.queue import MessageQueue
from pydotcompute.ring_kernels.runtime import RingKernelRuntime

if TYPE_CHECKING:
    pass


# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def accelerator() -> Generator[Accelerator, None, None]:
    """Provide a fresh accelerator instance."""
    # Reset singleton for testing
    Accelerator._instance = None
    Accelerator._initialized = False
    acc = Accelerator()
    yield acc


@pytest.fixture
def memory_pool() -> Generator[MemoryPool, None, None]:
    """Provide a fresh memory pool instance."""
    # Reset singleton
    MemoryPool._instance = None
    pool = MemoryPool()
    yield pool
    pool.free_all_blocks()


@pytest.fixture
def unified_buffer() -> Generator[UnifiedBuffer[np.float32], None, None]:
    """Provide a test unified buffer."""
    buffer: UnifiedBuffer[np.float32] = UnifiedBuffer((100,), dtype=np.float32)
    buffer.allocate()
    yield buffer
    buffer.free()


@pytest.fixture
async def runtime() -> AsyncGenerator[RingKernelRuntime, None]:
    """Provide a clean runtime for each test."""
    async with RingKernelRuntime() as rt:
        yield rt


@pytest.fixture
def message_queue() -> Generator[MessageQueue[RingKernelMessage], None, None]:
    """Provide a test message queue."""
    queue: MessageQueue[RingKernelMessage] = MessageQueue(maxsize=100)
    yield queue
    queue.clear()


@pytest.fixture
def sample_message() -> RingKernelMessage:
    """Provide a sample message for testing."""
    return RingKernelMessage(priority=100)


# Test message classes
@message
class TestRequest:
    """Test request message."""

    value: float = 0.0


@message
class TestResponse:
    """Test response message."""

    result: float = 0.0


@pytest.fixture
def test_request() -> TestRequest:
    """Provide a test request."""
    return TestRequest(value=42.0)


@pytest.fixture
def test_response() -> TestResponse:
    """Provide a test response."""
    return TestResponse(result=84.0)


# Markers for CUDA tests
def pytest_configure(config: pytest.Config) -> None:
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip CUDA tests if CUDA is not available."""
    cuda_available = False
    try:
        import cupy as cp

        cuda_available = cp.cuda.runtime.getDeviceCount() > 0
    except ImportError:
        pass

    if not cuda_available:
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)
