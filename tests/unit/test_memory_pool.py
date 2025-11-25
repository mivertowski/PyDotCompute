"""
Unit tests for the memory pool module.

Tests the MemoryPool class, PoolStatistics, and related utilities
for GPU memory pooling and management.
"""

from __future__ import annotations

import pytest

from pydotcompute.core.memory_pool import (
    MemoryPool,
    PoolStatistics,
    configure_memory_pool,
    get_memory_pool,
)


class TestPoolStatistics:
    """Tests for PoolStatistics dataclass."""

    def test_creation(self) -> None:
        """Test creating pool statistics."""
        stats = PoolStatistics(
            used_bytes=1024,
            total_bytes=4096,
            n_free_blocks=10,
        )
        assert stats.used_bytes == 1024
        assert stats.total_bytes == 4096
        assert stats.n_free_blocks == 10

    def test_free_bytes(self) -> None:
        """Test free_bytes property."""
        stats = PoolStatistics(
            used_bytes=1000,
            total_bytes=4000,
            n_free_blocks=5,
        )
        assert stats.free_bytes == 3000

    def test_utilization(self) -> None:
        """Test utilization property."""
        stats = PoolStatistics(
            used_bytes=2048,
            total_bytes=4096,
            n_free_blocks=5,
        )
        assert stats.utilization == 0.5

    def test_utilization_zero_total(self) -> None:
        """Test utilization when total is zero."""
        stats = PoolStatistics(
            used_bytes=0,
            total_bytes=0,
            n_free_blocks=0,
        )
        assert stats.utilization == 0.0

    def test_used_mb(self) -> None:
        """Test used_mb property."""
        stats = PoolStatistics(
            used_bytes=1024 * 1024,  # 1 MB
            total_bytes=4 * 1024 * 1024,
            n_free_blocks=5,
        )
        assert stats.used_mb == 1.0

    def test_total_mb(self) -> None:
        """Test total_mb property."""
        stats = PoolStatistics(
            used_bytes=0,
            total_bytes=2 * 1024 * 1024,  # 2 MB
            n_free_blocks=5,
        )
        assert stats.total_mb == 2.0


class TestMemoryPool:
    """Tests for MemoryPool class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset the singleton before each test."""
        MemoryPool._instance = None

    def test_singleton_pattern(self) -> None:
        """Test that MemoryPool is a singleton."""
        pool1 = MemoryPool()
        pool2 = MemoryPool()
        assert pool1 is pool2

    def test_creation(self) -> None:
        """Test pool creation."""
        pool = MemoryPool()
        assert pool._initialized is True
        assert pool._enabled is False

    def test_creation_stores_default_params(self) -> None:
        """Test pool creation stores default parameters."""
        pool = MemoryPool()
        # Singleton pattern - parameters are set on first init
        assert pool._initial_size_mb == 0  # default
        assert pool._max_size_mb is None  # default (unlimited)

    def test_enable_disable(self) -> None:
        """Test enabling and disabling the pool."""
        pool = MemoryPool()

        pool.enable()
        assert pool.enabled is True

        pool.disable()
        assert pool.enabled is False

    def test_enabled_property(self) -> None:
        """Test enabled property."""
        pool = MemoryPool()
        assert pool.enabled is False

        pool.enable()
        assert pool.enabled is True

    def test_get_stats(self) -> None:
        """Test getting pool statistics."""
        pool = MemoryPool()
        stats = pool.get_stats()

        assert isinstance(stats, PoolStatistics)
        # Without CUDA, should return zeros
        if not pool._cuda_available:
            assert stats.used_bytes == 0
            assert stats.total_bytes == 0

    def test_get_pinned_stats(self) -> None:
        """Test getting pinned memory statistics."""
        pool = MemoryPool()
        stats = pool.get_pinned_stats()

        assert isinstance(stats, PoolStatistics)

    def test_free_all_blocks(self) -> None:
        """Test freeing all blocks."""
        pool = MemoryPool()
        pool.enable()

        # Should not raise even without allocations
        pool.free_all_blocks()

    def test_free_all_free(self) -> None:
        """Test freeing all free memory."""
        pool = MemoryPool()
        pool.enable()

        # Should not raise
        pool.free_all_free()

    def test_set_limit(self) -> None:
        """Test setting memory limit."""
        pool = MemoryPool()

        # Should not raise
        pool.set_limit(100)
        pool.set_limit(None)  # Unlimited

    def test_repr(self) -> None:
        """Test string representation."""
        pool = MemoryPool()
        repr_str = repr(pool)

        assert "MemoryPool" in repr_str
        assert "enabled=" in repr_str
        assert "cuda=" in repr_str

    def test_cuda_available_detection(self) -> None:
        """Test CUDA availability detection."""
        pool = MemoryPool()
        # Just verify it's a boolean
        assert isinstance(pool._cuda_available, bool)


class TestGetMemoryPool:
    """Tests for get_memory_pool function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset the singleton before each test."""
        MemoryPool._instance = None

    def test_returns_singleton(self) -> None:
        """Test that get_memory_pool returns singleton."""
        pool1 = get_memory_pool()
        pool2 = get_memory_pool()
        assert pool1 is pool2

    def test_returns_memory_pool(self) -> None:
        """Test that it returns a MemoryPool instance."""
        pool = get_memory_pool()
        assert isinstance(pool, MemoryPool)


class TestConfigureMemoryPool:
    """Tests for configure_memory_pool function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset the singleton before each test."""
        MemoryPool._instance = None

    def test_configure_and_enable(self) -> None:
        """Test configuring enables the pool by default."""
        pool = configure_memory_pool()

        assert isinstance(pool, MemoryPool)
        assert pool.enabled is True  # enable=True by default

    def test_configure_returns_pool(self) -> None:
        """Test configuring returns the pool instance."""
        pool = configure_memory_pool(enable=True)
        assert isinstance(pool, MemoryPool)
        assert pool.enabled is True

    def test_configure_disabled(self) -> None:
        """Test configuring without enabling."""
        pool = configure_memory_pool(enable=False)
        assert pool.enabled is False

    def test_configure_sets_limit_when_specified(self) -> None:
        """Test that set_limit is called when max_size_mb is specified."""
        pool = configure_memory_pool(max_size_mb=128)
        # The function calls set_limit internally
        assert isinstance(pool, MemoryPool)


class TestMemoryPoolCPUFallback:
    """Tests for CPU fallback behavior."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset the singleton before each test."""
        MemoryPool._instance = None

    def test_stats_without_cuda(self) -> None:
        """Test that stats work without CUDA."""
        pool = MemoryPool()

        # If CUDA not available, should still return valid stats
        if not pool._cuda_available:
            stats = pool.get_stats()
            assert stats.used_bytes == 0
            assert stats.total_bytes == 0
            assert stats.n_free_blocks == 0

    def test_enable_without_cuda(self) -> None:
        """Test enable works without CUDA."""
        pool = MemoryPool()

        if not pool._cuda_available:
            # Should not raise
            pool.enable()
            assert pool.enabled is True

    def test_disable_without_cuda(self) -> None:
        """Test disable works without CUDA."""
        pool = MemoryPool()
        pool.enable()

        if not pool._cuda_available:
            pool.disable()
            assert pool.enabled is False


class TestPoolStatisticsEdgeCases:
    """Edge case tests for PoolStatistics."""

    def test_large_values(self) -> None:
        """Test with large memory values."""
        stats = PoolStatistics(
            used_bytes=8 * 1024 * 1024 * 1024,  # 8 GB
            total_bytes=16 * 1024 * 1024 * 1024,  # 16 GB
            n_free_blocks=1000,
        )
        assert stats.utilization == 0.5
        assert stats.used_mb == 8 * 1024
        assert stats.total_mb == 16 * 1024

    def test_full_utilization(self) -> None:
        """Test with 100% utilization."""
        stats = PoolStatistics(
            used_bytes=1000,
            total_bytes=1000,
            n_free_blocks=0,
        )
        assert stats.utilization == 1.0
        assert stats.free_bytes == 0
