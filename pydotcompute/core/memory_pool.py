"""
Memory pooling for reduced allocation overhead.

Wraps CuPy's built-in memory pool with statistics tracking
and configuration options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class PoolStatistics:
    """Statistics for a memory pool."""

    used_bytes: int
    total_bytes: int
    n_free_blocks: int

    @property
    def free_bytes(self) -> int:
        """Get free bytes in the pool."""
        return self.total_bytes - self.used_bytes

    @property
    def utilization(self) -> float:
        """Get pool utilization as a fraction."""
        if self.total_bytes == 0:
            return 0.0
        return self.used_bytes / self.total_bytes

    @property
    def used_mb(self) -> float:
        """Get used memory in MB."""
        return self.used_bytes / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        """Get total memory in MB."""
        return self.total_bytes / (1024 * 1024)


class MemoryPool:
    """
    Memory pool for reduced allocation overhead.

    Wraps CuPy's built-in memory pool for GPU memory and provides
    a consistent interface for CPU fallback.

    Example:
        >>> pool = MemoryPool()
        >>> pool.enable()
        >>> stats = pool.get_stats()
        >>> print(f"Used: {stats.used_mb:.2f} MB")
    """

    _instance: MemoryPool | None = None

    def __new__(cls) -> MemoryPool:
        """Singleton pattern for memory pool."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        initial_size_mb: int = 0,
        max_size_mb: int | None = None,
    ) -> None:
        """
        Initialize the memory pool.

        Args:
            initial_size_mb: Initial pool size in MB (0 for on-demand).
            max_size_mb: Maximum pool size in MB (None for unlimited).
        """
        if getattr(self, "_initialized", False):
            return

        self._initial_size_mb = initial_size_mb
        self._max_size_mb = max_size_mb
        self._cuda_available = False
        self._pool = None
        self._pinned_pool = None
        self._enabled = False

        try:
            import cupy as cp

            self._cuda_available = cp.cuda.runtime.getDeviceCount() > 0
            if self._cuda_available:
                self._pool = cp.cuda.MemoryPool()
                self._pinned_pool = cp.cuda.PinnedMemoryPool()
        except ImportError:
            pass

        self._initialized = True

    def enable(self) -> None:
        """Enable the memory pool as the default allocator."""
        if not self._cuda_available or self._pool is None:
            self._enabled = True
            return

        try:
            import cupy as cp

            cp.cuda.set_allocator(self._pool.malloc)
            if self._pinned_pool is not None:
                cp.cuda.set_pinned_memory_allocator(self._pinned_pool.malloc)
            self._enabled = True
        except ImportError:
            self._enabled = True

    def disable(self) -> None:
        """Disable the memory pool and use default allocator."""
        if not self._cuda_available:
            self._enabled = False
            return

        try:
            import cupy as cp

            cp.cuda.set_allocator(None)
            cp.cuda.set_pinned_memory_allocator(None)
            self._enabled = False
        except ImportError:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if the memory pool is enabled."""
        return self._enabled

    def get_stats(self) -> PoolStatistics:
        """
        Get memory pool statistics.

        Returns:
            PoolStatistics with current usage information.
        """
        if not self._cuda_available or self._pool is None:
            return PoolStatistics(
                used_bytes=0,
                total_bytes=0,
                n_free_blocks=0,
            )

        return PoolStatistics(
            used_bytes=self._pool.used_bytes(),
            total_bytes=self._pool.total_bytes(),
            n_free_blocks=self._pool.n_free_blocks(),
        )

    def get_pinned_stats(self) -> PoolStatistics:
        """
        Get pinned memory pool statistics.

        Returns:
            PoolStatistics with current pinned memory usage.
        """
        if not self._cuda_available or self._pinned_pool is None:
            return PoolStatistics(
                used_bytes=0,
                total_bytes=0,
                n_free_blocks=0,
            )

        return PoolStatistics(
            used_bytes=self._pinned_pool.n_free_blocks(),  # PinnedMemoryPool has different API
            total_bytes=0,
            n_free_blocks=self._pinned_pool.n_free_blocks(),
        )

    def free_all_blocks(self) -> None:
        """Free all unused blocks in the pool."""
        if self._pool is not None:
            self._pool.free_all_blocks()
        if self._pinned_pool is not None:
            self._pinned_pool.free_all_blocks()

    def free_all_free(self) -> None:
        """Free all free (unused) memory in the pool."""
        if self._pool is not None:
            self._pool.free_all_free()

    def set_limit(self, size_mb: int | None) -> None:
        """
        Set the memory limit for the pool.

        Args:
            size_mb: Maximum pool size in MB (None for unlimited).
        """
        if self._pool is None:
            return

        if size_mb is None:
            # Set to device memory size (effectively unlimited)
            try:
                import cupy as cp

                mem_info = cp.cuda.runtime.memGetInfo()
                self._pool.set_limit(size=mem_info[1])
            except Exception:
                pass
        else:
            self._pool.set_limit(size=size_mb * 1024 * 1024)

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"MemoryPool(enabled={self._enabled}, "
            f"used={stats.used_mb:.2f}MB, "
            f"total={stats.total_mb:.2f}MB, "
            f"cuda={self._cuda_available})"
        )


# Convenience functions
def get_memory_pool() -> MemoryPool:
    """Get the global memory pool instance."""
    return MemoryPool()


def configure_memory_pool(
    initial_size_mb: int = 0,
    max_size_mb: int | None = None,
    enable: bool = True,
) -> MemoryPool:
    """
    Configure and optionally enable the global memory pool.

    Args:
        initial_size_mb: Initial pool size in MB.
        max_size_mb: Maximum pool size in MB.
        enable: Whether to enable the pool immediately.

    Returns:
        Configured MemoryPool instance.
    """
    pool = MemoryPool(initial_size_mb=initial_size_mb, max_size_mb=max_size_mb)
    if max_size_mb is not None:
        pool.set_limit(max_size_mb)
    if enable:
        pool.enable()
    return pool
