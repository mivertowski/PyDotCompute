"""
Unit tests for the kernel cache module.

Tests the KernelCache class, CacheEntry dataclass, and related utilities
for persistent kernel caching with LRU eviction.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from pydotcompute.compilation.cache import (
    CacheEntry,
    KernelCache,
    get_kernel_cache,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_creation(self) -> None:
        """Test creating a cache entry."""
        entry = CacheEntry(
            kernel_name="test_kernel",
            source_hash="abc123",
            signature="void(float32[:], float32[:])",
        )
        assert entry.kernel_name == "test_kernel"
        assert entry.source_hash == "abc123"
        assert entry.signature == "void(float32[:], float32[:])"
        assert entry.ptx_path is None
        assert entry.metadata == {}
        assert entry.hit_count == 0

    def test_creation_with_metadata(self) -> None:
        """Test creating a cache entry with metadata."""
        entry = CacheEntry(
            kernel_name="my_kernel",
            source_hash="def456",
            signature=None,
            metadata={"version": "1.0", "author": "test"},
            created_timestamp=1000.0,
            last_accessed=1001.0,
            hit_count=5,
        )
        assert entry.metadata == {"version": "1.0", "author": "test"}
        assert entry.created_timestamp == 1000.0
        assert entry.last_accessed == 1001.0
        assert entry.hit_count == 5

    def test_ptx_path(self) -> None:
        """Test cache entry with PTX path."""
        entry = CacheEntry(
            kernel_name="kernel",
            source_hash="hash",
            signature=None,
            ptx_path=Path("/tmp/test.ptx"),
        )
        assert entry.ptx_path == Path("/tmp/test.ptx")


class TestKernelCache:
    """Tests for KernelCache class."""

    @pytest.fixture
    def temp_cache_dir(self) -> Path:
        """Provide a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache(self, temp_cache_dir: Path) -> KernelCache:
        """Provide a fresh cache instance."""
        return KernelCache(cache_dir=temp_cache_dir, max_entries=10, max_size_mb=1)

    def test_creation(self, temp_cache_dir: Path) -> None:
        """Test cache creation."""
        cache = KernelCache(cache_dir=temp_cache_dir)
        assert cache._cache_dir == temp_cache_dir
        assert cache._max_entries == 128  # default
        assert cache._cache_dir.exists()

    def test_creation_with_custom_params(self, temp_cache_dir: Path) -> None:
        """Test cache creation with custom parameters."""
        cache = KernelCache(
            cache_dir=temp_cache_dir,
            max_entries=50,
            max_size_mb=100,
        )
        assert cache._max_entries == 50
        assert cache._max_size_bytes == 100 * 1024 * 1024

    def test_creation_default_dir(self) -> None:
        """Test cache creation with default directory."""
        cache = KernelCache()
        assert cache._cache_dir.exists()
        assert "pydotcompute_cache" in str(cache._cache_dir)

    def test_store_and_load(self, cache: KernelCache) -> None:
        """Test storing and loading PTX."""
        ptx = b"fake ptx content"

        # Store
        result = cache.store("my_kernel", "hash123", ptx)
        assert result is True

        # Load
        loaded = cache.load("my_kernel", "hash123")
        assert loaded == ptx

    def test_store_string_ptx(self, cache: KernelCache) -> None:
        """Test storing PTX as string."""
        ptx_str = "string ptx content"

        cache.store("kernel", "hash", ptx_str)
        loaded = cache.load("kernel", "hash")
        assert loaded == ptx_str.encode()

    def test_store_with_metadata(self, cache: KernelCache) -> None:
        """Test storing with signature and metadata."""
        ptx = b"ptx data"

        cache.store(
            "kernel",
            "hash",
            ptx,
            signature="void(int32)",
            metadata={"opt_level": 3},
        )

        # Verify entry exists with metadata
        assert cache.contains("kernel", "hash")
        cache_key = cache._get_cache_key("kernel", "hash")
        entry = cache._entries[cache_key]
        assert entry.signature == "void(int32)"
        assert entry.metadata == {"opt_level": 3}

    def test_load_nonexistent(self, cache: KernelCache) -> None:
        """Test loading non-existent kernel."""
        result = cache.load("nonexistent", "hash")
        assert result is None

    def test_contains(self, cache: KernelCache) -> None:
        """Test checking if kernel is cached."""
        cache.store("kernel", "hash", b"data")

        assert cache.contains("kernel", "hash") is True
        assert cache.contains("kernel", "other_hash") is False
        assert cache.contains("other_kernel", "hash") is False

    def test_remove(self, cache: KernelCache) -> None:
        """Test removing a kernel from cache."""
        cache.store("kernel", "hash", b"data")
        assert cache.contains("kernel", "hash")

        result = cache.remove("kernel", "hash")
        assert result is True
        assert cache.contains("kernel", "hash") is False

    def test_remove_nonexistent(self, cache: KernelCache) -> None:
        """Test removing non-existent kernel."""
        result = cache.remove("nonexistent", "hash")
        assert result is False

    def test_clear(self, cache: KernelCache) -> None:
        """Test clearing all cache entries."""
        cache.store("kernel1", "hash1", b"data1")
        cache.store("kernel2", "hash2", b"data2")
        cache.store("kernel3", "hash3", b"data3")

        count = cache.clear()
        assert count == 3
        assert cache.contains("kernel1", "hash1") is False
        assert cache.contains("kernel2", "hash2") is False

    def test_lru_eviction(self, temp_cache_dir: Path) -> None:
        """Test LRU eviction when max entries reached."""
        cache = KernelCache(cache_dir=temp_cache_dir, max_entries=3)

        # Store 3 entries
        cache.store("k1", "h1", b"d1")
        time.sleep(0.01)
        cache.store("k2", "h2", b"d2")
        time.sleep(0.01)
        cache.store("k3", "h3", b"d3")

        # Access k1 to make it recently used
        cache.load("k1", "h1")
        time.sleep(0.01)

        # Store 4th entry - should evict k2 (least recently used)
        cache.store("k4", "h4", b"d4")

        assert cache.contains("k1", "h1")  # Recently accessed
        assert cache.contains("k3", "h3")  # Recently stored
        assert cache.contains("k4", "h4")  # Just stored

    def test_get_stats(self, cache: KernelCache) -> None:
        """Test getting cache statistics."""
        cache.store("k1", "h1", b"data1")
        cache.store("k2", "h2", b"data2")
        cache.load("k1", "h1")  # Hit
        cache.load("k1", "h1")  # Another hit

        stats = cache.get_stats()
        assert stats["entries"] == 2
        assert stats["max_entries"] == 10
        assert stats["total_hits"] == 2
        assert stats["total_size_bytes"] > 0

    def test_repr(self, cache: KernelCache) -> None:
        """Test string representation."""
        cache.store("k1", "h1", b"x" * 100)

        repr_str = repr(cache)
        assert "KernelCache" in repr_str
        assert "entries=" in repr_str

    def test_index_persistence(self, temp_cache_dir: Path) -> None:
        """Test that cache index persists across instances."""
        # Create cache and store data
        cache1 = KernelCache(cache_dir=temp_cache_dir)
        cache1.store("kernel", "hash", b"persistent data")

        # Create new cache instance with same directory
        cache2 = KernelCache(cache_dir=temp_cache_dir)

        # Should be able to load from persisted index
        loaded = cache2.load("kernel", "hash")
        assert loaded == b"persistent data"

    def test_load_invalid_ptx_file(self, cache: KernelCache) -> None:
        """Test loading when PTX file is deleted."""
        cache.store("kernel", "hash", b"data")

        # Delete the PTX file manually
        cache_key = cache._get_cache_key("kernel", "hash")
        ptx_path = cache._get_ptx_path(cache_key)
        ptx_path.unlink()

        # Load should return None and clean up entry
        result = cache.load("kernel", "hash")
        assert result is None
        assert cache.contains("kernel", "hash") is False

    def test_hit_count_increment(self, cache: KernelCache) -> None:
        """Test that hit count increments on load."""
        cache.store("kernel", "hash", b"data")

        cache_key = cache._get_cache_key("kernel", "hash")
        assert cache._entries[cache_key].hit_count == 0

        cache.load("kernel", "hash")
        assert cache._entries[cache_key].hit_count == 1

        cache.load("kernel", "hash")
        assert cache._entries[cache_key].hit_count == 2


class TestGetKernelCache:
    """Tests for global cache getter."""

    def test_returns_singleton(self) -> None:
        """Test that get_kernel_cache returns singleton."""
        # Reset global cache
        import pydotcompute.compilation.cache as cache_module
        cache_module._global_cache = None

        cache1 = get_kernel_cache()
        cache2 = get_kernel_cache()

        assert cache1 is cache2

    def test_creates_cache_if_none(self) -> None:
        """Test that cache is created if not exists."""
        import pydotcompute.compilation.cache as cache_module
        cache_module._global_cache = None

        cache = get_kernel_cache()
        assert cache is not None
        assert isinstance(cache, KernelCache)


class TestCacheEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def temp_cache_dir(self) -> Path:
        """Provide a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_corrupted_index_file(self, temp_cache_dir: Path) -> None:
        """Test handling corrupted index file."""
        index_file = temp_cache_dir / "index.json"
        index_file.write_text("invalid json {{{")

        # Should handle gracefully
        cache = KernelCache(cache_dir=temp_cache_dir)
        assert len(cache._entries) == 0

    def test_size_eviction(self, temp_cache_dir: Path) -> None:
        """Test eviction based on size limit."""
        # Very small size limit
        cache = KernelCache(
            cache_dir=temp_cache_dir,
            max_entries=100,
            max_size_mb=1,  # 1MB limit
        )

        # Store large data
        large_data = b"x" * (512 * 1024)  # 512KB
        cache.store("k1", "h1", large_data)
        cache.store("k2", "h2", large_data)

        # Third should trigger eviction
        cache.store("k3", "h3", large_data)

        # At least one should be evicted
        total_entries = sum(1 for k in ["k1", "k2", "k3"]
                          if cache.contains(k, f"h{k[-1]}"))
        assert total_entries <= 2

    def test_empty_cache_stats(self, temp_cache_dir: Path) -> None:
        """Test stats on empty cache."""
        cache = KernelCache(cache_dir=temp_cache_dir)
        stats = cache.get_stats()

        assert stats["entries"] == 0
        assert stats["total_hits"] == 0
        assert stats["total_size_bytes"] == 0
