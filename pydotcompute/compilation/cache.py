"""
Kernel caching system for PyDotCompute.

Provides persistent caching of compiled kernels to reduce
startup time and avoid recompilation.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass


@dataclass
class CacheEntry:
    """Entry in the kernel cache."""

    kernel_name: str
    source_hash: str
    signature: str | None
    ptx_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_timestamp: float = 0.0
    last_accessed: float = 0.0
    hit_count: int = 0


class KernelCache:
    """
    Persistent cache for compiled kernels.

    Stores compiled PTX/CUBIN files on disk to avoid recompilation.
    Uses LRU eviction when cache size exceeds limit.

    Example:
        >>> cache = KernelCache()
        >>> cache.store("my_kernel", source_hash, ptx_bytes)
        >>> ptx = cache.load("my_kernel", source_hash)
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_entries: int = 128,
        max_size_mb: int = 256,
    ) -> None:
        """
        Initialize the kernel cache.

        Args:
            cache_dir: Directory for cache files (default: temp dir).
            max_entries: Maximum number of cached kernels.
            max_size_mb: Maximum total cache size in MB.
        """
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "pydotcompute_cache"

        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._max_entries = max_entries
        self._max_size_bytes = max_size_mb * 1024 * 1024

        self._index_file = self._cache_dir / "index.json"
        self._entries: dict[str, CacheEntry] = {}

        self._load_index()

    def _load_index(self) -> None:
        """Load the cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file) as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        ptx_path = entry_data.get("ptx_path")
                        self._entries[key] = CacheEntry(
                            kernel_name=entry_data["kernel_name"],
                            source_hash=entry_data["source_hash"],
                            signature=entry_data.get("signature"),
                            ptx_path=Path(ptx_path) if ptx_path else None,
                            metadata=entry_data.get("metadata", {}),
                            created_timestamp=entry_data.get("created_timestamp", 0.0),
                            last_accessed=entry_data.get("last_accessed", 0.0),
                            hit_count=entry_data.get("hit_count", 0),
                        )
            except Exception:
                self._entries = {}

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        try:
            data = {}
            for key, entry in self._entries.items():
                data[key] = {
                    "kernel_name": entry.kernel_name,
                    "source_hash": entry.source_hash,
                    "signature": entry.signature,
                    "ptx_path": str(entry.ptx_path) if entry.ptx_path else None,
                    "metadata": entry.metadata,
                    "created_timestamp": entry.created_timestamp,
                    "last_accessed": entry.last_accessed,
                    "hit_count": entry.hit_count,
                }
            with open(self._index_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _get_cache_key(self, kernel_name: str, source_hash: str) -> str:
        """Generate a unique cache key."""
        return f"{kernel_name}_{source_hash}"

    def _get_ptx_path(self, cache_key: str) -> Path:
        """Get the path for a PTX file."""
        return self._cache_dir / f"{cache_key}.ptx"

    def _get_cubin_path(self, cache_key: str) -> Path:
        """Get the path for a CUBIN file."""
        return self._cache_dir / f"{cache_key}.cubin"

    def store(
        self,
        kernel_name: str,
        source_hash: str,
        ptx: bytes | str,
        *,
        signature: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Store compiled PTX in the cache.

        Args:
            kernel_name: Name of the kernel.
            source_hash: Hash of the kernel source.
            ptx: Compiled PTX code.
            signature: Optional kernel signature.
            metadata: Optional metadata.

        Returns:
            True if stored successfully.
        """
        import time

        cache_key = self._get_cache_key(kernel_name, source_hash)
        ptx_path = self._get_ptx_path(cache_key)

        try:
            # Ensure we don't exceed limits
            self._evict_if_needed()

            # Write PTX file
            if isinstance(ptx, str):
                ptx = ptx.encode()

            with open(ptx_path, "wb") as f:
                f.write(ptx)

            # Create cache entry
            entry = CacheEntry(
                kernel_name=kernel_name,
                source_hash=source_hash,
                signature=signature,
                ptx_path=ptx_path,
                metadata=metadata or {},
                created_timestamp=time.time(),
                last_accessed=time.time(),
                hit_count=0,
            )
            self._entries[cache_key] = entry

            self._save_index()
            return True

        except Exception:
            return False

    def load(
        self,
        kernel_name: str,
        source_hash: str,
    ) -> bytes | None:
        """
        Load PTX from the cache.

        Args:
            kernel_name: Name of the kernel.
            source_hash: Hash of the kernel source.

        Returns:
            PTX bytes or None if not cached.
        """
        import time

        cache_key = self._get_cache_key(kernel_name, source_hash)
        entry = self._entries.get(cache_key)

        if entry is None:
            return None

        if entry.ptx_path is None or not entry.ptx_path.exists():
            # Remove invalid entry
            del self._entries[cache_key]
            self._save_index()
            return None

        try:
            with open(entry.ptx_path, "rb") as f:
                ptx = f.read()

            # Update access stats
            entry.last_accessed = time.time()
            entry.hit_count += 1
            self._save_index()

            return ptx

        except Exception:
            return None

    def contains(self, kernel_name: str, source_hash: str) -> bool:
        """
        Check if a kernel is in the cache.

        Args:
            kernel_name: Name of the kernel.
            source_hash: Hash of the kernel source.

        Returns:
            True if kernel is cached.
        """
        cache_key = self._get_cache_key(kernel_name, source_hash)
        return cache_key in self._entries

    def remove(self, kernel_name: str, source_hash: str) -> bool:
        """
        Remove a kernel from the cache.

        Args:
            kernel_name: Name of the kernel.
            source_hash: Hash of the kernel source.

        Returns:
            True if removed.
        """
        cache_key = self._get_cache_key(kernel_name, source_hash)
        entry = self._entries.pop(cache_key, None)

        if entry:
            # Delete file
            if entry.ptx_path and entry.ptx_path.exists():
                try:
                    entry.ptx_path.unlink()
                except Exception:
                    pass
            self._save_index()
            return True

        return False

    def _evict_if_needed(self) -> None:
        """Evict entries if cache exceeds limits."""
        # Check entry count
        while len(self._entries) >= self._max_entries:
            self._evict_lru()

        # Check total size
        while self._get_total_size() >= self._max_size_bytes:
            if not self._evict_lru():
                break

    def _evict_lru(self) -> bool:
        """Evict the least recently used entry."""
        if not self._entries:
            return False

        # Find LRU entry
        lru_key = min(
            self._entries.keys(),
            key=lambda k: self._entries[k].last_accessed,
        )

        entry = self._entries.pop(lru_key)

        # Delete file
        if entry.ptx_path and entry.ptx_path.exists():
            try:
                entry.ptx_path.unlink()
            except Exception:
                pass

        return True

    def _get_total_size(self) -> int:
        """Get total cache size in bytes."""
        total = 0
        for entry in self._entries.values():
            if entry.ptx_path and entry.ptx_path.exists():
                total += entry.ptx_path.stat().st_size
        return total

    def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries cleared.
        """
        count = len(self._entries)

        for entry in self._entries.values():
            if entry.ptx_path and entry.ptx_path.exists():
                try:
                    entry.ptx_path.unlink()
                except Exception:
                    pass

        self._entries.clear()
        self._save_index()

        return count

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        total_hits = sum(e.hit_count for e in self._entries.values())

        return {
            "entries": len(self._entries),
            "max_entries": self._max_entries,
            "total_size_bytes": self._get_total_size(),
            "max_size_bytes": self._max_size_bytes,
            "total_hits": total_hits,
            "cache_dir": str(self._cache_dir),
        }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"KernelCache(entries={stats['entries']}/{stats['max_entries']}, "
            f"size={stats['total_size_bytes'] // 1024}KB)"
        )


# Global cache instance
_global_cache: KernelCache | None = None


def get_kernel_cache() -> KernelCache:
    """Get the global kernel cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = KernelCache()
    return _global_cache


def warmup_kernels(kernels: list[Callable[..., Any]]) -> None:
    """
    Pre-compile a list of kernels to warm up the cache.

    Args:
        kernels: List of kernel functions to compile.
    """
    from pydotcompute.compilation.compiler import get_compiler

    compiler = get_compiler()
    for kernel in kernels:
        try:
            compiler.compile(kernel)
        except Exception:
            pass
