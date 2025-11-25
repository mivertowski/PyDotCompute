"""
Cython extensions for high-performance ring kernels.

This package contains optional Cython implementations that provide
sub-10μs latency for message passing operations.

If the Cython extensions are not built, pure-Python fallbacks are used.
"""

try:
    from pydotcompute.ring_kernels._cython.fast_spsc import FastSPSCQueue, is_cython_available
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    FastSPSCQueue = None  # type: ignore[assignment, misc]

    def is_cython_available() -> bool:
        """Check if Cython extension is loaded."""
        return False


__all__ = [
    "FastSPSCQueue",
    "is_cython_available",
    "CYTHON_AVAILABLE",
]
