"""
Backend implementations for PyDotCompute.
"""

from pydotcompute.backends.base import Backend, BackendType
from pydotcompute.backends.cpu import CPUBackend

__all__ = [
    "Backend",
    "BackendType",
    "CPUBackend",
]

# Conditionally export CUDA backend if available
try:
    from pydotcompute.backends.cuda import CUDABackend  # noqa: F401

    __all__.append("CUDABackend")
except ImportError:
    pass

# Conditionally export Metal backend if available (macOS only)
try:
    from pydotcompute.backends.metal import MetalBackend  # noqa: F401

    __all__.append("MetalBackend")
except ImportError:
    pass
