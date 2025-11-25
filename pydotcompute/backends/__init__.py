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
    from pydotcompute.backends.cuda import CUDABackend

    __all__.append("CUDABackend")
except ImportError:
    pass
