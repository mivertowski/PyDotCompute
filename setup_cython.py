"""
Setup script for building Cython extensions.

Usage:
    python setup_cython.py build_ext --inplace

This builds the high-performance Cython extensions for sub-10μs latency.
The extensions are optional - pure Python fallbacks are available.
"""

import os
import sys
from pathlib import Path

# Disable pyproject.toml parsing to avoid conflicts
os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = "0.1.0"
os.environ["PIP_IGNORE_REQUIRES_PYTHON"] = "1"

# Check for Cython
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    cythonize = None
    build_ext = None

from distutils.core import setup
from distutils.extension import Extension

# Get the project root
PROJECT_ROOT = Path(__file__).parent


def get_extensions():
    """Get list of Cython extensions to build."""
    if not CYTHON_AVAILABLE:
        print("WARNING: Cython not installed. Extensions will not be built.")
        print("Install with: pip install cython")
        return []

    extensions = []

    # FastSPSC Queue
    fast_spsc_path = PROJECT_ROOT / "pydotcompute" / "ring_kernels" / "_cython" / "fast_spsc.pyx"
    if fast_spsc_path.exists():
        extensions.append(
            Extension(
                "pydotcompute.ring_kernels._cython.fast_spsc",
                sources=[str(fast_spsc_path)],
                extra_compile_args=[
                    "-O3",
                    "-march=native",
                    "-ffast-math",
                    "-std=c11",
                ],
                define_macros=[
                    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
                ],
            )
        )

    return cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    )


if __name__ == "__main__":
    if not CYTHON_AVAILABLE:
        print("ERROR: Cython is required to build extensions.")
        print("Install with: pip install cython")
        sys.exit(1)

    setup(
        name="pydotcompute-cython",
        ext_modules=get_extensions(),
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
    )
