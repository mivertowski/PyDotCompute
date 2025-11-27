# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-11-27

### Added

- **Metal Backend**: Full GPU acceleration support for macOS/Apple Silicon via Apple's MLX framework
  - `MetalBackend` class implementing the `Backend` ABC interface
  - Memory allocation, copy operations, and synchronization for Metal GPU
  - Kernel execution and compilation with MLX operations
  - Pre-built kernels: `get_vector_add_kernel()`, `get_matrix_multiply_kernel()`, `get_elementwise_kernel()`
  - Support for 12 elementwise operations: add, sub, mul, div, sqrt, exp, log, sin, cos, abs, square, negative
  - Thread-safe buffer registry with `MetalBufferInfo` tracking
  - Automatic dtype conversion (NumPy to MLX)
- **UnifiedBuffer Metal Integration**: Added `.metal` property for seamless Metal GPU access
  - Automatic synchronization from host when buffer is dirty
  - `mark_metal_dirty()` method for marking Metal data as modified
  - Full state machine integration (HOST_DIRTY, DEVICE_DIRTY, SYNCHRONIZED)
- **Accelerator Metal Support**: Metal device discovery on macOS
  - `metal_available` property and `metal_available()` convenience function
  - Apple Silicon GPU core detection for M1/M2/M3/M4 chips
  - Metal device properties in `DeviceProperties` dataclass
- **Metal Exceptions**: `MetalError` and `MSLCompilationError` in exceptions module
- **Metal Benchmarks**: `benchmarks/metal_benchmark.py` for performance testing
  - Memory copy, vector addition, matrix multiplication benchmarks
  - Elementwise and reduction operation benchmarks
  - Backend API performance tests
- **Comprehensive Metal Test Suite**: 42+ tests covering all Metal functionality
  - Backend operations, kernel execution, edge cases
  - UnifiedBuffer and Accelerator integration
  - Stress tests, error handling, buffer tracking

### Changed

- Updated `pyproject.toml` with `metal` optional dependency (`mlx>=0.4.0`)
- Added `@pytest.mark.metal` marker for Metal-specific tests
- Updated documentation (README.md, CLAUDE.md) with Metal backend usage

## [0.1.0] - 2025-01-01

### Added

- Initial release of PyDotCompute
- Ring Kernel System with persistent GPU kernels and message queues
- `@ring_kernel` decorator for defining actor kernels
- `@message` decorator for type-safe message serialization with msgpack
- `RingKernelRuntime` for managing kernel lifecycle
- Two-phase kernel launch (launch -> activate) with graceful shutdown
- `UnifiedBuffer` for transparent host-device memory with lazy synchronization
- `MemoryPool` for efficient buffer reuse
- `Accelerator` for GPU device abstraction
- CPU simulation backend
- CUDA backend via Numba JIT and CuPy arrays
- Telemetry and performance monitoring
- Backpressure strategies: block, reject, drop_oldest
- Comprehensive test suite with pytest-asyncio
- Documentation with MkDocs

[Unreleased]: https://github.com/mivertowski/PyDotCompute/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mivertowski/PyDotCompute/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mivertowski/PyDotCompute/releases/tag/v0.1.0
