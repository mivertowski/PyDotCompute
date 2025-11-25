# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/mivertowski/PyDotCompute/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mivertowski/PyDotCompute/releases/tag/v0.1.0
