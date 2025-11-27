# Installation

This guide covers all installation options for PyDotCompute.

## Requirements

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **GPU (optional)**:
  - NVIDIA GPU with CUDA 12.x for CUDA acceleration
  - Apple Silicon (M1/M2/M3/M4) for Metal acceleration (macOS only)

## Installation Options

### Basic Installation (CPU only)

For development, testing, or systems without NVIDIA GPUs:

```bash
pip install pydotcompute
```

This installs the core package with CPU backend support. All features work, but GPU kernels run on CPU for simulation.

### With Performance Optimizations (Recommended)

For optimal message latency on Linux/macOS:

```bash
pip install pydotcompute[fast]
```

This includes **uvloop**, which provides:

- **21μs** message latency (p50)
- 20-40% faster event loop performance
- Automatic installation at import time

!!! tip "uvloop Auto-Installation"
    PyDotCompute automatically installs uvloop when you import `pydotcompute.ring_kernels`. No manual setup required!

### With CUDA Support

For full GPU acceleration:

```bash
pip install pydotcompute[cuda]
```

This includes:

- **CuPy**: GPU array library (NumPy-compatible)
- **Numba**: JIT compiler with CUDA support
- **pynvml**: NVIDIA GPU monitoring

!!! note "CUDA Version"
    The `cuda` extra installs `cupy-cuda12x`. If you need a different CUDA version, install CuPy manually:

    ```bash
    pip install pydotcompute
    pip install cupy-cuda11x  # For CUDA 11.x
    ```

### With Metal Support (macOS)

For Apple Silicon GPU acceleration on macOS:

```bash
pip install pydotcompute[metal]
```

This includes:

- **MLX**: Apple's machine learning framework for Metal

!!! note "Apple Silicon Only"
    Metal support requires macOS with Apple Silicon (M1, M2, M3, M4 chips). Intel Macs are not supported for Metal acceleration.

### With Cython Extensions (Maximum Performance)

For multi-process scenarios requiring ultimate performance:

```bash
pip install pydotcompute[cython]
python setup_cython.py build_ext --inplace
```

This provides:

- **0.33μs** queue operations (vs 1.8μs for pure Python)
- Lock-free SPSC queues
- Best for multi-process IPC scenarios

### Combined Installation

For full GPU acceleration with performance optimizations:

=== "NVIDIA GPU"

    ```bash
    pip install pydotcompute[cuda,fast]
    ```

=== "Apple Silicon"

    ```bash
    pip install pydotcompute[metal,fast]
    ```

### Development Installation

For contributing to PyDotCompute:

```bash
git clone https://github.com/mivertowski/PyDotCompute.git
cd PyDotCompute
pip install -e ".[dev]"
```

Development dependencies include:

- **pytest**: Testing framework
- **pytest-asyncio**: Async test support
- **mypy**: Static type checking
- **ruff**: Linting and formatting

### Documentation Build

To build the documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Verifying Installation

### Check Basic Installation

```python
import pydotcompute
print(f"PyDotCompute version: {pydotcompute.__version__}")
```

### Check CUDA Availability

```python
from pydotcompute.core.accelerator import cuda_available, get_accelerator

print(f"CUDA available: {cuda_available()}")

acc = get_accelerator()
print(f"Devices: {acc.device_count}")
for device in acc.devices:
    print(f"  - {device.name} ({device.device_type.name})")
```

### Run a Quick Test

```python
import asyncio
from pydotcompute import RingKernelRuntime, ring_kernel, message

@message
class Ping:
    value: int = 0

@ring_kernel(kernel_id="echo", auto_register=False)
async def echo_actor(ctx):
    while not ctx.should_terminate:
        try:
            msg = await ctx.receive(timeout=0.1)
            await ctx.send(msg)
        except:
            continue

async def test():
    async with RingKernelRuntime() as runtime:
        await runtime.launch("echo", echo_actor)
        await runtime.activate("echo")

        await asyncio.sleep(0.1)  # Let actor start

        await runtime.send("echo", Ping(value=42))
        response = await runtime.receive("echo", timeout=1.0)

        print(f"Echo test: {response.value == 42}")
        return response.value == 42

result = asyncio.run(test())
print(f"Installation verified: {result}")
```

## Troubleshooting

### ImportError: No module named 'cupy'

CuPy is not installed. Install it with:

```bash
pip install cupy-cuda12x  # Match your CUDA version
```

### CUDA driver version is insufficient

Your NVIDIA driver is too old. Update your GPU drivers from [NVIDIA's website](https://www.nvidia.com/drivers).

### No CUDA-capable device is detected

1. Verify GPU with `nvidia-smi`
2. Check CUDA installation with `nvcc --version`
3. Ensure the GPU is visible to Python

### Tests fail with "CUDA not available"

This is expected on systems without NVIDIA GPUs. Tests automatically skip CUDA-specific tests.

## Optional Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `uvloop` | Fast event loop (Linux/macOS) | `pip install uvloop` |
| `cupy-cuda12x` | GPU arrays | `pip install cupy-cuda12x` |
| `numba` | CUDA JIT compiler | `pip install numba` |
| `pynvml` | GPU monitoring | `pip install pynvml` |
| `cython` | Maximum performance queues | `pip install cython` |
| `pytest` | Testing | `pip install pytest` |
| `mypy` | Type checking | `pip install mypy` |

## Disabling uvloop

If you need to disable uvloop auto-installation (e.g., for debugging or compatibility):

```bash
PYDOTCOMPUTE_NO_UVLOOP=1 python my_script.py
```

Or set it in your Python code before importing:

```python
import os
os.environ["PYDOTCOMPUTE_NO_UVLOOP"] = "1"

from pydotcompute import RingKernelRuntime  # uvloop NOT installed
```

## Performance Tiers

PyDotCompute offers three performance tiers:

| Tier | Implementation | Latency (p50) | Use Case |
|------|---------------|---------------|----------|
| **1 (Default)** | uvloop + FastMessageQueue | **21μs** | Async Python code |
| 2 | ThreadedRingKernel | ~100μs | Blocking I/O, C extensions |
| 3 | CythonRingKernel | **0.33μs** queue ops | Multi-process IPC |

See the [Performance Tiers Guide](../articles/guides/performance-tiers.md) for detailed usage.

## Next Steps

- [Quick Start](quickstart.md): Get up and running
- [First Ring Kernel](first-kernel.md): Build your first actor
- [Performance Tiers](../articles/guides/performance-tiers.md): Choose the right tier
