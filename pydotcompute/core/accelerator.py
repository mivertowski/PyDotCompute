"""
GPU accelerator abstraction.

Provides device discovery, selection, and properties access.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class DeviceType(Enum):
    """Type of compute device."""

    CPU = auto()
    CUDA = auto()
    METAL = auto()  # Future support


@dataclass(frozen=True)
class DeviceProperties:
    """Properties of a compute device."""

    device_id: int
    device_type: DeviceType
    name: str
    compute_capability: tuple[int, int] | None  # (major, minor) for CUDA
    total_memory: int  # bytes
    multiprocessor_count: int
    max_threads_per_block: int
    max_block_dims: tuple[int, int, int]
    max_grid_dims: tuple[int, int, int]
    warp_size: int
    is_available: bool

    @property
    def compute_capability_str(self) -> str:
        """Get compute capability as string (e.g., '8.9')."""
        if self.compute_capability is None:
            return "N/A"
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"

    @property
    def total_memory_gb(self) -> float:
        """Get total memory in GB."""
        return self.total_memory / (1024**3)


class Accelerator:
    """
    GPU accelerator abstraction.

    Provides device discovery, selection, and properties access.
    Supports both CUDA and CPU fallback.
    """

    _instance: Accelerator | None = None
    _initialized: bool = False

    def __new__(cls) -> Accelerator:
        """Singleton pattern for accelerator."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the accelerator."""
        if Accelerator._initialized:
            return

        self._devices: list[DeviceProperties] = []
        self._current_device_id: int = 0
        self._cuda_available: bool = False
        self._metal_available: bool = False
        self._discover_devices()
        Accelerator._initialized = True

    def _discover_devices(self) -> None:
        """Discover available compute devices."""
        # Always add CPU as fallback
        cpu_device = DeviceProperties(
            device_id=0,
            device_type=DeviceType.CPU,
            name="CPU",
            compute_capability=None,
            total_memory=0,  # Will be updated if psutil available
            multiprocessor_count=1,
            max_threads_per_block=1,
            max_block_dims=(1, 1, 1),
            max_grid_dims=(1, 1, 1),
            warp_size=1,
            is_available=True,
        )

        # Try to discover CUDA devices
        try:
            import cupy as cp

            num_devices = cp.cuda.runtime.getDeviceCount()
            self._cuda_available = num_devices > 0

            for i in range(num_devices):
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    device = DeviceProperties(
                        device_id=i,
                        device_type=DeviceType.CUDA,
                        name=props["name"].decode()
                        if isinstance(props["name"], bytes)
                        else props["name"],
                        compute_capability=(props["major"], props["minor"]),
                        total_memory=props["totalGlobalMem"],
                        multiprocessor_count=props["multiProcessorCount"],
                        max_threads_per_block=props["maxThreadsPerBlock"],
                        max_block_dims=(
                            props["maxThreadsDim"][0],
                            props["maxThreadsDim"][1],
                            props["maxThreadsDim"][2],
                        ),
                        max_grid_dims=(
                            props["maxGridSize"][0],
                            props["maxGridSize"][1],
                            props["maxGridSize"][2],
                        ),
                        warp_size=props["warpSize"],
                        is_available=True,
                    )
                    self._devices.append(device)

        except ImportError:
            # CuPy not available, use CPU only
            pass
        except Exception:
            # CUDA error, fall back to CPU
            pass

        # Try to discover Metal devices (macOS only)
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                self._metal_available = True
                metal_device = DeviceProperties(
                    device_id=len(self._devices),
                    device_type=DeviceType.METAL,
                    name="Apple Silicon GPU",
                    compute_capability=None,  # Metal uses feature sets, not compute capability
                    total_memory=0,  # Unified memory - not directly queryable
                    multiprocessor_count=self._get_metal_gpu_cores(),
                    max_threads_per_block=1024,  # Metal threadgroup size
                    max_block_dims=(1024, 1024, 64),
                    max_grid_dims=(2**30, 2**16, 2**16),
                    warp_size=32,  # SIMD group width for Apple GPUs
                    is_available=True,
                )
                self._devices.append(metal_device)
        except ImportError:
            # MLX not available
            pass
        except Exception:
            # Metal error
            pass

        # If no GPU devices found, add CPU device
        if not self._devices:
            self._devices.append(cpu_device)

    def _get_metal_gpu_cores(self) -> int:
        """Estimate GPU cores based on Apple Silicon chip."""
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            chip_name = result.stdout.strip()

            # Approximate GPU core counts for Apple Silicon
            core_map = {
                "M1": 8,
                "M1 Pro": 16,
                "M1 Max": 32,
                "M1 Ultra": 64,
                "M2": 10,
                "M2 Pro": 19,
                "M2 Max": 38,
                "M2 Ultra": 76,
                "M3": 10,
                "M3 Pro": 18,
                "M3 Max": 40,
                "M4": 10,
                "M4 Pro": 20,
                "M4 Max": 40,
            }
            for chip, cores in core_map.items():
                if chip in chip_name:
                    return cores
            return 8  # Default
        except Exception:
            return 8

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available

    @property
    def metal_available(self) -> bool:
        """Check if Metal is available."""
        return self._metal_available

    @property
    def device_count(self) -> int:
        """Get the number of available devices."""
        return len(self._devices)

    @property
    def current_device(self) -> DeviceProperties:
        """Get the current device properties."""
        return self._devices[self._current_device_id]

    @property
    def devices(self) -> list[DeviceProperties]:
        """Get all available devices."""
        return self._devices.copy()

    def get_device(self, device_id: int) -> DeviceProperties:
        """Get properties for a specific device."""
        if device_id < 0 or device_id >= len(self._devices):
            raise ValueError(
                f"Invalid device_id: {device_id}. Valid range: 0-{len(self._devices) - 1}"
            )
        return self._devices[device_id]

    def set_device(self, device_id: int) -> None:
        """Set the current device."""
        if device_id < 0 or device_id >= len(self._devices):
            raise ValueError(
                f"Invalid device_id: {device_id}. Valid range: 0-{len(self._devices) - 1}"
            )

        self._current_device_id = device_id

        # If CUDA device, set CuPy device
        if self._devices[device_id].device_type == DeviceType.CUDA:
            try:
                import cupy as cp

                cp.cuda.Device(device_id).use()
            except ImportError:
                pass

    def synchronize(self) -> None:
        """Synchronize the current device."""
        if self._cuda_available and self.current_device.device_type == DeviceType.CUDA:
            try:
                import cupy as cp

                cp.cuda.Stream.null.synchronize()
            except ImportError:
                pass
        elif self._metal_available and self.current_device.device_type == DeviceType.METAL:
            try:
                import mlx.core as mx

                mx.eval()  # Force evaluation of pending operations
            except ImportError:
                pass

    def get_memory_info(self) -> dict[str, int]:
        """Get memory information for the current device."""
        if self._cuda_available and self.current_device.device_type == DeviceType.CUDA:
            try:
                import cupy as cp

                mem_info = cp.cuda.runtime.memGetInfo()
                return {
                    "free": mem_info[0],
                    "total": mem_info[1],
                    "used": mem_info[1] - mem_info[0],
                }
            except ImportError:
                pass
        elif self._metal_available and self.current_device.device_type == DeviceType.METAL:
            try:
                import mlx.core as mx

                # Use new API if available, fallback to deprecated
                if hasattr(mx, "get_cache_memory"):
                    cache_memory = mx.get_cache_memory()
                    peak_memory = mx.get_peak_memory()
                else:
                    cache_memory = mx.metal.get_cache_memory()
                    peak_memory = mx.metal.get_peak_memory()
                return {
                    "free": 0,  # Unified memory - not directly queryable
                    "total": 0,  # Unified memory
                    "used": cache_memory,
                    "cache_memory": cache_memory,
                    "peak_memory": peak_memory,
                }
            except ImportError:
                pass

        # CPU fallback
        return {
            "free": 0,
            "total": 0,
            "used": 0,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Accelerator(devices={self.device_count}, "
            f"cuda_available={self.cuda_available}, "
            f"metal_available={self.metal_available}, "
            f"current_device={self.current_device.name})"
        )


# Convenience functions
def get_accelerator() -> Accelerator:
    """Get the global accelerator instance."""
    return Accelerator()


def cuda_available() -> bool:
    """Check if CUDA is available."""
    return get_accelerator().cuda_available


def metal_available() -> bool:
    """Check if Metal is available."""
    return get_accelerator().metal_available
