"""
Telemetry and monitoring for ring kernels.

Provides GPU monitoring and kernel performance metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class RingKernelTelemetry:
    """
    Telemetry data for a ring kernel.

    Tracks message counts, latency, and error statistics.
    """

    kernel_id: str = ""
    messages_processed: int = 0
    messages_dropped: int = 0
    queue_depth: int = 0
    total_latency_ns: int = 0
    max_latency_ns: int = 0
    min_latency_ns: int = 0
    error_count: int = 0
    last_error: str | None = None
    uptime_seconds: float = 0.0

    @property
    def avg_latency_ns(self) -> float:
        """Get average message latency in nanoseconds."""
        if self.messages_processed == 0:
            return 0.0
        return self.total_latency_ns / self.messages_processed

    @property
    def avg_latency_ms(self) -> float:
        """Get average message latency in milliseconds."""
        return self.avg_latency_ns / 1_000_000

    @property
    def throughput(self) -> float:
        """Get message throughput (messages per second)."""
        if self.uptime_seconds == 0:
            return 0.0
        return self.messages_processed / self.uptime_seconds

    def record_message(self, latency_ns: int) -> None:
        """
        Record a processed message.

        Args:
            latency_ns: Message processing latency in nanoseconds.
        """
        self.messages_processed += 1
        self.total_latency_ns += latency_ns

        if self.min_latency_ns == 0 or latency_ns < self.min_latency_ns:
            self.min_latency_ns = latency_ns

        if latency_ns > self.max_latency_ns:
            self.max_latency_ns = latency_ns

    def record_drop(self) -> None:
        """Record a dropped message."""
        self.messages_dropped += 1

    def record_error(self, error: str) -> None:
        """
        Record an error.

        Args:
            error: Error message.
        """
        self.error_count += 1
        self.last_error = error

    def reset(self) -> None:
        """Reset all telemetry counters."""
        self.messages_processed = 0
        self.messages_dropped = 0
        self.queue_depth = 0
        self.total_latency_ns = 0
        self.max_latency_ns = 0
        self.min_latency_ns = 0
        self.error_count = 0
        self.last_error = None
        self.uptime_seconds = 0.0

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "kernel_id": self.kernel_id,
            "messages_processed": self.messages_processed,
            "messages_dropped": self.messages_dropped,
            "queue_depth": self.queue_depth,
            "avg_latency_ms": self.avg_latency_ms,
            "max_latency_ns": self.max_latency_ns,
            "min_latency_ns": self.min_latency_ns,
            "throughput": self.throughput,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "uptime_seconds": self.uptime_seconds,
        }


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    index: int
    name: str
    uuid: str
    memory_total: int  # bytes
    memory_used: int  # bytes
    memory_free: int  # bytes
    utilization_gpu: float  # 0.0-1.0
    utilization_memory: float  # 0.0-1.0
    temperature: float  # Celsius
    power_usage: float  # Watts
    power_limit: float  # Watts

    @property
    def memory_utilization(self) -> float:
        """Get memory utilization as fraction."""
        if self.memory_total == 0:
            return 0.0
        return self.memory_used / self.memory_total


class GPUMonitor:
    """
    GPU telemetry using NVML.

    Provides real-time GPU monitoring including utilization,
    memory usage, temperature, and power consumption.

    Example:
        >>> monitor = GPUMonitor()
        >>> if monitor.is_available:
        ...     info = monitor.get_gpu_info()
        ...     print(f"GPU: {info.name}, Util: {info.utilization_gpu:.1%}")
    """

    def __init__(self) -> None:
        """Initialize the GPU monitor."""
        self._initialized = False
        self._device_count = 0
        self._nvml_available = False

        try:
            import pynvml

            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._nvml_available = True
            self._initialized = True
        except ImportError:
            pass
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._nvml_available and self._device_count > 0

    @property
    def device_count(self) -> int:
        """Get the number of available GPU devices."""
        return self._device_count

    def get_gpu_info(self, device_index: int = 0) -> GPUInfo:
        """
        Get information about a GPU device.

        Args:
            device_index: Index of the GPU device.

        Returns:
            GPUInfo with current device state.
        """
        if not self._nvml_available:
            return self._empty_gpu_info(device_index)

        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

            # Get basic info
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            # Get temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            # Get power info
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except pynvml.NVMLError:
                power = 0.0
                power_limit = 0.0

            return GPUInfo(
                index=device_index,
                name=name,
                uuid=uuid,
                memory_total=mem_info.total,
                memory_used=mem_info.used,
                memory_free=mem_info.free,
                utilization_gpu=util.gpu / 100.0,
                utilization_memory=util.memory / 100.0,
                temperature=float(temp),
                power_usage=power,
                power_limit=power_limit,
            )

        except Exception:
            return self._empty_gpu_info(device_index)

    def _empty_gpu_info(self, device_index: int) -> GPUInfo:
        """Create empty GPU info for unavailable devices."""
        return GPUInfo(
            index=device_index,
            name="N/A",
            uuid="",
            memory_total=0,
            memory_used=0,
            memory_free=0,
            utilization_gpu=0.0,
            utilization_memory=0.0,
            temperature=0.0,
            power_usage=0.0,
            power_limit=0.0,
        )

    def get_all_gpu_info(self) -> list[GPUInfo]:
        """
        Get information about all GPU devices.

        Returns:
            List of GPUInfo for all devices.
        """
        return [self.get_gpu_info(i) for i in range(self._device_count)]

    def get_utilization(self, device_index: int = 0) -> float:
        """
        Get GPU utilization.

        Args:
            device_index: Index of the GPU device.

        Returns:
            GPU utilization as fraction (0.0-1.0).
        """
        return self.get_gpu_info(device_index).utilization_gpu

    def get_memory_info(self, device_index: int = 0) -> dict[str, int]:
        """
        Get memory information for a device.

        Args:
            device_index: Index of the GPU device.

        Returns:
            Dictionary with used, free, and total memory.
        """
        info = self.get_gpu_info(device_index)
        return {
            "used": info.memory_used,
            "free": info.memory_free,
            "total": info.memory_total,
        }

    def shutdown(self) -> None:
        """Shutdown NVML."""
        if self._nvml_available:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_available = False

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.shutdown()

    def __repr__(self) -> str:
        """String representation."""
        return f"GPUMonitor(available={self._nvml_available}, devices={self._device_count})"


@dataclass
class TelemetryCollector:
    """
    Collects telemetry from multiple ring kernels.

    Aggregates telemetry data for monitoring dashboards.
    """

    _kernel_telemetry: dict[str, RingKernelTelemetry] = field(default_factory=dict)
    _gpu_monitor: GPUMonitor = field(default_factory=GPUMonitor)
    _start_time: float = field(default_factory=time.time)

    def register_kernel(self, kernel_id: str) -> RingKernelTelemetry:
        """
        Register a kernel for telemetry collection.

        Args:
            kernel_id: ID of the kernel.

        Returns:
            Telemetry instance for the kernel.
        """
        if kernel_id not in self._kernel_telemetry:
            self._kernel_telemetry[kernel_id] = RingKernelTelemetry(kernel_id=kernel_id)
        return self._kernel_telemetry[kernel_id]

    def unregister_kernel(self, kernel_id: str) -> None:
        """
        Unregister a kernel from telemetry collection.

        Args:
            kernel_id: ID of the kernel.
        """
        self._kernel_telemetry.pop(kernel_id, None)

    def get_kernel_telemetry(self, kernel_id: str) -> RingKernelTelemetry | None:
        """
        Get telemetry for a specific kernel.

        Args:
            kernel_id: ID of the kernel.

        Returns:
            Telemetry data or None if not registered.
        """
        return self._kernel_telemetry.get(kernel_id)

    def get_all_telemetry(self) -> dict[str, RingKernelTelemetry]:
        """
        Get telemetry for all registered kernels.

        Returns:
            Dictionary mapping kernel IDs to telemetry.
        """
        # Update uptime
        current_time = time.time()
        uptime = current_time - self._start_time
        for telemetry in self._kernel_telemetry.values():
            telemetry.uptime_seconds = uptime

        return self._kernel_telemetry.copy()

    def get_summary(self) -> dict[str, object]:
        """
        Get aggregated telemetry summary.

        Returns:
            Summary dictionary with aggregated metrics.
        """
        all_telemetry = self.get_all_telemetry()

        total_processed = sum(t.messages_processed for t in all_telemetry.values())
        total_dropped = sum(t.messages_dropped for t in all_telemetry.values())
        total_errors = sum(t.error_count for t in all_telemetry.values())

        # Get GPU info if available
        gpu_info = None
        if self._gpu_monitor.is_available:
            gpu_info = self._gpu_monitor.get_gpu_info().utilization_gpu

        return {
            "kernel_count": len(all_telemetry),
            "total_messages_processed": total_processed,
            "total_messages_dropped": total_dropped,
            "total_errors": total_errors,
            "uptime_seconds": time.time() - self._start_time,
            "gpu_utilization": gpu_info,
        }

    def reset_all(self) -> None:
        """Reset all kernel telemetry."""
        for telemetry in self._kernel_telemetry.values():
            telemetry.reset()
        self._start_time = time.time()
