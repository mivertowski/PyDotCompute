"""
Unit tests for the telemetry module.

Tests the RingKernelTelemetry, GPUInfo, GPUMonitor, and TelemetryCollector
classes for performance monitoring and GPU telemetry.
"""

from __future__ import annotations

import time

import pytest

from pydotcompute.ring_kernels.telemetry import (
    GPUInfo,
    GPUMonitor,
    RingKernelTelemetry,
    TelemetryCollector,
)


class TestRingKernelTelemetry:
    """Tests for RingKernelTelemetry dataclass."""

    def test_default_creation(self) -> None:
        """Test creating telemetry with defaults."""
        telemetry = RingKernelTelemetry()
        assert telemetry.kernel_id == ""
        assert telemetry.messages_processed == 0
        assert telemetry.messages_dropped == 0
        assert telemetry.queue_depth == 0
        assert telemetry.error_count == 0
        assert telemetry.last_error is None

    def test_creation_with_kernel_id(self) -> None:
        """Test creating telemetry with kernel ID."""
        telemetry = RingKernelTelemetry(kernel_id="my_kernel")
        assert telemetry.kernel_id == "my_kernel"

    def test_avg_latency_ns(self) -> None:
        """Test average latency in nanoseconds."""
        telemetry = RingKernelTelemetry(
            messages_processed=10,
            total_latency_ns=1000000,  # 1ms total
        )
        assert telemetry.avg_latency_ns == 100000  # 100us average

    def test_avg_latency_ns_zero_messages(self) -> None:
        """Test average latency with zero messages."""
        telemetry = RingKernelTelemetry(messages_processed=0)
        assert telemetry.avg_latency_ns == 0.0

    def test_avg_latency_ms(self) -> None:
        """Test average latency in milliseconds."""
        telemetry = RingKernelTelemetry(
            messages_processed=4,
            total_latency_ns=4_000_000,  # 4ms total
        )
        assert telemetry.avg_latency_ms == 1.0  # 1ms average

    def test_throughput(self) -> None:
        """Test throughput calculation."""
        telemetry = RingKernelTelemetry(
            messages_processed=100,
            uptime_seconds=10.0,
        )
        assert telemetry.throughput == 10.0  # 10 msgs/sec

    def test_throughput_zero_uptime(self) -> None:
        """Test throughput with zero uptime."""
        telemetry = RingKernelTelemetry(
            messages_processed=100,
            uptime_seconds=0.0,
        )
        assert telemetry.throughput == 0.0

    def test_record_message(self) -> None:
        """Test recording a message."""
        telemetry = RingKernelTelemetry()

        telemetry.record_message(1000)  # 1us
        assert telemetry.messages_processed == 1
        assert telemetry.total_latency_ns == 1000
        assert telemetry.min_latency_ns == 1000
        assert telemetry.max_latency_ns == 1000

    def test_record_multiple_messages(self) -> None:
        """Test recording multiple messages."""
        telemetry = RingKernelTelemetry()

        telemetry.record_message(1000)
        telemetry.record_message(2000)
        telemetry.record_message(500)

        assert telemetry.messages_processed == 3
        assert telemetry.total_latency_ns == 3500
        assert telemetry.min_latency_ns == 500
        assert telemetry.max_latency_ns == 2000

    def test_record_drop(self) -> None:
        """Test recording dropped messages."""
        telemetry = RingKernelTelemetry()

        telemetry.record_drop()
        telemetry.record_drop()

        assert telemetry.messages_dropped == 2

    def test_record_error(self) -> None:
        """Test recording errors."""
        telemetry = RingKernelTelemetry()

        telemetry.record_error("First error")
        assert telemetry.error_count == 1
        assert telemetry.last_error == "First error"

        telemetry.record_error("Second error")
        assert telemetry.error_count == 2
        assert telemetry.last_error == "Second error"

    def test_reset(self) -> None:
        """Test resetting telemetry."""
        telemetry = RingKernelTelemetry(kernel_id="test")
        telemetry.record_message(1000)
        telemetry.record_drop()
        telemetry.record_error("error")
        telemetry.uptime_seconds = 100.0

        telemetry.reset()

        assert telemetry.kernel_id == "test"  # ID preserved
        assert telemetry.messages_processed == 0
        assert telemetry.messages_dropped == 0
        assert telemetry.error_count == 0
        assert telemetry.last_error is None
        assert telemetry.uptime_seconds == 0.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        telemetry = RingKernelTelemetry(
            kernel_id="test_kernel",
            messages_processed=50,
            messages_dropped=2,
            uptime_seconds=10.0,
        )

        data = telemetry.to_dict()

        assert data["kernel_id"] == "test_kernel"
        assert data["messages_processed"] == 50
        assert data["messages_dropped"] == 2
        assert data["throughput"] == 5.0
        assert "avg_latency_ms" in data
        assert "error_count" in data


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_creation(self) -> None:
        """Test creating GPU info."""
        info = GPUInfo(
            index=0,
            name="NVIDIA GeForce RTX 4090",
            uuid="GPU-12345678",
            memory_total=24 * 1024 * 1024 * 1024,  # 24GB
            memory_used=8 * 1024 * 1024 * 1024,    # 8GB
            memory_free=16 * 1024 * 1024 * 1024,   # 16GB
            utilization_gpu=0.75,
            utilization_memory=0.33,
            temperature=65.0,
            power_usage=250.0,
            power_limit=450.0,
        )

        assert info.index == 0
        assert info.name == "NVIDIA GeForce RTX 4090"
        assert info.utilization_gpu == 0.75
        assert info.temperature == 65.0

    def test_memory_utilization(self) -> None:
        """Test memory utilization property."""
        info = GPUInfo(
            index=0,
            name="GPU",
            uuid="",
            memory_total=1000,
            memory_used=250,
            memory_free=750,
            utilization_gpu=0.0,
            utilization_memory=0.0,
            temperature=0.0,
            power_usage=0.0,
            power_limit=0.0,
        )

        assert info.memory_utilization == 0.25

    def test_memory_utilization_zero_total(self) -> None:
        """Test memory utilization with zero total."""
        info = GPUInfo(
            index=0,
            name="",
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

        assert info.memory_utilization == 0.0


class TestGPUMonitor:
    """Tests for GPUMonitor class."""

    def test_creation(self) -> None:
        """Test monitor creation."""
        monitor = GPUMonitor()
        # Should not raise
        assert isinstance(monitor._nvml_available, bool)
        assert isinstance(monitor._device_count, int)

    def test_is_available_property(self) -> None:
        """Test is_available property."""
        monitor = GPUMonitor()
        # Will be True only if NVML is available and devices exist
        assert isinstance(monitor.is_available, bool)

    def test_device_count_property(self) -> None:
        """Test device_count property."""
        monitor = GPUMonitor()
        assert monitor.device_count >= 0

    def test_get_gpu_info(self) -> None:
        """Test getting GPU info."""
        monitor = GPUMonitor()
        info = monitor.get_gpu_info(0)

        assert isinstance(info, GPUInfo)
        assert info.index == 0

    def test_get_gpu_info_not_available(self) -> None:
        """Test get_gpu_info when not available."""
        monitor = GPUMonitor()

        if not monitor.is_available:
            info = monitor.get_gpu_info(0)
            assert info.name == "N/A"
            assert info.memory_total == 0

    def test_get_all_gpu_info(self) -> None:
        """Test getting info for all GPUs."""
        monitor = GPUMonitor()
        all_info = monitor.get_all_gpu_info()

        assert isinstance(all_info, list)
        assert len(all_info) == monitor.device_count

    def test_get_utilization(self) -> None:
        """Test getting GPU utilization."""
        monitor = GPUMonitor()
        util = monitor.get_utilization(0)

        assert isinstance(util, float)
        assert 0.0 <= util <= 1.0

    def test_get_memory_info(self) -> None:
        """Test getting memory info."""
        monitor = GPUMonitor()
        mem_info = monitor.get_memory_info(0)

        assert "used" in mem_info
        assert "free" in mem_info
        assert "total" in mem_info
        assert isinstance(mem_info["used"], int)

    def test_shutdown(self) -> None:
        """Test shutdown."""
        monitor = GPUMonitor()
        monitor.shutdown()
        assert monitor._nvml_available is False

    def test_repr(self) -> None:
        """Test string representation."""
        monitor = GPUMonitor()
        repr_str = repr(monitor)

        assert "GPUMonitor" in repr_str
        assert "available=" in repr_str
        assert "devices=" in repr_str


class TestTelemetryCollector:
    """Tests for TelemetryCollector class."""

    @pytest.fixture
    def collector(self) -> TelemetryCollector:
        """Provide a fresh collector instance."""
        return TelemetryCollector()

    def test_creation(self, collector: TelemetryCollector) -> None:
        """Test collector creation."""
        assert collector._kernel_telemetry == {}
        assert isinstance(collector._gpu_monitor, GPUMonitor)

    def test_register_kernel(self, collector: TelemetryCollector) -> None:
        """Test registering a kernel."""
        telemetry = collector.register_kernel("kernel1")

        assert isinstance(telemetry, RingKernelTelemetry)
        assert telemetry.kernel_id == "kernel1"
        assert "kernel1" in collector._kernel_telemetry

    def test_register_kernel_idempotent(self, collector: TelemetryCollector) -> None:
        """Test that registering same kernel returns same instance."""
        t1 = collector.register_kernel("kernel")
        t2 = collector.register_kernel("kernel")

        assert t1 is t2

    def test_unregister_kernel(self, collector: TelemetryCollector) -> None:
        """Test unregistering a kernel."""
        collector.register_kernel("kernel")
        assert "kernel" in collector._kernel_telemetry

        collector.unregister_kernel("kernel")
        assert "kernel" not in collector._kernel_telemetry

    def test_unregister_nonexistent(self, collector: TelemetryCollector) -> None:
        """Test unregistering non-existent kernel."""
        # Should not raise
        collector.unregister_kernel("nonexistent")

    def test_get_kernel_telemetry(self, collector: TelemetryCollector) -> None:
        """Test getting kernel telemetry."""
        collector.register_kernel("kernel")
        telemetry = collector.get_kernel_telemetry("kernel")

        assert telemetry is not None
        assert telemetry.kernel_id == "kernel"

    def test_get_kernel_telemetry_nonexistent(self, collector: TelemetryCollector) -> None:
        """Test getting non-existent kernel telemetry."""
        result = collector.get_kernel_telemetry("nonexistent")
        assert result is None

    def test_get_all_telemetry(self, collector: TelemetryCollector) -> None:
        """Test getting all telemetry."""
        collector.register_kernel("k1")
        collector.register_kernel("k2")
        collector.register_kernel("k3")

        all_telemetry = collector.get_all_telemetry()

        assert len(all_telemetry) == 3
        assert "k1" in all_telemetry
        assert "k2" in all_telemetry

    def test_get_all_telemetry_updates_uptime(self, collector: TelemetryCollector) -> None:
        """Test that get_all_telemetry updates uptime."""
        t = collector.register_kernel("kernel")
        time.sleep(0.1)

        collector.get_all_telemetry()

        assert t.uptime_seconds > 0

    def test_get_summary(self, collector: TelemetryCollector) -> None:
        """Test getting telemetry summary."""
        t1 = collector.register_kernel("k1")
        t2 = collector.register_kernel("k2")

        t1.record_message(1000)
        t1.record_message(2000)
        t2.record_message(3000)
        t1.record_drop()
        t2.record_error("error")

        summary = collector.get_summary()

        assert summary["kernel_count"] == 2
        assert summary["total_messages_processed"] == 3
        assert summary["total_messages_dropped"] == 1
        assert summary["total_errors"] == 1
        assert "uptime_seconds" in summary
        assert "gpu_utilization" in summary

    def test_reset_all(self, collector: TelemetryCollector) -> None:
        """Test resetting all telemetry."""
        t1 = collector.register_kernel("k1")
        t2 = collector.register_kernel("k2")

        t1.record_message(1000)
        t2.record_message(2000)

        collector.reset_all()

        assert t1.messages_processed == 0
        assert t2.messages_processed == 0


class TestTelemetryEdgeCases:
    """Edge case tests for telemetry."""

    def test_high_throughput_tracking(self) -> None:
        """Test tracking high throughput."""
        telemetry = RingKernelTelemetry(kernel_id="high_throughput")

        # Simulate 1 million messages
        for _ in range(1000):
            telemetry.record_message(100)  # 100ns each

        assert telemetry.messages_processed == 1000
        assert telemetry.avg_latency_ns == 100

    def test_latency_edge_values(self) -> None:
        """Test latency with edge values."""
        telemetry = RingKernelTelemetry()

        telemetry.record_message(0)  # Zero latency
        assert telemetry.min_latency_ns == 0

        telemetry.record_message(1_000_000_000)  # 1 second
        assert telemetry.max_latency_ns == 1_000_000_000

    def test_collector_with_many_kernels(self) -> None:
        """Test collector with many kernels."""
        collector = TelemetryCollector()

        for i in range(100):
            collector.register_kernel(f"kernel_{i}")

        summary = collector.get_summary()
        assert summary["kernel_count"] == 100
