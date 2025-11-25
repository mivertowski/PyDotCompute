"""
Unit tests for Accelerator.
"""

from __future__ import annotations

import pytest

from pydotcompute.core.accelerator import (
    Accelerator,
    DeviceProperties,
    DeviceType,
    cuda_available,
    get_accelerator,
)


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_device_types(self) -> None:
        """Test all device types are defined."""
        assert DeviceType.CPU is not None
        assert DeviceType.CUDA is not None
        assert DeviceType.METAL is not None


class TestDeviceProperties:
    """Tests for DeviceProperties dataclass."""

    def test_cpu_properties(self) -> None:
        """Test CPU device properties."""
        props = DeviceProperties(
            device_id=0,
            device_type=DeviceType.CPU,
            name="CPU",
            compute_capability=None,
            total_memory=0,
            multiprocessor_count=1,
            max_threads_per_block=1,
            max_block_dims=(1, 1, 1),
            max_grid_dims=(1, 1, 1),
            warp_size=1,
            is_available=True,
        )

        assert props.device_type == DeviceType.CPU
        assert props.compute_capability_str == "N/A"
        assert props.total_memory_gb == 0.0

    def test_cuda_properties(self) -> None:
        """Test CUDA device properties."""
        props = DeviceProperties(
            device_id=0,
            device_type=DeviceType.CUDA,
            name="Test GPU",
            compute_capability=(8, 9),
            total_memory=8 * 1024**3,  # 8 GB
            multiprocessor_count=60,
            max_threads_per_block=1024,
            max_block_dims=(1024, 1024, 64),
            max_grid_dims=(2**31 - 1, 65535, 65535),
            warp_size=32,
            is_available=True,
        )

        assert props.device_type == DeviceType.CUDA
        assert props.compute_capability_str == "8.9"
        assert props.total_memory_gb == 8.0


class TestAccelerator:
    """Tests for Accelerator class."""

    def test_singleton(self) -> None:
        """Test that Accelerator is a singleton."""
        # Reset singleton for testing
        Accelerator._instance = None
        Accelerator._initialized = False

        acc1 = Accelerator()
        acc2 = Accelerator()

        assert acc1 is acc2

    def test_device_count(self) -> None:
        """Test device count is at least 1 (CPU)."""
        Accelerator._instance = None
        Accelerator._initialized = False

        acc = Accelerator()

        assert acc.device_count >= 1

    def test_current_device(self) -> None:
        """Test current device access."""
        Accelerator._instance = None
        Accelerator._initialized = False

        acc = Accelerator()

        device = acc.current_device
        assert isinstance(device, DeviceProperties)
        assert device.is_available

    def test_devices_list(self) -> None:
        """Test devices list access."""
        Accelerator._instance = None
        Accelerator._initialized = False

        acc = Accelerator()

        devices = acc.devices
        assert isinstance(devices, list)
        assert len(devices) >= 1

        # Should be a copy
        devices.append(None)  # type: ignore
        assert len(acc.devices) >= 1

    def test_get_device(self) -> None:
        """Test get_device method."""
        Accelerator._instance = None
        Accelerator._initialized = False

        acc = Accelerator()

        device = acc.get_device(0)
        assert isinstance(device, DeviceProperties)

        with pytest.raises(ValueError):
            acc.get_device(-1)

        with pytest.raises(ValueError):
            acc.get_device(1000)

    def test_set_device(self) -> None:
        """Test set_device method."""
        Accelerator._instance = None
        Accelerator._initialized = False

        acc = Accelerator()

        # Setting to 0 should always work (at least CPU)
        acc.set_device(0)

        with pytest.raises(ValueError):
            acc.set_device(-1)

    def test_synchronize(self) -> None:
        """Test synchronize method doesn't raise."""
        Accelerator._instance = None
        Accelerator._initialized = False

        acc = Accelerator()

        # Should not raise
        acc.synchronize()

    def test_get_memory_info(self) -> None:
        """Test get_memory_info method."""
        Accelerator._instance = None
        Accelerator._initialized = False

        acc = Accelerator()

        mem_info = acc.get_memory_info()

        assert "free" in mem_info
        assert "total" in mem_info
        assert "used" in mem_info

    def test_repr(self) -> None:
        """Test string representation."""
        Accelerator._instance = None
        Accelerator._initialized = False

        acc = Accelerator()

        repr_str = repr(acc)

        assert "Accelerator" in repr_str
        assert "devices=" in repr_str


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_accelerator(self) -> None:
        """Test get_accelerator function."""
        Accelerator._instance = None
        Accelerator._initialized = False

        acc = get_accelerator()

        assert isinstance(acc, Accelerator)

    def test_cuda_available(self) -> None:
        """Test cuda_available function."""
        Accelerator._instance = None
        Accelerator._initialized = False

        # Should return a boolean
        result = cuda_available()

        assert isinstance(result, bool)
