# Accelerator

Device detection and management for CPU and GPU compute resources.

## Overview

The `Accelerator` class provides a unified interface for detecting and managing compute devices. It abstracts away the differences between CPU and GPU hardware.

```python
from pydotcompute import get_accelerator, Accelerator

# Get the global accelerator instance
acc = get_accelerator()

# Check available devices
print(f"Device count: {acc.device_count}")
for device in acc.devices:
    print(f"  {device.name}: {device.device_type.name}")
```

## Classes

### DeviceType

```python
class DeviceType(Enum):
    """Type of compute device."""
    CPU = "cpu"
    CUDA = "cuda"
```

### DeviceInfo

```python
@dataclass
class DeviceInfo:
    """Information about a compute device."""
    device_id: int
    device_type: DeviceType
    name: str
    total_memory: int  # bytes
    compute_capability: tuple[int, int] | None  # CUDA only
```

### Accelerator

```python
class Accelerator:
    """Manages compute devices and provides device information."""
```

## Functions

### get_accelerator

```python
def get_accelerator() -> Accelerator:
    """Get the global accelerator instance (singleton)."""
```

### cuda_available

```python
def cuda_available() -> bool:
    """Check if CUDA is available."""
```

## Properties

### devices

```python
@property
def devices(self) -> list[DeviceInfo]:
    """List of all available compute devices."""
```

### device_count

```python
@property
def device_count(self) -> int:
    """Number of available devices."""
```

### cuda_available

```python
@property
def cuda_available(self) -> bool:
    """Whether CUDA devices are available."""
```

### current_device

```python
@property
def current_device(self) -> DeviceInfo:
    """The currently selected device."""
```

## Methods

### select_device

```python
def select_device(self, device_id: int) -> None:
    """Select a device by ID for subsequent operations."""
```

### get_device

```python
def get_device(self, device_id: int) -> DeviceInfo:
    """Get information about a specific device."""
```

### get_memory_info

```python
def get_memory_info(self, device_id: int | None = None) -> tuple[int, int]:
    """Get (free, total) memory in bytes for a device."""
```

## Usage Examples

### Basic Device Detection

```python
from pydotcompute import get_accelerator

acc = get_accelerator()

# List all devices
for device in acc.devices:
    print(f"Device {device.device_id}: {device.name}")
    print(f"  Type: {device.device_type.name}")
    print(f"  Memory: {device.total_memory / 1e9:.1f} GB")
    if device.compute_capability:
        print(f"  Compute Capability: {device.compute_capability}")
```

### Check CUDA Availability

```python
from pydotcompute.core.accelerator import cuda_available

if cuda_available():
    print("CUDA is available!")
else:
    print("Running in CPU mode")
```

### Memory Monitoring

```python
acc = get_accelerator()

# Get memory for current device
free, total = acc.get_memory_info()
print(f"Memory: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

# Get memory for specific GPU
if acc.cuda_available:
    free, total = acc.get_memory_info(device_id=0)
    print(f"GPU 0: {free / 1e9:.1f} GB free")
```

## Notes

- The `Accelerator` is a singleton - use `get_accelerator()` to access it
- CPU is always available as device 0
- CUDA devices are numbered starting from 1 when available
- Memory info for CPU returns system RAM information
