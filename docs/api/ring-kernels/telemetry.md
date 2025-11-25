# Telemetry

Performance monitoring and metrics collection for ring kernels.

## Overview

The telemetry system collects performance metrics for ring kernels, including message throughput, latency, queue statistics, and resource utilization.

```python
from pydotcompute import RingKernelRuntime

async with RingKernelRuntime(enable_telemetry=True) as runtime:
    # ... run kernels ...

    # Get per-kernel metrics
    telemetry = runtime.get_telemetry("my_kernel")
    print(f"Throughput: {telemetry.throughput:.2f} msg/s")

    # Get runtime summary
    summary = runtime.get_summary()
    print(f"Total messages: {summary['total_messages']}")
```

## Classes

### KernelTelemetry

```python
@dataclass
class KernelTelemetry:
    """Telemetry data for a single kernel."""

    kernel_id: str
    state: KernelState
    messages_received: int = 0
    messages_sent: int = 0
    messages_processed: int = 0
    errors: int = 0
    start_time: float | None = None
    last_message_time: float | None = None

    # Derived metrics
    @property
    def throughput(self) -> float:
        """Messages processed per second."""

    @property
    def uptime(self) -> float:
        """Seconds since kernel started."""

    @property
    def error_rate(self) -> float:
        """Errors per message processed."""
```

### TelemetryCollector

```python
class TelemetryCollector:
    """Collects and aggregates telemetry from multiple kernels."""

    def __init__(self, enabled: bool = True) -> None:
        """
        Create a telemetry collector.

        Args:
            enabled: Whether to collect metrics
        """
```

## KernelTelemetry Properties

### messages_received

```python
messages_received: int = 0
```

Total messages received by the kernel's input queue.

### messages_sent

```python
messages_sent: int = 0
```

Total messages sent to the kernel's output queue.

### messages_processed

```python
messages_processed: int = 0
```

Messages successfully processed (received and responded to).

### errors

```python
errors: int = 0
```

Number of errors encountered during processing.

### throughput

```python
@property
def throughput(self) -> float:
    """Calculate messages processed per second."""
```

### uptime

```python
@property
def uptime(self) -> float:
    """Calculate seconds since kernel started."""
```

### error_rate

```python
@property
def error_rate(self) -> float:
    """Calculate errors as fraction of processed messages."""
```

## TelemetryCollector Methods

### record_receive

```python
def record_receive(self, kernel_id: str) -> None:
    """Record a message received event."""
```

### record_send

```python
def record_send(self, kernel_id: str) -> None:
    """Record a message sent event."""
```

### record_error

```python
def record_error(self, kernel_id: str, error: Exception) -> None:
    """Record an error event."""
```

### get_telemetry

```python
def get_telemetry(self, kernel_id: str) -> KernelTelemetry | None:
    """Get telemetry for a specific kernel."""
```

### get_all_telemetry

```python
def get_all_telemetry(self) -> dict[str, KernelTelemetry]:
    """Get telemetry for all kernels."""
```

### get_summary

```python
def get_summary(self) -> dict[str, Any]:
    """
    Get aggregated summary across all kernels.

    Returns:
        Dictionary with:
        - kernel_count: Number of kernels
        - total_messages: Total messages processed
        - total_errors: Total errors
        - aggregate_throughput: Combined throughput
        - per_kernel: Individual kernel summaries
    """
```

### reset

```python
def reset(self, kernel_id: str | None = None) -> None:
    """
    Reset telemetry counters.

    Args:
        kernel_id: Specific kernel to reset, or None for all
    """
```

## Usage Examples

### Enable Telemetry

```python
from pydotcompute import RingKernelRuntime

# Telemetry is disabled by default for performance
async with RingKernelRuntime(enable_telemetry=True) as runtime:
    await runtime.launch("worker")
    await runtime.activate("worker")

    # Process messages...
    for i in range(100):
        await runtime.send("worker", Request(data=i))
        await runtime.receive("worker", timeout=1.0)

    # Get metrics
    telemetry = runtime.get_telemetry("worker")
    print(f"Processed: {telemetry.messages_processed}")
    print(f"Throughput: {telemetry.throughput:.1f} msg/s")
    print(f"Uptime: {telemetry.uptime:.1f}s")
```

### Monitor Multiple Kernels

```python
async with RingKernelRuntime(enable_telemetry=True) as runtime:
    # Launch multiple kernels
    for i in range(3):
        await runtime.launch(f"worker_{i}")
        await runtime.activate(f"worker_{i}")

    # Process work...

    # Get summary
    summary = runtime.get_summary()
    print(f"Active kernels: {summary['kernel_count']}")
    print(f"Total messages: {summary['total_messages']}")
    print(f"Total throughput: {summary['aggregate_throughput']:.1f} msg/s")

    # Individual stats
    for kernel_id, stats in summary['per_kernel'].items():
        print(f"  {kernel_id}: {stats['throughput']:.1f} msg/s")
```

### Error Tracking

```python
telemetry = runtime.get_telemetry("worker")

if telemetry.error_rate > 0.01:  # More than 1% errors
    print(f"Warning: High error rate: {telemetry.error_rate:.1%}")
    print(f"Errors: {telemetry.errors}")
```

### Periodic Monitoring

```python
import asyncio

async def monitor(runtime: RingKernelRuntime, interval: float = 5.0):
    """Periodically print telemetry."""
    while True:
        await asyncio.sleep(interval)

        summary = runtime.get_summary()
        print(f"\n=== Telemetry @ {time.time():.0f} ===")
        print(f"Kernels: {summary['kernel_count']}")
        print(f"Messages: {summary['total_messages']}")
        print(f"Throughput: {summary['aggregate_throughput']:.1f} msg/s")
        print(f"Errors: {summary['total_errors']}")

# Run alongside your application
async with RingKernelRuntime(enable_telemetry=True) as runtime:
    monitor_task = asyncio.create_task(monitor(runtime))

    # Main work...

    monitor_task.cancel()
```

### Reset Counters

```python
# Reset specific kernel
runtime.get_telemetry("worker")  # Get current stats
# ... analyze ...
runtime.reset_telemetry("worker")  # Start fresh count

# Reset all
runtime.reset_telemetry()
```

## Metrics Reference

| Metric | Type | Description |
|--------|------|-------------|
| `messages_received` | Counter | Input queue receives |
| `messages_sent` | Counter | Output queue sends |
| `messages_processed` | Counter | Successful request-response pairs |
| `errors` | Counter | Exceptions during processing |
| `start_time` | Timestamp | When kernel started |
| `last_message_time` | Timestamp | Most recent message |
| `throughput` | Gauge | Messages/second (derived) |
| `uptime` | Gauge | Seconds running (derived) |
| `error_rate` | Gauge | Errors/message (derived) |

## Performance Notes

- Telemetry collection adds minimal overhead (~1-2%)
- Counters use atomic operations for thread safety
- Disable telemetry in production if not needed
- Summary calculations are done on-demand

## Notes

- Telemetry is opt-in via `enable_telemetry=True`
- Metrics persist until reset or runtime exit
- GPU-specific metrics require pynvml (separate module)
- Throughput is calculated over the kernel's uptime
