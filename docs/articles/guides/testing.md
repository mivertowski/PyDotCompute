# Testing Ring Kernels

Comprehensive guide to testing PyDotCompute actors.

## Overview

Testing ring kernels requires handling async operations, lifecycle management, and potentially GPU resources. This guide covers strategies for unit tests, integration tests, and performance tests.

## Test Setup

### pytest Configuration

```python
# conftest.py
import pytest
import asyncio

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def runtime():
    """Provide a fresh runtime for each test."""
    from pydotcompute import RingKernelRuntime

    async with RingKernelRuntime() as rt:
        yield rt
```

### pyproject.toml

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "gpu: marks tests requiring GPU",
]
```

## Unit Testing Actors

### Basic Actor Test

```python
import pytest
from pydotcompute import RingKernelRuntime, ring_kernel, message
from dataclasses import dataclass, field
from uuid import UUID, uuid4

@message
@dataclass
class EchoRequest:
    value: str
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None

@message
@dataclass
class EchoResponse:
    value: str
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None

@ring_kernel(kernel_id="echo", auto_register=False)
async def echo_actor(ctx):
    while not ctx.should_terminate:
        if not ctx.is_active:
            await ctx.wait_active()
            continue
        try:
            msg = await ctx.receive(timeout=0.1)
            await ctx.send(EchoResponse(
                value=msg.value,
                correlation_id=msg.message_id,
            ))
        except:
            continue

@pytest.mark.asyncio
async def test_echo_basic():
    """Test basic echo functionality."""
    async with RingKernelRuntime() as runtime:
        await runtime.launch("echo", echo_actor)
        await runtime.activate("echo")

        await asyncio.sleep(0.1)

        request = EchoRequest(value="hello")
        await runtime.send("echo", request)

        response = await runtime.receive("echo", timeout=1.0)

        assert response.value == "hello"
        assert response.correlation_id == request.message_id
```

### Testing Error Cases

```python
@pytest.mark.asyncio
async def test_calculator_divide_by_zero():
    """Test division by zero handling."""
    async with RingKernelRuntime() as runtime:
        await runtime.launch("calculator", calculator_actor)
        await runtime.activate("calculator")

        await asyncio.sleep(0.1)

        request = CalculateRequest(a=10, b=0, operation="div")
        await runtime.send("calculator", request)

        response = await runtime.receive("calculator", timeout=1.0)

        assert not response.success
        assert "zero" in response.error.lower()

@pytest.mark.asyncio
async def test_calculator_invalid_operation():
    """Test invalid operation handling."""
    async with RingKernelRuntime() as runtime:
        await runtime.launch("calculator", calculator_actor)
        await runtime.activate("calculator")

        await asyncio.sleep(0.1)

        request = CalculateRequest(a=1, b=2, operation="invalid")
        await runtime.send("calculator", request)

        response = await runtime.receive("calculator", timeout=1.0)

        assert not response.success
```

### Testing Multiple Operations

```python
@pytest.mark.asyncio
async def test_calculator_multiple_operations():
    """Test multiple operations in sequence."""
    test_cases = [
        (10, 5, "add", 15.0),
        (10, 5, "sub", 5.0),
        (10, 5, "mul", 50.0),
        (10, 2, "div", 5.0),
    ]

    async with RingKernelRuntime() as runtime:
        await runtime.launch("calculator", calculator_actor)
        await runtime.activate("calculator")

        await asyncio.sleep(0.1)

        for a, b, op, expected in test_cases:
            request = CalculateRequest(a=a, b=b, operation=op)
            await runtime.send("calculator", request)

            response = await runtime.receive("calculator", timeout=1.0)

            assert response.success, f"Failed for {a} {op} {b}"
            assert response.result == expected, f"Expected {expected}, got {response.result}"
```

## Testing Lifecycle

### Deactivation and Reactivation

```python
@pytest.mark.asyncio
async def test_deactivate_reactivate():
    """Test pause and resume functionality."""
    async with RingKernelRuntime() as runtime:
        await runtime.launch("worker", worker_actor)
        await runtime.activate("worker")

        await asyncio.sleep(0.1)

        # Send while active
        await runtime.send("worker", Request(data="first"))
        response1 = await runtime.receive("worker", timeout=1.0)
        assert response1.success

        # Deactivate
        await runtime.deactivate("worker")
        assert runtime.get_state("worker") == KernelState.DEACTIVATED

        # Reactivate
        await runtime.reactivate("worker")
        assert runtime.get_state("worker") == KernelState.ACTIVE

        await asyncio.sleep(0.1)

        # Send after reactivation
        await runtime.send("worker", Request(data="second"))
        response2 = await runtime.receive("worker", timeout=1.0)
        assert response2.success
```

### Graceful Shutdown

```python
@pytest.mark.asyncio
async def test_graceful_shutdown():
    """Test that shutdown completes gracefully."""
    async with RingKernelRuntime() as runtime:
        await runtime.launch("worker", worker_actor)
        await runtime.activate("worker")

        await asyncio.sleep(0.1)

        # Terminate should complete without error
        await runtime.terminate("worker", timeout=5.0)

        assert runtime.get_state("worker") == KernelState.TERMINATED
```

## Testing Pipelines

### Two-Stage Pipeline

```python
@pytest.mark.asyncio
async def test_two_stage_pipeline():
    """Test a two-stage processing pipeline."""
    async with RingKernelRuntime() as runtime:
        await runtime.launch("stage1", stage1_actor)
        await runtime.launch("stage2", stage2_actor)

        await runtime.activate("stage1")
        await runtime.activate("stage2")

        await asyncio.sleep(0.1)

        # Send to stage 1
        await runtime.send("stage1", RawData(values=[1, 2, 3]))

        # Get from stage 1
        stage1_output = await runtime.receive("stage1", timeout=1.0)
        assert stage1_output.processed is not None

        # Send to stage 2
        await runtime.send("stage2", stage1_output)

        # Get final result
        final = await runtime.receive("stage2", timeout=1.0)
        assert final.result is not None
```

## GPU Testing

### Skip if No GPU

```python
import pytest

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

@pytest.mark.skipif(not HAS_GPU, reason="CUDA not available")
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_gpu_computation():
    """Test GPU-accelerated computation."""
    async with RingKernelRuntime(backend="cuda") as runtime:
        await runtime.launch("gpu_worker", gpu_worker_actor)
        await runtime.activate("gpu_worker")

        await asyncio.sleep(0.1)

        # Large array to ensure GPU is used
        data = list(range(100000))
        await runtime.send("gpu_worker", ComputeRequest(data=data))

        response = await runtime.receive("gpu_worker", timeout=5.0)
        assert response.success
```

### GPU Memory Tests

```python
@pytest.mark.skipif(not HAS_GPU, reason="CUDA not available")
@pytest.mark.gpu
def test_unified_buffer_sync():
    """Test UnifiedBuffer host-device synchronization."""
    from pydotcompute import UnifiedBuffer

    buf = UnifiedBuffer((1000,), dtype=np.float32)

    # Write on host
    buf.host[:] = np.arange(1000, dtype=np.float32)

    # Access on device (triggers sync)
    device_data = buf.device

    # Verify
    cp.testing.assert_array_equal(device_data, cp.arange(1000, dtype=cp.float32))
```

## Performance Testing

### Throughput Test

```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_throughput():
    """Measure message throughput."""
    import time

    num_messages = 1000

    async with RingKernelRuntime(enable_telemetry=True) as runtime:
        await runtime.launch("worker", fast_worker_actor)
        await runtime.activate("worker")

        await asyncio.sleep(0.1)

        start = time.perf_counter()

        # Send all messages
        for i in range(num_messages):
            await runtime.send("worker", Request(data=i))

        # Receive all responses
        for _ in range(num_messages):
            await runtime.receive("worker", timeout=5.0)

        elapsed = time.perf_counter() - start

        throughput = num_messages / elapsed
        print(f"Throughput: {throughput:.0f} msg/s")

        # Assert minimum throughput
        assert throughput > 100, f"Throughput too low: {throughput}"
```

### Latency Test

```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_latency():
    """Measure round-trip latency."""
    import time
    import statistics

    latencies = []

    async with RingKernelRuntime() as runtime:
        await runtime.launch("worker", fast_worker_actor)
        await runtime.activate("worker")

        await asyncio.sleep(0.1)

        for _ in range(100):
            start = time.perf_counter()

            await runtime.send("worker", Request(data=1))
            await runtime.receive("worker", timeout=1.0)

            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)  # ms

        avg_latency = statistics.mean(latencies)
        p99_latency = sorted(latencies)[99]

        print(f"Avg latency: {avg_latency:.2f} ms")
        print(f"P99 latency: {p99_latency:.2f} ms")

        assert avg_latency < 10, f"Average latency too high: {avg_latency}"
```

## Test Fixtures

### Common Fixtures

```python
# conftest.py
import pytest
from pydotcompute import RingKernelRuntime

@pytest.fixture
async def runtime():
    """Fresh runtime for each test."""
    async with RingKernelRuntime() as rt:
        yield rt

@pytest.fixture
async def telemetry_runtime():
    """Runtime with telemetry enabled."""
    async with RingKernelRuntime(enable_telemetry=True) as rt:
        yield rt

@pytest.fixture
async def active_calculator(runtime):
    """Pre-configured calculator actor."""
    await runtime.launch("calculator", calculator_actor)
    await runtime.activate("calculator")
    await asyncio.sleep(0.1)
    return runtime
```

### Using Fixtures

```python
@pytest.mark.asyncio
async def test_with_fixture(active_calculator):
    """Test using the pre-configured fixture."""
    await active_calculator.send("calculator", CalculateRequest(a=1, b=2))
    response = await active_calculator.receive("calculator", timeout=1.0)
    assert response.result == 3
```

## Mocking

### Mock Messages

```python
from unittest.mock import Mock, patch

def test_message_serialization():
    """Test message serialization without runtime."""
    msg = CalculateRequest(a=1.5, b=2.5, operation="add")

    data = msg.serialize()
    restored = CalculateRequest.deserialize(data)

    assert restored.a == 1.5
    assert restored.b == 2.5
    assert restored.operation == "add"
```

## Best Practices

1. **Isolate Tests**: Use fresh runtime per test

2. **Use Timeouts**: Prevent hanging tests

3. **Test Error Cases**: Verify error handling

4. **Mark Slow Tests**: Use `@pytest.mark.slow`

5. **Skip GPU Tests**: Use `skipif` for CI without GPU

6. **Measure Performance**: Track regressions

7. **Clean Up**: Use async context managers

## Next Steps

- [Building Actors](building-actors.md): Actor design
- [GPU Optimization](gpu-optimization.md): Performance tuning
