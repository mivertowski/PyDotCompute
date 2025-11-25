# DotCompute Comparison

PyDotCompute's relationship to the DotCompute project.

## What is DotCompute?

**DotCompute** is a .NET library for GPU-native actor systems, implementing the Ring Kernel System for high-performance GPU computing with the actor model.

## PyDotCompute: A Python Port

PyDotCompute brings DotCompute's concepts to Python, adapting them for the Python ecosystem.

### Design Philosophy

Both projects share core principles:

1. **Persistent Kernels**: GPU actors that run continuously
2. **Message Passing**: Typed, async communication
3. **Unified Memory**: Seamless host-device data management
4. **Two-Phase Launch**: Separate allocation from activation
5. **Graceful Lifecycle**: Structured state transitions

## Feature Comparison

| Feature | DotCompute (.NET) | PyDotCompute (Python) |
|---------|-------------------|----------------------|
| Language | C#, F# | Python |
| Runtime | .NET CLR | asyncio |
| GPU Backend | CUDA.NET | Numba, CuPy |
| Serialization | Binary | msgpack |
| Type Safety | Compile-time | Runtime + type hints |
| Memory | Spans, Memory<T> | NumPy arrays |
| Async Model | Task, async/await | asyncio, async/await |

## API Comparison

### Actor Definition

**DotCompute (C#):**

```csharp
[RingKernel("processor")]
public class ProcessorKernel : IRingKernel<Request, Response>
{
    public async Task RunAsync(IKernelContext<Request, Response> ctx)
    {
        while (!ctx.ShouldTerminate)
        {
            var request = await ctx.ReceiveAsync();
            var response = Process(request);
            await ctx.SendAsync(response);
        }
    }
}
```

**PyDotCompute (Python):**

```python
@ring_kernel(kernel_id="processor")
async def processor(ctx: KernelContext[Request, Response]):
    while not ctx.should_terminate:
        request = await ctx.receive()
        response = process(request)
        await ctx.send(response)
```

### Runtime Usage

**DotCompute (C#):**

```csharp
await using var runtime = new RingKernelRuntime();
await runtime.LaunchAsync("processor");
await runtime.ActivateAsync("processor");

await runtime.SendAsync("processor", new Request { Data = "hello" });
var response = await runtime.ReceiveAsync<Response>("processor");
```

**PyDotCompute (Python):**

```python
async with RingKernelRuntime() as runtime:
    await runtime.launch("processor")
    await runtime.activate("processor")

    await runtime.send("processor", Request(data="hello"))
    response = await runtime.receive("processor")
```

### Message Types

**DotCompute (C#):**

```csharp
public record Request(
    string Data,
    Guid MessageId = default,
    int Priority = 128,
    Guid? CorrelationId = null
) : IRingKernelMessage;
```

**PyDotCompute (Python):**

```python
@message
@dataclass
class Request:
    data: str
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None
```

## Architecture Comparison

### DotCompute Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DotCompute (.NET)                         │
├─────────────────────────────────────────────────────────────┤
│  Managed Code (C#/F#)                                        │
│  ├── RingKernelRuntime                                       │
│  ├── Channel<T> (System.Threading.Channels)                  │
│  └── IRingKernel<TIn, TOut>                                  │
├─────────────────────────────────────────────────────────────┤
│  Interop Layer                                               │
│  ├── CUDA.NET (ManagedCuda)                                  │
│  └── Memory<T>, Span<T>                                      │
├─────────────────────────────────────────────────────────────┤
│  Native CUDA                                                 │
│  └── PTX Kernels, cuBLAS, cuDNN                              │
└─────────────────────────────────────────────────────────────┘
```

### PyDotCompute Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PyDotCompute (Python)                     │
├─────────────────────────────────────────────────────────────┤
│  Python Code                                                 │
│  ├── RingKernelRuntime                                       │
│  ├── asyncio.Queue                                           │
│  └── @ring_kernel decorator                                  │
├─────────────────────────────────────────────────────────────┤
│  Scientific Python Stack                                     │
│  ├── NumPy (arrays)                                          │
│  ├── CuPy (GPU arrays)                                       │
│  └── Numba (JIT compilation)                                 │
├─────────────────────────────────────────────────────────────┤
│  Native CUDA                                                 │
│  └── PTX Kernels via Numba/CuPy                              │
└─────────────────────────────────────────────────────────────┘
```

## Python-Specific Adaptations

### 1. Decorators Instead of Interfaces

Python uses decorators for cleaner syntax:

```python
# Python's decorator pattern
@ring_kernel(kernel_id="worker")
async def worker(ctx):
    ...

# vs C#'s interface pattern
public class Worker : IRingKernel<TIn, TOut>
{
    ...
}
```

### 2. Duck Typing

Python's flexibility allows simpler message types:

```python
@message
@dataclass
class MyMessage:
    data: str
    # Works without explicit interface implementation
```

### 3. asyncio Integration

Native Python async:

```python
async with RingKernelRuntime() as runtime:
    # Natural async context manager
    await runtime.launch("worker")
```

### 4. NumPy Ecosystem

Leverage scientific Python:

```python
from pydotcompute import UnifiedBuffer
import numpy as np

buf = UnifiedBuffer((1000,), dtype=np.float32)
buf.host[:] = np.random.randn(1000)  # NumPy operations
```

## Why PyDotCompute?

### For Python Developers

- **Familiar Tools**: NumPy, asyncio, type hints
- **Easy Installation**: `pip install pydotcompute`
- **Interactive Development**: Works in Jupyter notebooks
- **Rich Ecosystem**: Integration with ML frameworks

### For DotCompute Users

- **Cross-Platform**: Share concepts across teams
- **Prototyping**: Quick experimentation in Python
- **ML Integration**: Access to Python ML libraries

### For New Users

- **Gentle Learning Curve**: Python is accessible
- **Great Documentation**: Examples and guides
- **Active Community**: Python GPU computing

## Migration Path

### From DotCompute to PyDotCompute

1. **Concepts Transfer**: Same mental model
2. **API Similarity**: Similar method names
3. **Message Types**: Adapt to dataclasses
4. **Runtime**: Use asyncio instead of Tasks

### From Traditional Python GPU

1. **Keep NumPy**: UnifiedBuffer wraps NumPy
2. **Keep CuPy**: Works alongside PyDotCompute
3. **Add Structure**: Actors provide organization
4. **Add Async**: Embrace async/await

## Future Directions

Both projects continue to evolve:

| Area | DotCompute | PyDotCompute |
|------|------------|--------------|
| Multi-GPU | Native support | Planned |
| Clustering | In development | Future |
| Tensor Cores | Available | Via CuPy |
| ROCm (AMD) | Investigating | Via CuPy |

## Resources

- **DotCompute**: Original .NET implementation
- **PyDotCompute**: This project
- **CUDA**: NVIDIA GPU computing
- **CuPy**: GPU array library
- **Numba**: JIT compiler

## Next Steps

- [Actor Model Background](actor-model.md): Theoretical foundation
- [Quick Start](../../getting-started/quickstart.md): Get started now
