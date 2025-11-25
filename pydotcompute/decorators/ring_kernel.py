"""
Ring kernel decorator for PyDotCompute.

Provides the @ring_kernel decorator for defining persistent
GPU actor kernels.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from pydotcompute.ring_kernels.runtime import register_ring_kernel

if TYPE_CHECKING:
    from pydotcompute.ring_kernels.message import RingKernelMessage

F = TypeVar("F", bound=Callable[..., Any])


def ring_kernel(
    func: F | None = None,
    *,
    kernel_id: str | None = None,
    input_type: type[RingKernelMessage] | None = None,
    output_type: type[RingKernelMessage] | None = None,
    queue_size: int = 4096,
    input_queue_size: int | None = None,
    output_queue_size: int | None = None,
    grid_size: int = 1,
    block_size: int = 256,
    backpressure: str = "block",
    auto_register: bool = True,
) -> F | Callable[[F], F]:
    """
    Decorator to define a ring kernel (persistent GPU actor).

    Ring kernels are persistent actors that process messages in a loop.
    They support two-phase launch (launch → activate) and graceful shutdown.

    Args:
        func: The actor function to decorate.
        kernel_id: Unique identifier (defaults to function name).
        input_type: Type of input messages.
        output_type: Type of output messages.
        queue_size: Default size for both queues.
        input_queue_size: Size of input queue (overrides queue_size).
        output_queue_size: Size of output queue (overrides queue_size).
        grid_size: CUDA grid size.
        block_size: CUDA block size.
        backpressure: Strategy: "block", "reject", or "drop_oldest".
        auto_register: Automatically register with runtime.

    Returns:
        Decorated ring kernel function.

    Example:
        >>> @ring_kernel(
        ...     kernel_id="vector_add",
        ...     input_type=VectorAddRequest,
        ...     output_type=VectorAddResponse,
        ...     queue_size=1000,
        ... )
        ... async def vector_add_actor(ctx):
        ...     while not ctx.should_terminate:
        ...         msg = await ctx.receive()
        ...         result = VectorAddResponse(result=msg.a + msg.b)
        ...         await ctx.send(result)
        ...
        >>> # Use with runtime
        >>> async with RingKernelRuntime() as runtime:
        ...     await runtime.launch("vector_add")
        ...     await runtime.activate("vector_add")
    """

    def decorator(fn: F) -> F:
        k_id = kernel_id or fn.__name__

        # Build configuration
        config = {
            "input_queue_size": input_queue_size or queue_size,
            "output_queue_size": output_queue_size or queue_size,
            "grid_size": grid_size,
            "block_size": block_size,
            "backpressure_strategy": backpressure,
        }

        # Store metadata on function
        fn._ring_kernel_meta = {  # type: ignore
            "kernel_id": k_id,
            "input_type": input_type,
            "output_type": output_type,
            "config": config,
        }

        # Register with global registry
        if auto_register:
            register_ring_kernel(
                kernel_id=k_id,
                func=fn,
                input_type=input_type,
                output_type=output_type,
                **config,
            )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Ring kernels should be launched via runtime,
            # but allow direct invocation for testing
            return fn(*args, **kwargs)

        # Copy metadata
        wrapper._ring_kernel_meta = fn._ring_kernel_meta  # type: ignore
        wrapper._original_func = fn  # type: ignore

        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


def get_ring_kernel_meta(func: Callable[..., Any]) -> dict[str, Any] | None:
    """
    Get ring kernel metadata from a decorated function.

    Args:
        func: Potentially decorated function.

    Returns:
        Metadata dictionary or None if not a ring kernel.
    """
    return getattr(func, "_ring_kernel_meta", None)


def is_ring_kernel(func: Callable[..., Any]) -> bool:
    """
    Check if a function is a ring kernel.

    Args:
        func: Function to check.

    Returns:
        True if decorated with @ring_kernel.
    """
    return get_ring_kernel_meta(func) is not None


class RingKernelBuilder:
    """
    Builder for creating ring kernels programmatically.

    Provides a fluent interface for kernel configuration.

    Example:
        >>> kernel = (
        ...     RingKernelBuilder("my_kernel")
        ...     .with_input_type(MyRequest)
        ...     .with_output_type(MyResponse)
        ...     .with_queue_size(2000)
        ...     .with_handler(my_handler_func)
        ...     .build()
        ... )
    """

    def __init__(self, kernel_id: str) -> None:
        """
        Initialize the builder.

        Args:
            kernel_id: Unique kernel identifier.
        """
        self._kernel_id = kernel_id
        self._input_type: type | None = None
        self._output_type: type | None = None
        self._input_queue_size = 4096
        self._output_queue_size = 4096
        self._grid_size = 1
        self._block_size = 256
        self._backpressure = "block"
        self._handler: Callable[..., Any] | None = None

    def with_input_type(self, input_type: type) -> RingKernelBuilder:
        """Set the input message type."""
        self._input_type = input_type
        return self

    def with_output_type(self, output_type: type) -> RingKernelBuilder:
        """Set the output message type."""
        self._output_type = output_type
        return self

    def with_queue_size(self, size: int) -> RingKernelBuilder:
        """Set both queue sizes."""
        self._input_queue_size = size
        self._output_queue_size = size
        return self

    def with_input_queue_size(self, size: int) -> RingKernelBuilder:
        """Set input queue size."""
        self._input_queue_size = size
        return self

    def with_output_queue_size(self, size: int) -> RingKernelBuilder:
        """Set output queue size."""
        self._output_queue_size = size
        return self

    def with_grid_size(self, size: int) -> RingKernelBuilder:
        """Set CUDA grid size."""
        self._grid_size = size
        return self

    def with_block_size(self, size: int) -> RingKernelBuilder:
        """Set CUDA block size."""
        self._block_size = size
        return self

    def with_backpressure(self, strategy: str) -> RingKernelBuilder:
        """Set backpressure strategy."""
        self._backpressure = strategy
        return self

    def with_handler(self, handler: Callable[..., Any]) -> RingKernelBuilder:
        """Set the handler function."""
        self._handler = handler
        return self

    def build(self) -> Callable[..., Any]:
        """
        Build and register the ring kernel.

        Returns:
            The registered kernel handler.

        Raises:
            ValueError: If handler is not set.
        """
        if self._handler is None:
            raise ValueError("Handler function is required")

        # Apply decorator
        decorated: Callable[..., Any] = ring_kernel(
            kernel_id=self._kernel_id,
            input_type=self._input_type,
            output_type=self._output_type,
            input_queue_size=self._input_queue_size,
            output_queue_size=self._output_queue_size,
            grid_size=self._grid_size,
            block_size=self._block_size,
            backpressure=self._backpressure,
        )(self._handler)  # type: ignore[arg-type]

        return decorated

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RingKernelBuilder(id={self._kernel_id!r}, "
            f"input={self._input_type}, output={self._output_type})"
        )
