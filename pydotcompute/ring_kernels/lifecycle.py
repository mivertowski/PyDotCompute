"""
Lifecycle management for ring kernels.

Provides state machine for kernel lifecycle:
CREATED → LAUNCHED → ACTIVE → DEACTIVATED → TERMINATED
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydotcompute.exceptions import KernelStateError
from pydotcompute.ring_kernels.fast_queue import FastMessageQueue
from pydotcompute.ring_kernels.message import RingKernelMessage
from pydotcompute.ring_kernels.queue import MessageQueue

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


TIn = TypeVar("TIn", bound=RingKernelMessage)
TOut = TypeVar("TOut", bound=RingKernelMessage)


class KernelState(Enum):
    """State of a ring kernel."""

    CREATED = auto()
    LAUNCHED = auto()
    ACTIVE = auto()
    DEACTIVATED = auto()
    TERMINATING = auto()
    TERMINATED = auto()


@dataclass
class RingKernelConfig:
    """Configuration for a ring kernel."""

    kernel_id: str
    grid_size: int = 1
    block_size: int = 256
    input_queue_size: int = 4096
    output_queue_size: int = 4096
    backpressure_strategy: str = "block"  # block, reject, drop_oldest
    use_fast_queue: bool = True  # Use FastMessageQueue for better performance

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.grid_size < 1:
            raise ValueError(f"grid_size must be >= 1, got {self.grid_size}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.input_queue_size < 1:
            raise ValueError(f"input_queue_size must be >= 1, got {self.input_queue_size}")
        if self.output_queue_size < 1:
            raise ValueError(f"output_queue_size must be >= 1, got {self.output_queue_size}")


@dataclass
class KernelContext(Generic[TIn, TOut]):
    """
    Context passed to ring kernel actors.

    Provides access to message queues and control events.
    """

    kernel_id: str
    input_queue: MessageQueue[TIn] | FastMessageQueue[TIn]
    output_queue: MessageQueue[TOut] | FastMessageQueue[TOut]
    _shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    _active_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def should_terminate(self) -> bool:
        """Check if the kernel should terminate."""
        return self._shutdown_event.is_set()

    @property
    def is_active(self) -> bool:
        """Check if the kernel is active."""
        return self._active_event.is_set()

    async def receive(self, timeout: float | None = None) -> TIn:
        """
        Receive a message from the input queue.

        Args:
            timeout: Maximum time to wait.

        Returns:
            Received message.
        """
        return await self.input_queue.get(timeout=timeout)

    async def send(self, message: TOut) -> None:
        """
        Send a message to the output queue.

        Args:
            message: Message to send.
        """
        await self.output_queue.put(message)

    async def wait_active(self) -> None:
        """Wait until the kernel is active."""
        await self._active_event.wait()


class RingKernel(Generic[TIn, TOut]):
    """
    Managed ring kernel with lifecycle support.

    Implements the two-phase launch pattern:
    1. launch() - Create resources but don't start processing
    2. activate() - Begin message processing

    Example:
        >>> kernel = RingKernel("my_kernel", my_actor_func)
        >>> await kernel.launch()
        >>> await kernel.activate()
        >>> await kernel.send(MyRequest(value=42))
        >>> response = await kernel.receive()
        >>> await kernel.terminate()
    """

    def __init__(
        self,
        kernel_id: str,
        kernel_func: Callable[[KernelContext[TIn, TOut]], Any],
        *,
        config: RingKernelConfig | None = None,
        input_type: type[TIn] | None = None,
        output_type: type[TOut] | None = None,
    ) -> None:
        """
        Initialize a ring kernel.

        Args:
            kernel_id: Unique identifier for the kernel.
            kernel_func: Async function implementing the kernel logic.
            config: Kernel configuration.
            input_type: Type of input messages.
            output_type: Type of output messages.
        """
        self.kernel_id = kernel_id
        self._kernel_func = kernel_func
        self._config = config or RingKernelConfig(kernel_id=kernel_id)
        self._input_type = input_type
        self._output_type = output_type

        self._state = KernelState.CREATED
        self._task: asyncio.Task[Any] | None = None
        self._context: KernelContext[TIn, TOut] | None = None

    @property
    def state(self) -> KernelState:
        """Get the current kernel state."""
        return self._state

    @property
    def config(self) -> RingKernelConfig:
        """Get the kernel configuration."""
        return self._config

    async def launch(self) -> None:
        """
        Phase 1: Setup resources but don't start processing.

        Creates message queues and prepares the kernel context.

        Raises:
            KernelStateError: If kernel is not in CREATED state.
        """
        if self._state != KernelState.CREATED:
            raise KernelStateError(self.kernel_id, self._state.name, "launch")

        # Create message queues - use FastMessageQueue for better performance
        # serialize=False enables zero-copy in-process message passing
        if self._config.use_fast_queue:
            input_queue: MessageQueue[TIn] | FastMessageQueue[TIn] = FastMessageQueue(
                maxsize=self._config.input_queue_size,
                kernel_id=self.kernel_id,
                serialize=False,  # Zero-copy: pass object references directly
            )
            output_queue: MessageQueue[TOut] | FastMessageQueue[TOut] = FastMessageQueue(
                maxsize=self._config.output_queue_size,
                kernel_id=self.kernel_id,
                serialize=False,  # Zero-copy: pass object references directly
            )
        else:
            input_queue = MessageQueue(
                maxsize=self._config.input_queue_size,
                kernel_id=self.kernel_id,
            )
            output_queue = MessageQueue(
                maxsize=self._config.output_queue_size,
                kernel_id=self.kernel_id,
            )

        # Create context
        self._context = KernelContext(
            kernel_id=self.kernel_id,
            input_queue=input_queue,
            output_queue=output_queue,
        )

        self._state = KernelState.LAUNCHED

    async def activate(self) -> None:
        """
        Phase 2: Begin message processing.

        Starts the kernel task and enables message processing.

        Raises:
            KernelStateError: If kernel is not in LAUNCHED state.
        """
        if self._state != KernelState.LAUNCHED:
            raise KernelStateError(self.kernel_id, self._state.name, "activate")

        if self._context is None:
            raise RuntimeError("Context not initialized")

        # Set active and start task
        self._context._active_event.set()
        self._task = asyncio.create_task(
            self._run_loop(),
            name=f"ring_kernel_{self.kernel_id}",
        )

        self._state = KernelState.ACTIVE

    async def deactivate(self) -> None:
        """
        Pause message processing (preserve state).

        The kernel remains alive but stops processing messages.

        Raises:
            KernelStateError: If kernel is not in ACTIVE state.
        """
        if self._state != KernelState.ACTIVE:
            raise KernelStateError(self.kernel_id, self._state.name, "deactivate")

        if self._context:
            self._context._active_event.clear()

        self._state = KernelState.DEACTIVATED

    async def reactivate(self) -> None:
        """
        Resume message processing after deactivation.

        Raises:
            KernelStateError: If kernel is not in DEACTIVATED state.
        """
        if self._state != KernelState.DEACTIVATED:
            raise KernelStateError(self.kernel_id, self._state.name, "reactivate")

        if self._context:
            self._context._active_event.set()

        self._state = KernelState.ACTIVE

    async def terminate(self, timeout: float = 5.0) -> None:
        """
        Gracefully shutdown the kernel.

        Args:
            timeout: Maximum time to wait for shutdown.

        Raises:
            KernelStateError: If kernel is already terminated.
        """
        if self._state == KernelState.TERMINATED:
            return

        self._state = KernelState.TERMINATING

        # Signal shutdown
        if self._context:
            self._context._shutdown_event.set()
            self._context._active_event.set()  # Unblock any waiting

        # Wait for task completion
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except TimeoutError:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

        # Cleanup
        self._cleanup_resources()
        self._state = KernelState.TERMINATED

    async def _run_loop(self) -> None:
        """Main kernel execution loop."""
        if self._context is None:
            return

        try:
            # Wait for activation
            await self._context.wait_active()

            # Run the kernel function
            result = self._kernel_func(self._context)
            if asyncio.iscoroutine(result):
                await result

        except asyncio.CancelledError:
            pass
        except Exception:
            # Log error but don't crash
            pass

    def _cleanup_resources(self) -> None:
        """Cleanup kernel resources."""
        if self._context:
            self._context.input_queue.clear()
            self._context.output_queue.clear()
        self._context = None
        self._task = None

    async def send(self, message: TIn, timeout: float | None = None) -> None:
        """
        Send a message to the kernel's input queue.

        Args:
            message: Message to send.
            timeout: Maximum time to wait.

        Raises:
            KernelStateError: If kernel is not active.
        """
        if self._state not in (KernelState.ACTIVE, KernelState.DEACTIVATED):
            raise KernelStateError(self.kernel_id, self._state.name, "send message")

        if self._context:
            await self._context.input_queue.put(message, timeout=timeout)

    async def receive(self, timeout: float | None = None) -> TOut:
        """
        Receive a message from the kernel's output queue.

        Args:
            timeout: Maximum time to wait.

        Returns:
            Received message.

        Raises:
            KernelStateError: If kernel is not active.
        """
        if self._state not in (KernelState.ACTIVE, KernelState.DEACTIVATED):
            raise KernelStateError(self.kernel_id, self._state.name, "receive message")

        if self._context:
            return await self._context.output_queue.get(timeout=timeout)

        raise RuntimeError("Context not initialized")

    def __repr__(self) -> str:
        """String representation."""
        return f"RingKernel(id={self.kernel_id!r}, state={self._state.name})"


@asynccontextmanager
async def managed_kernel(
    kernel_id: str,
    kernel_func: Callable[[KernelContext[TIn, TOut]], Any],
    **kwargs: Any,
) -> AsyncIterator[RingKernel[TIn, TOut]]:
    """
    Context manager for automatic kernel lifecycle management.

    Example:
        >>> async with managed_kernel("my_kernel", my_func) as kernel:
        ...     await kernel.send(request)
        ...     response = await kernel.receive()
    """
    kernel: RingKernel[TIn, TOut] = RingKernel(kernel_id, kernel_func, **kwargs)
    try:
        await kernel.launch()
        await kernel.activate()
        yield kernel
    finally:
        await kernel.terminate()
