"""
Cython Ring Kernel - Ultimate performance via Cython SPSC queues.

Combines the Cython FastSPSCQueue (0.33μs operations) with a dedicated
worker thread for sub-10μs full roundtrip latency.

This is the highest-performance kernel available in PyDotCompute.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydotcompute.exceptions import KernelStateError
from pydotcompute.ring_kernels.message import RingKernelMessage

# Try to import Cython queue, fall back to pure Python
try:
    from pydotcompute.ring_kernels._cython import CYTHON_AVAILABLE, FastSPSCQueue
except ImportError:
    CYTHON_AVAILABLE = False
    FastSPSCQueue = None  # type: ignore[assignment, misc]

if not CYTHON_AVAILABLE:
    # Fallback to pure Python SPSC
    from pydotcompute.ring_kernels.sync_queue import (
        SPSCQueue as FastSPSCQueue,  # type: ignore[no-redef]
    )

if TYPE_CHECKING:
    pass

TIn = TypeVar("TIn", bound=RingKernelMessage)
TOut = TypeVar("TOut", bound=RingKernelMessage)


class CythonKernelState(Enum):
    """State of a Cython kernel."""

    CREATED = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()


@dataclass
class CythonKernelConfig:
    """Configuration for a Cython kernel."""

    kernel_id: str
    input_queue_size: int = 4096
    output_queue_size: int = 4096
    spin_wait_ns: int = 100  # Nanoseconds to spin before yielding
    thread_name: str | None = None
    daemon: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.input_queue_size < 1:
            raise ValueError("input_queue_size must be >= 1")
        if self.output_queue_size < 1:
            raise ValueError("output_queue_size must be >= 1")


@dataclass
class CythonKernelContext(Generic[TIn, TOut]):
    """
    Context passed to Cython kernel functions.

    Uses FastSPSCQueue for sub-microsecond message operations.
    """

    kernel_id: str
    input_queue: Any  # FastSPSCQueue[TIn]
    output_queue: Any  # FastSPSCQueue[TOut]
    _shutdown_event: threading.Event = field(default_factory=threading.Event)

    @property
    def should_terminate(self) -> bool:
        """Check if the kernel should terminate."""
        return self._shutdown_event.is_set()

    def receive(self, timeout: float | None = None) -> TIn | None:
        """
        Receive a message (sub-microsecond when available).

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            Message or None if timeout/shutdown.
        """
        if timeout is None:
            timeout = 1.0  # Default timeout

        # Use Cython queue's wait method if available
        if hasattr(self.input_queue, "get_wait"):
            return self.input_queue.get_wait(timeout)
        else:
            return self.input_queue.get(timeout=timeout)

    def receive_nowait(self) -> TIn | None:
        """Receive without waiting (fastest path)."""
        return self.input_queue.get()

    def send(self, message: TOut, timeout: float | None = None) -> bool:
        """
        Send a message (sub-microsecond when space available).

        Args:
            message: Message to send.
            timeout: Maximum wait time.

        Returns:
            True if sent successfully.
        """
        if timeout is None:
            return self.input_queue.put(message)

        if hasattr(self.output_queue, "put_wait"):
            return self.output_queue.put_wait(message, timeout)
        else:
            return self.output_queue.put(message, timeout=timeout)

    def send_nowait(self, message: TOut) -> bool:
        """Send without waiting (fastest path)."""
        return self.output_queue.put(message)


class CythonRingKernel(Generic[TIn, TOut]):
    """
    Ultimate-performance ring kernel using Cython FastSPSCQueue.

    Performance characteristics:
    - Queue operations: ~0.33μs (Cython) vs ~1.8μs (Python)
    - Full roundtrip: <10μs typical (depends on kernel function)

    Example:
        >>> def echo_kernel(ctx: CythonKernelContext):
        ...     while not ctx.should_terminate:
        ...         msg = ctx.receive_nowait()
        ...         if msg is not None:
        ...             ctx.send_nowait(msg)
        ...
        >>> with CythonRingKernel("echo", echo_kernel) as kernel:
        ...     kernel.send(MyMessage())
        ...     response = kernel.receive()
    """

    def __init__(
        self,
        kernel_id: str,
        kernel_func: Callable[[CythonKernelContext[TIn, TOut]], Any],
        *,
        config: CythonKernelConfig | None = None,
    ) -> None:
        """
        Initialize a Cython kernel.

        Args:
            kernel_id: Unique identifier for the kernel.
            kernel_func: Function implementing the kernel logic.
            config: Kernel configuration.
        """
        self.kernel_id = kernel_id
        self._kernel_func = kernel_func
        self._config = config or CythonKernelConfig(kernel_id=kernel_id)

        self._state = CythonKernelState.CREATED
        self._thread: threading.Thread | None = None
        self._context: CythonKernelContext[TIn, TOut] | None = None
        self._error: Exception | None = None
        self._state_lock = threading.Lock()

        # Check Cython availability
        self._using_cython = CYTHON_AVAILABLE

    @property
    def state(self) -> CythonKernelState:
        """Get the current kernel state."""
        with self._state_lock:
            return self._state

    @property
    def is_running(self) -> bool:
        """Check if kernel is running."""
        with self._state_lock:
            return self._state == CythonKernelState.RUNNING

    @property
    def using_cython(self) -> bool:
        """Check if using Cython queue (vs Python fallback)."""
        return self._using_cython

    @property
    def error(self) -> Exception | None:
        """Get any error that occurred during execution."""
        return self._error

    def start(self) -> None:
        """
        Start the kernel thread.

        Raises:
            KernelStateError: If kernel is not in CREATED state.
        """
        with self._state_lock:
            if self._state != CythonKernelState.CREATED:
                raise KernelStateError(self.kernel_id, self._state.name, "start")

            # Create Cython SPSC queues
            input_queue = FastSPSCQueue(self._config.input_queue_size)
            output_queue = FastSPSCQueue(self._config.output_queue_size)

            # Create context
            self._context = CythonKernelContext(
                kernel_id=self.kernel_id,
                input_queue=input_queue,
                output_queue=output_queue,
            )

            # Create and start thread
            thread_name = self._config.thread_name or f"cython_kernel_{self.kernel_id}"
            self._thread = threading.Thread(
                target=self._run_loop,
                name=thread_name,
                daemon=self._config.daemon,
            )
            self._state = CythonKernelState.RUNNING
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the kernel thread gracefully.

        Args:
            timeout: Maximum time to wait for thread to stop.
        """
        with self._state_lock:
            if self._state in (CythonKernelState.STOPPED, CythonKernelState.CREATED):
                return

            self._state = CythonKernelState.STOPPING

        # Signal shutdown
        if self._context:
            self._context._shutdown_event.set()
            self._context.input_queue.shutdown()
            self._context.output_queue.shutdown()

        # Wait for thread
        if self._thread:
            self._thread.join(timeout=timeout)

        with self._state_lock:
            self._state = CythonKernelState.STOPPED

    def send(self, message: TIn, timeout: float | None = None) -> bool:
        """
        Send a message to the kernel.

        Args:
            message: Message to send.
            timeout: Maximum time to wait.

        Returns:
            True if sent successfully.
        """
        if not self.is_running:
            raise KernelStateError(self.kernel_id, self._state.name, "send message")

        if self._context:
            if timeout is None:
                return self._context.input_queue.put(message)
            if hasattr(self._context.input_queue, "put_wait"):
                return self._context.input_queue.put_wait(message, timeout)
            return self._context.input_queue.put(message, timeout=timeout)
        return False

    def send_nowait(self, message: TIn) -> bool:
        """Send without waiting (fastest)."""
        if self._context:
            return self._context.input_queue.put(message)
        return False

    def receive(self, timeout: float | None = None) -> TOut | None:
        """
        Receive a message from the kernel.

        Args:
            timeout: Maximum time to wait.

        Returns:
            Message or None if timeout.
        """
        if self._context:
            if timeout is None:
                # Non-blocking get
                return self._context.output_queue.get()
            if hasattr(self._context.output_queue, "get_wait"):
                return self._context.output_queue.get_wait(timeout)
            return self._context.output_queue.get(timeout=timeout)
        return None

    def receive_nowait(self) -> TOut | None:
        """Receive without waiting (fastest)."""
        if self._context:
            return self._context.output_queue.get()
        return None

    def _run_loop(self) -> None:
        """Main kernel execution loop (runs in background thread)."""
        if self._context is None:
            return

        try:
            self._kernel_func(self._context)
        except Exception as e:
            self._error = e
        finally:
            with self._state_lock:
                if self._state != CythonKernelState.STOPPED:
                    self._state = CythonKernelState.STOPPED

    def __enter__(self) -> CythonRingKernel[TIn, TOut]:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.stop()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CythonRingKernel(id={self.kernel_id!r}, "
            f"state={self._state.name}, "
            f"cython={self._using_cython})"
        )


def is_cython_kernel_available() -> bool:
    """Check if Cython kernel is available (vs Python fallback)."""
    return CYTHON_AVAILABLE
