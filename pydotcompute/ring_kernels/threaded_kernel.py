"""
Threaded Ring Kernel - Maximum performance via native threading.

Bypasses asyncio entirely for sub-100μs latency by running the kernel
in a dedicated background thread with synchronous queues.

This is ideal for latency-critical applications where the overhead
of asyncio task scheduling is unacceptable.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from pydotcompute.exceptions import KernelStateError
from pydotcompute.ring_kernels.message import RingKernelMessage
from pydotcompute.ring_kernels.sync_queue import SPSCQueue, SyncQueue

if TYPE_CHECKING:
    pass

TIn = TypeVar("TIn", bound=RingKernelMessage)
TOut = TypeVar("TOut", bound=RingKernelMessage)


class ThreadedKernelState(Enum):
    """State of a threaded kernel."""

    CREATED = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()


@dataclass
class ThreadedKernelConfig:
    """Configuration for a threaded kernel."""

    kernel_id: str
    input_queue_size: int = 4096
    output_queue_size: int = 4096
    use_spsc: bool = True  # Use lock-free SPSC queue (faster but single producer/consumer only)
    thread_name: str | None = None
    daemon: bool = True  # Daemon thread dies with main process

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.input_queue_size < 1:
            raise ValueError(f"input_queue_size must be >= 1, got {self.input_queue_size}")
        if self.output_queue_size < 1:
            raise ValueError(f"output_queue_size must be >= 1, got {self.output_queue_size}")


@dataclass
class ThreadedKernelContext(Generic[TIn, TOut]):
    """
    Context passed to threaded kernel functions.

    Provides synchronous (blocking) access to message queues.
    """

    kernel_id: str
    input_queue: SyncQueue[TIn] | SPSCQueue[TIn]
    output_queue: SyncQueue[TOut] | SPSCQueue[TOut]
    _shutdown_event: threading.Event = field(default_factory=threading.Event)

    @property
    def should_terminate(self) -> bool:
        """Check if the kernel should terminate."""
        return self._shutdown_event.is_set()

    def receive(self, timeout: float | None = None) -> TIn | None:
        """
        Receive a message from the input queue (blocking).

        Args:
            timeout: Maximum time to wait. None = wait forever.

        Returns:
            Message or None if timeout/shutdown.
        """
        try:
            if isinstance(self.input_queue, SPSCQueue):
                return self.input_queue.get(timeout=timeout)
            return self.input_queue.get(timeout=timeout)
        except Exception:
            if self._shutdown_event.is_set():
                return None
            raise

    def send(self, message: TOut, timeout: float | None = None) -> bool:
        """
        Send a message to the output queue (blocking).

        Args:
            message: Message to send.
            timeout: Maximum time to wait.

        Returns:
            True if sent successfully.
        """
        try:
            if isinstance(self.output_queue, SPSCQueue):
                return self.output_queue.put(message, timeout=timeout)
            return self.output_queue.put(message, timeout=timeout)
        except Exception:
            if self._shutdown_event.is_set():
                return False
            raise

    def wait_shutdown(self, timeout: float | None = None) -> bool:
        """Wait for shutdown signal."""
        return self._shutdown_event.wait(timeout=timeout)


class ThreadedRingKernel(Generic[TIn, TOut]):
    """
    High-performance ring kernel running in a dedicated thread.

    Bypasses asyncio entirely for maximum performance. Uses native
    threading with synchronous queues for sub-100μs latency.

    Key features:
    - No asyncio overhead
    - Native threading with efficient sync queues
    - Optional SPSC lock-free queue for single producer/consumer
    - Background thread with optional daemon mode

    Example:
        >>> def echo_kernel(ctx: ThreadedKernelContext):
        ...     while not ctx.should_terminate:
        ...         msg = ctx.receive(timeout=0.1)
        ...         if msg:
        ...             ctx.send(msg)
        ...
        >>> kernel = ThreadedRingKernel("echo", echo_kernel)
        >>> kernel.start()
        >>> kernel.send(MyMessage(value=42))
        >>> response = kernel.receive()
        >>> kernel.stop()
    """

    def __init__(
        self,
        kernel_id: str,
        kernel_func: Callable[[ThreadedKernelContext[TIn, TOut]], Any],
        *,
        config: ThreadedKernelConfig | None = None,
        input_type: type[TIn] | None = None,
        output_type: type[TOut] | None = None,
    ) -> None:
        """
        Initialize a threaded kernel.

        Args:
            kernel_id: Unique identifier for the kernel.
            kernel_func: Function implementing the kernel logic.
                         Receives a ThreadedKernelContext and should
                         loop until ctx.should_terminate is True.
            config: Kernel configuration.
            input_type: Type of input messages (for validation).
            output_type: Type of output messages (for validation).
        """
        self.kernel_id = kernel_id
        self._kernel_func = kernel_func
        self._config = config or ThreadedKernelConfig(kernel_id=kernel_id)
        self._input_type = input_type
        self._output_type = output_type

        self._state = ThreadedKernelState.CREATED
        self._thread: threading.Thread | None = None
        self._context: ThreadedKernelContext[TIn, TOut] | None = None
        self._error: Exception | None = None
        self._state_lock = threading.Lock()

    @property
    def state(self) -> ThreadedKernelState:
        """Get the current kernel state."""
        with self._state_lock:
            return self._state

    @property
    def config(self) -> ThreadedKernelConfig:
        """Get the kernel configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if kernel is running."""
        with self._state_lock:
            return self._state == ThreadedKernelState.RUNNING

    @property
    def error(self) -> Exception | None:
        """Get any error that occurred during execution."""
        return self._error

    def start(self) -> None:
        """
        Start the kernel thread.

        Creates queues and starts the background thread.

        Raises:
            KernelStateError: If kernel is not in CREATED state.
        """
        with self._state_lock:
            if self._state != ThreadedKernelState.CREATED:
                raise KernelStateError(
                    self.kernel_id, self._state.name, "start"
                )

            # Create queues
            if self._config.use_spsc:
                input_queue: SyncQueue[TIn] | SPSCQueue[TIn] = SPSCQueue(
                    capacity=self._config.input_queue_size
                )
                output_queue: SyncQueue[TOut] | SPSCQueue[TOut] = SPSCQueue(
                    capacity=self._config.output_queue_size
                )
            else:
                input_queue = SyncQueue(maxsize=self._config.input_queue_size)
                output_queue = SyncQueue(maxsize=self._config.output_queue_size)

            # Create context
            self._context = ThreadedKernelContext(
                kernel_id=self.kernel_id,
                input_queue=input_queue,
                output_queue=output_queue,
            )

            # Create and start thread
            thread_name = self._config.thread_name or f"threaded_kernel_{self.kernel_id}"
            self._thread = threading.Thread(
                target=self._run_loop,
                name=thread_name,
                daemon=self._config.daemon,
            )
            self._state = ThreadedKernelState.RUNNING
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the kernel thread gracefully.

        Args:
            timeout: Maximum time to wait for thread to stop.
        """
        with self._state_lock:
            if self._state in (ThreadedKernelState.STOPPED, ThreadedKernelState.CREATED):
                return

            self._state = ThreadedKernelState.STOPPING

        # Signal shutdown
        if self._context:
            self._context._shutdown_event.set()
            # Wake up any blocked queue operations
            if isinstance(self._context.input_queue, SPSCQueue):
                self._context.input_queue.shutdown()
            else:
                self._context.input_queue.shutdown()
            if isinstance(self._context.output_queue, SPSCQueue):
                self._context.output_queue.shutdown()
            else:
                self._context.output_queue.shutdown()

        # Wait for thread
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # Thread didn't stop gracefully
                pass  # Can't force-kill threads in Python

        with self._state_lock:
            self._state = ThreadedKernelState.STOPPED

    def send(self, message: TIn, timeout: float | None = None) -> bool:
        """
        Send a message to the kernel (blocking).

        Args:
            message: Message to send.
            timeout: Maximum time to wait.

        Returns:
            True if sent successfully.

        Raises:
            KernelStateError: If kernel is not running.
        """
        if not self.is_running:
            raise KernelStateError(
                self.kernel_id, self._state.name, "send message"
            )

        if self._context:
            if isinstance(self._context.input_queue, SPSCQueue):
                return self._context.input_queue.put(message, timeout=timeout)
            return self._context.input_queue.put(message, timeout=timeout)
        return False

    def receive(self, timeout: float | None = None) -> TOut | None:
        """
        Receive a message from the kernel (blocking).

        Args:
            timeout: Maximum time to wait.

        Returns:
            Message or None if timeout.

        Raises:
            KernelStateError: If kernel is not running.
        """
        if not self.is_running and self._context and (
            isinstance(self._context.output_queue, SPSCQueue) and self._context.output_queue.empty
            or isinstance(self._context.output_queue, SyncQueue) and self._context.output_queue.empty
        ):
            raise KernelStateError(
                self.kernel_id, self._state.name, "receive message"
            )

        if self._context:
            if isinstance(self._context.output_queue, SPSCQueue):
                return self._context.output_queue.get(timeout=timeout)
            try:
                return self._context.output_queue.get(timeout=timeout)
            except Exception:
                return None
        return None

    def send_nowait(self, message: TIn) -> bool:
        """Send without blocking."""
        if self._context:
            return self._context.input_queue.put_nowait(message)
        return False

    def receive_nowait(self) -> TOut | None:
        """Receive without blocking."""
        if self._context:
            return self._context.output_queue.get_nowait()
        return None

    def _run_loop(self) -> None:
        """Main kernel execution loop (runs in background thread)."""
        if self._context is None:
            return

        try:
            # Run the kernel function
            self._kernel_func(self._context)
        except Exception as e:
            self._error = e
        finally:
            with self._state_lock:
                if self._state != ThreadedKernelState.STOPPED:
                    self._state = ThreadedKernelState.STOPPED

    def __enter__(self) -> ThreadedRingKernel[TIn, TOut]:
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
            f"ThreadedRingKernel(id={self.kernel_id!r}, "
            f"state={self._state.name}, "
            f"spsc={self._config.use_spsc})"
        )


class ThreadedKernelPool:
    """
    Pool of threaded kernels for parallel processing.

    Manages multiple threaded kernels and provides round-robin
    message distribution.
    """

    def __init__(
        self,
        kernel_id_prefix: str,
        kernel_func: Callable[[ThreadedKernelContext[Any, Any]], Any],
        num_workers: int = 4,
        **config_kwargs: Any,
    ) -> None:
        """
        Initialize kernel pool.

        Args:
            kernel_id_prefix: Prefix for kernel IDs.
            kernel_func: Kernel function for all workers.
            num_workers: Number of worker kernels.
            **config_kwargs: Additional config for ThreadedKernelConfig.
        """
        self._kernels: list[ThreadedRingKernel[Any, Any]] = []
        self._next_kernel = 0
        self._lock = threading.Lock()

        for i in range(num_workers):
            kernel_id = f"{kernel_id_prefix}_{i}"
            config = ThreadedKernelConfig(kernel_id=kernel_id, **config_kwargs)
            kernel: ThreadedRingKernel[Any, Any] = ThreadedRingKernel(
                kernel_id=kernel_id,
                kernel_func=kernel_func,
                config=config,
            )
            self._kernels.append(kernel)

    def start_all(self) -> None:
        """Start all kernels."""
        for kernel in self._kernels:
            kernel.start()

    def stop_all(self, timeout: float = 5.0) -> None:
        """Stop all kernels."""
        for kernel in self._kernels:
            kernel.stop(timeout=timeout / len(self._kernels))

    def send(self, message: Any, timeout: float | None = None) -> bool:
        """Send to next kernel (round-robin)."""
        with self._lock:
            kernel = self._kernels[self._next_kernel]
            self._next_kernel = (self._next_kernel + 1) % len(self._kernels)
        return kernel.send(message, timeout=timeout)

    def receive_any(self, timeout: float | None = None) -> Any | None:
        """Receive from any kernel that has output."""
        # Try each kernel
        for kernel in self._kernels:
            msg = kernel.receive_nowait()
            if msg is not None:
                return msg

        # If none had output, wait on first kernel
        return self._kernels[0].receive(timeout=timeout)

    def __enter__(self) -> ThreadedKernelPool:
        """Context manager entry."""
        self.start_all()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.stop_all()

    def __len__(self) -> int:
        """Get number of kernels."""
        return len(self._kernels)
