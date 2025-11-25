"""
Ring Kernel Runtime - Main coordinator for kernel lifecycle.

Provides high-level API for managing ring kernels, message routing,
and resource coordination.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from pydotcompute.exceptions import (
    KernelAlreadyExistsError,
    KernelNotFoundError,
    KernelStateError,
)
from pydotcompute.ring_kernels._loop import (
    install_uvloop,
    install_eager_task_factory,
    get_loop_info,
)
from pydotcompute.ring_kernels.lifecycle import (
    KernelContext,
    KernelState,
    RingKernel,
    RingKernelConfig,
)
from pydotcompute.ring_kernels.message import RingKernelMessage
from pydotcompute.ring_kernels.telemetry import RingKernelTelemetry, TelemetryCollector

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


TIn = TypeVar("TIn", bound=RingKernelMessage)
TOut = TypeVar("TOut", bound=RingKernelMessage)


# Registry of decorated kernel functions
_kernel_registry: dict[str, dict[str, Any]] = {}


def register_ring_kernel(
    kernel_id: str,
    func: Callable[..., Any],
    input_type: type | None = None,
    output_type: type | None = None,
    **config: Any,
) -> None:
    """
    Register a kernel function in the global registry.

    Args:
        kernel_id: Unique identifier for the kernel.
        func: The kernel function.
        input_type: Type of input messages.
        output_type: Type of output messages.
        **config: Additional configuration.
    """
    _kernel_registry[kernel_id] = {
        "func": func,
        "input_type": input_type,
        "output_type": output_type,
        "config": config,
    }


def get_registered_kernel(kernel_id: str) -> dict[str, Any] | None:
    """Get a registered kernel by ID."""
    return _kernel_registry.get(kernel_id)


class RingKernelRuntime:
    """
    Ring Kernel Runtime - Main coordinator for kernel lifecycle.

    Manages kernel registration, lifecycle, message routing, and telemetry.
    Use as an async context manager for automatic resource cleanup.

    Example:
        >>> async with RingKernelRuntime() as runtime:
        ...     await runtime.launch("my_kernel")
        ...     await runtime.activate("my_kernel")
        ...     await runtime.send("my_kernel", MyRequest(value=42))
        ...     response = await runtime.receive("my_kernel")
    """

    def __init__(
        self,
        *,
        enable_telemetry: bool = True,
        use_uvloop: bool = True,
        use_eager_tasks: bool = True,
    ) -> None:
        """
        Initialize the runtime.

        Args:
            enable_telemetry: Whether to enable telemetry collection.
            use_uvloop: Whether to install uvloop for faster event loop (20-40% improvement).
                        Only works on Linux/macOS. No-op on Windows.
            use_eager_tasks: Whether to enable eager task factory (Python 3.12+).
                             Reduces task scheduling overhead by 20-30%.
        """
        self._kernels: dict[str, RingKernel[Any, Any]] = {}
        self._enable_telemetry = enable_telemetry
        self._telemetry = TelemetryCollector() if enable_telemetry else None
        self._active = False
        self._use_uvloop = use_uvloop
        self._use_eager_tasks = use_eager_tasks
        self._uvloop_installed = False
        self._eager_tasks_installed = False

        # Install uvloop early (must be before event loop is created)
        if self._use_uvloop:
            self._uvloop_installed = install_uvloop()

    async def __aenter__(self) -> RingKernelRuntime:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.shutdown()

    async def start(self) -> None:
        """
        Start the runtime.

        This method:
        1. Installs eager task factory if enabled and Python 3.12+
        2. Sets the runtime to active state
        """
        # Install eager task factory (must be done after loop is running)
        if self._use_eager_tasks:
            self._eager_tasks_installed = install_eager_task_factory()

        self._active = True

    async def shutdown(self, timeout: float = 10.0) -> None:
        """
        Shutdown the runtime and all kernels.

        Args:
            timeout: Maximum time to wait for kernel shutdown.
        """
        if not self._active:
            return

        # Terminate all kernels
        for kernel_id in list(self._kernels.keys()):
            try:
                await self.terminate(kernel_id, timeout=timeout / max(len(self._kernels), 1))
            except Exception:
                pass

        self._kernels.clear()
        self._active = False

    @property
    def is_active(self) -> bool:
        """Check if the runtime is active."""
        return self._active

    @property
    def kernel_ids(self) -> list[str]:
        """Get list of registered kernel IDs."""
        return list(self._kernels.keys())

    def _get_kernel(self, kernel_id: str) -> RingKernel[Any, Any]:
        """Get a kernel by ID or raise KernelNotFoundError."""
        if kernel_id not in self._kernels:
            raise KernelNotFoundError(kernel_id, self.kernel_ids)
        return self._kernels[kernel_id]

    async def launch(
        self,
        kernel_id: str,
        kernel_func: Callable[[KernelContext[TIn, TOut]], Any] | None = None,
        *,
        config: RingKernelConfig | None = None,
        input_type: type[TIn] | None = None,
        output_type: type[TOut] | None = None,
    ) -> RingKernel[TIn, TOut]:
        """
        Launch a kernel (Phase 1: setup resources).

        If kernel_func is not provided, looks up the kernel in the registry.

        Args:
            kernel_id: Unique identifier for the kernel.
            kernel_func: Kernel function (or None to use registry).
            config: Kernel configuration.
            input_type: Type of input messages.
            output_type: Type of output messages.

        Returns:
            The launched RingKernel.

        Raises:
            KernelAlreadyExistsError: If kernel ID is already in use.
            KernelNotFoundError: If kernel_func is None and not in registry.
        """
        if kernel_id in self._kernels:
            raise KernelAlreadyExistsError(kernel_id)

        # Look up in registry if no function provided
        if kernel_func is None:
            registered = get_registered_kernel(kernel_id)
            if registered is None:
                raise KernelNotFoundError(
                    kernel_id,
                    list(_kernel_registry.keys()),
                )
            kernel_func = registered["func"]
            input_type = input_type or registered.get("input_type")
            output_type = output_type or registered.get("output_type")

            # Merge config
            if config is None:
                reg_config = registered.get("config", {})
                config = RingKernelConfig(kernel_id=kernel_id, **reg_config)

        # Create kernel
        if config is None:
            config = RingKernelConfig(kernel_id=kernel_id)

        kernel: RingKernel[TIn, TOut] = RingKernel(
            kernel_id=kernel_id,
            kernel_func=kernel_func,
            config=config,
            input_type=input_type,
            output_type=output_type,
        )

        # Launch
        await kernel.launch()

        # Register
        self._kernels[kernel_id] = kernel

        # Register telemetry
        if self._telemetry:
            self._telemetry.register_kernel(kernel_id)

        return kernel

    async def activate(self, kernel_id: str) -> None:
        """
        Activate a kernel (Phase 2: start processing).

        Args:
            kernel_id: ID of the kernel to activate.

        Raises:
            KernelNotFoundError: If kernel is not registered.
            KernelStateError: If kernel is not in LAUNCHED state.
        """
        kernel = self._get_kernel(kernel_id)
        await kernel.activate()

    async def deactivate(self, kernel_id: str) -> None:
        """
        Deactivate a kernel (pause processing).

        Args:
            kernel_id: ID of the kernel to deactivate.

        Raises:
            KernelNotFoundError: If kernel is not registered.
            KernelStateError: If kernel is not in ACTIVE state.
        """
        kernel = self._get_kernel(kernel_id)
        await kernel.deactivate()

    async def reactivate(self, kernel_id: str) -> None:
        """
        Reactivate a deactivated kernel.

        Args:
            kernel_id: ID of the kernel to reactivate.

        Raises:
            KernelNotFoundError: If kernel is not registered.
            KernelStateError: If kernel is not in DEACTIVATED state.
        """
        kernel = self._get_kernel(kernel_id)
        await kernel.reactivate()

    async def terminate(self, kernel_id: str, timeout: float = 5.0) -> None:
        """
        Terminate a kernel.

        Args:
            kernel_id: ID of the kernel to terminate.
            timeout: Maximum time to wait for shutdown.

        Raises:
            KernelNotFoundError: If kernel is not registered.
        """
        kernel = self._get_kernel(kernel_id)
        await kernel.terminate(timeout=timeout)

        # Unregister
        del self._kernels[kernel_id]

        # Unregister telemetry
        if self._telemetry:
            self._telemetry.unregister_kernel(kernel_id)

    async def send(
        self,
        kernel_id: str,
        message: TIn,
        *,
        timeout: float | None = None,
    ) -> None:
        """
        Send a message to a kernel's input queue.

        Args:
            kernel_id: ID of the target kernel.
            message: Message to send.
            timeout: Maximum time to wait.

        Raises:
            KernelNotFoundError: If kernel is not registered.
            KernelStateError: If kernel is not active.
        """
        kernel = self._get_kernel(kernel_id)
        await kernel.send(message, timeout=timeout)

    async def receive(
        self,
        kernel_id: str,
        *,
        timeout: float | None = None,
    ) -> TOut:
        """
        Receive a message from a kernel's output queue.

        Args:
            kernel_id: ID of the source kernel.
            timeout: Maximum time to wait.

        Returns:
            Received message.

        Raises:
            KernelNotFoundError: If kernel is not registered.
            KernelStateError: If kernel is not active.
            QueueTimeoutError: If timeout expires.
        """
        kernel = self._get_kernel(kernel_id)
        return await kernel.receive(timeout=timeout)

    def get_kernel_state(self, kernel_id: str) -> KernelState:
        """
        Get the state of a kernel.

        Args:
            kernel_id: ID of the kernel.

        Returns:
            Current kernel state.

        Raises:
            KernelNotFoundError: If kernel is not registered.
        """
        kernel = self._get_kernel(kernel_id)
        return kernel.state

    def get_telemetry(self, kernel_id: str) -> RingKernelTelemetry | None:
        """
        Get telemetry for a kernel.

        Args:
            kernel_id: ID of the kernel.

        Returns:
            Telemetry data or None if telemetry is disabled.
        """
        if self._telemetry:
            return self._telemetry.get_kernel_telemetry(kernel_id)
        return None

    def get_all_telemetry(self) -> dict[str, RingKernelTelemetry]:
        """
        Get telemetry for all kernels.

        Returns:
            Dictionary mapping kernel IDs to telemetry.
        """
        if self._telemetry:
            return self._telemetry.get_all_telemetry()
        return {}

    def get_summary(self) -> dict[str, object]:
        """
        Get aggregated telemetry summary.

        Returns:
            Summary dictionary with aggregated metrics.
        """
        if self._telemetry:
            return self._telemetry.get_summary()
        return {
            "kernel_count": len(self._kernels),
            "total_messages_processed": 0,
            "total_messages_dropped": 0,
            "total_errors": 0,
        }

    async def send_and_receive(
        self,
        kernel_id: str,
        message: TIn,
        *,
        timeout: float | None = None,
    ) -> TOut:
        """
        Send a message and wait for a response.

        Convenience method for request-response patterns.

        Args:
            kernel_id: ID of the target kernel.
            message: Message to send.
            timeout: Maximum time to wait.

        Returns:
            Response message.
        """
        await self.send(kernel_id, message, timeout=timeout)
        return await self.receive(kernel_id, timeout=timeout)

    @asynccontextmanager
    async def kernel_scope(
        self,
        kernel_id: str,
        kernel_func: Callable[[KernelContext[TIn, TOut]], Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RingKernel[TIn, TOut]]:
        """
        Context manager for a single kernel's lifecycle.

        Args:
            kernel_id: ID for the kernel.
            kernel_func: Kernel function.
            **kwargs: Additional kernel configuration.

        Yields:
            The active kernel.
        """
        kernel = await self.launch(kernel_id, kernel_func, **kwargs)
        try:
            await self.activate(kernel_id)
            yield kernel
        finally:
            await self.terminate(kernel_id)

    def get_loop_info(self) -> dict[str, str | bool]:
        """
        Get information about the current event loop configuration.

        Returns:
            Dictionary containing loop class, uvloop/eager_tasks status, etc.
        """
        info = get_loop_info()
        # Add runtime-specific info
        info["runtime_uvloop_installed"] = self._uvloop_installed
        info["runtime_eager_tasks_installed"] = self._eager_tasks_installed
        return info

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RingKernelRuntime(active={self._active}, "
            f"kernels={len(self._kernels)}, "
            f"telemetry={self._enable_telemetry}, "
            f"uvloop={self._uvloop_installed}, "
            f"eager_tasks={self._eager_tasks_installed})"
        )
