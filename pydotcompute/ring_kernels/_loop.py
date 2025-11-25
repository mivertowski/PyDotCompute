"""
Event loop optimization utilities.

Provides automatic uvloop installation and eager task factory support
for improved asyncio performance.

Performance improvements:
- uvloop: 20-40% faster event loop (Linux/macOS only)
- eager_task_factory: 20-30% faster task startup (Python 3.12+)
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Track installation state
_uvloop_installed: bool = False
_eager_tasks_installed: bool = False


def install_uvloop() -> bool:
    """
    Install uvloop as the default event loop policy if available.

    uvloop is a drop-in replacement for asyncio's event loop that provides
    20-40% performance improvement on Unix systems. It's based on libuv
    (the same library that powers Node.js).

    Returns:
        True if uvloop was installed, False otherwise.

    Note:
        - uvloop is not available on Windows (returns False)
        - Safe to call multiple times (idempotent)
        - Must be called before creating any event loops
    """
    global _uvloop_installed

    if _uvloop_installed:
        return True

    if sys.platform == "win32":
        return False

    try:
        import uvloop

        uvloop.install()
        _uvloop_installed = True
        return True
    except ImportError:
        return False


def install_eager_task_factory(loop: asyncio.AbstractEventLoop | None = None) -> bool:
    """
    Install eager task factory for faster task startup (Python 3.12+).

    Eager task factory allows coroutines to start executing immediately
    (synchronously) until they hit their first await, rather than waiting
    for the next event loop iteration. This reduces task scheduling overhead
    by 20-30%.

    Args:
        loop: Event loop to configure. If None, uses the running loop.

    Returns:
        True if eager task factory was installed, False otherwise.

    Note:
        - Only available in Python 3.12 and later
        - Safe to call on earlier Python versions (no-op)
        - Must be called after the event loop is created
    """
    global _eager_tasks_installed

    if sys.version_info < (3, 12):
        return False

    try:
        if loop is None:
            loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop
        return False

    try:
        # asyncio.eager_task_factory was added in Python 3.12
        loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[attr-defined]
        _eager_tasks_installed = True
        return True
    except AttributeError:
        return False


def is_uvloop_available() -> bool:
    """
    Check if uvloop is available (installed and on supported platform).

    Returns:
        True if uvloop can be used, False otherwise.
    """
    if sys.platform == "win32":
        return False

    try:
        import uvloop  # noqa: F401

        return True
    except ImportError:
        return False


def is_uvloop_installed() -> bool:
    """
    Check if uvloop has been installed as the event loop policy.

    Returns:
        True if uvloop is active, False otherwise.
    """
    return _uvloop_installed


def is_eager_tasks_installed() -> bool:
    """
    Check if eager task factory has been installed.

    Returns:
        True if eager task factory is active, False otherwise.
    """
    return _eager_tasks_installed


def get_loop_info() -> dict[str, str | bool]:
    """
    Get information about the current event loop configuration.

    Returns:
        Dictionary containing:
        - loop_class: Name of the current event loop class
        - uvloop_available: Whether uvloop can be used
        - uvloop_installed: Whether uvloop is currently active
        - eager_tasks_available: Whether eager task factory is available
        - eager_tasks_installed: Whether eager task factory is active
        - platform: Current platform identifier
        - python_version: Python version string
    """
    try:
        loop = asyncio.get_running_loop()
        loop_class = type(loop).__name__
    except RuntimeError:
        loop_class = "no_running_loop"

    return {
        "loop_class": loop_class,
        "uvloop_available": is_uvloop_available(),
        "uvloop_installed": _uvloop_installed,
        "eager_tasks_available": sys.version_info >= (3, 12),
        "eager_tasks_installed": _eager_tasks_installed,
        "platform": sys.platform,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def create_optimized_loop() -> asyncio.AbstractEventLoop:
    """
    Create an optimized event loop with all available enhancements.

    Applies in order:
    1. uvloop policy (if available, Unix only)
    2. Creates new event loop
    3. Installs eager_task_factory (Python 3.12+)

    Returns:
        Configured event loop ready for use.

    Example:
        >>> loop = create_optimized_loop()
        >>> asyncio.set_event_loop(loop)
        >>> loop.run_until_complete(main())
    """
    # Install uvloop policy first (before creating loop)
    install_uvloop()

    # Create new loop (will use uvloop if installed)
    loop = asyncio.new_event_loop()

    # Install eager task factory if available
    if sys.version_info >= (3, 12):
        try:
            loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[attr-defined]
        except AttributeError:
            pass

    return loop


def optimize_current_loop() -> dict[str, bool]:
    """
    Apply all available optimizations to the current running event loop.

    This is useful when you don't control loop creation but want to
    enable optimizations at runtime.

    Returns:
        Dictionary with optimization results:
        - eager_tasks: Whether eager task factory was installed

    Note:
        uvloop must be installed before the loop is created, so this
        function cannot enable uvloop on an existing loop.
    """
    results = {
        "eager_tasks": False,
    }

    try:
        loop = asyncio.get_running_loop()

        # Try to install eager task factory
        if sys.version_info >= (3, 12):
            try:
                loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[attr-defined]
                results["eager_tasks"] = True
            except AttributeError:
                pass

    except RuntimeError:
        pass

    return results
