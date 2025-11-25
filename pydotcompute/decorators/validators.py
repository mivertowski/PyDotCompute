"""
Runtime validators for PyDotCompute.

Provides validation decorators and utilities for message types
and kernel configurations.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any, TypeVar, get_args, get_origin

from pydotcompute.exceptions import InvalidConfigurationError, TypeValidationError

if TYPE_CHECKING:
    pass

F = TypeVar("F", bound=Callable[..., Any])


def validate_message(cls: type) -> type:
    """
    Decorator to validate a message class.

    Ensures the class is a valid message type with required fields
    and serialization support.

    Args:
        cls: Class to validate.

    Returns:
        The validated class.

    Raises:
        TypeValidationError: If validation fails.

    Example:
        >>> @validate_message
        ... @dataclass
        ... class MyMessage:
        ...     value: float
    """
    # Check if dataclass
    if not is_dataclass(cls):
        raise TypeValidationError(
            expected=type,
            actual=cls,
            context="Message must be a dataclass",
        )

    # Check for serialization support
    if not hasattr(cls, "serialize"):
        import warnings

        warnings.warn(
            f"Message class '{cls.__name__}' does not have serialize method. "
            "Consider using @message decorator.",
            stacklevel=2,
        )

    return cls


def validate_config(config_cls: type) -> Callable[[F], F]:
    """
    Decorator to validate function arguments against a config class.

    Args:
        config_cls: Configuration class with field specifications.

    Returns:
        Decorator function.

    Example:
        >>> @validate_config(RingKernelConfig)
        ... def create_kernel(kernel_id: str, queue_size: int = 4096):
        ...     pass
    """

    def decorator(func: F) -> F:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Bind arguments
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate each argument
            for param_name, value in bound.arguments.items():
                _validate_param(param_name, value, config_cls)

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def _validate_param(name: str, value: Any, config_cls: type) -> None:
    """Validate a parameter against config class hints."""
    hints = getattr(config_cls, "__annotations__", {})

    if name in hints:
        expected_type = hints[name]
        if not _check_type(value, expected_type):
            raise TypeValidationError(
                expected=expected_type,
                actual=type(value),
                context=f"parameter '{name}'",
            )


def _check_type(value: Any, expected: type) -> bool:
    """Check if a value matches an expected type."""
    if expected is Any:
        return True

    origin = get_origin(expected)

    if origin is None:
        # Simple type
        return isinstance(value, expected)

    # Handle generic types
    if origin is list:
        if not isinstance(value, list):
            return False
        args = get_args(expected)
        if args:
            return all(_check_type(item, args[0]) for item in value)
        return True

    if origin is dict:
        if not isinstance(value, dict):
            return False
        args = get_args(expected)
        if len(args) == 2:
            return all(
                _check_type(k, args[0]) and _check_type(v, args[1]) for k, v in value.items()
            )
        return True

    # Union types
    if origin is type(None | int):  # UnionType
        args = get_args(expected)
        return any(_check_type(value, arg) for arg in args)

    return isinstance(value, expected)


def validate_queue_size(size: int, name: str = "queue_size") -> int:
    """
    Validate a queue size parameter.

    Args:
        size: Queue size to validate.
        name: Parameter name for error messages.

    Returns:
        Validated size.

    Raises:
        InvalidConfigurationError: If size is invalid.
    """
    if not isinstance(size, int):
        raise InvalidConfigurationError(name, size, "must be an integer")

    if size < 1:
        raise InvalidConfigurationError(name, size, "must be >= 1")

    if size > 1_000_000:
        raise InvalidConfigurationError(name, size, "must be <= 1,000,000")

    return size


def validate_grid_size(size: int | tuple[int, ...], name: str = "grid_size") -> tuple[int, ...]:
    """
    Validate a CUDA grid size parameter.

    Args:
        size: Grid size to validate.
        name: Parameter name for error messages.

    Returns:
        Validated size as tuple.

    Raises:
        InvalidConfigurationError: If size is invalid.
    """
    if isinstance(size, int):
        size = (size,)

    if not isinstance(size, tuple):
        raise InvalidConfigurationError(name, size, "must be int or tuple of ints")

    for dim in size:
        if not isinstance(dim, int) or dim < 1:
            raise InvalidConfigurationError(name, size, "all dimensions must be positive integers")

    if len(size) > 3:
        raise InvalidConfigurationError(name, size, "maximum 3 dimensions")

    return size


def validate_block_size(
    size: int | tuple[int, ...],
    name: str = "block_size",
    max_threads: int = 1024,
) -> tuple[int, ...]:
    """
    Validate a CUDA block size parameter.

    Args:
        size: Block size to validate.
        name: Parameter name for error messages.
        max_threads: Maximum threads per block.

    Returns:
        Validated size as tuple.

    Raises:
        InvalidConfigurationError: If size is invalid.
    """
    if isinstance(size, int):
        size = (size,)

    if not isinstance(size, tuple):
        raise InvalidConfigurationError(name, size, "must be int or tuple of ints")

    total_threads = 1
    for dim in size:
        if not isinstance(dim, int) or dim < 1:
            raise InvalidConfigurationError(name, size, "all dimensions must be positive integers")
        total_threads *= dim

    if total_threads > max_threads:
        raise InvalidConfigurationError(
            name, size, f"total threads ({total_threads}) exceeds max ({max_threads})"
        )

    if len(size) > 3:
        raise InvalidConfigurationError(name, size, "maximum 3 dimensions")

    return size


def validate_priority(priority: int) -> int:
    """
    Validate a message priority.

    Args:
        priority: Priority value (0-255).

    Returns:
        Validated priority.

    Raises:
        InvalidConfigurationError: If priority is invalid.
    """
    if not isinstance(priority, int):
        raise InvalidConfigurationError("priority", priority, "must be an integer")

    if not 0 <= priority <= 255:
        raise InvalidConfigurationError("priority", priority, "must be 0-255")

    return priority


def validate_backpressure_strategy(strategy: str) -> str:
    """
    Validate a backpressure strategy.

    Args:
        strategy: Strategy name.

    Returns:
        Validated strategy.

    Raises:
        InvalidConfigurationError: If strategy is invalid.
    """
    valid = {"block", "reject", "drop_oldest"}

    if strategy not in valid:
        raise InvalidConfigurationError(
            "backpressure_strategy",
            strategy,
            f"must be one of {valid}",
        )

    return strategy


class RuntimeValidator:
    """
    Runtime validator for kernel and message validation.

    Provides centralized validation with configurable strictness.
    """

    def __init__(self, strict: bool = True) -> None:
        """
        Initialize the validator.

        Args:
            strict: Whether to raise exceptions on validation failure.
        """
        self._strict = strict
        self._warnings: list[str] = []

    @property
    def strict(self) -> bool:
        """Check if strict mode is enabled."""
        return self._strict

    @property
    def warnings(self) -> list[str]:
        """Get accumulated warnings."""
        return self._warnings.copy()

    def validate_message_type(self, cls: type) -> bool:
        """
        Validate a message type.

        Args:
            cls: Class to validate.

        Returns:
            True if valid.
        """
        if not is_dataclass(cls):
            msg = f"Message type '{cls.__name__}' should be a dataclass"
            if self._strict:
                raise TypeValidationError(type, cls, msg)
            self._warnings.append(msg)
            return False

        return True

    def validate_kernel_config(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate kernel configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Validated configuration.
        """
        validated: dict[str, Any] = {}

        if "queue_size" in config:
            validated["queue_size"] = validate_queue_size(config["queue_size"])

        if "input_queue_size" in config:
            validated["input_queue_size"] = validate_queue_size(
                config["input_queue_size"], "input_queue_size"
            )

        if "output_queue_size" in config:
            validated["output_queue_size"] = validate_queue_size(
                config["output_queue_size"], "output_queue_size"
            )

        if "grid_size" in config:
            validated["grid_size"] = validate_grid_size(config["grid_size"])

        if "block_size" in config:
            validated["block_size"] = validate_block_size(config["block_size"])

        if "backpressure_strategy" in config:
            validated["backpressure_strategy"] = validate_backpressure_strategy(
                config["backpressure_strategy"]
            )

        return validated

    def clear_warnings(self) -> None:
        """Clear accumulated warnings."""
        self._warnings.clear()


# Global validator instance
_validator: RuntimeValidator | None = None


def get_validator(strict: bool = True) -> RuntimeValidator:
    """Get the global runtime validator."""
    global _validator
    if _validator is None:
        _validator = RuntimeValidator(strict=strict)
    return _validator
