"""
Message infrastructure for ring kernels.

Provides base message class and serialization support.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, get_type_hints
from uuid import UUID, uuid4

import msgpack

from pydotcompute.exceptions import (
    MessageDeserializationError,
    MessageSerializationError,
)

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound="RingKernelMessage")


# Registry of message types for deserialization
_message_registry: dict[str, type[RingKernelMessage]] = {}


def _uuid_encoder(obj: Any) -> Any:
    """Encode UUID objects for msgpack."""
    if isinstance(obj, UUID):
        return {"__uuid__": str(obj)}
    raise TypeError(f"Unknown type: {type(obj)}")


def _uuid_decoder(obj: dict[str, Any]) -> Any:
    """Decode UUID objects from msgpack."""
    if "__uuid__" in obj:
        return UUID(obj["__uuid__"])
    return obj


@dataclass
class RingKernelMessage:
    """
    Base class for all ring kernel messages.

    Provides common fields and serialization/deserialization support.
    All user messages should inherit from this class or use the @message decorator.

    Example:
        >>> @dataclass
        ... class MyMessage(RingKernelMessage):
        ...     value: float
        ...
        >>> msg = MyMessage(value=42.0)
        >>> data = msg.serialize()
        >>> restored = MyMessage.deserialize(data)
    """

    message_id: UUID = field(default_factory=uuid4)
    priority: int = field(default=128)  # 0-255, higher = more important
    correlation_id: UUID | None = field(default=None)

    def serialize(self) -> bytes:
        """
        Serialize the message to bytes using msgpack.

        Returns:
            Serialized message bytes.

        Raises:
            MessageSerializationError: If serialization fails.
        """
        try:
            # Convert dataclass to dict, handling UUIDs
            data = self._to_dict()
            data["__type__"] = self.__class__.__name__
            return msgpack.packb(data, default=_uuid_encoder, strict_types=False)
        except Exception as e:
            raise MessageSerializationError(type(self), e) from e

    @classmethod
    def deserialize(cls: type[T], data: bytes) -> T:
        """
        Deserialize a message from bytes.

        Args:
            data: Serialized message bytes.

        Returns:
            Deserialized message instance.

        Raises:
            MessageDeserializationError: If deserialization fails.
        """
        try:
            unpacked = msgpack.unpackb(
                data,
                object_hook=_uuid_decoder,
                raw=False,
            )

            # Remove type marker if present
            unpacked.pop("__type__", None)

            return cls._from_dict(unpacked)
        except Exception as e:
            raise MessageDeserializationError(cls, e) from e

    def _to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        result = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, UUID):
                value = str(value)
            result[f.name] = value
        return result

    @classmethod
    def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create message from dictionary."""
        # Convert string UUIDs back to UUID objects
        if "message_id" in data and isinstance(data["message_id"], str):
            data["message_id"] = UUID(data["message_id"])
        if "correlation_id" in data and isinstance(data["correlation_id"], str):
            data["correlation_id"] = UUID(data["correlation_id"])

        # Get only fields that exist in this class
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}

        return cls(**filtered_data)

    def with_correlation(self, correlation_id: UUID) -> RingKernelMessage:
        """
        Create a copy of this message with a correlation ID.

        Args:
            correlation_id: The correlation ID to set.

        Returns:
            New message instance with correlation ID.
        """
        data = self._to_dict()
        data["correlation_id"] = correlation_id
        return self.__class__._from_dict(data)

    def reply(self: T, **kwargs: Any) -> T:
        """
        Create a reply message with the same correlation ID.

        Args:
            **kwargs: Fields for the reply message.

        Returns:
            New message with correlation ID set.
        """
        response_class = kwargs.pop("response_class", self.__class__)
        data = {
            "correlation_id": self.message_id,
            **kwargs,
        }
        return response_class(**data)

    def __post_init__(self) -> None:
        """Validate message after initialization."""
        if not 0 <= self.priority <= 255:
            raise ValueError(f"Priority must be 0-255, got {self.priority}")


def message(cls: type[T] | None = None) -> type[T]:
    """
    Decorator to create a ring kernel message class.

    Automatically applies @dataclass and registers the message type.

    Example:
        >>> @message
        ... class VectorAddRequest:
        ...     a: float
        ...     b: float
        ...
        >>> msg = VectorAddRequest(a=1.0, b=2.0)
    """

    def decorator(cls: type[T]) -> type[T]:
        # Check if already a dataclass
        if not dataclasses.is_dataclass(cls):
            # Add message fields if not present
            annotations = getattr(cls, "__annotations__", {})
            if "message_id" not in annotations:
                annotations["message_id"] = UUID
            if "priority" not in annotations:
                annotations["priority"] = int
            if "correlation_id" not in annotations:
                annotations["correlation_id"] = UUID | None

            cls.__annotations__ = annotations

            # Add defaults
            if not hasattr(cls, "message_id"):
                cls.message_id = field(default_factory=uuid4)  # type: ignore
            if not hasattr(cls, "priority"):
                cls.priority = field(default=128)  # type: ignore
            if not hasattr(cls, "correlation_id"):
                cls.correlation_id = field(default=None)  # type: ignore

            # Apply dataclass decorator
            cls = dataclass(cls)

        # Add serialization methods
        if not hasattr(cls, "serialize"):
            cls.serialize = RingKernelMessage.serialize  # type: ignore
        if not hasattr(cls, "deserialize"):
            cls.deserialize = classmethod(RingKernelMessage.deserialize.__func__)  # type: ignore
        if not hasattr(cls, "_to_dict"):
            cls._to_dict = RingKernelMessage._to_dict  # type: ignore
        if not hasattr(cls, "_from_dict"):
            cls._from_dict = classmethod(RingKernelMessage._from_dict.__func__)  # type: ignore

        # Register message type
        _message_registry[cls.__name__] = cls  # type: ignore

        return cls

    if cls is not None:
        return decorator(cls)
    return decorator  # type: ignore


def get_message_type(name: str) -> type[RingKernelMessage] | None:
    """
    Get a registered message type by name.

    Args:
        name: Name of the message class.

    Returns:
        Message class or None if not registered.
    """
    return _message_registry.get(name)


def deserialize_message(data: bytes) -> RingKernelMessage:
    """
    Deserialize a message of unknown type.

    Uses the __type__ field in the serialized data to determine
    the correct message class.

    Args:
        data: Serialized message bytes.

    Returns:
        Deserialized message instance.

    Raises:
        MessageDeserializationError: If deserialization fails.
    """
    try:
        unpacked = msgpack.unpackb(
            data,
            object_hook=_uuid_decoder,
            raw=False,
        )

        type_name = unpacked.pop("__type__", None)
        if type_name and type_name in _message_registry:
            cls = _message_registry[type_name]
            return cls._from_dict(unpacked)

        # Fall back to base class
        return RingKernelMessage._from_dict(unpacked)

    except Exception as e:
        raise MessageDeserializationError(RingKernelMessage, e) from e
