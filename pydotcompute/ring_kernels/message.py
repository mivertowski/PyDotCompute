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
    """Encode UUID and numpy objects for msgpack."""
    if isinstance(obj, UUID):
        # Use binary format: 16 bytes instead of 36-byte string
        return {"__uuid_bin__": obj.bytes}
    # Handle numpy arrays (for standard messages that include arrays)
    if hasattr(obj, "tobytes") and hasattr(obj, "dtype") and hasattr(obj, "shape"):
        # numpy/cupy array - serialize with data
        return {
            "__ndarray__": obj.tobytes(),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    # Handle numpy scalar types
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Unknown type: {type(obj)}")


def _uuid_decoder(obj: dict[str, Any]) -> Any:
    """Decode UUID and numpy objects from msgpack."""
    import numpy as np

    # Support new binary format
    if "__uuid_bin__" in obj:
        return UUID(bytes=obj["__uuid_bin__"])
    # Backward compatibility with string format
    if "__uuid__" in obj:
        return UUID(obj["__uuid__"])
    # Decode numpy arrays
    if "__ndarray__" in obj:
        return np.frombuffer(
            obj["__ndarray__"],
            dtype=obj["dtype"],
        ).reshape(obj["shape"]).copy()  # copy to make writeable
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

    # Class-level cache for field metadata to avoid reflection on every call
    _cached_fields: tuple[dataclasses.Field[Any], ...] | None = None
    _cached_field_names: frozenset[str] | None = None

    message_id: UUID = field(default_factory=uuid4)
    priority: int = field(default=128)  # 0-255, higher = more important
    correlation_id: UUID | None = field(default=None)

    @classmethod
    def _get_fields(cls) -> tuple[dataclasses.Field[Any], ...]:
        """Get cached field metadata for this class."""
        if cls._cached_fields is None:
            cls._cached_fields = dataclasses.fields(cls)
        return cls._cached_fields

    @classmethod
    def _get_field_names(cls) -> frozenset[str]:
        """Get cached field names for this class."""
        if cls._cached_field_names is None:
            cls._cached_field_names = frozenset(f.name for f in cls._get_fields())
        return cls._cached_field_names

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
        for f in self._get_fields():
            # Skip class-level cache fields
            if f.name.startswith("_cached"):
                continue
            value = getattr(self, f.name)
            # UUIDs are handled by msgpack encoder with binary format
            result[f.name] = value
        return result

    @classmethod
    def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create message from dictionary."""
        # Convert string/bytes UUIDs back to UUID objects (backward compatibility)
        for uuid_field in ("message_id", "correlation_id"):
            if uuid_field in data:
                val = data[uuid_field]
                if isinstance(val, str):
                    data[uuid_field] = UUID(val)
                elif isinstance(val, bytes):
                    data[uuid_field] = UUID(bytes=val)
                # Already UUID - no conversion needed

        # Get only fields that exist in this class (using cached field names)
        field_names = cls._get_field_names()
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

        # Add caching attributes for field metadata
        cls._cached_fields = None  # type: ignore
        cls._cached_field_names = None  # type: ignore

        # Add caching methods
        if not hasattr(cls, "_get_fields"):
            cls._get_fields = classmethod(RingKernelMessage._get_fields.__func__)  # type: ignore
        if not hasattr(cls, "_get_field_names"):
            cls._get_field_names = classmethod(RingKernelMessage._get_field_names.__func__)  # type: ignore

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
