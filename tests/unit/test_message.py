"""
Unit tests for message serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from pydotcompute.exceptions import MessageDeserializationError, MessageSerializationError
from pydotcompute.ring_kernels.message import (
    RingKernelMessage,
    deserialize_message,
    get_message_type,
    message,
)


class TestRingKernelMessage:
    """Tests for RingKernelMessage base class."""

    def test_default_creation(self) -> None:
        """Test creating message with defaults."""
        msg = RingKernelMessage()

        assert isinstance(msg.message_id, UUID)
        assert msg.priority == 128
        assert msg.correlation_id is None

    def test_custom_creation(self) -> None:
        """Test creating message with custom values."""
        custom_id = uuid4()
        correlation = uuid4()

        msg = RingKernelMessage(
            message_id=custom_id,
            priority=255,
            correlation_id=correlation,
        )

        assert msg.message_id == custom_id
        assert msg.priority == 255
        assert msg.correlation_id == correlation

    def test_priority_validation(self) -> None:
        """Test priority validation."""
        with pytest.raises(ValueError):
            RingKernelMessage(priority=-1)

        with pytest.raises(ValueError):
            RingKernelMessage(priority=256)

    def test_serialization(self) -> None:
        """Test message serialization."""
        msg = RingKernelMessage(priority=100)

        data = msg.serialize()

        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_deserialization(self) -> None:
        """Test message deserialization."""
        original = RingKernelMessage(priority=100)
        data = original.serialize()

        restored = RingKernelMessage.deserialize(data)

        assert restored.priority == original.priority
        assert restored.message_id == original.message_id

    def test_roundtrip(self) -> None:
        """Test serialization roundtrip."""
        original = RingKernelMessage(priority=50)

        data = original.serialize()
        restored = RingKernelMessage.deserialize(data)

        assert restored.message_id == original.message_id
        assert restored.priority == original.priority

    def test_with_correlation(self) -> None:
        """Test with_correlation method."""
        msg = RingKernelMessage()
        correlation_id = uuid4()

        correlated = msg.with_correlation(correlation_id)

        assert correlated.correlation_id == correlation_id
        assert correlated.message_id == msg.message_id


class TestMessageDecorator:
    """Tests for @message decorator."""

    def test_decorator_creates_dataclass(self) -> None:
        """Test that decorator creates proper dataclass."""

        @message
        class TestMsg:
            value: float = 0.0

        msg = TestMsg(value=42.0)

        assert msg.value == 42.0
        assert hasattr(msg, "message_id")
        assert hasattr(msg, "priority")
        assert hasattr(msg, "correlation_id")

    def test_decorated_message_serialization(self) -> None:
        """Test serialization of decorated messages."""

        @message
        class CustomMessage:
            x: float = 0.0
            y: float = 0.0
            name: str = ""

        original = CustomMessage(x=1.0, y=2.0, name="test")

        data = original.serialize()
        restored = CustomMessage.deserialize(data)

        assert restored.x == original.x
        assert restored.y == original.y
        assert restored.name == original.name

    def test_decorator_registers_message(self) -> None:
        """Test that decorator registers message type."""

        @message
        class RegisteredMessage:
            data: str = ""

        retrieved = get_message_type("RegisteredMessage")

        assert retrieved is not None
        assert retrieved.__name__ == "RegisteredMessage"

    def test_nested_data_serialization(self) -> None:
        """Test serialization with nested data types."""

        @message
        class ComplexMessage:
            values: list = None  # type: ignore
            mapping: dict = None  # type: ignore

            def __post_init__(self) -> None:
                if self.values is None:
                    self.values = []
                if self.mapping is None:
                    self.mapping = {}

        original = ComplexMessage(
            values=[1, 2, 3],
            mapping={"a": 1, "b": 2},
        )

        data = original.serialize()
        restored = ComplexMessage.deserialize(data)

        assert restored.values == [1, 2, 3]
        assert restored.mapping == {"a": 1, "b": 2}


class TestDeserializeMessage:
    """Tests for deserialize_message function."""

    def test_deserialize_unknown_type(self) -> None:
        """Test deserializing message with unknown type."""
        msg = RingKernelMessage(priority=100)
        data = msg.serialize()

        restored = deserialize_message(data)

        assert isinstance(restored, RingKernelMessage)
        assert restored.priority == 100

    def test_deserialize_registered_type(self) -> None:
        """Test deserializing message with registered type."""

        @message
        class KnownMessage:
            value: int = 0

        original = KnownMessage(value=42)
        data = original.serialize()

        restored = deserialize_message(data)

        assert hasattr(restored, "value")
        assert restored.value == 42


class TestMessageErrors:
    """Tests for message error handling."""

    def test_deserialization_error(self) -> None:
        """Test deserialization with invalid data."""
        with pytest.raises(MessageDeserializationError):
            RingKernelMessage.deserialize(b"invalid data")

    def test_deserialization_error_empty(self) -> None:
        """Test deserialization with empty data."""
        with pytest.raises(MessageDeserializationError):
            RingKernelMessage.deserialize(b"")
