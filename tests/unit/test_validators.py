"""
Unit tests for the validators module.

Tests validation decorators, type checking utilities, and the
RuntimeValidator class for configuration and message validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from pydotcompute.decorators.validators import (
    RuntimeValidator,
    _check_type,
    get_validator,
    validate_backpressure_strategy,
    validate_block_size,
    validate_config,
    validate_grid_size,
    validate_message,
    validate_priority,
    validate_queue_size,
)
from pydotcompute.exceptions import InvalidConfigurationError, TypeValidationError


class TestValidateMessage:
    """Tests for validate_message decorator."""

    def test_valid_dataclass(self) -> None:
        """Test validating a valid dataclass."""

        @validate_message
        @dataclass
        class ValidMessage:
            value: int

        # Should not raise
        assert ValidMessage is not None

    def test_non_dataclass_raises(self) -> None:
        """Test that non-dataclass raises TypeValidationError."""

        with pytest.raises(TypeValidationError):

            @validate_message
            class NotADataclass:
                pass

    def test_dataclass_without_serialize_warns(self) -> None:
        """Test that dataclass without serialize method warns."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @validate_message
            @dataclass
            class MessageWithoutSerialize:
                data: str

            # Should have warning
            assert len(w) >= 1
            assert "serialize" in str(w[-1].message)


class TestValidateConfig:
    """Tests for validate_config decorator."""

    def test_function_executes_with_decorator(self) -> None:
        """Test that decorated function executes."""

        @dataclass
        class SimpleConfig:
            pass

        @validate_config(SimpleConfig)
        def func(x: int) -> int:
            return x * 2

        # Function should still execute
        result = func(5)
        assert result == 10

    def test_decorator_preserves_function(self) -> None:
        """Test that decorator preserves function behavior."""

        @dataclass
        class Config:
            pass

        @validate_config(Config)
        def add(a: int, b: int) -> int:
            return a + b

        assert add(3, 4) == 7

    def test_decorator_with_kwargs(self) -> None:
        """Test decorator with keyword arguments."""

        @dataclass
        class Config:
            pass

        @validate_config(Config)
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = greet(name="World", greeting="Hi")
        assert result == "Hi, World!"


class TestCheckType:
    """Tests for _check_type utility function."""

    def test_simple_types(self) -> None:
        """Test checking simple types."""
        assert _check_type(42, int) is True
        assert _check_type("hello", str) is True
        assert _check_type(3.14, float) is True
        assert _check_type(True, bool) is True

    def test_simple_type_mismatch(self) -> None:
        """Test mismatched simple types."""
        assert _check_type("42", int) is False
        assert _check_type(42, str) is False

    def test_any_type(self) -> None:
        """Test Any type accepts anything."""
        assert _check_type(42, Any) is True
        assert _check_type("hello", Any) is True
        assert _check_type(None, Any) is True

    def test_list_type(self) -> None:
        """Test list type checking."""
        assert _check_type([1, 2, 3], list[int]) is True
        assert _check_type(["a", "b"], list[str]) is True
        assert _check_type([1, "mixed"], list[int]) is False

    def test_dict_type(self) -> None:
        """Test dict type checking."""
        assert _check_type({"a": 1, "b": 2}, dict[str, int]) is True
        assert _check_type({1: "a"}, dict[int, str]) is True
        assert _check_type({"a": "b"}, dict[str, int]) is False

    def test_union_type(self) -> None:
        """Test union type checking."""
        assert _check_type(42, int | str) is True
        assert _check_type("hello", int | str) is True
        assert _check_type(3.14, int | str) is False

    def test_optional_type(self) -> None:
        """Test optional (None union) type checking."""
        assert _check_type(None, int | None) is True
        assert _check_type(42, int | None) is True


class TestValidateQueueSize:
    """Tests for validate_queue_size function."""

    def test_valid_sizes(self) -> None:
        """Test valid queue sizes."""
        assert validate_queue_size(1) == 1
        assert validate_queue_size(100) == 100
        assert validate_queue_size(1_000_000) == 1_000_000

    def test_zero_raises(self) -> None:
        """Test that zero raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_queue_size(0)

    def test_negative_raises(self) -> None:
        """Test that negative raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_queue_size(-1)

    def test_too_large_raises(self) -> None:
        """Test that too large value raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_queue_size(1_000_001)

    def test_non_int_raises(self) -> None:
        """Test that non-integer raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_queue_size(3.14)  # type: ignore

    def test_custom_name(self) -> None:
        """Test custom parameter name in error."""
        with pytest.raises(InvalidConfigurationError) as exc_info:
            validate_queue_size(-1, name="my_queue")

        assert "my_queue" in str(exc_info.value)


class TestValidateGridSize:
    """Tests for validate_grid_size function."""

    def test_int_converted_to_tuple(self) -> None:
        """Test that int is converted to tuple."""
        result = validate_grid_size(64)
        assert result == (64,)

    def test_valid_1d(self) -> None:
        """Test valid 1D grid size."""
        result = validate_grid_size((128,))
        assert result == (128,)

    def test_valid_2d(self) -> None:
        """Test valid 2D grid size."""
        result = validate_grid_size((64, 64))
        assert result == (64, 64)

    def test_valid_3d(self) -> None:
        """Test valid 3D grid size."""
        result = validate_grid_size((8, 8, 8))
        assert result == (8, 8, 8)

    def test_4d_raises(self) -> None:
        """Test that 4D raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_grid_size((2, 2, 2, 2))

    def test_zero_dimension_raises(self) -> None:
        """Test that zero dimension raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_grid_size((0, 64))

    def test_negative_dimension_raises(self) -> None:
        """Test that negative dimension raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_grid_size((-1, 64))

    def test_invalid_type_raises(self) -> None:
        """Test that invalid type raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_grid_size("invalid")  # type: ignore


class TestValidateBlockSize:
    """Tests for validate_block_size function."""

    def test_int_converted_to_tuple(self) -> None:
        """Test that int is converted to tuple."""
        result = validate_block_size(256)
        assert result == (256,)

    def test_valid_1d(self) -> None:
        """Test valid 1D block size."""
        result = validate_block_size((512,))
        assert result == (512,)

    def test_valid_2d(self) -> None:
        """Test valid 2D block size."""
        result = validate_block_size((16, 16))
        assert result == (16, 16)

    def test_valid_3d(self) -> None:
        """Test valid 3D block size."""
        result = validate_block_size((8, 8, 8))
        assert result == (8, 8, 8)

    def test_exceeds_max_threads_raises(self) -> None:
        """Test that exceeding max threads raises."""
        # Default max is 1024
        with pytest.raises(InvalidConfigurationError):
            validate_block_size((32, 64))  # 2048 threads

    def test_custom_max_threads(self) -> None:
        """Test custom max threads limit."""
        result = validate_block_size((32, 64), max_threads=2048)
        assert result == (32, 64)

    def test_zero_dimension_raises(self) -> None:
        """Test that zero dimension raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_block_size((0,))

    def test_4d_raises(self) -> None:
        """Test that 4D raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_block_size((2, 2, 2, 2))


class TestValidatePriority:
    """Tests for validate_priority function."""

    def test_valid_priorities(self) -> None:
        """Test valid priority values."""
        assert validate_priority(0) == 0
        assert validate_priority(128) == 128
        assert validate_priority(255) == 255

    def test_negative_raises(self) -> None:
        """Test that negative raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_priority(-1)

    def test_too_large_raises(self) -> None:
        """Test that > 255 raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_priority(256)

    def test_non_int_raises(self) -> None:
        """Test that non-integer raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_priority(3.14)  # type: ignore


class TestValidateBackpressureStrategy:
    """Tests for validate_backpressure_strategy function."""

    def test_valid_strategies(self) -> None:
        """Test valid strategies."""
        assert validate_backpressure_strategy("block") == "block"
        assert validate_backpressure_strategy("reject") == "reject"
        assert validate_backpressure_strategy("drop_oldest") == "drop_oldest"

    def test_invalid_strategy_raises(self) -> None:
        """Test that invalid strategy raises."""
        with pytest.raises(InvalidConfigurationError):
            validate_backpressure_strategy("invalid")

    def test_case_sensitive(self) -> None:
        """Test that validation is case-sensitive."""
        with pytest.raises(InvalidConfigurationError):
            validate_backpressure_strategy("BLOCK")


class TestRuntimeValidator:
    """Tests for RuntimeValidator class."""

    def test_creation_strict(self) -> None:
        """Test creating strict validator."""
        validator = RuntimeValidator(strict=True)
        assert validator.strict is True
        assert validator.warnings == []

    def test_creation_non_strict(self) -> None:
        """Test creating non-strict validator."""
        validator = RuntimeValidator(strict=False)
        assert validator.strict is False

    def test_validate_message_type_valid(self) -> None:
        """Test validating valid message type."""
        validator = RuntimeValidator()

        @dataclass
        class ValidMessage:
            data: str

        result = validator.validate_message_type(ValidMessage)
        assert result is True

    def test_validate_message_type_invalid_strict(self) -> None:
        """Test validating invalid message type in strict mode."""
        validator = RuntimeValidator(strict=True)

        class NotDataclass:
            pass

        with pytest.raises(TypeValidationError):
            validator.validate_message_type(NotDataclass)

    def test_validate_message_type_invalid_non_strict(self) -> None:
        """Test validating invalid message type in non-strict mode."""
        validator = RuntimeValidator(strict=False)

        class NotDataclass:
            pass

        result = validator.validate_message_type(NotDataclass)
        assert result is False
        assert len(validator.warnings) == 1

    def test_validate_kernel_config(self) -> None:
        """Test validating kernel configuration."""
        validator = RuntimeValidator()

        config = {
            "queue_size": 4096,
            "grid_size": 64,
            "block_size": 256,
            "backpressure_strategy": "block",
        }

        validated = validator.validate_kernel_config(config)

        assert validated["queue_size"] == 4096
        assert validated["grid_size"] == (64,)
        assert validated["block_size"] == (256,)
        assert validated["backpressure_strategy"] == "block"

    def test_validate_kernel_config_partial(self) -> None:
        """Test validating partial configuration."""
        validator = RuntimeValidator()

        config = {"queue_size": 1000}
        validated = validator.validate_kernel_config(config)

        assert validated["queue_size"] == 1000
        assert "grid_size" not in validated

    def test_validate_kernel_config_with_queue_variants(self) -> None:
        """Test validating input/output queue sizes."""
        validator = RuntimeValidator()

        config = {
            "input_queue_size": 2048,
            "output_queue_size": 4096,
        }

        validated = validator.validate_kernel_config(config)

        assert validated["input_queue_size"] == 2048
        assert validated["output_queue_size"] == 4096

    def test_clear_warnings(self) -> None:
        """Test clearing warnings."""
        validator = RuntimeValidator(strict=False)

        class NotDataclass:
            pass

        validator.validate_message_type(NotDataclass)
        assert len(validator.warnings) == 1

        validator.clear_warnings()
        assert validator.warnings == []

    def test_warnings_property_returns_copy(self) -> None:
        """Test that warnings property returns a copy."""
        validator = RuntimeValidator(strict=False)

        class NotDataclass:
            pass

        validator.validate_message_type(NotDataclass)

        warnings1 = validator.warnings
        warnings2 = validator.warnings

        assert warnings1 == warnings2
        assert warnings1 is not warnings2


class TestGetValidator:
    """Tests for get_validator function."""

    def test_returns_validator(self) -> None:
        """Test that get_validator returns a RuntimeValidator."""
        # Reset global
        import pydotcompute.decorators.validators as validators_module

        validators_module._validator = None

        validator = get_validator()
        assert isinstance(validator, RuntimeValidator)

    def test_returns_singleton(self) -> None:
        """Test that get_validator returns singleton."""
        import pydotcompute.decorators.validators as validators_module

        validators_module._validator = None

        v1 = get_validator()
        v2 = get_validator()

        assert v1 is v2

    def test_respects_strict_param(self) -> None:
        """Test that strict parameter is respected on creation."""
        import pydotcompute.decorators.validators as validators_module

        validators_module._validator = None

        validator = get_validator(strict=True)
        assert validator.strict is True


class TestValidatorEdgeCases:
    """Edge case tests for validators."""

    def test_empty_grid_size_tuple_passes_basic_check(self) -> None:
        """Test empty tuple passes basic check (no dimensions to validate)."""
        # Empty tuple has 0 dimensions which is <= 3, passes basic checks
        result = validate_grid_size(())
        assert result == ()

    def test_empty_block_size_tuple_passes_basic_check(self) -> None:
        """Test empty tuple passes basic check (total threads = 1)."""
        # Empty tuple has no dimensions, total_threads starts at 1
        result = validate_block_size(())
        assert result == ()

    def test_single_element_grid(self) -> None:
        """Test single element grid size."""
        result = validate_grid_size((1,))
        assert result == (1,)

    def test_single_element_block(self) -> None:
        """Test single element block size."""
        result = validate_block_size((1,))
        assert result == (1,)

    def test_check_type_with_nested_list(self) -> None:
        """Test type checking with nested lists."""
        assert _check_type([[1, 2], [3, 4]], list) is True

    def test_check_type_with_empty_containers(self) -> None:
        """Test type checking with empty containers."""
        assert _check_type([], list[int]) is True
        assert _check_type({}, dict[str, int]) is True

    def test_check_type_none_value(self) -> None:
        """Test type checking with None value."""
        assert _check_type(None, type(None)) is True

    def test_large_grid_size(self) -> None:
        """Test large but valid grid size."""
        result = validate_grid_size((65535, 65535, 65535))
        assert result == (65535, 65535, 65535)

    def test_max_block_threads(self) -> None:
        """Test block size at maximum threads."""
        result = validate_block_size((1024,), max_threads=1024)
        assert result == (1024,)
