# Validators

Type validation utilities for messages and kernels.

## Overview

The validators module provides runtime type checking for message types and kernel signatures. This helps catch type errors early and provides clear error messages.

```python
from pydotcompute.decorators.validators import validate_message_type

# Validate message has required attributes
validate_message_type(MyMessage)
```

## Functions

### validate_message_type

```python
def validate_message_type(cls: type) -> None:
    """
    Validate that a class is a valid message type.

    Checks for:
    - Is a dataclass
    - Has message_id field (UUID)
    - Has priority field (int)
    - Has correlation_id field (UUID | None)
    - All fields are serializable types

    Args:
        cls: Class to validate

    Raises:
        TypeError: If class is not a valid message type
    """
```

### validate_kernel_signature

```python
def validate_kernel_signature(func: Callable) -> None:
    """
    Validate that a function is a valid kernel.

    Checks for:
    - Is callable
    - Is async function (for ring kernels)
    - Has ctx parameter

    Args:
        func: Function to validate

    Raises:
        TypeError: If function is not a valid kernel
    """
```

### is_serializable_type

```python
def is_serializable_type(tp: type) -> bool:
    """
    Check if a type can be serialized by msgpack.

    Supported types:
    - Primitives: int, float, str, bool, bytes, None
    - Collections: list, tuple, dict
    - Special: UUID, datetime
    - Nested message types

    Args:
        tp: Type to check

    Returns:
        Whether the type is serializable
    """
```

### get_type_signature

```python
def get_type_signature(func: Callable) -> tuple[type, ...]:
    """
    Extract type signature from function annotations.

    Args:
        func: Annotated function

    Returns:
        Tuple of argument types
    """
```

## Usage Examples

### Validating Message Types

```python
from pydotcompute.decorators.validators import validate_message_type
from dataclasses import dataclass

@dataclass
class ValidMessage:
    data: str
    message_id: UUID = field(default_factory=uuid4)
    priority: int = 128
    correlation_id: UUID | None = None

# This passes
validate_message_type(ValidMessage)

@dataclass
class InvalidMessage:
    data: str
    # Missing required fields!

# This raises TypeError
try:
    validate_message_type(InvalidMessage)
except TypeError as e:
    print(f"Invalid: {e}")
```

### Checking Serializable Types

```python
from pydotcompute.decorators.validators import is_serializable_type

# Serializable types
assert is_serializable_type(int)
assert is_serializable_type(str)
assert is_serializable_type(list)
assert is_serializable_type(dict)
assert is_serializable_type(UUID)

# Non-serializable
assert not is_serializable_type(lambda x: x)
assert not is_serializable_type(object)
```

### Validating Kernel Functions

```python
from pydotcompute.decorators.validators import validate_kernel_signature

async def valid_kernel(ctx):
    pass

# This passes
validate_kernel_signature(valid_kernel)

def sync_kernel(ctx):  # Not async!
    pass

# This raises TypeError for ring kernels
try:
    validate_kernel_signature(sync_kernel)
except TypeError as e:
    print(f"Invalid: {e}")
```

### Extracting Type Signatures

```python
from pydotcompute.decorators.validators import get_type_signature
import numpy as np

def my_kernel(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
    pass

sig = get_type_signature(my_kernel)
# sig = (np.ndarray, np.ndarray, np.ndarray)
```

## Validation Rules

### Message Requirements

| Field | Type | Required |
|-------|------|----------|
| `message_id` | `UUID` | Yes |
| `priority` | `int` | Yes |
| `correlation_id` | `UUID \| None` | Yes |
| Other fields | Serializable | User-defined |

### Serializable Types

| Category | Types |
|----------|-------|
| Primitives | `int`, `float`, `str`, `bool`, `bytes`, `None` |
| Collections | `list`, `tuple`, `dict`, `set` |
| Special | `UUID`, `datetime`, `date`, `Decimal` |
| Nested | Other `@message` decorated types |

### Kernel Requirements

| Type | Requirements |
|------|--------------|
| `@kernel` | Callable, numeric/array params |
| `@ring_kernel` | Async function, `ctx` parameter |

## Error Messages

The validators provide clear error messages:

```python
# Missing message_id
TypeError: Message type 'MyMessage' must have a 'message_id' field of type UUID

# Not a dataclass
TypeError: Message type must be a dataclass, got <class 'dict'>

# Not async
TypeError: Ring kernel 'my_kernel' must be an async function

# Non-serializable field
TypeError: Field 'callback' of type '<class 'function'>' is not serializable
```

## Notes

- Validation runs at decoration time, not runtime
- Use type annotations for best results
- Custom types can implement `__msgpack__` for serialization
- Validation can be disabled for performance in production
