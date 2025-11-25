# Contributing to PyDotCompute

Thank you for your interest in contributing to PyDotCompute!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PyDotCompute.git
   cd PyDotCompute
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pydotcompute

# Skip CUDA tests if no GPU available
pytest -m "not cuda"
```

### Code Quality

```bash
# Type checking
mypy pydotcompute

# Linting
ruff check pydotcompute

# Format check
ruff format --check pydotcompute
```

### Pre-commit Checks

Before submitting a PR, ensure:
- All tests pass
- Type checking passes (`mypy pydotcompute`)
- Linting passes (`ruff check pydotcompute`)
- New code has appropriate test coverage

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Update documentation if needed
5. Run all checks locally
6. Submit a pull request

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- CUDA version (if applicable)
- Steps to reproduce
- Expected vs actual behavior

## Project Roadmap

See `docs/IMPLEMENTATION_PLAN.md` for the project roadmap and planned features.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
