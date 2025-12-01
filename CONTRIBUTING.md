# Contributing to HST

Thank you for your interest in contributing to the Hierarchical Spatial-Temporal (HST) project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please read and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

## How to Contribute

### Reporting Bugs

**Before Submitting a Bug Report:**

- Check the existing [issues](https://github.com/ilicilicc/HST/issues) to avoid duplicates
- Check [CHANGELOG.md](CHANGELOG.md) to see if the issue is already known
- Verify the issue persists with the latest version

**How to Submit a Bug Report:**

1. Use the bug report template when opening an issue
2. Provide a clear, descriptive title
3. Include detailed steps to reproduce
4. Provide example code that demonstrates the issue
5. Describe the observed behavior and what you expected
6. Include your environment details (Python version, OS, GPU info)

### Suggesting Enhancements

**Before Suggesting an Enhancement:**

- Check existing issues and discussions
- Consider if the enhancement aligns with HST's core purpose

**How to Submit an Enhancement:**

1. Use the feature request template
2. Provide a clear description of the enhancement
3. Explain the use case and benefits
4. Provide examples of how it would be used

### Code Contributions

#### Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/HST.git
   cd HST
   ```
3. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Making Changes

1. Create a new branch with a descriptive name:
   ```bash
   git checkout -b feature/descriptive-name
   # or
   git checkout -b fix/issue-description
   ```

2. Make your changes following the code style guidelines (see below)

3. Add or update tests as needed

4. Ensure tests pass:
   ```bash
   pytest
   ```

5. Run linting checks:
   ```bash
   flake8 .
   black --check .
   mypy .
   ```

#### Code Style Guidelines

**Python Style:**
- Follow PEP 8 conventions
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use meaningful variable and function names

**Formatting:**
```python
# Use Black for formatting
black hst_*.py

# Check with flake8
flake8 hst_*.py

# Type checking
mypy hst_*.py
```

**Example:**
```python
from typing import Tuple, Optional
import torch

def process_sequence(
    sequence: torch.Tensor,
    levels: int,
    dropout: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process a temporal sequence through the model.

    Args:
        sequence: Input tensor of shape (batch, length, features)
        levels: Number of lattice levels
        dropout: Dropout rate for regularization

    Returns:
        Tuple of (output, hidden_state)
    """
    # Implementation
    pass
```

#### Commit Messages

Write clear, descriptive commit messages:

```
feat: add harmonic component optimization

- Improve calculation efficiency by 30%
- Add caching mechanism for repeated components
- Update tests for new optimization

Fixes #123
```

**Format:**
- Use imperative mood ("add" not "adds" or "added")
- First line should be <= 50 characters
- Reference issues when applicable (Fixes #123)
- Separate subject from body with blank line

### Pull Request Process

1. **Ensure your code is ready:**
   - All tests pass
   - Code follows style guidelines
   - Documentation is updated

2. **Push to your fork:**
   ```bash
   git push origin feature/descriptive-name
   ```

3. **Open a Pull Request:**
   - Use the pull request template
   - Provide clear description of changes
   - Reference related issues
   - Include screenshots/examples if relevant

4. **Respond to Review:**
   - Address feedback promptly
   - Request re-review after making changes
   - Be open to discussion and suggestions

#### PR Requirements

- Title: Clear, descriptive summary
- Description: What changed and why
- Tests: Added/updated tests for new functionality
- Documentation: Updated docs if needed
- No breaking changes without discussion

### Testing

**Write Tests For:**
- New features
- Bug fixes
- Edge cases
- Error conditions

**Test Structure:**
```python
# tests/test_lattice.py
import pytest
from hst_v8_2 import HST, LatticeBuilder

def test_lattice_initialization():
    """Test that lattice initializes correctly."""
    builder = LatticeBuilder(levels=4)
    lattice = builder.build()
    assert lattice is not None
    assert len(lattice.levels) == 4

def test_lattice_with_invalid_levels():
    """Test that invalid levels raise error."""
    with pytest.raises(ValueError):
        LatticeBuilder(levels=0)
```

**Run Tests:**
```bash
# All tests
pytest

# Specific test file
pytest tests/test_lattice.py

# With coverage
pytest --cov=. --cov-report=html
```

### Documentation

**Update Documentation For:**
- New features
- API changes
- Significant improvements
- Bug fixes that affect behavior

**Documentation Files:**
- `README.md` - Project overview
- `INSTALL.md` - Installation guide
- `USAGE.md` - Usage examples
- `API.md` - API reference
- Docstrings in code

**Documentation Style:**
```python
def get_lattice_info(self) -> Dict[str, Any]:
    """
    Retrieve information about the lattice structure.

    Returns detailed information about the current lattice
    configuration including number of levels, nodes, and
    total connections.

    Returns:
        Dictionary with keys:
            - levels: Number of hierarchical levels
            - nodes_per_level: List of node counts
            - total_connections: Total edges in lattice
            - harmonic_components: Number of harmonics

    Example:
        >>> model = HST()
        >>> info = model.get_lattice_info()
        >>> print(info['levels'])  # 4
    """
```

## Development Workflow

### Local Testing

```bash
# Full test suite
pytest tests/ -v

# Specific test
pytest tests/test_hst.py::test_forward_pass -v

# With coverage report
pytest --cov=. --cov-report=term-missing
```

### Code Quality Checks

```bash
# Format code
black hst_*.py tests/

# Lint
flake8 .

# Type check
mypy hst_*.py

# All checks
black --check . && flake8 . && mypy .
```

### Building Documentation

```bash
# Generate docs (if sphinx is available)
cd docs
make html
```

## Community

### Getting Help

- **Questions:** Open a Discussion in GitHub
- **Issues:** Use GitHub Issues
- **Chat:** Check project's communication channels

### Where to Find Us

- [GitHub Repository](https://github.com/ilicilicc/HST)
- [Issue Tracker](https://github.com/ilicilicc/HST/issues)
- [Discussions](https://github.com/ilicilicc/HST/discussions)

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- GitHub contributors page

## Additional Resources

- [USAGE.md](USAGE.md) - How to use HST
- [API.md](API.md) - API documentation
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community guidelines

## Questions?

If you have questions about contributing:

1. Check existing documentation
2. Search closed issues for similar questions
3. Open a new discussion
4. Contact the maintainers

Thank you for contributing to HST!
