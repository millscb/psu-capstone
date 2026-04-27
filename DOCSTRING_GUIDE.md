# Docstring Guidelines

This project uses **Google-style docstrings** with **type hints** for all Python code.

## Standards

1. **Style**: Google-style docstrings ([see examples](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html))
2. **Type Hints**: Use Python type hints in function signatures; do NOT duplicate types in docstrings
3. **Documentation Tool**: pdoc for generating HTML documentation
4. **Enforcement**: Automated checks via pre-commit and GitHub Actions

## Format

### Module Docstrings

```python
"""Brief module description.

More detailed explanation of what the module provides.
Can span multiple paragraphs.

Attributes:
    MODULE_VAR: Description of module-level variable.
"""
```

### Function Docstrings

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief one-line summary.

    Longer description if needed. Explain what the function does,
    not how it does it (that's what code is for).

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        Description of return value.

    Raises:
        ValueError: When and why this exception is raised.
        TypeError: When and why this exception is raised.

    Examples:
        >>> function_name("hello", 42)
        True
    """
```

### Class Docstrings

```python
class MyClass:
    """Brief class description.

    Longer description explaining the class purpose and behavior.

    Args:
        param1: Description of constructor parameter.

    Attributes:
        CLASS_CONSTANT: Description of class-level constant.
    """

    CLASS_CONSTANT = 42  # Only document class-level attributes

    def __init__(self, param1: str):
        # Instance attributes set here (like self._param1)
        # should NOT be in the Attributes: section
        self._param1 = param1
```

**Important:** The `Attributes:` section should ONLY document class-level attributes (constants, class variables). Do NOT document instance attributes (set in `__init__`) there. Use the `Args:` section to document constructor parameters instead.

## Key Rules

✅ **DO:**
- Write docstrings for all public modules, classes, functions, and methods
- Use one-line summaries ending with a period
- Put types in function signatures, not docstrings (we have type hints!)
- Use `Args:`, `Returns:`, `Raises:`, `Examples:` sections as needed
- Keep docstrings concise but informative
- Add examples for complex functions

❌ **DON'T:**
- Duplicate type information (it's in the hints!)
- Write docstrings that just repeat the function name
- Use other styles (NumPy, Sphinx, etc.) - stick to Google style
- Document `self` parameter (it's implied)
- Add docstrings to private methods (unless complex)

## Automated Checks

### Pre-commit Hooks

```bash
make lint-docs        # Check docstring style and coverage
make lint-docs-strict # Strict validation with pydoclint
```

### Running Locally

```bash
# Check docstring style (Google convention)
uv run pydocstyle src/psu_capstone --convention=google

# Check docstring coverage (80% threshold)
uv run interrogate -vv src/psu_capstone --fail-under=80

# Strict validation (args match signatures; isolated deps avoid docstring_parser conflicts)
uv run --with pydoclint --with docstring-parser-fork pydoclint --style=google src/
```

### GitHub Actions

Documentation checks run automatically on all PRs and will fail if:
- Docstring coverage falls below 80%
- Docstrings don't follow Google style conventions
- Function arguments don't match docstring Args sections

## Configuration

All docstring tools are configured in `pyproject.toml`:
- `[tool.pydocstyle]` - Google style enforcement
- `[tool.interrogate]` - Coverage thresholds
- `[tool.pydoclint]` - Argument validation

## Generating Documentation

To generate and view HTML documentation locally:

```bash
# Generate docs for the entire package
uv run pdoc src/psu_capstone -o ./docs

# Start a local web server to view docs
python3 -m http.server 8000 --directory docs
# Then open http://localhost:8000 in your browser
```

### Updating Documentation

After making docstring changes, regenerate and view:

```bash
# 1. Regenerate the HTML docs
uv run pdoc src/psu_capstone -o ./docs

# 2. If the server is already running, stop it first:
#    - Find the server process: lsof -i :8000
#    - Kill it: kill <PID>
#    Or use Ctrl+C if running in foreground

# 3. Restart the server
python3 -m http.server 8000 --directory docs

# 4. Refresh your browser (hard refresh: Cmd+Shift+R on Mac, Ctrl+Shift+R on Linux/Windows)
```

**Tip:** Run the server in the background or a separate terminal window so you can continue working:

```bash
# Background server (add & at the end)
python3 -m http.server 8000 --directory docs &

# To stop background server later
pkill -f "http.server 8000"
```

## Examples from This Project

See these files for good examples:
- `src/utils.py` - Utility functions with complete docs
- `src/psu_capstone/encoder_layer/base_encoder.py` - Abstract base class
- `src/psu_capstone/encoder_layer/category_encoder.py` - Concrete implementation
- `src/psu_capstone/log.py` - Module-level documentation

## References

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Napoleon Extension Examples](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- [pdoc Documentation](https://pdoc.dev/)
