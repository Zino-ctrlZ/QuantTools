# Docstring Standards Skill

Use this skill when adding or revising Python docstrings in QuantTools.

## Required Rules
- Follow PEP 257.
- Use Google-style docstrings.
- Add a module docstring as the first statement in every Python module.
- Add docstrings for every class.
- Add docstrings for every function and method, including `__init__` and private helpers.
- Keep summary lines concise and imperative where possible.
- Keep docs aligned with implementation; do not leave stale descriptions.

## Required Sections
Use sections only when applicable:
- Args:
- Returns:
- Raises:
- Examples:

## Module Docstring Template
```python
"""One-line module summary.

Short 2-4 line overview of scope and purpose.

Core Classes:
        ClassA: Purpose.

Core Functions:
        fn_a: Purpose.

Usage:
        >>> result = fn_a(...)
"""
```

## Function Docstring Template
```python
def func(a: int, b: int) -> int:
    """Return the combined score for two inputs.

    Args:
        a: First input value.
        b: Second input value.

    Returns:
        Combined score.

    Raises:
        ValueError: If either input is negative.

    Examples:
        >>> func(2, 3)
        5
    """
```

## Class and __init__ Template
```python
class Example:
    """Represent an example service.

    Examples:
        >>> ex = Example(name="demo")
        >>> ex.name
        'demo'
    """

    def __init__(self, name: str) -> None:
        """Initialize the example service.

        Args:
            name: Service name.
        """
```

## Quality Checklist
- Summary line ends with a period.
- Line wrapping is readable and consistent.
- Sections are present only when needed.
- Types in docstrings do not contradict type hints.
- Example snippets are realistic and runnable where practical.
