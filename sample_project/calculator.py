import math

"""Simple calculator utility module with intentional bugs for testing."""

def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    # Fixed: Raise ValueError as per project specification for division by zero.
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b