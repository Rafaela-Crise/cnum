import numpy as np

def dp(f, x, h):
    """Diferença progressiva de primeira ordem."""
    return (f(x + h) - f(x)) / h


def dr(f, x, h):
    """Diferença regressiva de primeira ordem."""
    return (f(x) - f(x - h)) / h


def dc(f, x, h):
    """Diferença central de segunda ordem."""
    return (f(x + h) - f(x - h)) / (2 * h)
