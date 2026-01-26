"""
Array utility functions for safe math operations.
Used for batch NumPy operations in feature building.
"""
import numpy as np


def safe_clip(x: np.ndarray, lo: float = -1e3, hi: float = 1e3) -> np.ndarray:
    """
    Clip array values and set non-finite values to 0.
    
    Args:
        x: Input array (float64)
        lo: Lower bound (default -1e3)
        hi: Upper bound (default 1e3)
    
    Returns:
        Clipped array with non-finite values set to 0
    """
    result = np.clip(x, lo, hi)
    mask = ~np.isfinite(result)
    result[mask] = 0.0
    return result


def safe_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two arrays, setting non-finite results to 0.
    
    Args:
        a: First array (float64)
        b: Second array (float64)
    
    Returns:
        Product array with non-finite values set to 0
    """
    result = a * b
    mask = ~np.isfinite(result)
    result[mask] = 0.0
    return result


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Divide two arrays with epsilon protection, setting non-finite results to 0.
    
    Args:
        a: Numerator array (float64)
        b: Denominator array (float64)
        eps: Epsilon for division protection (default 1e-12)
    
    Returns:
        Quotient array with non-finite values set to 0
    """
    result = a / (b + eps)
    mask = ~np.isfinite(result)
    result[mask] = 0.0
    return result

