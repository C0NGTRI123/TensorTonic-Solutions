import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    coefficients = np.asarray(coefficients)

    powers = np.arange(len(coefficients))
    fx = np.sum(coefficients * x ** powers)

    fx_h = np.sum(coefficients * (x + h) ** powers)

    derivative = (fx_h - fx) / h
    return fx, fx_h, derivative
