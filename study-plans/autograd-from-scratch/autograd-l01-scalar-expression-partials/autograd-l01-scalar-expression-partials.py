import numpy as np

def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """
    def f(a, b, c):
        return a*b + c
    d = f(a, b, c)
    da = (f(a + h, b, c) - f(a, b, c)) / h
    db = (f(a, b + h, c) - f(a, b, c)) / h
    dc = (f(a, b, c + h) - f(a, b, c)) / h

    return d, da, db, dc