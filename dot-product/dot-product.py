import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y using vectorization.
    Must return a float.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("Same shape")

    result = np.sum(x * y)

    return float(result)