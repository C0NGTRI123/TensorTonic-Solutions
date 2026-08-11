import numpy as np

def gradient_check_product_chain(a, b, c, f, h):
    """
    Returns: the loss, analytic gradients, numerical gradients, and maximum absolute disagreement
    """
    def L(a, b, c, f):
        e = a * b + c
        return e * f

    loss = L(a, b, c, f)

    analytic_gradients = [
        b * f,
        a * f,
        f,
        a * b + c
    ]

    numerical_gradients = [
        (L(a + h, b, c, f) - L(a, b, c, f)) / h,
        (L(a, b + h, c, f) - L(a, b, c, f)) / h,
        (L(a, b, c + h, f) - L(a, b, c, f)) / h,
        (L(a, b, c, f + h) - L(a, b, c, f)) / h
    ]

    absolute_disagreement = max(
        abs(x - y)
        for x, y in zip (analytic_gradients, numerical_gradients)
    )

    return loss, analytic_gradients, numerical_gradients, absolute_disagreement
