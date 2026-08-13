import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    # Write code here
    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)
    v_new = momentum * v + lr * grad
    w_new = w - v_new
    return np.round(w_new,6).tolist(), np.round(v_new, 6).tolist()