import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    A = np.asarray(A)
    num_rows, num_cols = A.shape
    n = min(num_rows, num_cols)
    trace_val = 0
    for i in range(0, n):
        trace_val += A[i, i]
    return trace_val
        