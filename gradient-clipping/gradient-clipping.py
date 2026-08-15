import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.asarray(g, dtype=float)
    g_norm = np.linalg.norm(g)
    if max_norm <= 0:
        return g
    if g_norm <= max_norm:
        g_clipped = g
    else:
        g_clipped = g * max_norm / g_norm
    return g_clipped