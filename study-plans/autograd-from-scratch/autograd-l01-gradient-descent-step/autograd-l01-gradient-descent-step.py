import numpy as np

def gradient_descent_step(values, gradients, learning_rate):
    """
    Returns: updated values and the predicted first-order objective change
    """
    values = np.asarray(values)
    gradients = np.asarray(gradients)

    updated_values = values - learning_rate * gradients
    objective_change = -1 * (learning_rate * np.sum(gradients ** 2))

    return updated_values.tolist(), objective_change
