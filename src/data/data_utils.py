import numpy as np
from scipy.integrate import simps


def rbf_kernel(sensor_points: np.array, length_scale: int) -> np.array:
    """
    Radial Basis Function (RBF) kernel (Gaussian kernel).

    :param sensor_points: Numpy array with the x-values at which the RBF should be evaluated
    :param length_scale:
    """
    sensor_points = sensor_points[:, np.newaxis]
    sqdist = (
        np.sum(sensor_points**2, 1).reshape(-1, 1)
        + np.sum(sensor_points**2, 1)
        - 2 * np.dot(sensor_points, sensor_points.T)
    )
    return np.exp(-0.5 * (1 / length_scale**2) * sqdist)


def numerical_integration(
    y_values: np.array, x_values: np.array, method: str = "trapezoidal"
) -> np.array:
    """
    Compute the cumulative numerical integration of y_values with respect to x_values.

    :param y_values: Function values at each x (numpy array).
    :param x_values: Points at which the function is evaluated (numpy array).
    :param method: Method of numerical integration ('trapezoidal', 'cumsum', or 'simpson').
    :return: Cumulative integral (antiderivative) at each point in x_values.
    """
    antiderivative = np.zeros_like(y_values)

    if method == "trapezoidal":
        for i in range(1, len(x_values)):
            antiderivative[i] = antiderivative[i - 1] + np.trapz(
                y_values[i - 1 : i + 1], x_values[i - 1 : i + 1]
            )
    elif method == "cumsum":
        dx = x_values[1] - x_values[0]
        antiderivative = np.cumsum(y_values) * dx
    elif method == "simpson":
        if len(x_values) % 2 == 0:
            raise ValueError("Simpson's rule requires an odd number of points.")
        for i in range(1, len(x_values), 2):
            antiderivative[i] = simps(y_values[: i + 1], x_values[: i + 1])
            if i + 1 < len(x_values):
                antiderivative[i + 1] = antiderivative[i]
    else:
        raise ValueError(
            "Invalid integration method. Choose 'trapezoidal', 'cumsum', or 'simpson'."
        )

    return antiderivative


def train_test_split(data: np.array, train_fraction: float = 0.8):
    """
    Split the data into training and testing sets.

    :param data: The data to split.
    :param train_fraction: The fraction of data to use for training.
    :return: The training and testing sets.
    """
    n_train = int(data.shape[0] * train_fraction)
    train_data = data[:n_train, :, :]
    test_data = data[n_train:, :, :]
    return train_data, test_data
