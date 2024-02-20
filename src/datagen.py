from __future__ import annotations

import numpy as np
from scipy.integrate import simps

from plotting import plot_functions_and_antiderivatives, plot_functions_only


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


# def trapezoidal_rule(y_values: np.array, x_values: np.array) -> float:
#     """
#     Numerically integrate y_values with respect to x_values using the trapezoidal rule.

#     :param y_values: Function values at each x (numpy array).
#     :param x_values: Points at which the function is evaluated (numpy array).
#     :return: Approximate integral (antiderivative) of the function.
#     """
#     integral = np.trapz(y_values, x_values)
#     return integral


def generate_polynomial_data(
    num_samples: int,
    sensor_points: np.array,
    scale: float = 1.0,
    method: str = "trapezoidal",
) -> list[tuple[np.array, np.array]]:
    """
    Generate polynomial functions with randomly sampled coefficients.

    :param num_samples: Number of samples to generate.
    :param sensor_points: Points at which the functions are evaluated.
    :param scale: Range of the random coefficients.
    :param method: Method of numerical integration ('trapezoidal', 'cumsum', or 'simpson').
    :return: List of tuples of numpy arrays (polynomial, antiderivative).
    """
    data = []
    for _ in range(num_samples):
        # Random coefficients for a cubic polynomial
        coefficients = np.random.uniform(-scale, scale, 6)
        polynomial = np.poly1d(coefficients)
        poly = polynomial(sensor_points)

        # Compute antiderivative
        antiderivative = numerical_integration(poly, sensor_points, method=method)

        data.append((poly, antiderivative))

    return data


# Modified generate_polynomial_data function
def generate_polynomial_data_coeff(
    num_samples, sensor_points, scale=1, method="trapezoidal", order=5
):
    """Generate a polynomial dataset with random coefficients as in generate_polynomial_data,
    but return the coefficients of the input polynomial.
    """
    data = []
    for _ in range(num_samples):
        # Random coefficients for a cubic polynomial
        coefficients = np.random.uniform(-scale, scale, order + 1)
        polynomial = np.poly1d(coefficients)
        poly = polynomial(sensor_points)

        # Compute antiderivative
        antiderivative = numerical_integration(poly, sensor_points, method=method)

        data.append((poly, antiderivative, coefficients))

    return data


def generate_sine_data(
    num_samples: int, sensor_points: np.array, int_method: str = "trapezoidal"
) -> list[tuple[np.array, np.array]]:
    """
    Generate a superposition of two sine waves with random frequency and amplitude.

    :param num_samples: Number of samples to generate.
    :param sensor_points: Points at which the functions are evaluated.
    :param int_method: Method of numerical integration ('trapezoidal', 'cumsum', or 'simpson').
    :return: List of tuples of numpy arrays (sine wave, antiderivative) for each sample.
    """
    data = []
    for _ in range(num_samples):
        # Generate two sine waves with random frequency and phase
        frequency1 = np.random.uniform(0.1, 5)
        phase1 = np.random.uniform(0, 2 * np.pi)
        sine_wave1 = np.sin(2 * np.pi * frequency1 * sensor_points + phase1)

        frequency2 = np.random.uniform(1, 5)
        phase2 = np.random.uniform(0, 2 * np.pi)
        sine_wave2 = np.sin(2 * np.pi * frequency2 * sensor_points + phase2)

        # Add the two sine waves together
        sine_wave = 0.5 * (sine_wave1 + sine_wave2)

        # Compute its antiderivative
        antiderivative = numerical_integration(
            sine_wave, sensor_points, method=int_method
        )

        data.append((sine_wave, antiderivative))

    return data


def generate_GRF_data(
    num_samples: int,
    sensor_points: np.array,
    length_scale: float,
    int_method: str = "trapezoidal",
) -> list[tuple[np.array, np.array]]:
    """
    Generate samples from a Gaussian random field (GRF) with an RBF kernel.
    Note: Instead of discretizing a continuous GRF, we sample from a multivariate Gaussian

    :param num_samples: Number of samples to generate.
    :param sensor_points: Points at which the functions are evaluated.
    :param length_scale: Length scale of the RBF kernel.
    :param int_method: Method of numerical integration ('trapezoidal', 'cumsum', or 'simpson').
    :return: List of tuples of numpy arrays (GRF sample, antiderivative).
    """
    data = []
    # Covariance matrix based on the RBF kernel
    covariance_matrix = rbf_kernel(sensor_points, length_scale)

    for _ in range(num_samples):
        # Sample from a multivariate Gaussian
        field_sample = np.random.multivariate_normal(
            np.zeros(len(sensor_points)), covariance_matrix
        )

        # Calculate the antiderivative (cumulative sum as a simple approximation)
        antiderivative = numerical_integration(
            field_sample, sensor_points, method=int_method
        )

        data.append((field_sample, antiderivative))

    return data


def generate_decaying_polynomials(
    num_samples: int = 100,
    order: int = 5,
    sensor_locations: np.array = [],
    N_steps: int = 101,
    T_final: float = 1.0,
    decay_rate: float = 1.0,
    scale: float = 1.0,
) -> list[tuple[np.array, np.array, np.array]]:
    """
    Generate polynomials with exponential decay of the coefficients.
    Note: All coefficients are decayed at the same rate.

    :param num_samples: Number of samples to generate.
    :param order: Order of the polynomial.
    :param sensor_locations: Points at which the functions are evaluated.
    :param N_steps: Number of time steps.
    :param T_final: Final time.
    :param decay_rate: Rate of decay for the coefficients.
    :param scale: Range of the random coefficients.
    :return: List of tuples of numpy arrays (polynomial, coefficients, timesteps).
    """
    # Time steps
    timesteps = np.linspace(0, T_final, N_steps)

    # Sensor locations
    if len(sensor_locations) == 0:
        sensor_locations = np.linspace(0, 1, 101)

    # Initialize arrays
    polynomials = np.zeros((num_samples, len(sensor_locations), N_steps))
    coefficients = np.zeros((num_samples, order + 1, N_steps))

    # Generate polynomials and coefficients
    for i in range(num_samples):
        # Random initial coefficients from a uniform distribution
        initial_coeffs = np.random.uniform(-scale, scale, order + 1)

        for j, t in enumerate(timesteps):
            # Exponential decay of coefficients
            decayed_coeffs = initial_coeffs * np.exp(-decay_rate * t)
            coefficients[i, :, j] = decayed_coeffs

            # Evaluate polynomial at sensor locations
            polynomial = np.polyval(
                decayed_coeffs[::-1], sensor_locations
            )  # Reverse coeffs for np.polyval
            polynomials[i, :, j] = polynomial

    return polynomials, coefficients, timesteps


def generate_decaying_sines(
    num_samples: int = 100,
    N_sines: int = 3,
    sensor_locations: np.array = [],
    N_steps: int = 101,
    T_final: float = 1.0,
    decay_rate: float = 1.0,
    scale: float = 1.0,
) -> list[tuple[np.array, np.array, np.array, np.array]]:
    """
    Generate a superposition of sines with random frequency and amplitude, and exponential decay of the amplitudes.

    :param num_samples: Number of samples to generate.
    :param N_sines: Number of sines to sum.
    :param sensor_locations: Points at which the functions are evaluated.
    :param N_steps: Number of time steps.
    :param T_final: Final time.
    :param decay_rate: Rate of decay for the amplitudes.
    :param scale: Range of the random amplitudes.
    :return: List of tuples of numpy arrays (sine wave, amplitudes, frequencies, timesteps).
    """
    # Time steps
    timesteps = np.linspace(0, T_final, N_steps)

    # Sensor locations
    if len(sensor_locations) == 0:
        sensor_locations = np.linspace(0, 1, 101)

    # Initialize arrays
    sines = np.zeros((num_samples, len(sensor_locations), N_steps))
    amplitudes = np.zeros((num_samples, N_sines, N_steps))
    frequencies = np.zeros((num_samples, N_sines))

    # Generate sines and amplitudes
    for i in range(num_samples):
        # Random initial amplitudes and frequencies
        initial_amplitudes = np.random.uniform(-scale, scale, N_sines)
        frequencies[i, :] = np.random.uniform(1, 5, N_sines)

        for j, t in enumerate(timesteps):
            # Exponential decay of amplitudes
            decayed_amplitudes = initial_amplitudes * np.exp(-decay_rate * t)
            amplitudes[i, :, j] = decayed_amplitudes

            # Evaluate sum of sines at sensor locations
            sine_sum = np.sum(
                [
                    a * np.sin(2 * np.pi * f * sensor_locations)
                    for a, f in zip(decayed_amplitudes, frequencies[i, :])
                ],
                axis=0,
            )
            sines[i, :, j] = sine_sum

    return sines, amplitudes, frequencies, timesteps


def generate_random_decaying_sines(
    num_samples: int = 100,
    N_sines: int = 3,
    sensor_locations: np.array = [],
    N_steps: int = 101,
    T_final: float = 1.0,
    scale: float = 1.0,
) -> list[tuple[np.array, np.array, np.array, np.array]]:
    """
    Generate a superposition of sines with random frequency and amplitude, and exponential decay of the amplitudes.
    Note: Here the decay_rate is different for each sample (randomly sampled from a uniform distribution).

    :param num_samples: Number of samples to generate.
    :param N_sines: Number of sines to sum.
    :param sensor_locations: Points at which the functions are evaluated.
    :param N_steps: Number of time steps.
    :param T_final: Final time.
    :param scale: Range of the random amplitudes.
    :return: List of tuples of numpy arrays (sine wave, amplitudes, frequencies, timesteps).
    """
    # Time steps
    timesteps = np.linspace(0, T_final, N_steps)

    # Sensor locations
    if len(sensor_locations) == 0:
        sensor_locations = np.linspace(0, 1, 101)

    # Initialize arrays
    branch_inputs = np.zeros((num_samples, len(sensor_locations) + 1, N_steps))
    amplitudes = np.zeros((num_samples, N_sines, N_steps))
    frequencies = np.zeros((num_samples, N_sines))
    decay_rates = np.random.uniform(0.1, 10, num_samples)

    # Generate sines and amplitudes
    for i in range(num_samples):
        # Random initial amplitudes and frequencies
        initial_amplitudes = np.random.uniform(-scale, scale, N_sines)
        frequencies[i, :] = np.random.uniform(1, 5, N_sines)

        for j, t in enumerate(timesteps):
            # Exponential decay of amplitudes
            decayed_amplitudes = initial_amplitudes * np.exp(-decay_rates[i] * t)
            amplitudes[i, :, j] = decayed_amplitudes

            # Evaluate sum of sines at sensor locations
            sine_sum = np.sum(
                [
                    a * np.sin(2 * np.pi * f * sensor_locations)
                    for a, f in zip(decayed_amplitudes, frequencies[i, :])
                ],
                axis=0,
            )
            branch_inputs[i, :-1, j] = sine_sum
            branch_inputs[i, -1, j] = decay_rates[i]

    return branch_inputs, amplitudes, frequencies, timesteps


def generate_oscillating_sines(
    num_samples: int = 100,
    N_sines: int = 3,
    sensor_locations: np.array = [],
    N_steps: int = 101,
    T_final: float = 1.0,
    rate: float = 1.0,
    scale: float = 1.0,
) -> list[tuple[np.array, np.array, np.array, np.array]]:
    """
    Generate a superposition of sines with random frequency and amplitude, where the amplitude oscillates over time.

    :param num_samples: Number of samples to generate.
    :param N_sines: Number of sines to sum.
    :param sensor_locations: Points at which the functions are evaluated.
    :param N_steps: Number of time steps.
    :param T_final: Final time.
    :param rate: Rate of oscillation for the amplitudes.
    :param scale: Range of the random amplitudes.
    :return: List of tuples of numpy arrays (sine wave, amplitudes, frequencies, timesteps).
    """
    # Time steps
    timesteps = np.linspace(0, T_final, N_steps)

    # Sensor locations
    if len(sensor_locations) == 0:
        sensor_locations = np.linspace(0, 1, 101)

    # Initialize arrays
    sines = np.zeros((num_samples, len(sensor_locations), N_steps))
    amplitudes = np.zeros((num_samples, N_sines, N_steps))
    frequencies = np.zeros((num_samples, N_sines))

    # Generate sines and amplitudes
    for i in range(num_samples):
        # Random initial amplitudes and frequencies
        initial_amplitudes = np.random.uniform(-scale, scale, N_sines)
        frequencies[i, :] = np.random.uniform(1, 5, N_sines)

        for j, t in enumerate(timesteps):
            # Exponential decay of amplitudes
            decayed_amplitudes = initial_amplitudes * np.cos(2 * np.pi * rate * t)
            amplitudes[i, :, j] = decayed_amplitudes

            # Evaluate sum of sines at sensor locations
            sine_sum = np.sum(
                [
                    a * np.sin(2 * np.pi * f * sensor_locations)
                    for a, f in zip(decayed_amplitudes, frequencies[i, :])
                ],
                axis=0,
            )
            sines[i, :, j] = sine_sum

    return sines, amplitudes, frequencies, timesteps


def spectrum(
    p: np.array,
    a: float = 0.0,
    b: float = -4.0,
    c: float = 2.0,
    A: float = 1.0,
    p0: float = 1.0,
    eps: float = 1e-6,
) -> np.array:
    """
    Generate a power-law spectrum with a given power-law exponent and amplitude.

    :param p: Momentum values at which the spectrum is evaluated.
    :param a: Power-law exponent.
    :param b: Power-law exponent.
    :param c: Power-law exponent.
    :param A: Amplitude.
    :param p0: Reference momentum value.
    :param eps: Small value to avoid division by zero.
    :return: Numpy array with the spectrum values at the given momenta.
    """
    p = np.maximum(p, eps)
    return A * ((p / p0) ** (-a / c) + (p / p0) ** (-b / c)) ** (-c)


def generate_evolving_spectra(
    num_samples: int = 100,
    sensor_locations: np.array = [],
    N_steps: int = 101,
    T_final: float = 1.0,
    p0: float = 1.0,
) -> list[tuple[np.array, np.array, np.array]]:
    """
    Generate spectra evolving over time, with random coefficients and exponential decay.

    :param num_samples: Number of samples to generate.
    :param sensor_locations: Points at which the functions are evaluated.
    :param N_steps: Number of time steps.
    :param T_final: Final time.
    :param p0: Reference momentum value.
    :return: List of tuples of numpy arrays (spectrum, coefficients, timesteps).
    """
    # Time steps
    timesteps = np.linspace(0, T_final, N_steps)

    # Sensor locations
    if len(sensor_locations) == 0:
        sensor_locations = np.linspace(0, 1, 101)

    # Initialize arrays
    branch_inputs = np.zeros((num_samples, len(sensor_locations) + 1, N_steps))
    coefficients = np.zeros((num_samples, 4, N_steps))  # a, b, c, A
    decay_rates = np.random.uniform(5, 10, num_samples)

    # Sample parameters for the spectra
    a = np.random.uniform(1, 2, num_samples)
    coefficients[:, 0, 0] = a
    b = np.random.uniform(-1, -6, num_samples)
    coefficients[:, 1, 0] = b
    c = np.random.uniform(1, 3, num_samples)
    coefficients[:, 2, 0] = c
    A = np.ones(num_samples)
    coefficients[:, 3, 0] = A

    # Generate sines and amplitudes
    for i in range(num_samples):

        for j, t in enumerate(timesteps):
            # Exponential development of the coefficients
            coeffs = np.asarray((a[i], b[i], c[i], A[i])) * (
                0.3 + 1.4 * np.exp(-decay_rates[i] * t)
            )
            coefficients[i, :, j] = coeffs

            # Evaluate the spectrum at sensor locations
            spectrum_values = spectrum(
                sensor_locations, coeffs[0], coeffs[1], coeffs[2], coeffs[3], p0
            )
            branch_inputs[i, :-1, j] = spectrum_values
            branch_inputs[i, -1, j] = decay_rates[i]

    return branch_inputs, coefficients, timesteps


if __name__ == "__main__":
    num_samples = 1000
    sensor_points = np.linspace(0, 1, 100)
    length_scale = 0.1
    scale = 3
    method = "trapezoidal"  # choose from trapezoidal, cumsum, or simpson

    data_poly = generate_polynomial_data(num_samples, sensor_points, scale, method)
    data_GRF = generate_GRF_data(num_samples, sensor_points, length_scale, method)
    data_sine = generate_sine_data(num_samples, sensor_points, method)
    # data = generate_polynomial_data(num_samples, sensor_points)

    # Inspect how the data covers the domain
    plot_functions_only(data_poly, sensor_points, num_samples_to_plot=100)
    plot_functions_only(data_GRF, sensor_points, num_samples_to_plot=100)
    plot_functions_only(data_sine, sensor_points, num_samples_to_plot=100)

    # Plot some examples from the data
    plot_functions_and_antiderivatives(
        data_poly, data_GRF, data_sine, sensor_points, num_samples_to_plot=3
    )
