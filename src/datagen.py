import numpy as np
from scipy.integrate import simps

from plotting import plot_functions_and_antiderivatives, plot_functions_only


def rbf_kernel(sensor_points, length_scale):
    """Radial Basis Function (RBF) kernel (Gaussian kernel)."""
    sensor_points = sensor_points[:, np.newaxis]
    sqdist = (
        np.sum(sensor_points**2, 1).reshape(-1, 1)
        + np.sum(sensor_points**2, 1)
        - 2 * np.dot(sensor_points, sensor_points.T)
    )
    return np.exp(-0.5 * (1 / length_scale**2) * sqdist)


def numerical_integration(y_values, x_values, method="trapezoidal"):
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


def trapezoidal_rule(y_values, x_values):
    """
    Numerically integrate y_values with respect to x_values using the trapezoidal rule.

    :param y_values: Function values at each x (numpy array).
    :param x_values: Points at which the function is evaluated (numpy array).
    :return: Approximate integral (antiderivative) of the function.
    """
    integral = np.trapz(y_values, x_values)
    return integral


def generate_polynomial_data(num_samples, sensor_points, scale=1, method="trapezoidal"):
    data = []
    for _ in range(num_samples):
        # Random coefficients for a cubic polynomial
        coefficients = np.random.uniform(-scale, scale, 6)
        polynomial = np.poly1d(coefficients)
        poly = polynomial(sensor_points)

        # Compute antiderivative
        # antiderivative = polynomial.integ()
        antiderivative = numerical_integration(poly, sensor_points, method=method)

        data.append((poly, antiderivative))

    return data


def generate_sine_data(num_samples, sensor_points, int_method="trapezoidal"):
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
    num_samples, sensor_points, length_scale, int_method="trapezoidal"
):
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
    num_samples=100,
    order=5,
    sensor_locations=[],
    N_steps=101,
    T_final=1,
    decay_rate=1,
    scale=1,
):
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
    num_samples=100,
    N_sines=3,
    sensor_locations=[],
    N_steps=101,
    T_final=1,
    decay_rate=1,
    scale=1,
):
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
    num_samples=100,
    N_sines=3,
    sensor_locations=[],
    N_steps=101,
    T_final=1,
    scale=1,
):
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
    num_samples=100,
    N_sines=3,
    sensor_locations=[],
    N_steps=101,
    T_final=1,
    rate=1,
    scale=1,
):
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


def spectrum(p, a, b, c, A, p0, eps=1e-6):
    p = np.maximum(p, eps)
    return A * ((p / p0) ** (-a / c) + (p / p0) ** (-b / c)) ** (-c)


def generate_evolving_spectra(
    num_samples=100, sensor_locations=[], N_steps=101, T_final=1, p0=1
):
    # Time steps
    timesteps = np.linspace(0, T_final, N_steps)

    # Sensor locations
    if len(sensor_locations) == 0:
        sensor_locations = np.linspace(0, 1, 101)

    # Initialize arrays
    branch_inputs = np.zeros((num_samples, len(sensor_locations) + 1, N_steps))
    coefficients = np.zeros((num_samples, 4, N_steps))  # a, b, c, A
    decay_rates = np.random.uniform(0.1, 10, num_samples)

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
                0.7 + 0.6 * np.exp(-decay_rates[i] * t)
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
