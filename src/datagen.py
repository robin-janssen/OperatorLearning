import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.integrate import simps

def rbf_kernel(sensor_points, length_scale):
    """ Radial Basis Function (RBF) kernel (Gaussian kernel). """
    sensor_points = sensor_points[:, np.newaxis]
    sqdist = np.sum(sensor_points**2, 1).reshape(-1, 1) + np.sum(sensor_points**2, 1) - 2 * np.dot(sensor_points, sensor_points.T)
    return np.exp(-0.5 * (1/length_scale**2) * sqdist)

def numerical_integration(y_values, x_values, method='trapezoidal'):
    """
    Compute the cumulative numerical integration of y_values with respect to x_values.

    :param y_values: Function values at each x (numpy array).
    :param x_values: Points at which the function is evaluated (numpy array).
    :param method: Method of numerical integration ('trapezoidal', 'cumsum', or 'simpson').
    :return: Cumulative integral (antiderivative) at each point in x_values.
    """
    antiderivative = np.zeros_like(y_values)

    if method == 'trapezoidal':
        for i in range(1, len(x_values)):
            antiderivative[i] = antiderivative[i-1] + np.trapz(y_values[i-1:i+1], x_values[i-1:i+1])
    elif method == 'cumsum':
        dx = x_values[1] - x_values[0]
        antiderivative = np.cumsum(y_values) * dx
    elif method == 'simpson':
        if len(x_values) % 2 == 0:
            raise ValueError("Simpson's rule requires an odd number of points.")
        for i in range(1, len(x_values), 2):
            antiderivative[i] = simps(y_values[:i+1], x_values[:i+1])
            if i+1 < len(x_values):
                antiderivative[i+1] = antiderivative[i]
    else:
        raise ValueError("Invalid integration method. Choose 'trapezoidal', 'cumsum', or 'simpson'.")

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

def generate_polynomial_data(num_samples, sensor_points, scale=1, method='trapezoidal'):
    data = []
    for _ in range(num_samples):
        # Random coefficients for a cubic polynomial
        coefficients = np.random.uniform(-scale, scale, 6)
        polynomial = np.poly1d(coefficients)
        poly = polynomial(sensor_points)

        # Compute antiderivative
        #antiderivative = polynomial.integ()
        antiderivative = numerical_integration(poly, sensor_points, method=method)

        data.append((poly, antiderivative))

    return data

def generate_sine_data(num_samples, sensor_points, int_method='trapezoidal'):
    data = []
    for _ in range(num_samples):
        # Generate two sine waves with random frequency and phase
        frequency1 = np.random.uniform(0.1, 5)
        phase1 = np.random.uniform(0, 2*np.pi)
        sine_wave1 = np.sin(2 * np.pi * frequency1 * sensor_points + phase1)

        frequency2 = np.random.uniform(1, 5)
        phase2 = np.random.uniform(0, 2*np.pi)
        sine_wave2 = np.sin(2 * np.pi * frequency2 * sensor_points + phase2)

        # Add the two sine waves together
        sine_wave = 0.5 * (sine_wave1 + sine_wave2)

        # Compute its antiderivative
        antiderivative = numerical_integration(sine_wave, sensor_points, method=int_method)

        data.append((sine_wave, antiderivative))

    return data

def generate_GRF_data(num_samples, sensor_points, length_scale, int_method='trapezoidal'):
    data = []
    # Covariance matrix based on the RBF kernel
    covariance_matrix = rbf_kernel(sensor_points, length_scale)

    for _ in range(num_samples):
        # Sample from a multivariate Gaussian
        field_sample = np.random.multivariate_normal(np.zeros(len(sensor_points)), covariance_matrix)

        # Calculate the antiderivative (cumulative sum as a simple approximation)
        antiderivative = numerical_integration(field_sample, sensor_points, method=int_method)

        data.append((field_sample, antiderivative))

    return data

def plot_functions_and_antiderivatives(data_poly, data_GRF, data_sine, sensor_points, num_samples_to_plot=3):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Some samples of the data')
    
    colors = cycle(plt.cm.tab10.colors)  # Color cycle from a matplotlib colormap

    for i in range(num_samples_to_plot):
        color = next(colors)
        axs[0].plot(sensor_points, data_poly[i][0], label=f'u_{i+1}', color=color, linestyle='-')
        axs[0].plot(sensor_points, data_poly[i][1], color=color, linestyle='--')
        axs[1].plot(sensor_points, data_GRF[i][0], color=color, linestyle='-')
        axs[1].plot(sensor_points, data_GRF[i][1], color=color, linestyle='--')
        axs[2].plot(sensor_points, data_sine[i][0], color=color, linestyle='-')
        axs[2].plot(sensor_points, data_sine[i][1], color=color, linestyle='--')

    axs[0].set_title('Polynomial data')
    axs[1].set_title('Gaussian random field data')
    axs[2].set_title('Sine data')

    for ax in axs:
        ax.set_xlabel('Domain')
        ax.set_ylabel('Function Value')

    plt.tight_layout()
    plt.show()

def plot_functions_only(data, sensor_points, num_samples_to_plot):
    '''
    Plot some samples of the data
    
    :param data: numpy array of data
    :param sensor_points: Array of sensor locations in the domain.
    :param num_samples_to_plot: Number of samples to plot.
    
    '''
    plt.figure(figsize=(6, 4))
    plt.title('Function Coverage')

    for i in range(num_samples_to_plot):
        plt.plot(sensor_points, data[i], color='blue', alpha=0.2)  # Low opacity

    plt.xlabel('Domain')
    plt.ylabel('Function Value')
    plt.tight_layout()
    plt.show()

def generate_decaying_polynomials(num_samples=100, order=5, sensor_locations=[], N_steps=101, T_final=1, decay_rate=1, scale=1):
    # Time steps
    timesteps = np.linspace(0, T_final, N_steps)

    #Sensor locations
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
            polynomial = np.polyval(decayed_coeffs[::-1], sensor_locations)  # Reverse coeffs for np.polyval
            polynomials[i, :, j] = polynomial

    return polynomials, coefficients, timesteps

def generate_decaying_sines(num_samples=100, N_sines=3, sensor_locations=[], N_steps=101, T_final=1, decay_rate=1, scale=1):
    # Time steps
    timesteps = np.linspace(0, T_final, N_steps)

    #Sensor locations
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
            sine_sum = np.sum([a * np.sin(2 * np.pi * f * sensor_locations) for a, f in zip(decayed_amplitudes, frequencies[i, :])], axis=0)
            sines[i, :, j] = sine_sum

    return sines, amplitudes, frequencies, timesteps

def generate_random_decaying_sines(num_samples=100, N_sines=3, sensor_locations=[], N_steps=101, T_final=1, decay_rate=1, scale=1):
    # Time steps
    timesteps = np.linspace(0, T_final, N_steps)

    #Sensor locations
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
            sine_sum = np.sum([a * np.sin(2 * np.pi * f * sensor_locations) for a, f in zip(decayed_amplitudes, frequencies[i, :])], axis=0)
            sines[i, :, j] = sine_sum

    return sines, amplitudes, frequencies, timesteps

def generate_oscillating_sines(num_samples=100, N_sines=3, sensor_locations=[], N_steps=101, T_final=1, rate=1, scale=1):
    # Time steps
    timesteps = np.linspace(0, T_final, N_steps)

    #Sensor locations
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
            sine_sum = np.sum([a * np.sin(2 * np.pi * f * sensor_locations) for a, f in zip(decayed_amplitudes, frequencies[i, :])], axis=0)
            sines[i, :, j] = sine_sum

    return sines, amplitudes, frequencies, timesteps

def surface_plot(sensor_locations, timesteps, functions, num_samples_to_plot, predictions=None, title='Data Visualization'):
    """
    Plot the evolution of functions over time using a 3D surface plot.

    :param sensor_locations: Array of sensor locations.
    :param timesteps: Array of timesteps.
    :param functions: 3D array of ground truth function values.
    :param num_samples_to_plot: Number of samples to plot.
    :param title: Title of the plot.
    :param predictions: 3D array of predicted function values (optional).
    """

    X, Y = np.meshgrid(sensor_locations, timesteps)
    total_cols = num_samples_to_plot

    if predictions is None:
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(title)
        for i in range(num_samples_to_plot):
            Z = functions[i].T
            plot_single_surface(fig, 1, i, 1, 3, X, Y, Z, 'Sample', i)
    else:
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle(title)
        for i in range(num_samples_to_plot):
            # Plot ground truth
            Z_gt = functions[i].T
            plot_single_surface(fig, 1, i, 3, total_cols, X, Y, Z_gt, 'Ground Truth', i)

            # Plot predictions
            Z_pred = predictions[i].T
            plot_single_surface(fig, 2, i, 3, total_cols, X, Y, Z_pred, 'Prediction', i)

            # Plot error
            Z_err = Z_gt - Z_pred
            plot_single_surface(fig, 3, i, 3, total_cols, X, Y, Z_err, 'Error', i)
    
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.show()

def plot_single_surface(fig, row, col, total_rows, total_cols, X, Y, Z, title_prefix, sample_number):
    """
    Plot a single surface in a subplot.

    :param fig: The figure object to which the subplot is added.
    :param row: The row number of the subplot.
    :param col: The column number of the subplot.
    :param total_rows: Total number of rows in the subplot grid.
    :param total_cols: Total number of columns in the subplot grid.
    :param X: X-coordinates for the meshgrid.
    :param Y: Y-coordinates for the meshgrid.
    :param Z: Z-coordinates (function values) for the plot.
    :param title_prefix: Prefix for the subplot title.
    :param sample_number: The sample number (for the title).
    """
    if total_rows == 1:
        # When there's only one row, the index is the sample number plus one
        index = sample_number + 1
    else:
        # For multiple rows, calculate the index based on row and column
        index = (row - 1) * total_cols + col + 1

    ax = fig.add_subplot(total_rows, total_cols, index, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Sensor Location')
    ax.set_ylabel('Time')
    ax.set_zlabel('Value')
    ax.set_title(f'{title_prefix} Sample {sample_number + 1}')

def heatmap_plot(sensor_locations, timesteps, functions, num_samples_to_plot, predictions=None, title='Data Visualization'):
    """
    Plot the evolution of functions over time using heat maps with a common colorbar.

    :param sensor_locations: Array of sensor locations.
    :param timesteps: Array of timesteps.
    :param functions: 3D array of ground truth function values.
    :param num_samples_to_plot: Number of samples to plot.
    :param title: Title of the plot.
    :param predictions: 3D array of predicted function values (optional).
    """
    figsize = (10, 3) if predictions is None else (12, 8)
    fig, axs = plt.subplots(3 if predictions is not None else 1, num_samples_to_plot, figsize=figsize, squeeze=False, layout="compressed")
    fig.suptitle(title)
    functions = functions[:num_samples_to_plot].transpose(0, 2, 1)
    predictions = predictions[:num_samples_to_plot].transpose(0, 2, 1) if predictions is not None else None

    # Determine common vmin and vmax for color scaling
    vmin = min(functions.min(), predictions.min() if predictions is not None else functions.min())
    vmax = max(functions.max(), predictions.max() if predictions is not None else functions.max())

    for i in range(num_samples_to_plot):
        # Plot ground truth
        im = axs[0, i].imshow(functions[i], aspect='equal', origin='lower', extent=[sensor_locations[0], sensor_locations[-1], timesteps[0], timesteps[-1]], vmin=vmin, vmax=vmax)
        axs[0, i].set_title(f'Ground Truth Sample {i+1}')
        axs[0, i].set_xlabel('Sensor Location')
        axs[0, i].set_ylabel('Time')

        if predictions is not None:
            # Plot predictions
            axs[1, i].imshow(predictions[i], aspect='equal', origin='lower', extent=[sensor_locations[0], sensor_locations[-1], timesteps[0], timesteps[-1]], vmin=vmin, vmax=vmax)
            axs[1, i].set_title(f'Prediction Sample {i+1}')
            axs[1, i].set_xlabel('Sensor Location')
            axs[1, i].set_ylabel('Time')

            # Plot error
            error = functions[i] - predictions[i]
            axs[2, i].imshow(error, aspect='equal', origin='lower', extent=[sensor_locations[0], sensor_locations[-1], timesteps[0], timesteps[-1]], vmin=vmin, vmax=vmax)
            axs[2, i].set_title(f'Error Sample {i+1}')
            axs[2, i].set_xlabel('Sensor Location')
            axs[2, i].set_ylabel('Time')

    # Add a common colorbar
    #cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical')
    #cbar.set_label('Value')
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    #fig.subplots_adjust(right=2)  # Adjust as needed

    #plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_samples = 1000
    sensor_points = np.linspace(0, 1, 100)
    length_scale = 0.1
    scale = 3
    method = 'trapezoidal' #choose from trapezoidal, cumsum, or simpson

    data_poly = generate_polynomial_data(num_samples, sensor_points, scale, method)
    data_GRF = generate_GRF_data(num_samples, sensor_points, length_scale, method)
    data_sine = generate_sine_data(num_samples, sensor_points, method)
    # data = generate_polynomial_data(num_samples, sensor_points)

    #Inspect how the data covers the domain
    plot_functions_only(data_poly, sensor_points, num_samples_to_plot=100)
    plot_functions_only(data_GRF, sensor_points, num_samples_to_plot=100)
    plot_functions_only(data_sine, sensor_points, num_samples_to_plot=100)
    
    #Plot some examples from the data
    plot_functions_and_antiderivatives(data_poly, data_GRF, data_sine, sensor_points, num_samples_to_plot=3)

