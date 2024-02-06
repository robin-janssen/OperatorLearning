import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np


def plot_functions_and_antiderivatives(
    data_poly, data_GRF, data_sine, sensor_points, num_samples_to_plot=3
):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Some samples of the data")

    colors = cycle(plt.cm.tab10.colors)  # Color cycle from a matplotlib colormap

    for i in range(num_samples_to_plot):
        color = next(colors)
        axs[0].plot(
            sensor_points, data_poly[i][0], label=f"u_{i+1}", color=color, linestyle="-"
        )
        axs[0].plot(sensor_points, data_poly[i][1], color=color, linestyle="--")
        axs[1].plot(sensor_points, data_GRF[i][0], color=color, linestyle="-")
        axs[1].plot(sensor_points, data_GRF[i][1], color=color, linestyle="--")
        axs[2].plot(sensor_points, data_sine[i][0], color=color, linestyle="-")
        axs[2].plot(sensor_points, data_sine[i][1], color=color, linestyle="--")

    axs[0].set_title("Polynomial data")
    axs[1].set_title("Gaussian random field data")
    axs[2].set_title("Sine data")

    for ax in axs:
        ax.set_xlabel("Domain")
        ax.set_ylabel("Function Value")

    plt.tight_layout()
    plt.show()


def plot_functions_only(data, sensor_points, num_samples_to_plot):
    """
    Plot some samples of the data

    :param data: numpy array of data
    :param sensor_points: Array of sensor locations in the domain.
    :param num_samples_to_plot: Number of samples to plot.

    """
    plt.figure(figsize=(6, 4))
    plt.title("Function Coverage")

    for i in range(num_samples_to_plot):
        plt.plot(sensor_points, data[i], color="blue", alpha=0.2)  # Low opacity

    plt.xlabel("Domain")
    plt.ylabel("Function Value")
    plt.tight_layout()
    plt.show()


def surface_plot(
    sensor_locations,
    timesteps,
    functions,
    num_samples_to_plot,
    predictions=None,
    title="Data Visualization",
):
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
            plot_single_surface(fig, 1, i, 1, 3, X, Y, Z, "Sample", i)
    else:
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle(title)
        for i in range(num_samples_to_plot):
            # Plot ground truth
            Z_gt = functions[i].T
            plot_single_surface(fig, 1, i, 3, total_cols, X, Y, Z_gt, "Ground Truth", i)

            # Plot predictions
            Z_pred = predictions[i].T
            plot_single_surface(fig, 2, i, 3, total_cols, X, Y, Z_pred, "Prediction", i)

            # Plot error
            Z_err = Z_gt - Z_pred
            plot_single_surface(fig, 3, i, 3, total_cols, X, Y, Z_err, "Error", i)

    plt.gcf().subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.show()


def plot_single_surface(
    fig, row, col, total_rows, total_cols, X, Y, Z, title_prefix, sample_number
):
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

    ax = fig.add_subplot(total_rows, total_cols, index, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Sensor Location")
    ax.set_ylabel("Time")
    ax.set_zlabel("Value")
    ax.set_title(f"{title_prefix} Sample {sample_number + 1}")


def heatmap_plot(
    sensor_locations,
    timesteps,
    functions,
    num_samples_to_plot,
    predictions=None,
    title="Data Visualization",
):
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
    fig, axs = plt.subplots(
        3 if predictions is not None else 1,
        num_samples_to_plot,
        figsize=figsize,
        squeeze=False,
        layout="compressed",
    )
    fig.suptitle(title)
    functions = functions[:num_samples_to_plot].transpose(0, 2, 1)
    predictions = (
        predictions[:num_samples_to_plot].transpose(0, 2, 1)
        if predictions is not None
        else None
    )

    # Determine common vmin and vmax for color scaling
    vmin = min(
        functions.min(),
        predictions.min() if predictions is not None else functions.min(),
    )
    vmax = max(
        functions.max(),
        predictions.max() if predictions is not None else functions.max(),
    )

    for i in range(num_samples_to_plot):
        # Plot ground truth
        im = axs[0, i].imshow(
            functions[i],
            aspect="equal",
            origin="lower",
            extent=[
                sensor_locations[0],
                sensor_locations[-1],
                timesteps[0],
                timesteps[-1],
            ],
            vmin=vmin,
            vmax=vmax,
        )
        axs[0, i].set_title(f"Ground Truth Sample {i+1}")
        axs[0, i].set_xlabel("Sensor Location")
        axs[0, i].set_ylabel("Time")

        if predictions is not None:
            # Plot predictions
            axs[1, i].imshow(
                predictions[i],
                aspect="equal",
                origin="lower",
                extent=[
                    sensor_locations[0],
                    sensor_locations[-1],
                    timesteps[0],
                    timesteps[-1],
                ],
                vmin=vmin,
                vmax=vmax,
            )
            axs[1, i].set_title(f"Prediction Sample {i+1}")
            axs[1, i].set_xlabel("Sensor Location")
            axs[1, i].set_ylabel("Time")

            # Plot error
            error = functions[i] - predictions[i]
            axs[2, i].imshow(
                error,
                aspect="equal",
                origin="lower",
                extent=[
                    sensor_locations[0],
                    sensor_locations[-1],
                    timesteps[0],
                    timesteps[-1],
                ],
                vmin=vmin,
                vmax=vmax,
            )
            axs[2, i].set_title(f"Error Sample {i+1}")
            axs[2, i].set_xlabel("Sensor Location")
            axs[2, i].set_ylabel("Time")

    # Add a common colorbar
    # cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical')
    # cbar.set_label('Value')
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)
    # fig.subplots_adjust(right=2)  # Adjust as needed

    # plt.tight_layout()
    plt.show()
