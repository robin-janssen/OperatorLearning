from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
from matplotlib.colors import Normalize
from itertools import cycle
import numpy as np
import streamlit as st
from utils import create_date_based_directory
import os


def plot_functions_and_antiderivatives(
    data_poly: list,
    data_GRF: list,
    data_sine: list,
    sensor_points: np.array,
    num_samples_to_plot: int = 3,
) -> None:
    """
    Plot some samples of the generated data (poly, GRF, sine) and their antiderivatives.

    :param data_poly: List of tuples containing the polynomial data and its antiderivative.
    :param data_GRF: List of tuples containing the GRF data and its antiderivative.
    :param data_sine: List of tuples containing the sine data and its antiderivative.
    Note: Each tuple contains two numpy arrays: the function and its antiderivative evaluated at sensor points.
    :param sensor_points: Array of sensor locations in the domain.
    :param num_samples_to_plot: Number of samples to plot.
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Some samples of the data")

    colors = cycle(plt.cm.tab10.colors)  # Color cycle from a matplotlib colormap

    for i in range(num_samples_to_plot):
        color = next(colors)
        axs[0].plot(sensor_points, data_poly[i][0], label=f"poly_{i+1}", color=color)
        axs[0].plot(
            sensor_points,
            data_poly[i][1],
            label=f"poly_int_{i+1}",
            color=color,
            linestyle="--",
        )
        axs[1].plot(
            sensor_points,
            data_GRF[i][0],
            label=f"GRF_{i+1}",
            color=color,
            linestyle="-",
        )
        axs[1].plot(
            sensor_points,
            data_GRF[i][1],
            label=f"GRF_int_{i+1}",
            color=color,
            linestyle="--",
        )
        axs[2].plot(
            sensor_points,
            data_sine[i][0],
            label=f"sine_{i+1}",
            color=color,
            linestyle="-",
        )
        axs[2].plot(
            sensor_points,
            data_sine[i][1],
            label=f"sine_int_{i+1}",
            color=color,
            linestyle="--",
        )

    axs[0].set_title("Polynomial data")
    axs[1].set_title("Gaussian random field data")
    axs[2].set_title("Sine data")

    for ax in axs:
        ax.set_xlabel("Domain")
        ax.set_ylabel("Function Value")

    plt.tight_layout()
    plt.show()


def plot_functions_only(
    data: list, sensor_points: np.array, num_samples_to_plot: int = 3
) -> None:
    """
    Plot some samples of the data

    :param data: list of tuples containing the function and its antiderivative.
    Note: Each tuple contains two numpy arrays: the function and its antiderivative evaluated at sensor points.
    :param sensor_points: Array of sensor locations in the domain.
    :param num_samples_to_plot: Number of samples to plot.

    """
    plt.figure(figsize=(6, 4))
    plt.title("Function Coverage")

    for i in range(num_samples_to_plot):
        plt.plot(sensor_points, data[i, :, 0], color="blue", alpha=0.2)  # Low opacity

    plt.xlabel("Domain")
    plt.ylabel("Function Value")
    plt.tight_layout()
    plt.show()


def surface_plot(
    sensor_locations: np.array,
    timesteps: np.array,
    functions: np.array,
    num_samples_to_plot: int = 3,
    predictions: np.array = None,
    title: str = "Data Visualization",
) -> None:
    """
    Plot the evolution of functions over time using a 3D surface plot.
    If predictions are provided, plot the ground truth, predictions, and error in separate subplots.

    :param sensor_locations: Array of sensor locations.
    :param timesteps: Array of timesteps.
    :param functions: 3D array of ground truth function values - shape [N_samples, len(sensor_locations), len(timesteps)].
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
            plot_single_surface(fig, 1, i, 1, total_cols, X, Y, Z, "Data", i)
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
    fig: plt.Figure,
    row: int,
    col: int,
    total_rows: int,
    total_cols: int,
    X: np.array,
    Y: np.array,
    Z: np.array,
    title_prefix: str,
    sample_number: int,
) -> None:
    """
    Plot a single surface in a subplot.

    :param fig: The figure object to which the subplot is added.
    :param row: The row number of the subplot.
    :param col: The column number of the subplot.
    :param total_rows: Total number of rows in the subplot grid.
    :param total_cols: Total number of columns in the subplot grid.
    :param X: X-coordinates for the meshgrid (shape [len(sensor_points), len(timesteps)]).
    :param Y: Y-coordinates for the meshgrid (shape [len(sensor_points), len(timesteps)]).
    :param Z: Z-coordinates (function values) for the plot (shape [len(sensor_points), len(timesteps)]).
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
    sensor_locations: np.array,
    timesteps: np.array,
    functions: np.array,
    num_samples_to_plot: int = 3,
    predictions: np.array = None,
    title: str = "Data Visualization",
) -> None:
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
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    plt.show()


def heatmap_plot_errors(
    sensor_locations: np.array,
    timesteps: np.array,
    errors: list,
    num_samples_to_plot: int = 3,
    title: str = "Model Comparison via Error Visualization",
) -> None:
    """
    Plot the error evolution of different models over time using heat maps with a common colorbar.

    :param sensor_locations: Array of sensor locations.
    :param timesteps: Array of timesteps.
    :param errors: List of 3D arrays of error values for different models.
    :param num_samples_to_plot: Number of samples to plot.
    :param title: Title of the plot.
    """
    num_models = len(errors)  # Determine number of models based on the list length
    figsize = (12, 3 * num_models)  # Adjust figure size based on the number of models
    fig, axs = plt.subplots(
        num_models,
        num_samples_to_plot,
        figsize=figsize,
        squeeze=False,
    )
    fig.suptitle(title)

    # Determine common vmin and vmax for color scaling across all models
    vmin = min(error.min() for error in errors)
    vmax = max(error.max() for error in errors)

    for model_idx, model_errors in enumerate(errors):
        model_errors = model_errors[:num_samples_to_plot].transpose(
            0, 2, 1
        )  # Adjust data shape

        for sample_idx in range(num_samples_to_plot):
            im = axs[model_idx, sample_idx].imshow(
                model_errors[sample_idx],
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
            axs[model_idx, sample_idx].set_title(
                f"Model {model_idx+1} Error {sample_idx+1}"
            )
            axs[model_idx, sample_idx].set_xlabel("Sensor Location")
            axs[model_idx, sample_idx].set_ylabel("Time")

    # Adjust colorbar to fit the height of the plot dynamically
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust as necessary
    fig.colorbar(im, cax=cbar_ax)

    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to prevent overlap
    plt.show()


def plot_losses(loss_histories: tuple[np.array, ...], labels: tuple[str, ...]) -> None:
    """
    Plot the loss trajectories for the training of multiple models.

    :param loss_histories: List of loss history arrays.
    :param labels: List of labels for each loss history.
    """

    # Create the figure
    plt.figure(figsize=(12, 6))
    if len(loss_histories) != len(labels):
        plt.plot(loss_histories, label=labels)
    else:
        for loss, label in zip(loss_histories, labels):
            plt.plot(loss, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Loss")
    plt.legend()

    directory = create_date_based_directory(subfolder="plots")

    # Initialize filename and counter
    filename = "losses.png"
    filepath = os.path.join(directory, filename)
    filebase, fileext = filename.split(".")

    # Check if the file exists and modify the filename accordingly
    counter = 1
    while os.path.exists(filepath):
        filename = f"{filebase}_{counter}.{fileext}"
        filepath = os.path.join(directory, filename)
        counter += 1

    plt.savefig(filepath)
    plt.show()
    print(f"Plot saved as: {filepath}")


def plot_results(
    title,
    sensor_locations,
    function_values,
    ground_truth,
    y_values,
    predictions,
    num_samples=3,
):
    colors = cycle(plt.cm.tab10.colors)  # Color cycle from a matplotlib colormap

    plt.figure(figsize=(12, 6))
    plt.suptitle(title)

    # First subplot: Input function, Ground Truth, and Predictions
    plt.subplot(1, 2, 1)
    for i in range(num_samples):
        color = next(colors)
        # plt.plot(sensor_locations, function_values[i], label=f"Input Function {i+1}", color=color, linestyle='-', marker='o')
        plt.plot(
            sensor_locations,
            ground_truth[i],
            label=f"Ground Truth {i+1}",
            color=color,
            linestyle="--",
        )
        plt.plot(
            y_values,
            predictions[i],
            label=f"DeepONet Prediction {i+1}",
            color=color,
            linestyle=":",
            marker=".",
        )
    plt.xlabel("Domain")
    plt.ylabel("Function Value")
    plt.title("Function, Ground Truth, and Predictions")
    plt.legend()

    # Second subplot: Error
    plt.subplot(1, 2, 2)
    for i in range(num_samples):
        color = next(colors)
        error = ground_truth[i] - predictions[i]
        plt.plot(y_values, error, label=f"Error {i+1}", color=color)
    plt.xlabel("Domain")
    plt.ylabel("Error")
    plt.title("Prediction Error")
    plt.legend()

    plt.tight_layout()
    plt.show()


def streamlit_visualization_abserror(train_loss, test_loss, predictions, ground_truths):
    """
    Visualize the training and test loss, predictions, ground truths, and prediction errors using Streamlit.
    This version uses plt.imshow for better visualization with colormaps.

    :param train_loss: List of training loss values.
    :param test_loss: List of test loss values.
    :param predictions: Numpy array of shape [5, N, 21, 21] for the last five epochs' predictions.
    :param ground_truths: Numpy array of shape [N, 21, 21] for the corresponding ground truths.
    """
    # Access the placeholders from session state
    loss_plot_placeholder = st.session_state["loss_plot_placeholder"]
    prediction_placeholder = st.session_state["prediction_placeholder"]

    # Clear previous content
    loss_plot_placeholder.empty()
    prediction_placeholder.empty()

    # Plot training and test loss in the first placeholder
    with loss_plot_placeholder.container():
        st.write("Training and Test Loss")
        fig, ax = plt.subplots()
        ax.plot(train_loss, label="Training Loss")
        ax.plot(test_loss, label="Test Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

    # Display predictions, ground truths, and errors in the second placeholder
    with prediction_placeholder.container():
        st.write("Predictions, Ground Truths, and Errors")
        if predictions is not None:
            N = predictions.shape[1]  # Number of samples per epoch

            for n in range(N):
                cols = st.columns(7)
                # 7 columns: 5 for predictions, 1 for ground truth, 1 for error
                # Display predictions in the first 5 columns
                for i in range(5):
                    with cols[i]:
                        pred = predictions[i, n]
                        pred_normalized = (pred - np.min(pred)) / (
                            np.max(pred) - np.min(pred)
                        )
                        fig, ax = plt.subplots()
                        ax.imshow(pred_normalized, cmap="viridis")
                        ax.axis("off")  # Hide axes
                        st.pyplot(fig)

                # Display ground truth in the 6th column
                with cols[5]:
                    gt = ground_truths[n]
                    gt_normalized = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
                    fig, ax = plt.subplots()
                    ax.imshow(gt_normalized, cmap="viridis")
                    ax.axis("off")
                    st.pyplot(fig)

                # Display error in the 7th column
                with cols[6]:
                    error = np.abs(pred - gt)
                    fig, ax = plt.subplots()
                    ax.imshow(error, cmap="viridis")
                    ax.axis("off")
                    st.pyplot(fig)
        else:
            st.write("Predictions will be displayed here once available.")


def streamlit_visualization(train_loss, test_loss, predictions, ground_truths):
    """
    Visualize the training and test loss, predictions, ground truths, and relative errors using Streamlit.
    """
    # Access the placeholders from session state
    loss_plot_placeholder = st.session_state["loss_plot_placeholder"]
    prediction_placeholder = st.session_state["prediction_placeholder"]

    # Clear previous content
    loss_plot_placeholder.empty()
    prediction_placeholder.empty()

    # Plot training and test loss in the first placeholder
    with loss_plot_placeholder.container():
        st.write("Training and Test Loss")
        fig, ax = plt.subplots()
        ax.plot(train_loss, label="Training Loss")
        ax.plot(test_loss, label="Test Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

    # Display predictions, ground truths, and errors in the second placeholder
    with prediction_placeholder.container():
        st.write("Predictions, Ground Truths, and Errors")
        if predictions is not None:
            N = predictions.shape[1]  # Number of samples per epoch

            # Column labels
            cols = st.columns(7)  # Prepare 7 columns for labels
            column_labels = [
                "Prediction 1",
                "Prediction 2",
                "Prediction 3",
                "Prediction 4",
                "Prediction 5",
                "Ground Truth",
                "Relative Error",
            ]
            for i, label in enumerate(column_labels):
                with cols[i]:
                    st.caption(label)  # Place text above each column

            for n in range(N):
                cols = st.columns(7)  # 7 columns for content
                # 7 columns: 5 for predictions, 1 for ground truth, 1 for error
                # Display predictions in the first 5 columns
                for i in range(5):
                    with cols[i]:
                        pred = predictions[i, n]
                        pred_normalized = (pred - np.min(pred)) / (
                            np.max(pred) - np.min(pred)
                        )
                        fig, ax = plt.subplots()
                        ax.imshow(pred_normalized, cmap="viridis")
                        ax.axis("off")  # Hide axes
                        st.pyplot(fig)

                # Display ground truth in the 6th column
                with cols[5]:
                    gt = ground_truths[n]
                    gt_normalized = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
                    fig, ax = plt.subplots()
                    ax.imshow(gt_normalized, cmap="viridis")
                    ax.axis("off")
                    st.pyplot(fig)

                # Display relative error in the 7th column
                with cols[6]:
                    pred = predictions[
                        -1, n
                    ]  # Using the last prediction for error calculation
                    gt = ground_truths[n]
                    # Calculate relative error, add a small constant to avoid division by zero
                    epsilon = 1e-4
                    relative_error = np.abs(pred - gt) / (np.abs(gt) + epsilon)
                    fig, ax = plt.subplots()
                    ax.imshow(relative_error, cmap="viridis")
                    ax.axis("off")
                    st.pyplot(fig)
        else:
            st.write("Predictions will be displayed here once available.")


def streamlit_visualization_history(
    train_loss, test_loss, predictions, ground_truths, current_epoch
):
    """
    Visualize training and test loss, selected predictions, their absolute errors with a common scale,
    ground truths, and include a colorbar for error magnitude. Also, display headings for the current epoch
    and for each column.
    """
    rcParams["figure.max_open_warning"] = 30  # Suppress matplotlib max open warning

    # Create placeholders if they don't exist
    if "loss_plot_placeholder" not in st.session_state:
        st.session_state["loss_plot_placeholder"] = st.empty()
    if "prediction_placeholder" not in st.session_state:
        st.session_state["prediction_placeholder"] = st.empty()

    # Access placeholders from session state
    # title_placeholder = st.session_state["title_placeholder"]
    loss_plot_placeholder = st.session_state["loss_plot_placeholder"]
    prediction_placeholder = st.session_state["prediction_placeholder"]

    # Clear previous content
    # title_placeholder.empty()
    loss_plot_placeholder.empty()
    prediction_placeholder.empty()

    # Display the current epoch as a heading
    # with title_placeholder.container():
    #     st.title(f"Epoch {current_epoch + 1}")

    # Plot training and test loss
    with loss_plot_placeholder.container():
        plt.close("all")  # Close previous plots
        st.write(f"Training and Test Loss (Epoch {current_epoch + 1})")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(train_loss, label="Training Loss")
        ax.plot(test_loss, label="Test Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend()
        st.pyplot(fig)

    # Pre-calculate the maximum error for a common scale in error plots
    max_error = 0
    epoch_indices = [
        max(0, current_epoch - 10),
        max(0, current_epoch - 5),
        current_epoch,
    ]
    for epoch_idx in epoch_indices:
        for n in range(predictions.shape[1]):
            pred = predictions[epoch_idx, n]
            gt = ground_truths[n]
            error = np.abs(pred - gt)
            max_error = max(max_error, np.max(error))

    # Define a normalization instance for the colorbar based on the max error
    norm = Normalize(vmin=0, vmax=max_error)

    with prediction_placeholder.container():
        # st.write("Predictions, Ground Truths, and Errors")
        if predictions is not None:
            N = predictions.shape[1]

            for n in range(N):
                cols = st.columns(7)
                # Display column headings
                epochs_displayed = [
                    epoch_idx
                    for epoch_idx in epoch_indices
                    if epoch_idx <= current_epoch
                ]
                headings_string = []
                for epoch_idx in epochs_displayed:
                    headings_string.append(f"Pred (e{epoch_idx + 1})")
                    headings_string.append(f"Err (e{epoch_idx + 1})")
                headings = headings_string + ["GT"]
                for i, heading in enumerate(headings):
                    cols[i].write(heading)

                for i, epoch_idx in enumerate(epochs_displayed):
                    # Display prediction
                    with cols[2 * i]:
                        pred = predictions[epoch_idx, n]
                        fig, ax = plt.subplots()
                        ax.imshow(pred, cmap="viridis")
                        ax.axis("off")
                        st.pyplot(fig)

                    # Display absolute error and colorbar next to each prediction
                    with cols[2 * i + 1]:
                        gt = ground_truths[n]
                        error = np.abs(pred - gt)
                        fig, ax = plt.subplots()
                        ax.imshow(error, cmap="viridis", norm=norm)
                        ax.axis("off")
                        st.pyplot(fig)

                # Display ground truth in the last column
                with cols[-1]:
                    fig, ax = plt.subplots()
                    ax.imshow(gt, cmap="viridis")
                    ax.axis("off")
                    st.pyplot(fig)

            # Add a colorbar to quantify error magnitude, displayed at the bottom or side
            fig, cax = plt.subplots(
                figsize=(5, 0.4)
            )  # Adjust figsize for the plot used for colorbar
            plt.axis("off")  # Hide the main plot axes
            cbar_ax = fig.add_axes(
                [0.05, 0.5, 0.9, 0.3]
            )  # Adjust these values: [left, bottom, width, height] in figure coordinate
            cbar = plt.colorbar(
                cm.ScalarMappable(norm=norm, cmap="viridis"),
                cax=cbar_ax,
                orientation="horizontal",
            )
            cbar.set_label("Error Magnitude", labelpad=-40, y=0.45)
            st.pyplot(fig)
        else:
            st.write("Predictions will be displayed here once available.")
