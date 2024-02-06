import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from itertools import cycle

from datagen import generate_polynomial_data, generate_GRF_data, generate_sine_data
from deeponet import DeepONet


def train_deeponet(
    data_loader,
    branch_input_size,
    trunk_input_size,
    hidden_size,
    branch_hidden_layers=3,
    trunk_hidden_layers=1,
    num_epochs=1000,
    learning_rate=0.001,
    schedule=True,
    test_loader=None,
):
    """Train a DeepONet model.
    The function instantiates a DeepONet model and trains it using the provided DataLoader.
    Note that it assumes equal number of neurons in each hidden layer of the branch and trunk networks.

    :param data_loader: A DataLoader object.
    :param branch_input_size: Input size for the branch network.
    :param trunk_input_size: Input size for the trunk network.
    :param hidden_size: Number of hidden units in each layer.
    :param num_epochs: Number of epochs to train for.
    :param branch_hidden_layers: Number of hidden layers in the branch network.
    :param trunk_hidden_layers: Number of hidden layers in the trunk network.

    :return: Trained DeepONet model and loss history.
    """
    deeponet = DeepONet(
        branch_input_size,
        hidden_size,
        branch_hidden_layers,
        trunk_input_size,
        hidden_size,
        trunk_hidden_layers,
        hidden_size,
    )

    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(deeponet.parameters(), lr=learning_rate)
    if schedule:
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=0.1, total_iters=num_epochs
        )

    train_loss_history = np.zeros(num_epochs)
    test_loss_history = np.zeros(num_epochs // 10)

    total_predictions = sum(len(targets) for _, _, targets in data_loader)

    progress_bar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in progress_bar:
        epoch_loss = 0
        for branch_inputs, trunk_inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = deeponet(branch_inputs, trunk_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= total_predictions
        train_loss_history[epoch] = epoch_loss
        clr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({"loss": epoch_loss, "lr": clr})
        scheduler.step()
        if test_loader is not None:
            if epoch % 10 == 0:
                test_loss, _ = test_deeponet(deeponet, test_loader)
                test_loss_history[int(epoch / 10)] = test_loss

    if test_loader is not None:
        return deeponet, train_loss_history, test_loss_history
    else:
        return deeponet, train_loss_history


def test_deeponet(model, data_loader):
    """
    Test a DeepONet model.

    :param model: A DeepONet model (as instantiated using the DeepONet class).
    :param data_loader: A DataLoader object.

    :return: Total loss and predictions.
    """
    criterion = nn.MSELoss(reduction="sum")
    model.eval()

    # Calculate the total number of predictions to pre-allocate the buffer
    total_predictions = sum(len(targets) for _, _, targets in data_loader)
    predictions_buffer = np.empty(total_predictions)
    buffer_index = 0

    total_loss = 0
    with torch.no_grad():
        for branch_inputs, trunk_inputs, targets in data_loader:
            outputs = model(branch_inputs, trunk_inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Store predictions in the buffer
            num_predictions = len(outputs)
            predictions_buffer[buffer_index : buffer_index + num_predictions] = (
                outputs.cpu().numpy()
            )
            buffer_index += num_predictions

    # Calculate relative error
    total_loss /= total_predictions

    return total_loss, predictions_buffer


def create_dataloader(data, sensor_points, batch_size=32, shuffle=False):
    """
    Create a DataLoader for DeepONet.

    :param data: List of tuples (function_values, antiderivative_values).
    :param sensor_points: Array of sensor locations in the domain.
    :param batch_size: Batch size for the DataLoader.
    :return: A DataLoader object.
    """
    branch_inputs = []
    trunk_inputs = []
    targets = []

    for function_values, antiderivative_values in data:
        for i in range(len(sensor_points)):
            branch_inputs.append(function_values)
            trunk_inputs.append([sensor_points[i]])
            targets.append([antiderivative_values[i]])

    # Convert to PyTorch tensors
    branch_inputs_tensor = torch.tensor(branch_inputs, dtype=torch.float32)
    trunk_inputs_tensor = torch.tensor(trunk_inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(branch_inputs_tensor, trunk_inputs_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def plot_losses(loss_histories, labels):
    """
    Plot the loss trajectories for the training of multiple models.

    :param loss_histories: List of loss history arrays.
    :param labels: List of labels for each loss history.
    """

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
    plt.show()


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


def load_deeponet(
    path_to_state_dict,
    branch_input_size,
    trunk_input_size,
    hidden_size,
    branch_hidden_layers,
    trunk_hidden_layers,
):
    """
    Load a DeepONet model from a saved state dictionary.

    :param path_to_state_dict: Path to the saved state dictionary.
    :param branch_input_size: Input size for the branch network.
    :param trunk_input_size: Input size for the trunk network.
    :param hidden_size: Number of hidden units in each layer.
    :param num_hidden_layers: Number of hidden layers in each network.
    :return: Loaded DeepONet model.
    """
    # Recreate the model architecture
    # branch_net = BranchNet(branch_input_size, hidden_size, hidden_size, branch_hidden_layers)
    # trunk_net = TrunkNet(trunk_input_size, hidden_size, hidden_size, trunk_hidden_layers)
    deeponet = DeepONet(
        branch_input_size,
        hidden_size,
        branch_hidden_layers,
        trunk_input_size,
        hidden_size,
        trunk_hidden_layers,
        hidden_size,
    )

    # Load the state dictionary
    state_dict = torch.load(path_to_state_dict)
    deeponet.load_state_dict(state_dict)

    return deeponet


if __name__ == "__main__":

    # Hyperparameters
    TRAIN = False
    branch_input_size = 21
    trunk_input_size = 1
    hidden_size = 40
    branch_output_size = hidden_size
    trunk_output_size = hidden_size
    branch_hidden_layers = 3
    trunk_hidden_layers = 1
    dataset_size = 1000
    num_epochs = 10
    sensor_points = np.linspace(0, 1, branch_input_size)
    num_samples_to_plot = 3

    if TRAIN:
        # Generate polynomial and GRF data and create DataLoaders
        poly_data = generate_polynomial_data(dataset_size, sensor_points, scale=3)
        grf_data = generate_GRF_data(dataset_size, sensor_points, length_scale=0.3)
        poly_data_loader = create_dataloader(poly_data, sensor_points, shuffle=True)
        grf_data_loader = create_dataloader(grf_data, sensor_points, shuffle=True)

        # Train models
        poly_deeponet, poly_loss = train_deeponet(
            poly_data_loader,
            branch_input_size,
            trunk_input_size,
            hidden_size,
            num_epochs,
            branch_hidden_layers,
            trunk_hidden_layers,
        )
        grf_deeponet, grf_loss = train_deeponet(
            grf_data_loader,
            branch_input_size,
            trunk_input_size,
            hidden_size,
            num_epochs,
            branch_hidden_layers,
            trunk_hidden_layers,
        )
        torch.save(poly_deeponet.state_dict(), "poly_deeponet.pt")
        torch.save(grf_deeponet.state_dict(), "grf_deeponet.pt")

        # Plot the loss trajectories
        plot_losses([poly_loss, grf_loss], ["Polynomial Data", "GRF Data"])
    else:
        poly_deeponet = load_deeponet(
            "poly_deeponet.pt",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
        )
        grf_deeponet = load_deeponet(
            "grf_deeponet.pt",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
        )

    # Generate sine data for testing
    sine_data = generate_sine_data(1000, sensor_points)
    sine_data_loader = create_dataloader(sine_data, sensor_points)

    # Generate some more polynomial data and GRF data for testing
    poly_test_data = generate_polynomial_data(dataset_size, sensor_points, scale=3)
    grf_test_data = generate_GRF_data(dataset_size, sensor_points, length_scale=0.3)
    poly_test_data_loader = create_dataloader(poly_test_data, sensor_points)
    grf_test_data_loader = create_dataloader(grf_test_data, sensor_points)

    # Test models
    poly_test_loss, poly_predictions = test_deeponet(poly_deeponet, sine_data_loader)
    grf_test_loss, grf_predictions = test_deeponet(grf_deeponet, sine_data_loader)
    print(f"Polynomial Model Test Loss: {poly_test_loss}")
    print(f"GRF Model Test Loss: {grf_test_loss}")

    # Plot some examples from the test set
    poly_predictions = poly_predictions.reshape(-1, len(sensor_points))
    grf_predictions = grf_predictions.reshape(-1, len(sensor_points))
    sine_data = np.array(sine_data)
    plot_results(
        "Polynomial data",
        sensor_points,
        sine_data[:, 0],
        sine_data[:, 1],
        sensor_points,
        poly_predictions,
        num_samples_to_plot,
    )
    plot_results(
        "GRF data",
        sensor_points,
        sine_data[:, 0],
        sine_data[:, 1],
        sensor_points,
        grf_predictions,
        num_samples_to_plot,
    )
