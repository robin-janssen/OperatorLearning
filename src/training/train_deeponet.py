from __future__ import annotations
from time import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import DeepONet, OperatorNetworkType
from plotting import streamlit_visualization_history
from .train_utils import time_execution


# TODO Add a training continuation functionality to all training functions
@time_execution
def train_deeponet_visualized(
    data_loader: DataLoader,
    branch_input_size: int,
    trunk_input_size: int,
    hidden_size: int,
    branch_hidden_layers: int = 3,
    trunk_hidden_layers: int = 3,
    num_epochs: int = 1000,
    learning_rate: int = 0.001,
    schedule: bool = True,
    test_loader: DataLoader = None,
    N_sensors: int = 101,
    N_timesteps: int = 101,
    pretrained_model_path: str | None = None,
    device: str = "cpu",
) -> tuple:
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
        device=device,
    )

    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(deeponet.parameters(), lr=learning_rate)
    if schedule:
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=0.1, total_iters=num_epochs
        )
    else:
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1, total_iters=num_epochs
        )

    train_loss_history = np.zeros(num_epochs)
    test_loss_history = np.zeros(num_epochs)
    output_history = np.zeros((num_epochs, 3, N_sensors, N_timesteps))

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
        epoch_loss *= 10
        train_loss_history[epoch] = epoch_loss
        clr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({"loss": epoch_loss, "lr": clr})
        scheduler.step()
        # if epoch % 10 == 0:
        if test_loader is not None:
            test_loss, outputs, targets = test_deeponet(deeponet, test_loader)
            test_loss *= 10
            test_loss_history[epoch] = test_loss
            outputs = outputs.reshape(-1, N_sensors, N_timesteps)
            outputs = outputs[:3]
            if epoch == 0:
                targets_vis = targets.reshape(-1, N_sensors, N_timesteps)[:3]
            output_history[epoch] = outputs
        streamlit_visualization_history(
            train_loss_history[: epoch + 1],
            test_loss_history[: epoch + 1],
            output_history,
            targets_vis,
            epoch,
        )

    if test_loader is not None:
        return deeponet, train_loss_history, test_loss_history
    else:
        return deeponet, train_loss_history


@time_execution
def train_deeponet(
    data_loader: DataLoader,
    branch_input_size: int,
    trunk_input_size: int,
    hidden_size: int,
    branch_hidden_layers: int = 3,
    trunk_hidden_layers: int = 3,
    num_epochs: int = 1000,
    learning_rate: int = 0.001,
    schedule: bool = True,
    test_loader: DataLoader = None,
    N_sensors: int = 101,
    N_timesteps: int = 101,
    pretrained_model_path: str | None = None,
    device: str = "cpu",
) -> tuple:
    """Train a DeepONet model.
    The function instantiates a DeepONet model and trains it using the provided DataLoader.
    Note that it assumes equal number of neurons in each hidden layer of the branch and trunk networks.

    :param data_loader: A DataLoader object.
    :param branch_input_size: Input size for the branch network.
    :param trunk_input_size: Input size for the trunk network.
    :param hidden_size: Number of hidden units in each layer.
    :param branch_hidden_layers: Number of hidden layers in the branch network.
    :param trunk_hidden_layers: Number of hidden layers in the trunk network.
    :param num_epochs: Number of epochs to train for.
    :param learning_rate: Learning rate for the optimizer.
    :param schedule: Whether to use a learning rate schedule.
    :param test_loader: A DataLoader object for testing the model.
    :param N_sensors: Number of sensors.
    :param N_timesteps: Number of timesteps.
    :param pretrained_model_path: Path to a pretrained model.
    :param device: The device to use for training.

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
        device=device,
    )

    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(deeponet.parameters(), lr=learning_rate)
    if schedule:
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=0.1, total_iters=num_epochs
        )
    else:
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1, total_iters=num_epochs
        )

    train_loss_history = np.zeros(num_epochs)
    test_loss_history = np.zeros(num_epochs)
    # output_history = np.zeros((num_epochs, 3, N_sensors, N_timesteps))

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
        # if epoch % 10 == 0:
        # if test_loader is not None:
        #     test_loss, outputs, targets = test_deeponet(deeponet, test_loader)
        #     test_loss_history[epoch] = test_loss
        #     outputs = outputs.reshape(-1, N_sensors, N_timesteps)
        #     outputs = outputs[:3]
        #     if epoch == 0:
        #         targets_vis = targets.reshape(-1, N_sensors, N_timesteps)[:3]
        #     output_history[epoch] = outputs
        # streamlit_visualization_history(
        #     train_loss_history[: epoch + 1],
        #     test_loss_history[: epoch + 1],
        #     output_history,
        #     targets_vis,
        #     epoch,
        # )

    if test_loader is not None:
        return deeponet, train_loss_history, test_loss_history
    else:
        return deeponet, train_loss_history


def test_deeponet(
    model: OperatorNetworkType,
    data_loader: DataLoader,
    device="cpu",
    criterion=nn.MSELoss(reduction="sum"),
    timing=False,
) -> tuple:
    """
    Test a DeepONet model.

    :param model: A DeepONet model (as instantiated using the DeepONet class).
    :param data_loader: A DataLoader object.
    :param device: Device to use for testing.
    :param criterion: Loss function to use for testing.
    :param timing: Whether to time the testing process.

    :return: Total loss and predictions.
    """
    device = torch.device(device)
    model.eval()
    model.to(device)

    # Calculate the total number of predictions to pre-allocate the buffer
    _, _, example_targets = next(iter(data_loader))
    dataset_size = len(data_loader.dataset)
    # Make sure the buffers have the correct shape for broadcasting
    if len(example_targets.size()) == 1:
        targetsize = 1
        predictions_buffer = np.empty(dataset_size)
        targets_buffer = np.empty(dataset_size)
    else:
        targetsize = example_targets.size(1)
        predictions_buffer = np.empty((dataset_size, targetsize))
        targets_buffer = np.empty((dataset_size, targetsize))

    buffer_index = 0
    total_loss = 0
    with torch.no_grad():
        start_time = time()
        for branch_inputs, trunk_inputs, targets in data_loader:
            if device != "cpu":
                branch_inputs = branch_inputs.to(device)
                trunk_inputs = trunk_inputs.to(device)
                targets = targets.to(device)
                model.to(device)
            outputs = model(branch_inputs, trunk_inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Store predictions in the buffer
            num_predictions = len(outputs)
            predictions_buffer[buffer_index : buffer_index + num_predictions] = (
                outputs.cpu().numpy()
            )
            targets_buffer[buffer_index : buffer_index + num_predictions] = (
                targets.cpu().numpy()
            )
            buffer_index += num_predictions
        end_time = time()

    if timing:
        print(f"Testing time: {end_time - start_time:.2f} seconds")
        print(
            f"Average time per sample: {(end_time - start_time) * 1000 / dataset_size:.3f} ms"
        )

    # Calculate relative error
    total_loss /= dataset_size * targetsize

    return total_loss, predictions_buffer, targets_buffer


def load_deeponet(
    path_to_state_dict: str,
    branch_input_size: int,
    trunk_input_size: int,
    hidden_size: int,
    branch_hidden_layers: int,
    trunk_hidden_layers: int,
    device: str = "cpu",
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
    deeponet = DeepONet(
        branch_input_size,
        hidden_size,
        branch_hidden_layers,
        trunk_input_size,
        hidden_size,
        trunk_hidden_layers,
        hidden_size,
        device,
    )

    # Load the state dictionary
    state_dict = torch.load(path_to_state_dict)
    deeponet.load_state_dict(state_dict)

    return deeponet
