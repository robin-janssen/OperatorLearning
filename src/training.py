from __future__ import annotations
from time import time

import numpy as np
from tqdm import tqdm
from typing import TypeVar
import matplotlib.pyplot as plt
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from deeponet import DeepONet, MultiONet, MultiONetB, MultiONetT, OperatorNetwork
from plotting import streamlit_visualization_history
from utils import time_execution

# Type hint for a DeepONet model: Accept any subclass of class OperatorNetwork
T = TypeVar("T", bound=OperatorNetwork)


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


@time_execution
def train_multionet_poly_coeff(
    data_loader: DataLoader,
    branch_input_size: int,
    trunk_input_size: int,
    hidden_size: int,
    branch_hidden_layers: int = 3,
    trunk_hidden_layers: int = 3,
    output_size: int = 40,
    N_outputs: int = 5,
    num_epochs: int = 1000,
    learning_rate: float = 0.001,
    schedule: bool = True,
    test_loader: DataLoader = None,
    N_sensors: int = 101,
    N_timesteps: int = 101,
    architecture: str = "both",
) -> tuple:
    """Train a DeepONet model.
    The function instantiates a DeepONet model (with multiple outputs) and trains it using the provided DataLoader.
    Note that it assumes equal number of neurons in each hidden layer of the branch and trunk networks.

    :param data_loader: A DataLoader object.
    :param branch_input_size: Input size for the branch network.
    :param trunk_input_size: Input size for the trunk network.
    :param hidden_size: Number of hidden units in each layer.
    :param num_epochs: Number of epochs to train for.
    :param branch_hidden_layers: Number of hidden layers in the branch network.
    :param trunk_hidden_layers: Number of hidden layers in the trunk network.
    :param output_size: Number of neurons in the last layer.
    :param N_outputs: Number of outputs.
    :param architecture: Architecture of the MODeepONet model. Can be "branch", "trunk", or "both".

    :return: Trained DeepONet model and loss history.
    """
    if architecture == "both":
        deeponet = MultiONet(
            branch_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_input_size,
            hidden_size,
            trunk_hidden_layers,
            output_size,
            N_outputs,
        )
    elif architecture == "branch":
        deeponet = MultiONetB(
            branch_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_input_size,
            hidden_size,
            trunk_hidden_layers,
            output_size,
            N_outputs,
        )
    elif architecture == "trunk":
        deeponet = MultiONetT(
            branch_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_input_size,
            hidden_size,
            trunk_hidden_layers,
            output_size,
            N_outputs,
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
    # output_history = np.zeros((num_epochs, 3, N_outputs, N_timesteps))
    output_history = np.zeros((num_epochs, 3, N_sensors, N_timesteps))

    total_predictions = len(data_loader.dataset)

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
        epoch_loss /= total_predictions * N_outputs
        train_loss_history[epoch] = epoch_loss
        clr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({"loss": epoch_loss, "lr": clr})
        scheduler.step()
        # if epoch % 10 == 0:
        if test_loader is not None:
            test_loss, outputs, targets = test_deeponet(deeponet, test_loader)
            test_loss_history[epoch] = test_loss
            outputs = outputs.reshape(-1, N_timesteps, N_outputs)
            outputs = outputs[:3]  # .transpose(0, 2, 1)
            # Calculate the polynomial values at the sensor locations
            outputs_vis = []
            for i in range(outputs.shape[0]):
                for j in range(outputs.shape[1]):
                    outputs_vis.append(
                        np.polyval(outputs[i, j, ::-1], np.linspace(0, 1, N_sensors))
                    )
            outputs_vis = np.array(outputs_vis)
            outputs_vis = outputs_vis.reshape(3, -1, N_sensors)
            outputs_vis = outputs_vis.transpose(0, 2, 1)
            if epoch == 0:
                targets = targets.reshape(-1, N_timesteps, N_outputs)[:3]
                # targets = targets.transpose(0, 2, 1)
                targets_vis = []
                for i in range(targets.shape[0]):
                    for j in range(targets.shape[1]):
                        targets_vis.append(
                            np.polyval(
                                targets[i, j, ::-1], np.linspace(0, 1, N_sensors)
                            )
                        )
                targets_vis = np.array(targets_vis).reshape(3, -1, N_sensors)
                targets_vis = targets_vis.transpose(0, 2, 1)
                # targets_vis = targets.reshape(-1, N_timesteps, N_outputs)[:3]
                # targets_vis = targets_vis.transpose(0, 2, 1)
            # output_history[epoch] = outputs
            output_history[epoch] = outputs_vis
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
def train_multionet_poly_values(
    data_loader: DataLoader,
    branch_input_size: int,
    trunk_input_size: int,
    hidden_size: int,
    branch_hidden_layers: int = 3,
    trunk_hidden_layers: int = 3,
    output_size: int = 40,
    N_outputs: int = 5,
    num_epochs: int = 1000,
    learning_rate: float = 0.001,
    schedule: bool = True,
    test_loader: DataLoader = None,
    sensor_locations: np.array = np.linspace(0, 1, 101),
    N_timesteps: int = 101,
    architecture: str = "both",
) -> tuple:
    """Train a DeepONet model.
    The function instantiates a DeepONet model (with multiple outputs) and trains it using the provided DataLoader.
    Note that it assumes equal number of neurons in each hidden layer of the branch and trunk networks.

    :param data_loader: A DataLoader object.
    :param branch_input_size: Input size for the branch network.
    :param trunk_input_size: Input size for the trunk network.
    :param hidden_size: Number of hidden units in each layer.
    :param num_epochs: Number of epochs to train for.
    :param branch_hidden_layers: Number of hidden layers in the branch network.
    :param trunk_hidden_layers: Number of hidden layers in the trunk network.
    :param output_size: Number of neurons in the last layer.
    :param N_outputs: Number of outputs.
    :param architecture: Architecture of the MODeepONet model. Can be "branch", "trunk", or "both".

    :return: Trained DeepONet model and loss history.
    """
    if architecture == "both":
        model = MultiONet
    elif architecture == "branch":
        model = MultiONetB
    elif architecture == "trunk":
        model = MultiONetT

    deeponet = model(
        branch_input_size,
        hidden_size,
        branch_hidden_layers,
        trunk_input_size,
        hidden_size,
        trunk_hidden_layers,
        output_size,
        N_outputs,
    )

    sensor_locations_tensor = torch.tensor(
        sensor_locations, dtype=torch.float32, device="cpu"
    )
    N_sensors = len(sensor_locations)

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
    # output_history = np.zeros((num_epochs, 3, N_outputs, N_timesteps))
    output_history = np.zeros((num_epochs, 3, N_sensors, N_timesteps))

    total_predictions = sum(len(targets) for _, _, targets in data_loader)

    progress_bar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in progress_bar:
        epoch_loss = 0
        for branch_inputs, trunk_inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = deeponet(branch_inputs, trunk_inputs)
            coeff_loss = criterion(outputs, targets)
            outputs_evaluated = poly_eval_torch(
                outputs.reshape(-1, outputs.shape[-1]), sensor_locations_tensor
            )
            targets_evaluated = poly_eval_torch(
                targets.reshape(-1, targets.shape[-1]), sensor_locations_tensor
            )
            poly_loss = criterion(outputs_evaluated, targets_evaluated)
            poly_loss.backward()
            optimizer.step()
            # epoch_loss += poly_loss.item()
            epoch_loss += coeff_loss.item()
        # epoch_loss /= total_predictions * N_sensors
        epoch_loss /= total_predictions * targets.size(1)
        train_loss_history[epoch] = epoch_loss
        clr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({"loss": epoch_loss, "lr": clr})
        scheduler.step()
        # if epoch % 10 == 0:
        if test_loader is not None:
            coeff_loss, poly_loss, outputs, targets = test_multionet_poly(
                deeponet, test_loader, sensor_locations
            )
            test_loss_history[epoch] = coeff_loss
            outputs = outputs.reshape(-1, N_timesteps, N_sensors)
            outputs = outputs[:3].transpose(0, 2, 1)
            if epoch == 0:
                targets = targets.reshape(-1, N_timesteps, N_sensors)[:3]
                targets_vis = targets.transpose(0, 2, 1)
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
def train_multionet_chemical(
    data_loader: DataLoader,
    masses: list | None = None,
    branch_input_size: int = 101,
    trunk_input_size: int = 1,
    hidden_size: int = 40,
    branch_hidden_layers: int = 3,
    trunk_hidden_layers: int = 3,
    output_size: int = 100,
    N_outputs: int = 10,
    num_epochs: int = 1000,
    learning_rate: float = 0.001,
    schedule: bool = True,
    test_loader: DataLoader = None,
    N_sensors: int = 101,
    N_timesteps: int = 101,
    architecture: str = "both",
    pretrained_model_path: str | None = None,
    device: str = "cpu",
    visualize: bool = True,
    optuna_trial: optuna.Trial | None = None,
    regularization_factor: float = 0.0,
    massloss_factor: float = 0.0,
) -> tuple:
    """Train a DeepONet model.
    The function instantiates a DeepONet model (with multiple outputs) and trains it using the provided DataLoader.
    Note that it assumes equal number of neurons in each hidden layer of the branch and trunk networks.

    :param data_loader: A DataLoader object.
    :param masses: A list of masses for the chemical species. If None, no mass conservation loss will be used.
    :param branch_input_size: Input size for the branch network.
    :param trunk_input_size: Input size for the trunk network.
    :param hidden_size: Number of hidden units in each layer.
    :param num_epochs: Number of epochs to train for.
    :param branch_hidden_layers: Number of hidden layers in the branch network.
    :param trunk_hidden_layers: Number of hidden layers in the trunk network.
    :param output_size: Number of neurons in the last layer.
    :param N_outputs: Number of outputs.
    :param architecture: Architecture of the MODeepONet model. Can be "branch", "trunk", or "both".
    :param pretrained_model_path: Path to a pretrained model.
    :param device: The device to use for training.
    :param visualize: Whether to visualize the training progress.
    :param optuna_trial: An Optuna trial object for hyperparameter optimization.
    :param regularization_factor: The regularization factor to use for the loss.

    :return: Trained DeepONet model and loss history.
    """
    device = torch.device(device)

    if pretrained_model_path is None:
        if architecture == "both":
            model = MultiONet
        elif architecture == "branch":
            model = MultiONetB
        elif architecture == "trunk":
            model = MultiONetT
        deeponet = model(
            branch_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_input_size,
            hidden_size,
            trunk_hidden_layers,
            output_size,
            N_outputs,
            device,
        )
    else:
        deeponet = load_multionet(
            pretrained_model_path,
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            output_size,
            N_outputs,
            architecture,
        )
        prev_losses = np.load(pretrained_model_path.replace(".pth", "_losses.npz"))
        prev_train_loss = prev_losses["train_loss"]
        prev_test_loss = prev_losses["test_loss"]

    crit = nn.MSELoss(reduction="sum")
    if masses is None:
        criterion = crit
    else:
        weights = (1.0, massloss_factor)
        criterion = mass_conservation_loss(masses, crit, weights, device)

    optimizer = optim.Adam(
        deeponet.parameters(), lr=learning_rate, weight_decay=regularization_factor
    )
    if schedule:
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=0.3, total_iters=num_epochs
        )
    else:
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1, total_iters=num_epochs
        )

    if pretrained_model_path is None:
        train_loss_history = np.zeros(num_epochs)
        test_loss_history = np.zeros(num_epochs)
    else:
        train_loss_history = np.concatenate((prev_train_loss, np.zeros(num_epochs)))
        test_loss_history = np.concatenate((prev_test_loss, np.zeros(num_epochs)))
    output_history = np.zeros((num_epochs, 3, N_sensors, N_timesteps))
    total_predictions = len(data_loader.dataset)

    progress_bar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in progress_bar:
        epoch_loss = 0
        for branch_inputs, trunk_inputs, targets in data_loader:
            if device != "cpu":
                branch_inputs = branch_inputs.to(device)
                trunk_inputs = trunk_inputs.to(device)
                targets = targets.to(device)
                deeponet.to(device)
                # TODO Why does the Deeponet need to be moved to the device in every iteration (and in the testing function also)?
            optimizer.zero_grad()
            outputs = deeponet(branch_inputs, trunk_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= total_predictions * N_outputs
        train_loss_history[epoch] = epoch_loss
        if optuna_trial is not None:
            optuna_trial.report(epoch_loss, epoch)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()
        clr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({"loss": epoch_loss, "lr": clr})
        scheduler.step()
        if test_loader is not None:
            test_loss, outputs, targets = test_deeponet(
                deeponet, test_loader, device, criterion
            )
            test_loss_history[epoch] = test_loss
            outputs = outputs.reshape(-1, N_timesteps, N_outputs)[:3]
            outputs = outputs.transpose(0, 2, 1)
            if epoch == 0:
                targets = targets.reshape(-1, N_timesteps, N_outputs)[:3]
                targets_vis = targets.transpose(0, 2, 1)
            output_history[epoch] = outputs
        if visualize:
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


def mass_conservation_loss(
    masses: list,
    criterion=nn.MSELoss(reduction="sum"),
    weights: tuple = (1, 1),
    device: torch.device = torch.device("cpu"),
):
    """
    Replaces the standard MSE loss with a sum of the standard MSE loss and a mass conservation loss.

    :param masses: A list of masses for the chemical species.
    :param criterion: The loss function to use for the standard loss.
    :param weights: A 2-tuple of weights for the standard loss and the mass conservation loss.
    :param device: The device to use for the loss function.

    :return: A new loss function that includes the mass conservation loss.
    """
    masses = torch.tensor(masses, dtype=torch.float32, device=device)

    def loss(outputs: torch.tensor, targets: torch.tensor) -> torch.tensor:
        """
        Loss function that includes the mass conservation loss.

        :param outputs: The predicted values.
        :param targets: The ground truth values.
        """
        standard_loss = criterion(outputs, targets)

        # Calculate the weighted sum of each chemical quantity for predicted and ground truth,
        # resulting in the total predicted mass and ground truth mass for each sample in the batch
        predicted_mass = torch.sum(outputs * masses, dim=1)
        true_mass = torch.sum(targets * masses, dim=1)

        # Calculate the mass conservation loss as the MSE of the predicted mass vs. true mass
        mass_loss = torch.abs(predicted_mass - true_mass).sum()
        # Sum up the standard MSE loss and the mass conservation loss
        total_loss = weights[0] * standard_loss + weights[1] * mass_loss

        # print(f"Standard loss: {standard_loss.item()}, Mass loss: {mass_loss.item()}")

        return total_loss

    return loss


def poly_eval_torch(p, x):
    """
    Evaluate a polynomial at given points in PyTorch.

    :param p: A tensor of shape [batch_size, n_coeffs] containing the coefficients of the polynomial.
    :param x: A tensor of shape [n_points] containing the x-values at which to evaluate the polynomial.
    :return: A tensor of shape [batch_size, n_points] with the evaluation of the polynomial at the x-values.
    """
    n = p.shape[1]  # Number of coefficients
    x = x.unsqueeze(0).repeat(p.shape[0], 1)  # Shape [batch_size, n_points]
    powers = torch.arange(
        n - 1, -1, -1, device=p.device
    )  # Exponents for each coefficient
    x_powers = x.unsqueeze(-1).pow(powers)  # Shape [batch_size, n_points, n_coeffs]
    return torch.sum(p.unsqueeze(1) * x_powers, dim=-1)  # Polynomial evaluation


def test_deeponet(
    model: T,
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


def test_multionet_poly(
    model: T, data_loader: DataLoader, sensor_locations: np.array
) -> tuple:
    criterion = nn.MSELoss(reduction="sum")
    model.eval()

    # Convert sensor locations to a PyTorch tensor
    sensor_locations_tensor = torch.tensor(
        sensor_locations, dtype=torch.float32, device="cpu"
    )  # Assuming model has .device attribute

    coeff_loss = 0
    poly_loss = 0
    total_predictions = len(data_loader.dataset)  # Number of total predictions
    # Pre-allocate buffers
    predictions_buffer = np.empty((total_predictions, len(sensor_locations)))
    targets_buffer = np.empty((total_predictions, len(sensor_locations)))

    buffer_index = 0

    with torch.no_grad():
        for branch_inputs, trunk_inputs, targets in data_loader:
            outputs = model(branch_inputs, trunk_inputs)
            loss = criterion(outputs, targets)
            coeff_loss += loss.item()

            # Evaluate the polynomial at the sensor locations for both outputs and targets
            polynomial_values = poly_eval_torch(outputs, sensor_locations_tensor)
            target_values = poly_eval_torch(targets, sensor_locations_tensor)

            # Compute the loss
            loss = criterion(polynomial_values, target_values)
            poly_loss += loss.item()

            # Store predictions and targets in buffers
            n = branch_inputs.size(0)  # Number of samples in the batch
            predictions_buffer[buffer_index : buffer_index + n, :] = (
                polynomial_values.cpu().numpy()
            )
            targets_buffer[buffer_index : buffer_index + n, :] = (
                target_values.cpu().numpy()
            )
            buffer_index += n

    # Calculate average loss
    coeff_loss /= total_predictions * targets.size(1)
    poly_loss /= total_predictions * len(sensor_locations)

    return coeff_loss, poly_loss, predictions_buffer, targets_buffer


def test_multionet_polynomial_old(
    model: T, data_loader: DataLoader, output_locations: np.array
) -> tuple:
    """
    Test a DeepONet model.
    This function is specifically designed for the MODeepONet model and assumes that the targets are polynomials.
    It takes the predicted coefficients and calculates the polynomial values at the sensor locations.

    :param model: A DeepONet model (as instantiated using the DeepONet class).
    :param data_loader: A DataLoader object.
    :param sensor_locations: Array of sensor locations in the domain.

    :return: Total loss and predictions.
    """
    criterion = nn.MSELoss(reduction="sum")
    model.eval()

    # Calculate the total number of predictions to pre-allocate the buffer
    total_predictions = len(data_loader.dataset)
    predictions_buffer = np.empty((total_predictions, len(output_locations)))
    targets_buffer = np.empty((total_predictions, len(output_locations)))
    buffer_index = 0

    total_loss = 0
    with torch.no_grad():
        for branch_inputs, trunk_inputs, targets in data_loader:
            outputs = model(branch_inputs, trunk_inputs)
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            for i in range(outputs.shape[0]):
                # Calculate the polynomial values at the sensor locations
                polynomial_values = np.polyval(outputs[i, ::-1], output_locations)
                # Calculate the target polynomial values at the sensor locations
                target_values = np.polyval(targets[i, ::-1], output_locations)
                loss = criterion(
                    torch.tensor(polynomial_values), torch.tensor(target_values)
                )
                total_loss += loss.item()
                # Store predictions in the buffer
                predictions_buffer[buffer_index : buffer_index + 1] = polynomial_values
                targets_buffer[buffer_index : buffer_index + 1] = target_values
                buffer_index += 1

    # Calculate relative error
    total_loss /= total_predictions * len(output_locations)

    return total_loss, predictions_buffer, targets_buffer


def create_dataloader(
    data: list, sensor_points: np.array, batch_size: int = 32, shuffle: bool = False
):
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


def create_dataloader_modified(
    data, sensor_points, batch_size=32, shuffle=False, coeff=False
):
    """
    Create a DataLoader for DeepONet.

    :param data: List of tuples (function_values, antiderivative_values, coefficients).
    :param sensor_points: Array of sensor locations in the domain.
    :param batch_size: Batch size for the DataLoader.
    :return: A DataLoader object.
    """
    branch_inputs = []
    trunk_inputs = []
    targets = []

    if coeff:
        for function_values, antiderivative_values, coefficients in data:
            for i in range(len(sensor_points)):
                branch_inputs.append(coefficients)
                trunk_inputs.append([sensor_points[i]])
                targets.append(antiderivative_values[i])
    else:
        for function_values, antiderivative_values, coefficients in data:
            for i in range(len(sensor_points)):
                branch_inputs.append(function_values)
                trunk_inputs.append([sensor_points[i]])
                targets.append(antiderivative_values[i])

    # Convert to PyTorch tensors
    branch_inputs_tensor = torch.tensor(np.array(branch_inputs), dtype=torch.float32)
    trunk_inputs_tensor = torch.tensor(trunk_inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(branch_inputs_tensor, trunk_inputs_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def subsampling_grid(
    num_sensor_points: int,
    num_timesteps: int,
    fraction: float = 1,
    visualize: bool = True,
):
    """
    Create a subsampling grid based on a given fraction.

    :param num_sensor_points: Length of the sensor points array.
    :param num_timesteps: Length of the timesteps array.
    :param fraction: The fraction of the grid that should be ones.
    :return: A 2D numpy array representing the subsampling grid.
    """
    # Ensure the fraction is within the valid range [0, 1]
    if not 0 <= fraction <= 1:
        raise ValueError("Fraction must be between 0 and 1.")

    # Calculate the probability for a grid point to be a one
    probability_of_one = fraction

    # Create the grid by sampling from {0, 1} based on the calculated probability
    if fraction == 1:
        # If the fraction is 1, we can just create a grid of ones
        grid = np.ones((num_sensor_points, num_timesteps), dtype=int)
    else:
        # Otherwise, we need to sample from {0, 1} with the given probability
        grid = np.random.choice(
            [0, 1],
            size=(num_sensor_points, num_timesteps),
            p=[1 - probability_of_one, probability_of_one],
        )

    # Visualize the grid
    if visualize:
        frac = np.sum(grid) / (num_sensor_points * num_timesteps)
        plt.imshow(grid)
        plt.title(f"Subsampling grid with fraction = {frac:.3f}")
        plt.show()

    return grid


def create_dataloader_2D(
    functions: np.array,
    sensor_points: np.array,
    timesteps: np.array,
    subsampling_grid=[],
    batch_size=32,
    shuffle=False,
):
    # TODO: What does the optional subsampling_grid input do exactly?
    """
    Create a DataLoader with optional subsampling on a grid for time-dependent data for DeepONet.

    :param functions: 3D numpy array with shape (num_samples, len(sensor_points), len(timesteps))
                      representing the function values over time.
    :param sensor_points: 1D numpy array of sensor locations in the domain.
    :param timesteps: 1D numpy array of timesteps.
    :param fraction: Fraction of the grid points to sample.
    :param batch_size: Batch size for the DataLoader.
    :param shuffle: Whether to shuffle the data.
    :return: A DataLoader object.
    """
    # Create subsampling grid

    branch_inputs = []
    trunk_inputs = []
    targets = []

    if len(subsampling_grid) == 0:
        subsampling_grid = np.ones((len(sensor_points), len(timesteps)), dtype=int)

    # Iterate through the grid to select the samples
    for sample in functions:
        for i, sensor_point in enumerate(sensor_points):
            for j, time in enumerate(timesteps):
                if subsampling_grid[i, j] == 1:
                    branch_inputs.append(sample[:, 0])  # Initial state of the function
                    trunk_inputs.append([sensor_point, time])
                    targets.append(
                        sample[i, j]
                    )  # Function value at this sensor point and time

    # Convert to PyTorch tensors
    branch_inputs_tensor = torch.tensor(np.array(branch_inputs), dtype=torch.float32)
    trunk_inputs_tensor = torch.tensor(np.array(trunk_inputs), dtype=torch.float32)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(branch_inputs_tensor, trunk_inputs_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_dataloader_2D_frac(
    functions,
    sensor_points,
    timesteps,
    fraction=1,
    batch_size=32,
    shuffle=False,
):
    """
    Create a DataLoader with optional fractional subsampling for time-dependent data for DeepONet.

    :param functions: 3D numpy array with shape (num_samples, len(sensor_points), len(timesteps))
                      representing the function values over time.
    :param sensor_points: 1D numpy array of sensor locations in the domain.
    :param timesteps: 1D numpy array of timesteps.
    :param fraction: Fraction of the grid points to sample.
    :param batch_size: Batch size for the DataLoader.
    :param shuffle: Whether to shuffle the data.
    :return: A DataLoader object.
    """
    # Create subsampling grid

    branch_inputs = []
    trunk_inputs = []
    targets = []

    # Iterate through the grid to select the samples
    if fraction == 1:
        for sample in functions:
            for i, sensor_point in enumerate(sensor_points):
                for j, time in enumerate(timesteps):
                    branch_inputs.append(sample[:, 0])  # Initial state of the function
                    trunk_inputs.append([sensor_point, time])
                    targets.append(
                        sample[i, j]
                    )  # Function value at this sensor point and time
    else:
        for sample in functions:
            for i, sensor_point in enumerate(sensor_points):
                for j, time in enumerate(timesteps):
                    if np.random.uniform(0, 1) < fraction:
                        branch_inputs.append(sample[:, 0])
                        trunk_inputs.append([sensor_point, time])
                        targets.append(sample[i, j])

    # Convert to PyTorch tensors
    branch_inputs_tensor = torch.tensor(np.array(branch_inputs), dtype=torch.float32)
    trunk_inputs_tensor = torch.tensor(np.array(trunk_inputs), dtype=torch.float32)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(branch_inputs_tensor, trunk_inputs_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_dataloader_chemicals(
    data,
    timesteps,
    fraction=1,
    batch_size=32,
    shuffle=False,
):
    """
    Create a DataLoader with optional fractional subsampling for chemical evolution data for DeepONet.

    :param data: 3D numpy array with shape (num_samples, len(timesteps), num_chemicals)
    :param timesteps: 1D numpy array of timesteps.
    :param fraction: Fraction of the grid points to sample.
    :param batch_size: Batch size for the DataLoader.
    :param shuffle: Whether to shuffle the data.
    :return: A DataLoader object.
    """
    # Create subsampling grid

    branch_inputs = []
    trunk_inputs = []
    targets = []

    # Iterate through the grid to select the samples
    if fraction == 1:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                branch_inputs.append(data[i, 0, :])
                trunk_inputs.append([timesteps[j]])
                targets.append(data[i, j, :])
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.random.uniform(0, 1) < fraction:
                    branch_inputs.append(data[i, :, 0])
                    trunk_inputs.append([timesteps[j]])
                    targets.append(data[i, :, j])

    # Convert to PyTorch tensors
    branch_inputs_tensor = torch.tensor(np.array(branch_inputs), dtype=torch.float32)
    trunk_inputs_tensor = torch.tensor(np.array(trunk_inputs), dtype=torch.float32)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(branch_inputs_tensor, trunk_inputs_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_dataloader_2D_frac_coeff(
    functions,
    coefficients,
    sensor_points,
    timesteps,
    fraction=1,
    batch_size=32,
    shuffle=False,
):
    """
    Create a DataLoader with optional fractional subsampling for time-dependent data for DeepONet.
    This DataLoader returns the coefficients of the function instead of the function values.

    :param functions: 3D numpy array with shape (num_samples, len(sensor_points), len(timesteps))
                      representing the function values over time.
    :param coefficients: 3D numpy array with shape (num_samples, num_coefficients, len(timesteps))
    :param sensor_points: 1D numpy array of sensor locations in the domain.
    :param timesteps: 1D numpy array of timesteps.
    :param fraction: Fraction of the grid points to sample.
    :param batch_size: Batch size for the DataLoader.
    :param shuffle: Whether to shuffle the data.
    :return: A DataLoader object.
    """
    # Create subsampling grid

    branch_inputs = []
    trunk_inputs = []
    targets = []

    # Iterate through the grid to select the samples
    if fraction == 1:
        for i, sample in enumerate(functions):
            for j, time in enumerate(timesteps):
                branch_inputs.append(sample[:, 0])
                trunk_inputs.append([time])
                targets.append(
                    coefficients[i, :, j]
                )  # Function value at this sensor point and time
    else:
        for i, sample in enumerate(functions):
            for j, time in enumerate(timesteps):
                if np.random.uniform(0, 1) < fraction:
                    branch_inputs.append(sample[:, 0])
                    trunk_inputs.append([time])
                    targets.append(coefficients[i, :, j])

    # Convert to PyTorch tensors
    branch_inputs_tensor = torch.tensor(np.array(branch_inputs), dtype=torch.float32)
    trunk_inputs_tensor = torch.tensor(np.array(trunk_inputs), dtype=torch.float32)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(branch_inputs_tensor, trunk_inputs_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
        device,
    )

    # Load the state dictionary
    state_dict = torch.load(path_to_state_dict)
    deeponet.load_state_dict(state_dict)

    return deeponet


def load_multionet(
    path_to_state_dict: str,
    branch_input_size: int,
    trunk_input_size: int,
    hidden_size: int,
    branch_hidden_layers: int,
    trunk_hidden_layers: int,
    output_neurons: int,
    N_outputs: int,
    architecture: str = "both",
    device: str = "cpu",
):
    """
    Load a DeepONet model from a saved state dictionary.

    :param path_to_state_dict: Path to the saved state dictionary.
    :param branch_input_size: Input size for the branch network.
    :param trunk_input_size: Input size for the trunk network.
    :param hidden_size: Number of hidden units in each layer.
    :param num_hidden_layers: Number of hidden layers in each network.
    :param architecture: Architecture of the MODeepONet model. Can be "branch", "trunk", or "both".
    :return: Loaded DeepONet model.
    """
    # Recreate the model architecture
    if architecture == "both":
        model = MultiONet
    elif architecture == "branch":
        model = MultiONetB
    elif architecture == "trunk":
        model = MultiONetT
    deeponet = model(
        branch_input_size,
        hidden_size,
        branch_hidden_layers,
        trunk_input_size,
        hidden_size,
        trunk_hidden_layers,
        output_neurons,
        N_outputs,
        device,
    )
    # Load the state dictionary
    state_dict = torch.load(path_to_state_dict)
    deeponet.load_state_dict(state_dict)

    return deeponet
