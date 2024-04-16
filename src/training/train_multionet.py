from __future__ import annotations

import numpy as np
from tqdm import tqdm
import optuna
import dataclasses

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import (
    MultiONet,
    MultiONetB,
    MultiONetT,
    initialize_weights,
)
from models.deeponet import OperatorNetworkType
from plotting import streamlit_visualization_history
from .train_deeponet import test_deeponet
from .train_utils import (
    poly_eval_torch,
    mass_conservation_loss,
    time_execution,
    save_model,
    setup_optimizer_and_scheduler,
    setup_criterion,
    setup_losses,
    training_step,
)
from utils import get_project_path


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
    device: str = "cpu",
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
    :param device: The device to use for training.

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
            device,
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
            device,
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
            device,
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
    device: str = "cpu",
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
        device,
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
    conf: type[dataclasses.dataclass],
    data_loader: DataLoader,
    test_loader: DataLoader = None,
) -> tuple:
    """Train a DeepONet model.
    The function instantiates a DeepONet model (with multiple outputs) and trains it using the provided DataLoader.
    Note that it assumes equal number of neurons in each hidden layer of the branch and trunk networks.

    Args:
        conf (type[dataclass]): A dataclass object containing the training configuration. It should have the following attributes:
            - 'masses' (List[float] | None): List of masses for the chemical species. If None, no mass conservation loss will be used.
            - 'branch_input_size' (int): Input size for the branch network.
            - 'trunk_input_size' (int): Input size for the trunk network.
            - 'hidden_size' (int): Number of hidden units in each layer.
            - 'branch_hidden_layers' (int): Number of hidden layers in the branch network.
            - 'trunk_hidden_layers' (int): Number of hidden layers in the trunk network.
            - 'output_size' (int): Number of neurons in the last layer.
            - 'N_outputs' (int): Number of outputs.
            - 'num_epochs' (int): Number of epochs to train for.
            - 'learning_rate' (float): Learning rate for the optimizer.
            - 'schedule' (bool): Whether to use a learning rate schedule.
            - 'N_sensors' (int): Number of sensor locations.
            - 'N_timesteps' (int): Number of timesteps.
            - 'architecture' (str): Architecture type, e.g., 'both', 'branch', or 'trunk'.
            - 'pretrained_model_path' (str | None): Path to a pretrained model. None if training from scratch.
            - 'device' (str): The device to use for training, e.g., 'cpu', 'cuda:0'.
            - 'use_streamlit' (bool): Whether to use Streamlit for live visualizations.
            - 'optuna_trial' (optuna.Trial | None): Optuna trial object for hyperparameter optimization. None if not using Optuna.
            - 'regularization_factor' (float): Regularization factor for the loss function.
            - 'massloss_factor' (float): Weight of the mass conservation loss component.
        data_loader (DataLoader): A DataLoader object containing the training data.
        test_loader (DataLoader): A DataLoader object containing the test data.

    :return: Trained DeepONet model and loss history.
    """
    device = torch.device(conf.device)

    deeponet, train_loss, test_loss = load_multionet(conf, device)

    criterion = setup_criterion(conf)

    optimizer, scheduler = setup_optimizer_and_scheduler(conf, deeponet)

    train_loss_hist, test_loss_hist = setup_losses(conf, train_loss, test_loss)
    output_hist = np.zeros((conf.num_epochs, 3, conf.N_sensors, conf.N_timesteps))

    progress_bar = tqdm(range(conf.num_epochs), desc="Training Progress")
    for epoch in progress_bar:
        train_loss_hist[epoch] = training_step(
            deeponet, data_loader, criterion, optimizer, device, conf.N_outputs
        )
        if conf.optuna_trial is not None:
            conf.optuna_trial.report(train_loss_hist[epoch], epoch)
            if conf.optuna_trial.should_prune():
                raise optuna.TrialPruned()
        clr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({"loss": train_loss_hist[epoch], "lr": clr})
        scheduler.step()
        if test_loader is not None:
            test_loss_hist[epoch], outputs, targets = test_deeponet(
                deeponet, test_loader, device, criterion, conf.N_timesteps
            )
            output_hist[epoch] = outputs[:3]
            if epoch == 0:
                targets_vis = targets[:3]

        if conf.use_streamlit:
            streamlit_visualization_history(
                train_loss_hist[: epoch + 1],
                test_loss_hist[: epoch + 1],
                output_hist,
                targets_vis,
                epoch,
            )

    if test_loader is None:
        test_loss_hist = None

    return deeponet, train_loss_hist, test_loss_hist


@time_execution
def train_multionet_chemical_2(
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
    schedule: bool = False,
    test_loader: DataLoader = None,
    N_sensors: int = 101,
    N_timesteps: int = 101,
    architecture: str = "both",
    pretrained_model_path: str | None = None,
    device: str = "cpu",
    use_streamlit: bool = True,
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
    :param use_streamlit: Whether to use Streamlit for visualization.
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
        if use_streamlit:
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
def train_multionet_chemical_remote(
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
    schedule: bool = False,
    test_loader: DataLoader = None,
    N_sensors: int = 101,
    N_timesteps: int = 101,
    architecture: str = "both",
    pretrained_model_path: str | None = None,
    device: str = "cpu",
    device_id: int = 0,
    use_streamlit: bool = False,
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
    :param device_id: The GPU ID to use for training.
    :param use_streamlit: Whether to use Streamlit for visualization.
    :param optuna_trial: An Optuna trial object for hyperparameter optimization.
    :param regularization_factor: The regularization factor to use for the loss.

    :return: Trained DeepONet model and loss history.
    """
    print(f"Starting training on GPU {device_id}")
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
        deeponet.apply(initialize_weights)
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
        if use_streamlit:
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
def train_multionet_chemical_cosann(
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
    test_loader: DataLoader = None,
    N_sensors: int = 101,
    N_timesteps: int = 101,
    architecture: str = "both",
    pretrained_model_path: str | None = None,
    device: str = "cpu",
    use_streamlit: bool = True,
    optuna_trial: optuna.Trial | None = None,
    regularization_factor: float = 0.0,
    massloss_factor: float = 0.0,
    cycles: int = 1,
) -> tuple:
    """Train a DeepONet model.
    The function instantiates a DeepONet model (with multiple outputs) and trains it using the provided DataLoader.
    Note that it assumes equal number of neurons in each hidden layer of the branch and trunk networks.
    In this version of the train function, the learning rate schedule is a cosine annealing schedule.
    Additionally, we will save the models state dictionary each time before the learning rate jumps to a higher value.

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
    :param use_streamlit: Whether to use Streamlit for visualization.
    :param optuna_trial: An Optuna trial object for hyperparameter optimization.
    :param regularization_factor: The regularization factor to use for the loss.
    :param massloss_factor: The weight of the mass conservation loss.
    :param cycles: The number of cosine annealing cycles to use.

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

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=num_epochs // cycles, T_mult=1, eta_min=0.001 * learning_rate
    )
    lr_history = np.zeros(num_epochs)

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
            epoch_loss += loss.item()
        epoch_loss /= total_predictions * N_outputs
        train_loss_history[epoch] = epoch_loss
        if optuna_trial is not None:
            optuna_trial.report(epoch_loss, epoch)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()
        clr = optimizer.param_groups[0]["lr"]
        lr_history[epoch] = clr
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
        if use_streamlit:
            streamlit_visualization_history(
                train_loss_history[: epoch + 1],
                test_loss_history[: epoch + 1],
                output_history,
                targets_vis,
                epoch,
            )
        if epoch % (num_epochs // cycles) == 0 and epoch > 0:
            save_model(
                deeponet,
                "deeponet_chemical_epoch_{}.pth".format(epoch),
                {
                    "branch_input_size": branch_input_size,
                    "trunk_input_size": trunk_input_size,
                    "hidden_size": hidden_size,
                    "branch_hidden_layers": branch_hidden_layers,
                    "trunk_hidden_layers": trunk_hidden_layers,
                    "num_epochs": num_epochs,
                    "current_epoch": epoch,
                    "current cycle": epoch // (num_epochs // cycles),
                    "learning_rate": learning_rate,
                    "N_timesteps": N_timesteps,
                    "architecture": architecture,
                },
                train_loss=epoch_loss,
                test_loss=test_loss,
            )
        optimizer.step()

    if test_loader is not None:
        return deeponet, train_loss_history, test_loss_history, lr_history
    else:
        return deeponet, train_loss_history


def test_multionet_poly(
    model: OperatorNetworkType, data_loader: DataLoader, sensor_locations: np.array
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
    model: OperatorNetworkType, data_loader: DataLoader, output_locations: np.array
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


def load_multionet(
    conf: type[dataclasses.dataclass] | dict,
    device: str = "cpu",
    model_path: str | None = None,
) -> OperatorNetworkType | tuple:
    """
    Load a MultiONet model from a saved state dictionary.
    If the path_to_state_dict is None, the function will return a new MultiONet model.

    Args:
        conf (type[dataclass]): A dataclass object containing the training configuration. It should have the following attributes:
            - 'pretrained_model_path' (str): Path to the saved state dictionary.
            - 'branch_input_size' (int): Input size for the branch network.
            - 'trunk_input_size' (int): Input size for the trunk network.
            - 'hidden_size' (int): Number of hidden units in each layer.
            - 'branch_hidden_layers' (int): Number of hidden layers in the branch network.
            - 'trunk_hidden_layers' (int): Number of hidden layers in the trunk network.
            - 'output_neurons' (int): Number of neurons in the last layer.
            - 'N_outputs' (int): Number of outputs.
            - 'architecture' (str): Architecture type, e.g., 'both', 'branch', or 'trunk'.
            - 'device' (str): The device to use for the model, e.g., 'cpu', 'cuda:0'.
        device (str): The device to use for the model.
        model_path (str): Path to the saved state dictionary. Should have the extension '.pth'.

    Returns:
        deeponet: Loaded DeepONet model.
    """
    # If the conf is a dataclass, convert it to a dictionary
    if dataclasses.is_dataclass(conf):
        conf = dataclasses.asdict(conf)
    # Instantiate the model
    if conf["architecture"] == "both":
        model = MultiONet
    elif conf["architecture"] == "branch":
        model = MultiONetB
    elif conf["architecture"] == "trunk":
        model = MultiONetT
    deeponet = model(
        conf["branch_input_size"],
        conf["hidden_size"],
        conf["branch_hidden_layers"],
        conf["trunk_input_size"],
        conf["hidden_size"],
        conf["trunk_hidden_layers"],
        conf["output_neurons"],
        conf["N_outputs"],
        device,
    )

    # Load the state dictionary
    if (
        "pretrained_model_path" not in conf or conf["pretrained_model_path"] is None
    ) and model_path is None:
        prev_train_loss = None
        prev_test_loss = None
    else:
        if model_path is None:
            model_path = conf["pretrained_model_path"]
        absolute_path = get_project_path(model_path)
        state_dict = torch.load(absolute_path + ".pth", map_location=device)
        deeponet.load_state_dict(state_dict)
        prev_losses = np.load(absolute_path + "_losses.npz")
        prev_train_loss = prev_losses["train_loss"]
        prev_test_loss = prev_losses["test_loss"]

    return deeponet, prev_train_loss, prev_test_loss


def load_multionet_2(
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
    absolute_path = get_project_path(path_to_state_dict)
    state_dict = torch.load(absolute_path, map_location=device)
    deeponet.load_state_dict(state_dict)

    return deeponet
