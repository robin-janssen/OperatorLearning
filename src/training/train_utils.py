import functools
import time
import os
import yaml

import numpy as np
import torch
import torch.nn as nn

from models import OperatorNetworkType
from utils import create_date_based_directory


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


def time_execution(func):
    """
    Decorator to time the execution of a function and store the duration
    as an attribute of the function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.duration = end_time - start_time
        print(f"{func.__name__} executed in {wrapper.duration:.2f} seconds.")
        return result

    wrapper.duration = None
    return wrapper


def save_model(
    model: OperatorNetworkType,
    model_name,
    hyperparameters,
    subfolder="models",
    train_loss: np.ndarray | None = None,
    test_loss: np.ndarray | None = None,
):
    """
    Save the trained model and hyperparameters.

    :param model: The trained model.
    :param hyperparameters: Dictionary containing hyperparameters.
    :param base_dir: Base directory for saving the model.
    """
    # Create a directory based on the current date
    model_dir = create_date_based_directory(subfolder=subfolder)

    # Save the model state dict
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)

    # Save hyperparameters as a YAML file
    hyperparameters_path = os.path.join(model_dir, f"{model_name}.yaml")
    with open(hyperparameters_path, "w") as file:
        yaml.dump(hyperparameters, file)

    if train_loss is not None and test_loss is not None:
        # Save the losses as a numpy file
        losses_path = os.path.join(model_dir, f"{model_name}_losses.npz")
        np.savez(losses_path, train_loss=train_loss, test_loss=test_loss)

    print(f"Model, losses and hyperparameters saved to {model_dir}")
