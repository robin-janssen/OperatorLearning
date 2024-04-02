from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset


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
    normalize=False,
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

    if normalize:
        branch_inputs_tensor = (
            branch_inputs_tensor - branch_inputs_tensor.mean()
        ) / branch_inputs_tensor.std()
        trunk_inputs_tensor = (
            trunk_inputs_tensor - trunk_inputs_tensor.mean()
        ) / trunk_inputs_tensor.std()
        targets_tensor = (targets_tensor - targets_tensor.mean()) / targets_tensor.std()

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(branch_inputs_tensor, trunk_inputs_tensor, targets_tensor)

    def worker_init_fn(worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        np.random.seed(np_seed)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, worker_init_fn=worker_init_fn
    )


def create_dataloader_2D_coeff(
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
