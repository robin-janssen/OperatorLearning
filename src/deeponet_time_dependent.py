import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# from datagen import generate_decaying_sines, surface_plot
from datagen import generate_oscillating_sines, heatmap_plot, plot_functions_only
from deeponet_training import train_deeponet, plot_losses, load_deeponet, test_deeponet
from utils import save_model


def subsampling_grid(len_sensor_points, len_timesteps, fraction=1, visualize=True):
    """
    Create a subsampling grid based on a given fraction.

    :param len_sensor_points: Length of the sensor points array.
    :param len_timesteps: Length of the timesteps array.
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
        grid = np.ones((len_sensor_points, len_timesteps), dtype=int)
    else:
        # Otherwise, we need to sample from {0, 1} with the given probability
        grid = np.random.choice([0, 1], size=(len_sensor_points, len_timesteps), p=[1 - probability_of_one, probability_of_one])

    # Visualize the grid
    if visualize:
        frac = np.sum(grid) / (len_sensor_points * len_timesteps)
        plt.imshow(grid)
        plt.title(f"Subsampling grid with fraction = {frac:.3f}")
        plt.show()

    return grid


def create_dataloader_2D(functions, sensor_points, timesteps, subsampling_grid=[], batch_size=32, shuffle=False):
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
                    targets.append(sample[i, j])  # Function value at this sensor point and time

    # Convert to PyTorch tensors
    branch_inputs_tensor = torch.tensor(np.array(branch_inputs), dtype=torch.float32)
    trunk_inputs_tensor = torch.tensor(np.array(trunk_inputs), dtype=torch.float32)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(branch_inputs_tensor, trunk_inputs_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_dataloader_2D_frac(functions, sensor_points, timesteps, fraction=1, batch_size=32, shuffle=False):
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
                    targets.append(sample[i, j])  # Function value at this sensor point and time
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


if __name__ == "__main__":
    TRAIN = False
    branch_input_size = 41
    trunk_input_size = 2
    hidden_size = 40
    branch_hidden_layers = 3
    trunk_hidden_layers = 3
    num_epochs = 100
    learning_rate = 3e-4
    N_timesteps = 41
    decay_rate = 1
    fraction = 0.25
    num_samples_train = 800
    num_samples_test = 100

    sensor_locations = np.linspace(0, 1, branch_input_size)
    # surface_plot(sensor_locations, timesteps, polynomials, num_samples_to_plot=3)
    # train_data, amplitudes, frequencies, timesteps = generate_decaying_sines(
    #     sensor_locations=sensor_locations,
    #     decay_rate=decay_rate,
    #     num_samples=num_samples_train,
    #     N_steps=N_timesteps)
    # test_data, _, _, _ = generate_decaying_sines(
    #     sensor_locations=sensor_locations,
    #     decay_rate=decay_rate,
    #     num_samples=num_samples_test,
    #     N_steps=N_timesteps)
    train_data, amplitudes, frequencies, timesteps = generate_oscillating_sines(
        sensor_locations=sensor_locations,
        rate=decay_rate,
        num_samples=num_samples_train,
        N_steps=N_timesteps)
    test_data, _, _, _ = generate_oscillating_sines(
        sensor_locations=sensor_locations,
        rate=decay_rate,
        num_samples=num_samples_test,
        N_steps=N_timesteps)
    # surface_plot(sensor_locations, timesteps, sines, num_samples_to_plot=3, title="Decaying sines")
    heatmap_plot(sensor_locations, timesteps, train_data, num_samples_to_plot=3, title="Decaying sines")
    print("Data generated.")
    plot_functions_only(train_data[:, :, 0], sensor_locations, 100)

    # grid = subsampling_grid(len(sensor_locations), len(timesteps), fraction=fraction, visualize=True)

    # Create a DataLoader for DeepONet
    dataloader = create_dataloader_2D_frac(train_data, sensor_locations, timesteps, batch_size=32, shuffle=True, fraction=fraction)
    dataloader_test = create_dataloader_2D_frac(test_data, sensor_locations, timesteps, batch_size=32, shuffle=False)
    print("DataLoader created.")

    # Now we need to train/load the DeepONet
    if TRAIN:
        # Train the DeepONet
        vanilla_deeponet, loss, test_loss = train_deeponet(
            dataloader,
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            num_epochs,
            learning_rate,
            test_loader=dataloader_test)
        # Plot the loss
        plot_losses((loss, np.repeat(test_loss, 10)), ('train loss', 'test loss'))
        # Save the trained DeepONet
        save_model(vanilla_deeponet, "deeponet_time_dependent", {
            "branch_input_size": branch_input_size,
            "trunk_input_size": trunk_input_size,
            "hidden_size": hidden_size,
            "branch_hidden_layers": branch_hidden_layers,
            "trunk_hidden_layers": trunk_hidden_layers,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "N_timesteps": N_timesteps,
            "decay_rate": decay_rate,
            "fraction": fraction,
            "num_samples_train": num_samples_train,
            "num_samples_test": num_samples_test,
        })
    else:
        # Load the DeepONet
        vanilla_deeponet = load_deeponet(
            "models/31-01/deeponet_time_dependent.pth",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers)

    # Test the DeepONet
    total_loss, predictions = test_deeponet(vanilla_deeponet, dataloader_test)
    print(f"Total loss: {total_loss:.3E}")

    # Plot the results
    predictions = predictions.reshape(-1, len(sensor_locations), len(timesteps))

    # surface_plot(sensor_locations, timesteps, sines, 3, predictions, title="DeepONet results")
    heatmap_plot(sensor_locations, timesteps, test_data, 5, predictions, title="DeepONet results")

    print("Done.")
