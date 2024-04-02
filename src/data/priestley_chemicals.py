import os
import numpy as np


def load_chemicals_priestley(data_folder):
    # Known final array shape
    n_particles_per_file = 250
    n_files = 40
    n_particles_total = (
        n_particles_per_file * n_files
    )  # Total particles across all files
    n_timesteps = 128  # Number of timesteps per particle
    n_columns = 217  # Number of quantities

    # Allocate memory for the entire dataset
    full_data_array = np.zeros((n_particles_total, n_timesteps, n_columns))

    # Loop through the 40 files
    for i in range(1, 41):
        file_name = f"trainingdata_{i}.out"
        file_path = os.path.join(data_folder, file_name)

        with open(file_path, "r") as file:
            # Read the first line to skip it
            file.readline()

            # Calculate the index offset for this file
            index_offset = (i - 1) * n_particles_per_file

            # Read and store the data
            for timestep in range(n_timesteps):
                for particle in range(n_particles_per_file):
                    line = file.readline().strip()
                    data = list(map(float, line.split(",")))
                    # Store the data
                    full_data_array[index_offset + particle, timestep, :] = data

    return full_data_array


def load_and_save_chemicals_priestley(data_folder):
    # Load the chemical data
    data = load_chemicals_priestley(data_folder)

    # Print the shape of the data
    print(f"Loaded chemical data with shape: {data.shape}")

    # Save the data to a file
    save_path = os.path.join(data_folder, "chemicals.npy")
    np.save(save_path, data)

    print(f"Saved chemical data to: {save_path}")


def shuffle_and_split_data(data_folder, train_fraction=0.8):
    data = np.load(os.path.join(data_folder, "chemicals.npy"))
    print(f"Loaded chemical data with shape: {data.shape}")

    # Shuffle the individual datapoints (along the first axis)
    np.random.shuffle(data)
    print("Data shuffled.")

    # Split the data into training and testing sets
    split_index = int(train_fraction * data.shape[0])
    train_data = data[:split_index]
    test_data = data[split_index:]
    print(
        f"Data split into training and testing sets: {train_data.shape}, {test_data.shape}"
    )

    # Save the training and testing data
    train_save_path = os.path.join(data_folder, "chemicals_train.npy")
    test_save_path = os.path.join(data_folder, "chemicals_test.npy")
    np.save(train_save_path, train_data)
    np.save(test_save_path, test_data)
    print(f"Training data saved to: {train_save_path}")


def run(args):

    data_folder = "data/chemicals_priestley"

    # Initial loading of the data (only needed once to create the .npy file)
    # load_and_save_chemicals_priestley(data_folder)

    # Load, shuffle, and save the data
    shuffle_and_split_data(data_folder, train_fraction=0.8)
