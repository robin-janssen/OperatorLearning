from chemicals import masses
from utils import load_chemical_data
import matplotlib.pyplot as plt
import numpy as np


def plot_mass_conservation(ground_truth, masses, num_examples=5):
    # Ensure masses is a numpy array to utilize broadcasting
    masses = np.array(masses)

    plt.figure(figsize=(12, 8))

    # Calculate and plot the total mass for a selection of examples
    for i in range(num_examples):
        # Calculate the total mass for each timestep
        total_mass = np.sum(ground_truth[i] * masses, axis=1)  # Sum over chemicals

        # Plot
        plt.plot(total_mass, label=f"Example {i+1}")

    plt.xlabel("Timestep")
    plt.ylabel("Total Mass")
    plt.title("Total Mass Conservation Over Time")
    plt.legend()
    plt.show()


data = load_chemical_data("data/dataset1000")
# data = load_chemical_data("data/dataset1000")
data_shape = data.shape
print(f"Data shape: {data_shape}")

# Use only the amount of each chemical, not the gradients
data = data[:, :, :29]

N_timesteps = data.shape[1]
# Check to see if the mass is conserved over time

plot_mass_conservation(data, masses, num_examples=5)

data = np.power(10, data)

plot_mass_conservation(data, masses, num_examples=5)
