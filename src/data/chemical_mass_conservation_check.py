import numpy as np

from osu_chemicals import masses
from plotting import plot_mass_conservation
from utils import load_chemical_data


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
