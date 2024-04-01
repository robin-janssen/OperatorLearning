# MultioNet script for predicting the time evolution of free cooling spectra.

import matplotlib.pyplot as plt
import numpy as np

from data import load_fc_spectra


def run(args):
    data = load_fc_spectra("spectral-data-free-cooling-large.pkl")
    data = data.reshape(-1, 11, 2, 100, order="F")
    # data = np.log10(data)
    print(data[0, 0, 0, :])
    print(data[0, 1, 0, :])
    print(data[0, 2, 0, :])
    print(data[0, 3, 0, :])

    data_vis = data.transpose(0, 1, 3, 2)
    data_vis = data_vis.reshape(-1, 1100, 2)
    data_transformed = np.where(np.isnan(data_vis), 2, np.where(data_vis == 0, 1, 0))
    # Plotting the first channel with colorbar
    fig, ax = plt.subplots()
    im = ax.imshow(data_transformed[:, :, 0], aspect="auto")
    plt.colorbar(im, ax=ax)  # Add a colorbar to the current plot
    plt.show()

    # Plotting the second channel with colorbar
    fig, ax = plt.subplots()
    im = ax.imshow(data_transformed[:, :, 1], aspect="auto")
    plt.colorbar(im, ax=ax)  # Add a colorbar to the current plot
    plt.show()

    print("Done!")
