# MultioNet script for predicting the time evolution of free cooling spectra.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from data import load_fc_spectra, spectrum


def spectral_fit(p, a, b, c):
    return spectrum(p, a, b, c, A=1.0, p0=1.0, eps=1e-6)


def fit_fc_spectra(data):
    spectra_list = []
    for i in range(data.shape[1]):
        p_values = data[0, i, 0]
        # Obtain indices of nonzero values
        indices = np.where(p_values != 0)[0]
        # Obtain the corresponding values
        p_values = p_values[indices]
        # print(p_values)
        for j in range(data.shape[0]):
            e_values = data[j, i, 1][indices]
            # print(e_values[:10])
            # Fit the spectrum
            popt, _ = curve_fit(spectral_fit, p_values, e_values)
            spectra_list.append(popt)
            # print(popt)
    return np.array(spectra_list)


def plot_example_spectra(data, spectra):

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    p_values = np.logspace(1e-2, 1e3, 100)
    for i in range(4):
        coeffs = spectra[i]
        print(coeffs)
        e_values = spectral_fit(p_values, *coeffs)
        ax[i // 2, i % 2].plot(e_values, p_values, label="Fit")
        ax[i // 2, i % 2].plot(data[i, 5, 1], data[i, 5, 0], label="Data")
        ax[i // 2, i % 2].set_xscale("log")
        ax[i // 2, i % 2].set_yscale("log")
        ax[i // 2, i % 2].set_title(f"Example spectrum {i}")
    plt.show()


def run(args):
    data = load_fc_spectra("spectral-data-free-cooling-large.pkl")
    data = data.reshape(-1, 11, 2, 100, order="F")
    # data = np.log10(data)
    print(data[2, 5, 0, :40])
    # print(data[0, 0, 1, :])
    print(data[4, 5, 0, :40])
    # print(data[0, 1, 1, :])
    print(data[10, 5, 0, :40])

    data_vis = data.transpose(0, 1, 3, 2)
    data_vis = data_vis.reshape(-1, 1100, 2)
    # data_transformed = np.where(np.isnan(data_vis), 2, np.where(data_vis == 0, 1, 0))
    # # Plotting the first channel with colorbar
    # fig, ax = plt.subplots()
    # im = ax.imshow(data_transformed[:, :, 1], aspect="auto")
    # plt.colorbar(im, ax=ax)  # Add a colorbar to the current plot
    # plt.show()

    # # Plotting the second channel with colorbar
    # fig, ax = plt.subplots()
    # im = ax.imshow(data_transformed[:, :, 1], aspect="auto")
    # plt.colorbar(im, ax=ax)  # Add a colorbar to the current plot
    # plt.show()

    # Fit the spectra
    spectra = fit_fc_spectra(data)

    # Make some exemplary plots

    print("Done!")
