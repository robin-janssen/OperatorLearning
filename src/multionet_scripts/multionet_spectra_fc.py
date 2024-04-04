# MultioNet script for predicting the time evolution of free cooling spectra.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from data import load_fc_spectra, spectrum


def spectral_fit(p, a, b, c):
    return spectrum(p, a, b, c, A=1.0, p0=1.0, eps=1e-6)


def fit_fc_spectra(data):
    """This function expects input data of shape [n_sl, n_sh, n_w, n_timesteps, 2, 100]"""
    n_sl, n_sh, n_w, n_timesteps, _, _ = data.shape
    spectral_coeffs = np.zeros((n_sl, n_sh, n_w, n_timesteps, 3))
    counter = 0
    for sl in range(n_sl):
        for sh in range(n_sh):
            for w in range(n_w):
                for t in range(n_timesteps):
                    counter += 1
                    if counter % 100 == 0:
                        print(f"Processing spectrum {counter}...")
                    p_values = data[sl, sh, w, t, 0]
                    # Obtain indices of nonzero values
                    indices = np.where(p_values != 0)[0]
                    # Obtain the corresponding values
                    p_values = p_values[indices]
                    e_values = data[sl, sh, w, t, 1][indices]
                    # Fit the spectrum
                    bounds = ([-15.0, -15.0, 0], [15.0, 0, np.inf])
                    popt, _ = curve_fit(
                        spectral_fit, p_values, e_values, bounds=bounds, maxfev=2000
                    )
                    spectral_coeffs[sl, sh, w, t] = popt
    return spectral_coeffs


def fit_fc_spectra_2(data):
    """This function expects input data of shape [n_samples, n_timesteps, 2, 100]"""
    spectra_list = []
    for j in range(data.shape[0]):
        if j % 100 == 0:
            print(f"Processing spectrum {j}...")
        for i in range(data.shape[1]):
            p_values = data[j, i, 0]
            # Obtain indices of nonzero values
            indices = np.where(p_values != 0)[0]
            # Obtain the corresponding values
            p_values = p_values[indices]
            # print(p_values)
            e_values = data[j, i, 1][indices]
            # print(e_values[:10])
            # Fit the spectrum
            bounds = ([-15.0, -15.0, 0], [15.0, 0, np.inf])
            popt, _ = curve_fit(
                spectral_fit, p_values, e_values, bounds=bounds, maxfev=2000
            )
            spectra_list.append(popt)
            # print(popt)
    return np.array(spectra_list)


def plot_example_spectra(data, coeffs):
    """This function expects input data of shape [n_sl, n_sh, n_w, n_timesteps, 2, 100]"""
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    p_values = np.logspace(-2, 3, num=100, base=10.0)
    for i in range(4):
        sl, sh, w, t = 20, 5, 4, i
        coeff_plot = coeffs[sl, sh, w, t]
        e_values = spectral_fit(p_values, *coeff_plot)
        ax[i // 2, i % 2].plot(p_values, e_values, label="Fit")
        ax[i // 2, i % 2].plot(
            data[sl, sh, w, t, 0], data[sl, sh, w, t, 1], label="Data"
        )
        ax[i // 2, i % 2].set_xscale("log")
        ax[i // 2, i % 2].set_yscale("log")
        ax[i // 2, i % 2].set_title(f"Example spectrum {i}")
        ax[i // 2, i % 2].legend()
    plt.show()


def plot_example_spectra_2(data, spectra):
    """This function expects input data of shape [n_samples, n_timesteps, 2, 100]"""

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    p_values = np.logspace(-2, 3, num=100, base=10.0)
    for i in range(4):
        offset = 1220
        coeffs = spectra[offset * 11 + i * 2]
        print(coeffs)
        e_values = spectral_fit(p_values, *coeffs)
        ax[i // 2, i % 2].plot(p_values, e_values, label="Fit")
        ax[i // 2, i % 2].plot(
            data[offset, i * 2, 0], data[offset, i * 2, 1], label="Data"
        )
        ax[i // 2, i % 2].set_xscale("log")
        ax[i // 2, i % 2].set_yscale("log")
        ax[i // 2, i % 2].set_title(f"Example spectrum {i}")
        ax[i // 2, i % 2].legend()
    plt.show()


def run(args):
    data = load_fc_spectra("spectral-data-free-cooling-large.pkl")
    # data = data.reshape(-1, 11, 2, 100, order="F")

    # # Fit and save the spectra
    # coeffs = fit_fc_spectra(data)
    # np.save("data/free_cooling/spectra_coeffs.npy", coeffs)

    # Load the coefficients
    coeffs = np.load("data/free_cooling/spectra_coeffs.npy")

    # Make some exemplary plots
    plot_example_spectra(data, coeffs)

    print("Done!")
