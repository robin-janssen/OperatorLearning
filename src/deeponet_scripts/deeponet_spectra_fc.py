# MultioNet script for predicting the time evolution of free cooling spectra.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from data import load_fc_spectra, spectrum, create_dataloader_spectra, train_test_split
from training import (
    train_deeponet_spectra,
    SpectraTrainConfig,
    save_model,
    load_deeponet_from_conf,
    test_deeponet,
)
from plotting import plot_relative_errors_over_time
from utils import check_streamlit


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
    """
    Plot exemplary spectra with their fits.

    [n_sl, n_sh, n_w, n_timesteps, 2, 100]
    """
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
    data, timesteps, _, _, _ = load_fc_spectra(
        "spectral-data-free-cooling-large.pkl", interpolate=True
    )

    config = SpectraTrainConfig()
    config.device = args.device
    config.use_streamlit = check_streamlit(config.use_streamlit)
    TRAIN = True
    FIT = False
    args.vis = True

    if FIT:
        # # Fit and save the spectra
        coeffs = fit_fc_spectra(data)
        np.save("data/free_cooling/spectra_coeffs.npy", coeffs)

        # Load the coefficients
        coeffs = np.load("data/free_cooling/spectra_coeffs.npy")

        # Make some exemplary plots of the spectra and their fits
        plot_example_spectra(data, coeffs)

    # Shuffle the data and create a train-test split
    np.random.shuffle(data)
    # data = data[:50]
    train_data, test_data = train_test_split(data, 0.8)

    # Make a dataloader for the spectral data
    dataloader_train = create_dataloader_spectra(
        train_data, timesteps, batch_size=config.batch_size, shuffle=True
    )
    dataloader_test = create_dataloader_spectra(
        test_data, timesteps, batch_size=config.batch_size, shuffle=False
    )

    if TRAIN:
        # Train the DeepONet
        deeponet, train_loss, test_loss = train_deeponet_spectra(
            config, dataloader_train, dataloader_test
        )

        # Save the trained model
        save_model(
            deeponet,
            "deeponet_spectra",
            config,
            train_loss=train_loss,
            test_loss=test_loss,
        )
    else:
        # Load the trained model
        model_path = "dummy"
        deeponet = load_deeponet_from_conf(config, args.device, model_path)

    average_error, predictions, ground_truth = test_deeponet(
        deeponet, dataloader_test, N_timesteps=config.N_timesteps
    )

    print(f"Average prediction error: {average_error:.3E}")

    errors = np.abs(predictions - ground_truth)
    relative_errors = errors / np.abs(ground_truth)
    relative_errors = relative_errors.reshape(-1, config.N_timesteps, config.N_outputs)

    plot_relative_errors_over_time(
        relative_errors,
        "Relative errors over time (MultiONet for Chemicals)",
        save=True,
    )

    print("Done!")
