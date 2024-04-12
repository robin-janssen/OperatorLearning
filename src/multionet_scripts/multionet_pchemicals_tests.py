# Script to train a MultioNet model for the Priestley chemical dataset.

import os

import numpy as np
import matplotlib.pyplot as plt

from data import create_dataloader_chemicals
from training import (
    PChemicalTrainConfig,
    train_multionet_chemical,
    save_model,
    test_deeponet,
    load_multionet,
)
from plotting import (
    plot_chemical_examples,
    plot_chemicals_comparative,
    plot_relative_errors_over_time,
    plot_chemical_results,
)


def prepare_priestley_data(train_data, test_data, train_cut=300, test_cut=100):
    """
    Prepare the Priestley data for training.
    """
    timesteps = train_data[0, :, 0]
    timesteps = np.log10(timesteps)
    train_data = train_data[:train_cut, :, :]
    test_data = test_data[:test_cut, :, :]
    train_data = np.where(train_data == 0, 1e-10, train_data)
    test_data = np.where(test_data == 0, 1e-10, test_data)
    train_data = np.log10(train_data[:, :, 1:])
    test_data = np.log10(test_data[:, :, 1:])
    return train_data, test_data, timesteps


def run(args):

    config = PChemicalTrainConfig()
    config.device = args.device
    TRAIN = False
    args.vis = True

    data_folder = "data/chemicals_priestley"

    # Loading from the numpy array
    train_data = np.load(os.path.join(data_folder, "chemicals_train.npy"))
    test_data = np.load(os.path.join(data_folder, "chemicals_test.npy"))
    print(
        f"Loaded chemical train/test data with shape: {train_data.shape}/{test_data.shape}"
    )

    train_data, test_data, timesteps = prepare_priestley_data(
        train_data, test_data, train_cut=200, test_cut=50
    )

    if args.vis:
        fig, ax = plt.subplots()
        im = ax.imshow(train_data[1, :, :].T, aspect="auto")
        plt.colorbar(im, ax=ax)  # Add a colorbar to the current plot
        plt.show()

        plot_chemical_examples(train_data, num_chemicals=20)
        plot_chemicals_comparative(train_data, num_chemicals=20)

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=config.batch_size, shuffle=False
    )

    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=config.batch_size, shuffle=False
    )

    if TRAIN:
        multionet, train_loss, test_loss = train_multionet_chemical(
            config, dataloader_train, dataloader_test
        )

        # Save the MulitONet
        save_model(
            multionet,
            "multionet_pchemicals_opt1",
            config,
            train_loss=train_loss,
            test_loss=test_loss,
        )

    else:
        model_path = "models/04-07/multionet_pchemicals_opt1"
        multionet, train_loss, test_loss = load_multionet(
            config, config.device, model_path
        )

    # average_error, predictions, ground_truth = test_deeponet(
    #     multionet, dataloader_test, N_timesteps=config.N_timesteps
    # )

    average_error, predictions, ground_truth = test_deeponet(
        multionet, dataloader_train, N_timesteps=config.N_timesteps
    )

    print(f"Average prediction error: {average_error:.3E}")

    errors = np.abs(predictions - ground_truth)
    relative_errors = errors / np.abs(ground_truth)
    relative_errors = relative_errors.reshape(-1, config.N_timesteps, config.N_outputs)

    plot_relative_errors_over_time(
        relative_errors, "Relative errors over time (MultiONet for Chemicals)"
    )

    plot_chemical_results(
        predictions=predictions,
        ground_truth=ground_truth,
        # names=extracted_chemicals,
        num_chemicals=10,
        model_names="MultiONet",
    )

    # plot_chemical_results_and_errors(
    #     predictions=predictions,
    #     ground_truth=ground_truth,
    #     # names=extracted_chemicals,
    #     num_chemicals=4,
    #     model_names="MultiONet",
    # )

    print("Done!")
