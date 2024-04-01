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
)
from plotting import (
    plot_chemical_examples,
    plot_chemicals_comparative,
    plot_relative_errors_over_time,
)


def run(args):

    data_folder = "data/chemicals_priestley"

    # Loading from the numpy array
    train_data = np.load(os.path.join(data_folder, "chemicals_train.npy"))
    test_data = np.load(os.path.join(data_folder, "chemicals_test.npy"))
    print(
        f"Loaded chemical train/test data with shape: {train_data.shape}/{test_data.shape}"
    )

    # For now: Use only a subset of the data
    train_data = train_data[:2000]
    test_data = test_data[:200]
    # train_data[:, 0, 3]

    timesteps = train_data[0, :, 1]
    train_data = np.where(train_data == 0, 1e-10, train_data)
    test_data = np.where(test_data == 0, 1e-10, test_data)
    train_data = np.log10(train_data[:, :, 1:])
    test_data = np.log10(test_data[:, :, 1:])

    # data_transformed = np.where(
    #     np.isnan(train_data), 2, np.where(train_data == -np.inf, 1, 0)
    # )
    data_transformed = train_data
    # Plotting the first channel with colorbar
    fig, ax = plt.subplots()
    im = ax.imshow(data_transformed[1, :, :].T, aspect="auto")
    plt.colorbar(im, ax=ax)  # Add a colorbar to the current plot
    plt.show()

    if args.vis:
        plot_chemical_examples(train_data, num_chemicals=20)
        plot_chemicals_comparative(train_data, num_chemicals=20)

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=128, shuffle=True
    )

    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=128, shuffle=False
    )

    config = PChemicalTrainConfig(
        data_loader=dataloader_train, test_loader=dataloader_test
    )
    config.device = args.device

    multionet, train_loss, test_loss = train_multionet_chemical(config)

    # Save the MulitONet
    save_model(
        multionet,
        "multionet_pchemicals",
        config,
        train_loss=train_loss,
        test_loss=test_loss,
    )

    average_error, predictions, ground_truth = test_deeponet(
        multionet, dataloader_test, N_timesteps=config.N_timesteps
    )

    errors = np.abs(predictions - ground_truth)
    relative_errors = errors / np.abs(ground_truth)

    plot_relative_errors_over_time(
        relative_errors, "Relative errors over time (MultiONet for Chemicals)"
    )

    print("Done!")