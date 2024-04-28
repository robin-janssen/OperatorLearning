# This script is used to train a MultiONet model on the osu dataset to evaluate interpolation (and extrapolation) performance.
# To do so, the subset of data used for trainig will be increasingly sparse -
# either by leaving out some of the intermediate timesteps or by cutting off the end of the dataset.

import numpy as np

from data import (
    create_dataloader_chemicals,
    # analyze_branca_data,
    # prepare_branca_data,
)
from training import (
    OChemicalTrainConfig,
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
    plot_losses,
)


def run(args):

    config = OChemicalTrainConfig()
    # config.device = args.device
    TRAIN = True
    args.vis = False
    config.device = args.device
    interval = 2

    # Load the data
    # data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset_1.npy")
    # data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset_2.npy")
    # print(f"Loaded Branca data with shape: {data.shape}")
    # analyze_branca_data(data)
    # train_data, test_data, timesteps = prepare_branca_data(
    #     data, train_cut=500000, test_cut=100000
    # )
    train_data = np.load("data/osu_data/train_data.npy")
    config.train_size = train_data.shape[0]
    test_data = np.load("data/osu_data/test_data.npy")
    config.test_size = test_data.shape[0]
    print(f"Loaded Osu data with shape: {train_data.shape}/{test_data.shape}")

    # Modify the data for interpolation testing
    train_data = train_data[:, ::interval]
    test_data = test_data[:, ::interval]

    full_timesteps = np.linspace(0, 99, 100)
    timesteps = full_timesteps[::interval]

    if args.vis:
        plot_chemical_examples(
            train_data,
            num_chemicals=10,
            save=True,
            title="Chemical Examples (Osu Data)",
        )
        plot_chemicals_comparative(
            train_data,
            num_chemicals=10,
            save=True,
            title="Chemical Comparison (Osu Data)",
        )

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=config.batch_size, shuffle=True
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
            f"multionet_ochemicals_interp_{interval}",
            config,
            train_loss=train_loss,
            test_loss=test_loss,
        )

    else:
        model_path = f"models/04-29/multionet_ochemicals_interp_{interval}"
        multionet, train_loss, test_loss = load_multionet(
            config, config.device, model_path
        )

    plot_losses(
        (train_loss, test_loss),
        ("Train loss", "Test loss"),
        "Losses (MultiONet on Osu Data)",
        save=True,
    )

    average_error, predictions, ground_truth = test_deeponet(
        multionet,
        dataloader_test,
        N_timesteps=config.N_timesteps,
        reshape=True,
    )

    print(f"Average prediction error: {average_error:.3E}")

    errors = np.abs(predictions - ground_truth)
    relative_errors = errors / np.abs(ground_truth)

    plot_relative_errors_over_time(
        relative_errors,
        f"Relative errors over time (MultiONet on Osu Data, Interpolation interval: {interval})",
        save=True,
    )

    plot_chemical_results(
        predictions=predictions,
        ground_truth=ground_truth,
        # names=extracted_chemicals,
        num_chemicals=10,
        model_names=f"Predictions of MultiONet on Osu Data (Interpolation interval: {interval})",
        save=True,
    )

    # plot_chemical_results_and_errors(
    #     predictions=predictions,
    #     ground_truth=ground_truth,
    #     # names=extracted_chemicals,
    #     num_chemicals=4,
    #     model_names="MultiONet",
    # )

    print("Done!")
