# Script to train a MultioNet model for the Branca chemicals dataset.
import numpy as np

from data import (
    create_dataloader_chemicals,
    analyze_branca_data,
    prepare_branca_data,
)
from training import (
    BChemicalTrainConfig,
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

    config = BChemicalTrainConfig()
    # config.device = args.device
    TRAIN = True
    args.vis = True

    # Load the data

    data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset_3.npy")
    analyze_branca_data(data)
    train_data, test_data, timesteps = prepare_branca_data(
        data, train_cut=500000, test_cut=100000
    )

    print(f"Time steps: {timesteps}")

    print(
        f"Loaded chemical train/test data with shape: {train_data.shape}/{test_data.shape}"
    )

    if args.vis:
        plot_chemical_examples(
            train_data,
            num_chemicals=10,
            save=True,
            title="Chemical Examples (Branca Data)",
        )
        plot_chemicals_comparative(
            train_data,
            num_chemicals=10,
            save=True,
            title="Chemical Comparison (Branca Data)",
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
            "multionet_bchemicals_3",
            config,
            train_loss=train_loss,
            test_loss=test_loss,
        )

    else:
        model_path = "models/04-18/multionet_bchemicals_3"
        multionet, train_loss, test_loss = load_multionet(
            config, config.device, model_path
        )

    plot_losses(
        (train_loss, test_loss),
        ("Train loss", "Test loss"),
        "Losses (MultiONet on Branca Data)",
        save=True,
    )

    # average_error, predictions, ground_truth = test_deeponet(
    #     multionet, dataloader_test, N_timesteps=config.N_timesteps
    # )

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
        "Relative errors over time (MultiONet for Chemicals)",
        save=True,
    )

    plot_chemical_results(
        predictions=predictions,
        ground_truth=ground_truth,
        # names=extracted_chemicals,
        num_chemicals=10,
        model_names="MultiONet",
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
